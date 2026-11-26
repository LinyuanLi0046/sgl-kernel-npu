"""Compare the real DeepEP Triton pack with WeLM Ascend C packing on an NPU.

Default workload: BF16 hidden[4,2048], INT32 ids[4,10], FP32 weights[4,10],
capacity 128. No distributed process group or model checkpoint is required.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import torch_npu

import sgl_kernel_npu  # noqa: F401 -- registers the compiled Torch operator


def load_triton_pack(path):
    spec = importlib.util.spec_from_file_location("_welm_original_ops_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pack_low_latency_inputs


def make_inputs(rows, device, seed):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn((rows, 2048), generator=generator, dtype=torch.bfloat16)
    ids = torch.randint(0, 512, (rows, 10), generator=generator, dtype=torch.int32)
    weights = torch.rand((rows, 10), generator=generator, dtype=torch.float32)
    # Special bit patterns catch unintended numeric conversions in a copy op.
    if rows:
        hidden.view(torch.int16)[0, :4] = torch.tensor(
            [-32768, 0x7FC1, 0x7F80, -128], dtype=torch.int16
        )
        weights.view(torch.int32)[0, :4] = torch.tensor(
            [-2147483648, 0x7FC01234, 0x7F800000, -8388608], dtype=torch.int32
        )
    return tuple(t.to(device) for t in (hidden, ids, weights))


def assert_pack(inputs, outputs, capacity):
    for original, result, bit_dtype in zip(
        inputs, outputs, (torch.int16, torch.int32, torch.int32)
    ):
        assert result.shape == (capacity, original.shape[1])
        assert result.dtype == original.dtype and result.device == original.device
        expected = torch.zeros((capacity, original.shape[1]), dtype=bit_dtype)
        expected[: original.shape[0]].copy_(original.cpu().view(bit_dtype))
        actual = result.cpu().view(bit_dtype)
        assert torch.equal(actual, expected), (
            f"packing mismatch: N={original.shape[0]}, C={capacity}, dtype={original.dtype}"
        )


def check_eager(triton_pack, native_pack, device):
    # C=7 exercises non-32-byte route output sizes. N=0 is a participating
    # empty rank; N==C checks the functional Torch op's fresh-output contract.
    cases = (
        (0, 128), (1, 128), (4, 128), (4, 7),
        (7, 7), (16, 128), (127, 128), (128, 128),
    )
    for rows, capacity in cases:
        inputs = make_inputs(rows, device, 1000 + rows)
        before = tuple(t.clone() for t in inputs)
        for fn in (triton_pack, native_pack):
            result = fn(*inputs, capacity)
            assert_pack(inputs, result, capacity)
            if fn is native_pack:
                assert all(out.data_ptr() != src.data_ptr() for src, out in zip(inputs, result))
            for old, current, bit_dtype in zip(
                before, inputs, (torch.int16, torch.int32, torch.int32)
            ):
                assert torch.equal(old.view(bit_dtype), current.view(bit_dtype)), (
                    "input was modified"
                )


def compile_native(mode):
    fn = torch.ops.npu.welm_pack_decode_inputs.default
    if mode == "graph":
        return fn

    # Import explicitly so both torch_npu-packaged and standalone distributions
    # can register their backend, matching the project's regional-compile path.
    try:
        import npugraph_ex  # noqa: F401
    except ImportError:
        from torch_npu.dynamo import npugraph_ex  # noqa: F401

    def region(hidden, ids, weights, capacity):
        return torch.ops.npu.welm_pack_decode_inputs(hidden, ids, weights, capacity)

    return torch.compile(
        region,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options={
            "force_eager": True,
            "static_kernel_compile": True,
            "super_kernel_optimize": mode == "superkernel",
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
        },
    )


def capture(fn, inputs, capacity, copies, stream, superkernel=False):
    with torch.npu.stream(stream):
        for _ in range(5):
            fn(*inputs, capacity)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    keepalive = []
    with torch.npu.graph(graph, stream=stream):
        # Each invocation is explicitly scoped; do not benchmark one huge SK
        # made by fusing all independent packing repetitions together.
        for index in range(copies):
            scope = f"welm_pack_decode_{index}"
            if superkernel:
                torch.npu.super_kernel_scope_begin(scope)
            result = fn(*inputs, capacity)
            if superkernel:
                torch.npu.super_kernel_scope_end(scope)
            keepalive.append(result)
    if superkernel:
        graph.super_kernel_optimize()
    stream.synchronize()
    for column in zip(*keepalive):
        assert len({tensor.data_ptr() for tensor in column}) == copies, (
            "captured calls reused output storage; this would bias the comparison"
        )
    return graph, keepalive


def time_graph(graph, copies, repeats, stream):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    with torch.npu.stream(stream):
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (copies * repeats)


def profile_graphs(graphs, output, stream):
    output.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
    ):
        for name, (graph, _) in graphs.items():
            with torch.npu.stream(stream), torch.autograd.profiler.record_function(name):
                graph.replay()
            torch.npu.synchronize()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument(
        "--copies", type=int, default=128, help="independent calls per captured graph"
    )
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mode", choices=("graph", "static", "superkernel"), default="graph")
    parser.add_argument("--target-us", type=float, default=2.0)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, help="optional result file")
    parser.add_argument(
        "--triton-source", type=Path,
        default=root / "python/deep_ep/deep_ep/ops_padding.py",
    )
    args = parser.parse_args()
    if not 0 <= args.rows < args.capacity <= 128:
        parser.error(
            "timing requires 0 <= rows < capacity <= 128; N==C normally bypasses packing"
        )
    if min(args.copies, args.replays, args.rounds) <= 0:
        parser.error("copies, replays and rounds must be positive")

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    triton_pack = load_triton_pack(args.triton_source)
    raw_native = torch.ops.npu.welm_pack_decode_inputs.default
    check_eager(triton_pack, raw_native, device)
    inputs = make_inputs(args.rows, device, 2026)
    native = compile_native(args.mode)
    stream = torch.npu.Stream(device=device)
    torch.npu.synchronize()
    graphs = {
        "triton": capture(triton_pack, inputs, args.capacity, args.copies, stream),
        "ascendc": capture(
            native, inputs, args.capacity, args.copies, stream,
            args.mode == "superkernel",
        ),
    }

    # Same captured addresses, new contents: neither the Meta function nor
    # static compilation may freeze the first input values.
    updated = make_inputs(args.rows, device, 2027)
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        for dst, src in zip(inputs, updated):
            dst.copy_(src)
    for graph, outputs in graphs.values():
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        # Check the first and last invocation; the remaining identical calls
        # only increase the event timing window, not correctness coverage.
        assert_pack(inputs, outputs[0], args.capacity)
        if len(outputs) > 1:
            assert_pack(inputs, outputs[-1], args.capacity)
        with torch.npu.stream(stream):
            for _ in range(10):
                graph.replay()
        stream.synchronize()

    samples = {name: [] for name in graphs}
    for trial in range(args.rounds):
        # Alternate measurement order to reduce a systematic warm-cache/order bias.
        order = ("triton", "ascendc") if trial % 2 == 0 else ("ascendc", "triton")
        for name in order:
            samples[name].append(
                time_graph(graphs[name][0], args.copies, args.replays, stream)
            )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    passed = medians["ascendc"] < min(medians["triton"], args.target_us)
    report = {
        "device": torch.npu.get_device_name(device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "sgl_kernel_npu": sgl_kernel_npu.__file__,
        "triton_source": str(args.triton_source.resolve()),
        "mode": args.mode,
        "rows": args.rows,
        "capacity": args.capacity,
        "hidden": 2048,
        "topk": 10,
        "experts": 512,
        "ep": "no collective; same packing for EP 4/8/16/32",
        "copies_per_graph": args.copies,
        "replays_per_sample": args.replays,
        "measurement": "NPU event graph time / call count; includes graph task gaps",
        "median_us": medians,
        "samples_us": samples,
        "speedup": medians["triton"] / medians["ascendc"],
        "target_us": args.target_us,
        "below_target_and_faster_than_triton": passed,
    }
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.profile_dir:
        profile_graphs(graphs, args.profile_dir, stream)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
