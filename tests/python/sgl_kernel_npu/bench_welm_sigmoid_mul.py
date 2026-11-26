"""Compare WeLM's unchanged in-place Triton sigmoid-mul with the 950 Ascend C op.

No SGLang server or process group is needed. --batch-sizes counts tokens;
--heads is the number of LOCAL attention heads, after attention TP partitioning.
"""

import argparse
import ast
import hashlib
import json
import statistics
import sys
import types
from pathlib import Path

import torch
import torch_npu
import sgl_kernel_npu

# Reuse the existing event timing and optional graph profiler, without changing
# the pack benchmark or its baseline. Always time ONE replay before resetting.
from bench_welm_pack_decode_inputs import profile_graphs, time_graph


def load_triton(path):
    import triton
    import triton.language as tl

    source = path.read_text(encoding="utf-8")
    names = {"sigmoid_mul_kernel_npu", "inplace_sigmoid_mul_npu"}
    selected = [
        node for node in ast.parse(source, filename=str(path)).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != names:
        raise RuntimeError(f"Cannot find the original WeLM sigmoid-mul functions in {path}")
    module = types.ModuleType("_welm_original_sigmoid_mul")
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    sys.modules[module.__name__] = module
    # Preserve original source locations/decorators for Triton inspect/JIT.
    # Importing the whole model module would require unrelated server operators.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.inplace_sigmoid_mul_npu


def make_inputs(batch, heads, device, seed=2026, special=False):
    generator = torch.Generator().manual_seed(seed)
    gate = (torch.randn(batch, heads, 1, generator=generator) * 3).bfloat16()
    attention = torch.randn(batch, heads, 256, generator=generator).bfloat16()
    if special and gate.numel():
        values = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, 8.0, -8.0, 32.0, -32.0,
             float("inf"), -float("inf"), float("nan"), -90.0],
            dtype=torch.bfloat16,
        )
        count = min(gate.numel(), values.numel())
        gate.view(-1)[:count] = values[:count]
        attention.view(-1)[:4] = torch.tensor(
            [0.0, -0.0, float("inf"), float("nan")], dtype=torch.bfloat16
        )
    return gate.to(device), attention.to(device)


def reference(gate, attention):
    # Preserve both rounding boundaries in the actual WeLM definition.
    return torch.sigmoid(gate.cpu().float()).bfloat16() * attention.cpu()


def assert_result(actual, expected):
    actual = actual.cpu()
    assert actual.dtype == torch.bfloat16 and actual.shape == expected.shape
    # Check non-finites independently; relative tolerances can hide these bugs.
    assert torch.equal(torch.isnan(actual), torch.isnan(expected)), "NaN locations differ"
    assert torch.equal(torch.isinf(actual), torch.isinf(expected)), "Inf locations differ"
    infinite = torch.isinf(expected)
    assert torch.equal(actual[infinite], expected[infinite]), "Inf signs differ"
    finite = torch.isfinite(expected)
    a_bits = actual.view(torch.int16).int()
    e_bits = expected.view(torch.int16).int()
    # Monotonic encoding across the sign boundary, including +/- zero.
    a_order = torch.where(a_bits < 0, -32768 - a_bits, a_bits)
    e_order = torch.where(e_bits < 0, -32768 - e_bits, e_bits)
    max_ulp = int((a_order[finite] - e_order[finite]).abs().max()) if finite.any() else 0
    assert max_ulp <= 1, f"more than one BF16 ULP from the reference: {max_ulp}"
    return max_ulp


def check_eager(triton_fn, device):
    functional = torch.ops.npu.welm_sigmoid_mul.default
    inplace = torch.ops.npu.welm_sigmoid_mul_.default
    # Tail heads, empty/maximum token batches and the production TP1/2/4 shapes.
    cases = [(0, 24), (1, 1), (1, 3), (1, 6), (2, 12),
             (4, 24), (6, 24), (8, 24), (128, 24)]
    for batch, heads in cases:
        gate, attention = make_inputs(batch, heads, device, special=True)
        before = attention.clone()
        gate_before = gate.clone()
        expected = reference(gate, attention)
        output = functional(gate, attention)
        assert_result(output, expected)
        if (batch, heads) == (4, 24):
            # Isolate the mandatory intermediate BF16 rounding from possible
            # Exp/Div differences. A single-round fused product fails this
            # exact check even if it is within one ULP of the CPU reference.
            scale = functional(gate, torch.ones_like(attention)).cpu()
            staged = scale * before.cpu()
            actual = output.cpu()
            finite = torch.isfinite(staged)
            assert torch.equal(actual.view(torch.int16)[finite], staged.view(torch.int16)[finite]), (
                "sigmoid must round to BF16 before multiplication"
            )
        if attention.numel():
            assert output.data_ptr() != attention.data_ptr(), "functional output aliases input"
        assert torch.equal(attention.view(torch.int16), before.view(torch.int16))
        output = inplace(gate, attention)
        assert output.data_ptr() == attention.data_ptr(), "in-place output lost aliasing"
        assert_result(attention, expected)
        assert torch.equal(gate.view(torch.int16), gate_before.view(torch.int16))
        if batch:  # The original Triton wrapper has no zero-grid guard.
            attention.copy_(before)
            triton_fn(gate, attention)
            assert_result(attention, expected)
    meta_gate = torch.empty((4, 12, 1), dtype=torch.bfloat16, device="meta")
    meta_attn = torch.empty((4, 12, 256), dtype=torch.bfloat16, device="meta")
    assert functional(meta_gate, meta_attn).shape == meta_attn.shape
    assert inplace(meta_gate, meta_attn).shape == meta_attn.shape


def native_function(mode):
    if mode == "graph":
        return torch.ops.npu.welm_sigmoid_mul_.default

    def region(gate, attention):
        # The compiled attention tail consumes a fresh result. Use the
        # functional schema so AOT need not functionalize an alias-returning
        # custom in-place op. Both schemas launch exactly the same kernel.
        return torch.ops.npu.welm_sigmoid_mul(gate, attention)

    try:
        import npugraph_ex  # noqa: F401
    except ImportError:
        from torch_npu.dynamo import npugraph_ex  # noqa: F401
    return torch.compile(
        region, backend="npugraph_ex", fullgraph=True, dynamic=False,
        options={
            "force_eager": True,
            "static_kernel_compile": True,
            "super_kernel_optimize": mode == "superkernel",
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
        },
    )


def reset_inputs(state, base, stream):
    with torch.npu.stream(stream):
        state.copy_(base.unsqueeze(0).expand_as(state))


def capture(fn, gate, base, copies, stream, superkernel=False):
    state = torch.empty((copies, *base.shape), dtype=base.dtype, device=base.device)
    stream.wait_stream(torch.npu.current_stream())
    reset_inputs(state, base, stream)
    with torch.npu.stream(stream):
        for _ in range(5):
            fn(gate, state[0])
    stream.synchronize()
    reset_inputs(state, base, stream)
    graph = torch.npu.NPUGraph()
    outputs = []
    with torch.npu.graph(graph, stream=stream):
        for index in range(copies):
            scope = f"welm_sigmoid_mul_{index}"
            if superkernel:
                torch.npu.super_kernel_scope_begin(scope)
            result = fn(gate, state[index])
            if superkernel:
                torch.npu.super_kernel_scope_end(scope)
            outputs.append(state[index] if result is None else result)
    if superkernel:
        graph.super_kernel_optimize()
    stream.synchronize()
    assert len({state[i].data_ptr() for i in range(copies)}) == copies
    assert len({output.data_ptr() for output in outputs}) == copies
    return graph, (state, outputs)


def benchmark(batch, args, triton_fn, native_fn, device, stream):
    gate, base = make_inputs(batch, args.heads, device)
    torch.npu.synchronize()
    graphs = {
        "triton": capture(triton_fn, gate, base, args.copies, stream),
        "ascendc": capture(native_fn, gate, base, args.copies, stream, args.mode == "superkernel"),
    }
    # Change values while retaining capture addresses. This checks that static
    # compilation does not freeze inputs and that mutable graph outputs survive.
    new_gate, base = make_inputs(batch, args.heads, device, seed=2027)
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        gate.copy_(new_gate)
    stream.synchronize()
    expected = reference(new_gate, base)
    errors = {}
    for name, (graph, (state, outputs)) in graphs.items():
        reset_inputs(state, base, stream)
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        errors[name] = max(assert_result(outputs[0], expected), assert_result(outputs[-1], expected))
        if name == "ascendc" and args.mode != "graph":
            assert outputs[0].data_ptr() != state[0].data_ptr(), "functional graph output aliases input"
            assert torch.equal(state[0], base), "functional graph modified input"
        for _ in range(5):
            reset_inputs(state, base, stream)
            with torch.npu.stream(stream):
                graph.replay()
        stream.synchronize()

    samples = {name: [] for name in graphs}
    for trial in range(args.rounds):
        order = ("triton", "ascendc") if trial % 2 == 0 else ("ascendc", "triton")
        for name in order:
            graph, (state, _) = graphs[name]
            round_samples = []
            for _ in range(args.replays):
                # Triton and graph-mode native mutate attention. Reset both
                # inputs OUTSIDE every timed replay, preventing products from
                # converging to zeros/subnormals during performance measurement.
                reset_inputs(state, base, stream)
                round_samples.append(time_graph(graph, args.copies, 1, stream))
            samples[name].append(statistics.median(round_samples))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    speedup = medians["triton"] / medians["ascendc"]
    result = {
        "batch_size": batch, "local_heads": args.heads, "head_rows": batch * args.heads,
        "head_dim": 256, "mode": args.mode,
        "native_contract": "inplace" if args.mode == "graph" else "functional",
        "median_us": medians, "samples_us": samples,
        "speedup": speedup, "performance_pass": speedup > args.min_speedup,
        "max_bf16_ulp": errors,
    }
    if args.profile_dir:
        for _, (state, _) in graphs.values():
            reset_inputs(state, base, stream)
        profile_graphs(graphs, args.profile_dir / f"b{batch}_h{args.heads}", stream)
    return result


def main():
    root = Path(__file__).resolve().parents[3]
    relative = Path("python/sglang/srt/layers/welmv4_npu_op.py")
    candidates = [root.parent / "sglang" / relative, root.parent / "NEWSGLANG/sglang" / relative]
    default_source = next((path for path in candidates if path.is_file()), candidates[0])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--heads", type=int, default=24, choices=(1, 2, 3, 4, 6, 8, 12, 24))
    parser.add_argument("--copies", type=int, default=128)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mode", choices=("graph", "static", "superkernel"), default="graph")
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--triton-source", type=Path, default=default_source)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if any(not 1 <= batch <= 128 for batch in args.batch_sizes):
        parser.error("timing requires 1 <= batch size <= 128")
    if min(args.copies, args.replays, args.rounds) <= 0 or args.min_speedup < 1.0:
        parser.error("counts must be positive and --min-speedup must be >= 1")
    if not args.triton_source.is_file():
        parser.error("use --triton-source to point to the real welmv4_npu_op.py")
    if not hasattr(torch.ops.npu, "welm_sigmoid_mul_"):
        parser.error("install the new Ascend 950 sgl-kernel-npu wheel before benchmarking")
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    triton_fn = load_triton(args.triton_source.resolve())
    check_eager(triton_fn, device)
    stream = torch.npu.Stream(device=device)
    results = []
    for batch in args.batch_sizes:
        if args.mode != "graph":
            # Shapes are measured independently. Avoid hitting Dynamo's
            # recompilation limit when sweeping all nine Decode batch sizes.
            torch.compiler.reset()
        native_fn = native_function(args.mode)
        result = benchmark(batch, args, triton_fn, native_fn, device, stream)
        results.append(result)
        print(json.dumps(result), flush=True)
    library = Path(sgl_kernel_npu.__file__).parent / "lib/libsgl_kernel_npu.so"
    report = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "native_library": str(library),
        "native_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "triton_source": str(args.triton_source.resolve()),
        "triton_source_sha256": hashlib.sha256(args.triton_source.read_bytes()).hexdigest(),
        "copies": args.copies, "replays": args.replays, "rounds": args.rounds,
        "timing": "NPU events, independent calls, reset inputs before every timed graph replay",
        "results": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(result["performance_pass"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
