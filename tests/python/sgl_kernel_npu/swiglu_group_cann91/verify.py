"""Collect SwigluGroup execution and static/SuperKernel evidence on CANN 9.1.

Each mode runs in a fresh process/directory. Execution PASS never implies that
static-kernel selection or SuperKernel fusion succeeded; see README.md.
"""

import argparse
import csv
import importlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def collect_evidence(root, args):
    """Inventory observations only: no inference from an enabled option to success."""
    result = {"swiglu_compile_descriptions": [], "static_packages": [],
              "swiglu_binary_metadata": [], "sk_metadata": [], "device_kernel_rows": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        lower = rel.lower()
        if "static_kernel" in lower and path.suffix == ".run":
            result["static_packages"].append(rel)
        if path.suffix == ".json" and "swiglu" in lower:
            key = ("swiglu_compile_descriptions" if "opcompile" in lower or "_debug" in lower
                   else "swiglu_binary_metadata")
            result[key].append(rel)
        if "sk_meta" in lower:
            result["sk_metadata"].append(rel)
        if path.name == "kernel_details.csv":
            with path.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    result["device_kernel_rows"].append({"file": rel, **row})
    rows = result["device_kernel_rows"]
    ordinary = [row for row in rows if row.get("Type", "").lower() == "swiglugroup"]
    fused = [row for row in rows if row.get("Type", "").lower() == "superkernel"
             and "welm_swiglu_probe" in row.get("Name", "")
             and "swiglugroup" in row.get("Name", "").lower()]
    expected = args.replays * args.chain
    artifacts = bool(result["swiglu_compile_descriptions"] and result["static_packages"])
    ordinary_static = (len(ordinary) == expected
                       and all(row.get("OP State", "").lower() == "static" for row in ordinary))
    # This scope contains only SwigluGroup calls, and profiling contains only replays.
    # Fewer kernels alone is insufficient: also require the device Type, scope and op identity.
    fused_expected = expected if args.max_core_debug else args.replays
    fused_static = (len(fused) == fused_expected and not ordinary
                    and all(row.get("OP State", "").lower() == "static" for row in fused))
    result["static_execution"] = "CONFIRMED" if artifacts and (ordinary_static or fused_static) else "UNCONFIRMED"
    result["superkernel_execution"] = (
        ("CONFIRMED_PER_OP_DEBUG" if args.max_core_debug else "CONFIRMED")
        if artifacts and fused_static else "UNCONFIRMED"
    )
    result["observed_swiglu_tasks"] = len(ordinary)
    result["observed_scoped_superkernel_tasks"] = len(fused)
    result["expected_calls_per_replay"] = args.chain
    result["note"] = ("Unknown profiler formats/names remain UNCONFIRMED, not unsupported. "
                      "Review logs, binary metadata and device trace; never infer from config alone.")
    write_json(root / "evidence.json", result)
    return result


def run_device(args, root):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device_name = torch.npu.get_device_name()
    if not device_name.startswith("Ascend950"):
        raise RuntimeError(f"SwigluGroup requires Ascend950, got {device_name}")
    cann_version = str(torch_npu.npu.utils.get_cann_version())
    if not cann_version.startswith("9.1."):
        raise RuntimeError(f"Expected CANN 9.1.x, got {cann_version}")
    package = importlib.import_module(args.torch_package)
    op = torch.ops.cann_ops_nn.swiglu_group.default
    environment = {"python": sys.executable, "torch": torch.__version__,
                   "torch_npu": torch_npu.__version__, "CANN": cann_version,
                   "device": device_name, "package": package.__file__,
                   "schema": str(op._schema), "arguments": vars(args),
                   "ASCEND_CUSTOM_OPP_PATH": os.getenv("ASCEND_CUSTOM_OPP_PATH"),
                   "ASCEND_HOME_PATH": os.getenv("ASCEND_HOME_PATH")}
    write_json(root / "environment.json", environment)

    def infer_one_step(x, counts):
        y = torch.ops.cann_ops_nn.swiglu_group(x, group_index=counts, clamp_limit=-1.0)
        if args.chain == 2:
            # A real producer/consumer dependency, with no unrelated compute in the scope.
            y = torch.ops.cann_ops_nn.swiglu_group(y, group_index=counts, clamp_limit=-1.0)
        return y

    def inputs(valid, seed):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(args.rows, 2 * args.intermediate_size, generator=generator).bfloat16()
        x[valid:] = float("nan")  # Padding must never be consumed as active rows.
        counts = torch.full((args.groups,), valid // args.groups, dtype=torch.int64)
        counts[: valid % args.groups] += 1
        return x, counts

    def reference(x, valid):
        result = x[:valid]
        for _ in range(args.chain):
            a, b = result.float().chunk(2, dim=-1)
            result = (torch.nn.functional.silu(a) * b).bfloat16()
        return result

    def check(output, expected, valid):
        if output.dtype != torch.bfloat16 or tuple(output.shape) != (
            args.rows, 2 * args.intermediate_size // (2 ** args.chain)
        ):
            raise AssertionError("Unexpected output shape/dtype")
        # Only the active prefix is defined by SwigluGroup's contract.
        actual = output[:valid].cpu()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
        return float((actual.float() - expected.float()).abs().max())

    with torch.no_grad():
        host_x, host_counts = inputs(args.valid_rows, 2026)
        dummy_input = host_x.to(args.device)
        counts = host_counts.to(args.device)
        expected = reference(host_x, args.valid_rows)
        # Complete the JIT Torch bridge and lazy initialization outside compilation/capture.
        output = infer_one_step(dummy_input, counts)
        check(output, expected, args.valid_rows)
        config = {}
        compiled_model = infer_one_step
        if args.mode in ("static", "superkernel"):
            try:
                backend_module = importlib.import_module("npugraph_ex")
            except ModuleNotFoundError as exc:
                if exc.name != "npugraph_ex":
                    raise
                backend_module = importlib.import_module("torch_npu.dynamo.npugraph_ex")
            environment["npugraph_ex"] = backend_module.__file__
            write_json(root / "environment.json", environment)
            # Preserve the actual installed implementation alongside the diagnostic log.
            utils = importlib.import_module(backend_module.__name__ + ".core.utils")
            utils.logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(root / "npugraph_ex.log", encoding="utf-8")
            utils.logger.addHandler(handler)
            static_source = Path(backend_module.__file__).parent / "_acl_concrete_graph/static_kernel.py"
            if static_source.is_file():
                shutil.copy2(static_source, root / "installed_static_kernel.py")
            config = {"force_eager": True, "static_kernel_compile": True,
                      "super_kernel_optimize": args.mode == "superkernel",
                      "inplace_pass": False, "clone_input": False, "clone_output": False}
            if args.max_core_debug:
                config["super_kernel_debug_options"] = {"debug_per_op_max_core_num": 1}
            compiled_model = torch.compile(infer_one_step, backend="npugraph_ex",
                                           fullgraph=True, dynamic=False, options=config)
        write_json(root / "compile_options.json", config)

        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                output = compiled_model(dummy_input, counts)
        stream.synchronize()
        graph = None
        if args.mode != "eager":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                if args.mode == "superkernel":
                    torch.npu.super_kernel_scope_begin("welm_swiglu_probe")
                output = compiled_model(dummy_input, counts)
                if args.mode == "superkernel":
                    torch.npu.super_kernel_scope_end("welm_swiglu_probe")
            if args.mode == "superkernel":
                graph.super_kernel_optimize()
            stream.synchronize()

        checks = []
        # Same addresses/shapes, changed values AND device counts. Check growing and
        # shrinking prefixes; a frozen token count or stale graph output must fail.
        for index, valid in enumerate(dict.fromkeys((args.valid_rows, 1, args.rows))):
            host_x, host_counts = inputs(valid, 2027 + index)
            with torch.npu.stream(stream):
                dummy_input.copy_(host_x)
                counts.copy_(host_counts)
                if graph is None:
                    output = compiled_model(dummy_input, counts)
                else:
                    output.fill_(float("nan"))
                    graph.replay()
            stream.synchronize()
            error = check(output, reference(host_x, valid), valid)
            checks.append({"valid_rows": valid, "max_abs_error": error})

        # Restore the requested active rows before the only profiled replay region.
        with torch.npu.stream(stream):
            dummy_input.copy_(inputs(args.valid_rows, 2026)[0])
            counts.copy_(inputs(args.valid_rows, 2026)[1])
            for _ in range(3):
                if graph is None:
                    output = compiled_model(dummy_input, counts)
                else:
                    graph.replay()
        stream.synchronize()
        experimental = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            experimental_config=experimental,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root / "profile")),
        ) as prof:
            for index in range(args.replays):
                with torch.npu.stream(stream), torch.autograd.profiler.record_function(
                    f"swiglu_{args.mode}_replay_{index}"
                ):
                    if graph is None:
                        output = compiled_model(dummy_input, counts)
                    else:
                        graph.replay()
                stream.synchronize()
                prof.step()
        check(output, expected, args.valid_rows)
        # Verify which custom library was actually loaded, not merely which env var was set.
        maps = Path("/proc/self/maps")
        if maps.is_file():
            (root / "loaded_libraries.txt").write_text("\n".join(
                line for line in maps.read_text().splitlines()
                if any(token in line for token in ("libcust_opapi", "liboptiling", "swiglu_group", "libascendsk"))
            ) + "\n", encoding="utf-8")
    return {"execution": "PASS", "checks": checks, "profile_replays": args.replays,
            "note": "PASS covers execution/precision only; review evidence.json and README for optimization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "eager", "graph", "static", "superkernel"), default="all")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing paths are rejected")
    parser.add_argument("--torch-package", default=os.getenv("WELM_SWIGLU_TORCH_PACKAGE", "cann_ops_nn_welm_swiglu91"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, default=64, help="Allocated rows of GMM1 output, not source decode BS")
    parser.add_argument("--valid-rows", type=int, default=40, help="Sum of per-expert received-token counts")
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--intermediate-size", type=int, default=2048, help="Local FFN width I; first input is [rows, 2*I]")
    parser.add_argument("--chain", type=int, choices=(1, 2), default=2)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--max-core-debug", action="store_true", help="Superkernel-only diagnostic; disables joint fusion")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (0 < args.valid_rows <= args.rows and args.groups > 0 and args.replays > 0
            and args.intermediate_size > 0 and (args.chain == 1 or args.intermediate_size % 2 == 0)):
        parser.error("Require 0 < valid-rows <= rows, positive dimensions/replays, even I for chain=2")
    if args.max_core_debug and args.mode != "superkernel":
        parser.error("--max-core-debug requires --mode superkernel")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if not args.worker:
        modes = ("eager", "graph", "static", "superkernel") if args.mode == "all" else (args.mode,)
        results = {}
        for mode in modes:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode,
                       "--output", str(root / mode), "--torch-package", args.torch_package,
                       "--device", args.device, "--rows", str(args.rows), "--valid-rows", str(args.valid_rows),
                       "--groups", str(args.groups), "--intermediate-size", str(args.intermediate_size),
                       "--chain", str(args.chain), "--replays", str(args.replays)]
            if args.max_core_debug:
                command.append("--max-core-debug")
            with (root / f"{mode}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                results[mode] = process.wait()
            print(f"{mode}: exit={results[mode]}, log={root / (mode + '.log')}", flush=True)
        write_json(root / "exit_codes.json", results)
        return int(any(results.values()))

    os.chdir(root)
    # Set before importing torch/torch_npu. No process group or launch-script rank is used.
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(root / "torch_compile_debug")
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "1"
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
    os.environ["ASCEND_PROCESS_LOG_PATH"] = str(root / "plog")
    (root / "plog").mkdir()
    try:
        result = run_device(args, root)
    except Exception:
        result = {"execution": "FAIL", "traceback": traceback.format_exc()}
        print(result["traceback"], file=sys.stderr)
    write_json(root / "result.json", result)
    evidence = collect_evidence(root, args)
    result["static_execution"] = evidence["static_execution"]
    result["superkernel_execution"] = evidence["superkernel_execution"]
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)
    if result["execution"] != "PASS":
        return 1
    if args.mode == "static" and evidence["static_execution"] != "CONFIRMED":
        return 2
    if args.mode == "superkernel" and not evidence["superkernel_execution"].startswith("CONFIRMED"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
