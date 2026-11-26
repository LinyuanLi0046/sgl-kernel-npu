"""Collect WeLM packing with msprof op / op simulator and create a tar.gz.

Run on the Linux NPU server with the rebuilt wheel installed:
    python tests/python/sgl_kernel_npu/collect_welm_pack_profiles.py

Defaults: rows 1 2 4 6 8, capacity 8, Ascend C + Triton, both collectors.
This collects individual eager kernels, not the benchmark's graph median_us.
The parent uses only the standard library; --dry-run also works without CANN.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
KERNEL_NAMES = {
    "ascendc": "welm_pack_decode_inputs_kernel",
    "triton": "_pack_low_latency_inputs_kernel",
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command, directory, env=None, timeout=60, name="run"):
    """Keep errors and partial output; terminate the owned process group on timeout."""
    record = {"argv": command, "cwd": str(directory)}
    (directory / f"{name}.command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    started = time.monotonic()
    with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        except OSError as error:
            record.update(returncode=None, error=str(error))
            log.write(str(error) + "\n")
        else:
            try:
                record["returncode"] = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                # SIGINT gives msprof a chance to export already collected data.
                for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
                    if process.poll() is not None:
                        break
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            break
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        continue
                process.wait()
                if os.name == "posix":
                    # A profiler can exit before its application; do not let
                    # timed-out children overlap the next collection.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if isinstance(error, KeyboardInterrupt):
                    raise
                record.update(returncode=process.returncode, error=f"timeout after {timeout}s")
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    return record


def worker(args):
    # Reuse the existing input generator, bitwise checker and Triton loader;
    # do not run the benchmark's full correctness matrix or graph loop here.
    import torch
    import torch_npu
    import sgl_kernel_npu

    from bench_welm_pack_decode_inputs import assert_pack, load_triton_pack, make_inputs

    rows = args.rows[0]
    implementation = args.implementations[0]
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    inputs = make_inputs(rows, torch.device(f"npu:{args.device}"), 2026)
    fn = torch.ops.npu.welm_pack_decode_inputs.default
    if implementation == "triton":
        fn = load_triton_pack(args.triton_source)
        if rows == args.capacity:
            # The production wrapper bypasses N == C. Diagnose the *same* JIT
            # kernel without changing its code, signature or launch geometry.
            kernel = fn.__globals__["_get_pack_kernel"]()

            def fn(hidden, ids, weights, capacity):
                outputs = tuple(
                    torch.empty((capacity, value.shape[1]), dtype=value.dtype, device=value.device)
                    for value in (hidden, ids, weights)
                )
                block_size = 1024
                x_blocks = (capacity * 2048 + block_size - 1) // block_size
                route_blocks = (capacity * 10 + block_size - 1) // block_size
                kernel[(x_blocks + route_blocks,)](
                    hidden, ids, weights, *outputs, rows,
                    HIDDEN=2048, TOPK=10, CAPACITY=capacity,
                    X_BLOCKS=x_blocks, BLOCK_SIZE=block_size,
                )
                return outputs

    library = Path(sgl_kernel_npu.__file__).parent / "lib/libsgl_kernel_npu.so"
    report = {
        "implementation": implementation, "rows": rows, "capacity": args.capacity,
        "calls": args.calls, "device": torch.npu.get_device_name(args.device),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "sgl_kernel_npu": sgl_kernel_npu.__file__,
        "library": str(library), "library_sha256": sha256(library),
        "triton_source": str(args.triton_source),
        "triton_source_sha256": sha256(args.triton_source),
        "full_capacity_diagnostic": rows == args.capacity,
        "production_bypasses_packing": rows == args.capacity,
        "measurement": "individual eager kernel; not graph median_us",
        "completed": False,
    }
    if implementation == "triton":
        import triton

        report["triton"] = triton.__version__
    write_json(args.worker_result, report)
    torch.npu.synchronize()
    # msprof op owns warmup/replay. Simulator must only execute one target call.
    outputs = [fn(*inputs, args.capacity) for _ in range(args.calls)]
    torch.npu.synchronize()
    for result in outputs:
        assert_pack(inputs, result, args.capacity)
    report["completed"] = True
    report["bitwise_check"] = "passed"
    write_json(args.worker_result, report)
    print("PACK_PROFILE_DONE", flush=True)


def artifact_summary(directory, stage):
    files = [path for path in directory.rglob("*") if path.is_file() and path.stat().st_size]
    csv_files = [path for path in files if path.suffix == ".csv"]
    populated = []
    for path in csv_files:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as source:
            reader = csv.reader(source)
            next(reader, None)
            if any(any(field.strip() for field in row) for row in reader):
                populated.append(path)
    if stage == "board":
        complete = all(
            any(path.name.lower().startswith(prefix) for path in populated)
            for prefix in ("opbasicinfo", "pipeutilization")
        )
    else:
        complete = any("instr_exe" in path.name for path in populated) and any(
            path.name.startswith("trace") and path.suffix == ".json"
            or path.name == "visualize_data.bin" for path in files
        )
    artifacts = [
        str(path.relative_to(directory)) for path in files
        if path.suffix == ".csv" or path.name == "visualize_data.bin"
        or path.name.startswith("trace") or path.name == "aicore_binary.o"
    ]
    return {"expected_data_present": complete, "artifacts": sorted(artifacts)}


def make_command(args, stage, implementation, rows, case):
    count = args.board_count if stage == "board" else 1
    command = [args.msprof, "op"]
    if stage == "simulator":
        command += ["simulator", f"--soc-version={args.soc_version}"]
        # Do not restrict to core 0: hidden-copy, hidden-zero and routing AIVs
        # do different work. Keep the simulator's default core coverage.
        # Keep default metrics, including synchronization-event information.
    else:
        command += [f"--warm-up={args.warm_up}", "--aic-metrics=Default"]
    command += [
        f"--kernel-name={KERNEL_NAMES[implementation]}", f"--launch-count={count}",
        f"--output={case / 'profiler'}",
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--rows", str(rows), "--capacity", str(args.capacity),
        "--implementations", implementation, "--calls", str(count),
        "--device", str(args.device if stage == "board" else 0),
        "--triton-source", str(args.triton_source),
        "--worker-result", str(case / "worker.json"),
    ]
    return command


def snapshot_sources(output, triton_source):
    source_root = output / "sources"
    sources = list((ROOT / "csrc/welm_pack_decode_inputs").rglob("*"))
    sources += [ROOT / "csrc/CMakeLists.txt", Path(__file__).resolve(),
                Path(__file__).with_name("bench_welm_pack_decode_inputs.py")]
    for source in sources:
        if source.is_file() and source.suffix in (".cpp", ".h", ".md", ".txt", ".py"):
            target = source_root / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copyfile(triton_source, source_root / "ops_padding.py")
    for name, command in (
        ("git_head", ["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        ("git_status", ["git", "-C", str(ROOT), "status", "--short"]),
    ):
        run_logged(command, output, name=name)


def collect(args):
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700)  # Never reuse an old collection.
    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items() if not key.startswith("worker")},
        "python": sys.version, "platform": sys.platform,
        "environment": {key: os.environ[key] for key in (
            "ASCEND_HOME_PATH", "ASCEND_HOME", "ASCEND_TOOLKIT_HOME", "LD_LIBRARY_PATH",
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "SOC_VERSION",
            "TRITON_DEBUG", "TRITON_DISABLE_LINE_INFO", "TRITON_ALLWAYS_COMPILE",
        ) if key in os.environ},
        "notes": [
            "Kernel-level eager collection; no NPUGraph/static/SuperKernel timing.",
            "N == capacity forces the original Triton JIT kernel; production bypasses packing.",
            "Simulator times are diagnostic model estimates, not hardware latency.",
        ],
        "cases": [],
    }
    failed = False
    try:
        snapshot_sources(output, args.triton_source)
        if not args.dry_run:
            run_logged(["npu-smi", "info"], output, name="npu_smi")
            run_logged([args.msprof, "--version"], output, name="msprof_version")
            # Bounded reads from the configured CANN installation only.
            manifest["cann_version_files"] = {}
            for key in ("ASCEND_HOME_PATH", "ASCEND_HOME", "ASCEND_TOOLKIT_HOME"):
                if os.environ.get(key):
                    for relative in ("compiler/version.info", "runtime/version.info"):
                        version_file = Path(os.environ[key]) / relative
                        if version_file.is_file():
                            manifest["cann_version_files"][str(version_file)] = version_file.read_text(
                                encoding="utf-8", errors="replace"
                            )
        stage_errors = {}
        for stage in args.stages:
            if args.dry_run:
                continue
            command = [args.msprof, "op"] + (["simulator"] if stage == "simulator" else [])
            result = run_logged(command + ["--help"], output, name=f"{stage}_help")
            help_text = (output / f"{stage}_help.log").read_text(encoding="utf-8")
            required = ["--output", "--kernel-name", "--launch-count"]
            required += ["--soc-version"] if stage == "simulator" else ["--warm-up", "--aic-metrics"]
            missing = [option for option in required if option not in help_text]
            if result["returncode"] != 0 or missing:
                stage_errors[stage] = f"msprof {stage} help failed or missing options: {missing}"
        if "simulator" in args.stages and not args.soc_version:
            if args.dry_run:
                args.soc_version = "YOUR_CANN_SOC_VERSION"
            else:
                probe = run_logged(
                    [sys.executable, "-c", "import acl; print(acl.get_soc_name())"],
                    output, name="soc_version",
                )
                lines = (output / "soc_version.log").read_text(encoding="utf-8").splitlines()
                matches = [line.strip() for line in lines if re.fullmatch(r"Ascend\w+", line.strip())]
                if probe["returncode"] == 0 and len(matches) == 1:
                    args.soc_version = matches[0]
                else:
                    stage_errors["simulator"] = "SoC detection failed; supply --soc-version from acl.get_soc_name()."
        manifest["soc_version"] = args.soc_version
        for stage in args.stages:
            for rows in args.rows:
                for implementation in args.implementations:
                    label = f"{stage}/{implementation}/rows_{rows}_capacity_{args.capacity}"
                    case = output / label
                    case.mkdir(parents=True, mode=0o700)
                    command = make_command(args, stage, implementation, rows, case)
                    env = os.environ.copy()
                    overrides = {"PYTHONUNBUFFERED": "1"}
                    if implementation == "triton":
                        # A private cache preserves compiler IR/binaries and avoids
                        # deleting or depending on the user's existing JIT cache.
                        overrides.update(TRITON_CACHE_DIR=str(case / "triton_cache"),
                                         TRITON_DISABLE_LINE_INFO="0")
                    env.update(overrides)
                    entry = {"case": label, "argv": command, "env_overrides": overrides,
                             "full_capacity_diagnostic": rows == args.capacity}
                    manifest["cases"].append(entry)
                    print(f"[{len(manifest['cases'])}] {label}", flush=True)
                    if args.dry_run or stage in stage_errors:
                        (case / "run.command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
                        entry["status"] = "dry_run" if args.dry_run else "unavailable"
                        if stage in stage_errors:
                            entry["error"] = stage_errors[stage]
                    else:
                        entry.update(run_logged(command, case, env, args.timeout))
                        try:
                            entry.update(artifact_summary(case / "profiler", stage))
                            result_path = case / "worker.json"
                            entry["worker"] = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
                        except (OSError, ValueError, csv.Error) as error:
                            entry["error"] = f"reading partial output: {error}"
                        ok = (entry["returncode"] == 0 and "error" not in entry
                              and entry.get("worker", {}).get("completed", False)
                              and entry.get("expected_data_present", False))
                        entry["status"] = "ok" if ok else "failed"
                    failed |= entry["status"] not in ("ok", "dry_run")
                    print(f"    {entry['status']} (log: {case / 'run.log'})", flush=True)
                    write_json(output / "manifest.json", manifest)
    except KeyboardInterrupt:
        failed = True
        manifest["error"] = "interrupted; unfinished cases are not successful"
    except Exception as error:
        failed = True
        manifest["error"] = f"{type(error).__name__}: {error}"
        print(manifest["error"], file=sys.stderr, flush=True)
    finally:
        manifest["status"] = "incomplete" if failed else "dry_run" if args.dry_run else "ok"
        write_json(output / "manifest.json", manifest)
        summary = ["case\tstatus\treturncode"]
        summary += [f"{entry['case']}\t{entry.get('status', 'interrupted')}\t{entry.get('returncode', '')}"
                    for entry in manifest["cases"]]
        (output / "summary.tsv").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print("Packing collection...", flush=True)
        archive = shutil.make_archive(str(output), "gztar", root_dir=output.parent, base_dir=output.name)
        print(f"Archive: {archive}\nStatus: {manifest['status']}", flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--device", type=int, default=0, help="logical board device; simulator always uses 0")
    parser.add_argument("--soc-version", default=os.environ.get("SOC_VERSION"), help="default: acl.get_soc_name()")
    parser.add_argument("--stages", nargs="+", choices=("board", "simulator"), default=["board", "simulator"])
    parser.add_argument("--implementations", nargs="+", choices=tuple(KERNEL_NAMES), default=list(KERNEL_NAMES))
    parser.add_argument("--board-count", type=int, default=3, help="target calls collected per board case")
    parser.add_argument("--warm-up", type=int, default=10, help="msprof op internal warmup")
    parser.add_argument("--timeout", type=int, default=600, help="wall-clock seconds per collection process")
    parser.add_argument("--msprof", default="msprof", help="executable or absolute msprof path")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / datetime.now().strftime("welm_pack_profiles_%Y%m%d_%H%M%S"))
    parser.add_argument("--triton-source", type=Path, default=ROOT / "python/deep_ep/deep_ep/ops_padding.py")
    parser.add_argument("--dry-run", action="store_true", help="write commands and archive without importing Torch or running profilers")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--calls", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.capacity <= 128 or any(not 0 <= rows <= args.capacity for rows in args.rows):
        parser.error("require 0 <= rows <= capacity <= 128 and capacity > 0")
    if args.device < 0 or not 1 <= args.board_count <= 100 or not 0 <= args.warm_up <= 500 or args.timeout <= 0:
        parser.error("require device >= 0, board-count in [1,100], warm-up in [0,500], timeout > 0")
    if len(set(args.rows)) != len(args.rows) or len(set(args.stages)) != len(args.stages) or len(set(args.implementations)) != len(args.implementations):
        parser.error("rows, stages and implementations must not contain duplicates")
    args.triton_source = args.triton_source.resolve()
    if not args.triton_source.is_file():
        parser.error(f"missing Triton source: {args.triton_source}")
    args.msprof = shutil.which(args.msprof) or args.msprof
    if args.worker:
        if len(args.rows) != 1 or len(args.implementations) != 1 or not args.worker_result or args.calls < 1:
            parser.error("worker requires one row count, one implementation and --worker-result")
        worker(args)
        return 0
    if args.output.exists() or Path(str(args.output) + ".tar.gz").exists():
        parser.error("output directory/archive already exists; choose a new --output")
    if not args.dry_run and sys.platform != "linux":
        parser.error("run collection on the Linux NPU server; use --dry-run for a local preview")
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
