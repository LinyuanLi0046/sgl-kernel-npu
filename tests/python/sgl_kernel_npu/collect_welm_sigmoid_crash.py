"""Collect one functional sigmoid-mul eager call and its launch/build artifacts.

Run with the SAME Python environment/wheel as the failing SGLang server:
    python tests/python/sgl_kernel_npu/collect_welm_sigmoid_crash.py

The parent uses only the standard library. A failed/aborted/timed-out NPU child
does not prevent packaging. No rebuild, installation, or model launch occurs.
The adjacent collect_welm_pack_profiles.py supplies existing collection helpers.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

from collect_welm_pack_profiles import run_logged, sha256, write_json


ROOT = Path(__file__).resolve().parents[3]
OP = "welm_sigmoid_mul"
CANN_ENV = ("ASCEND_HOME_PATH", "ASCEND_HOME", "ASCEND_TOOLKIT_HOME")
ENV_KEYS = CANN_ENV + (
    "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES", "ASCEND_CUSTOM_OPP_PATH", "TORCH_EXTENSIONS_DIR",
    "SOC_VERSION", "SGL_WELM_SIGMOID_MUL_SUPERKERNEL", "ASCEND_LAUNCH_BLOCKING",
    "ASCEND_PROCESS_LOG_PATH", "ASCEND_WORK_PATH", "NPU_COLLECT_PATH",
    "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_MODULE_LOG_LEVEL", "ASCEND_SLOG_PRINT_TO_STDOUT",
)
TEXT_SUFFIXES = {".h", ".hpp", ".c", ".cpp", ".cc", ".inc", ".cmake", ".txt", ".json"}


def worker(args):
    """Checkpoint before each device operation: fatal signals cannot run finally."""
    directory = args.output
    result = {"pid": os.getpid(), "status": "running", "phase": "start"}

    def checkpoint(phase):
        result["phase"] = phase
        write_json(directory / "worker.json", result)
        print(f"[{phase}]", flush=True)

    checkpoint("import")
    try:
        # Device-fault diagnostics are collected below; avoid a huge host core.
        if os.name == "posix":
            import resource
            _, hard = resource.getrlimit(resource.RLIMIT_CORE)
            resource.setrlimit(resource.RLIMIT_CORE, (0, hard))

        spec = importlib.util.find_spec("sgl_kernel_npu")
        result["package_origin"] = spec.origin if spec else None
        checkpoint("import_torch")
        import torch
        import torch_npu

        result["torch"] = {"version": torch.__version__, "file": torch.__file__}
        result["torch_npu"] = {"version": torch_npu.__version__, "file": torch_npu.__file__}
        checkpoint("import_sgl_kernel_npu")
        import sgl_kernel_npu

        result["package_origin"] = sgl_kernel_npu.__file__
        result["loaded_libraries"] = sorted(
            str(p) for p in torch.ops.loaded_libraries if "sgl_kernel_npu" in str(p)
        )
        maps = Path("/proc/self/maps")
        if maps.is_file():
            # Save only this extension's mappings, before any custom device launch.
            lines = [line for line in maps.read_text().splitlines()
                     if "sgl_kernel_npu" in line and ".so" in line]
            (directory / "extension_maps.txt").write_text("\n".join(lines) + "\n")
            result["mapped_libraries"] = sorted({
                line.split(maxsplit=5)[5] for line in lines if len(line.split(maxsplit=5)) == 6
            })
        # Fingerprint what this process actually loads, before the repro can fail.
        result["loaded_library_sha256"] = {
            p: sha256(Path(p)) for p in result["loaded_libraries"] if Path(p).is_file()
        }
        checkpoint("resolve_operator")
        op = torch.ops.npu.welm_sigmoid_mul.default
        result["schema"] = str(op._schema)
        checkpoint("initialize_device")
        torch.npu.set_device(args.device)
        torch.npu.set_compile_mode(jit_compile=False)

        generator = torch.Generator().manual_seed(2026)
        gate_cpu = torch.randn(args.batch, args.heads, 1, generator=generator).bfloat16()
        attention_cpu = torch.randn(args.batch, args.heads, 256, generator=generator).bfloat16()
        torch.save({"gate": gate_cpu, "attention": attention_cpu}, directory / "inputs.pt")
        checkpoint("copy_inputs")
        gate = gate_cpu.to(f"npu:{args.device}")
        attention = attention_cpu.to(f"npu:{args.device}")
        torch.npu.synchronize()
        result["inputs"] = {
            name: {"shape": list(t.shape), "stride": list(t.stride()),
                   "dtype": str(t.dtype), "storage_offset": t.storage_offset(),
                   "data_ptr": hex(t.data_ptr()), "device": str(t.device)}
            for name, t in (("gate", gate), ("attention", attention))
        }
        result["rows"] = args.batch * args.heads
        checkpoint("operator_call")
        out = op(gate, attention)  # Same functional schema as WeLM's compiled tail.
        result["output_ptr"] = hex(out.data_ptr())
        checkpoint("operator_synchronize")
        torch.npu.synchronize()
        result["launch_and_sync_completed"] = True
        checkpoint("copy_output")
        actual = out.cpu()
        expected = torch.sigmoid(gate_cpu.float()).bfloat16() * attention_cpu
        torch.save({"actual": actual, "reference": expected}, directory / "outputs.pt")
        result["numerical_summary"] = {
            "all_finite": bool(torch.isfinite(actual).all()),
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
            "exact_equal": bool(torch.equal(actual, expected)),
            "note": "Diagnostic only; this single random case is not a full accuracy test.",
        }
        result["status"] = "completed"
        checkpoint("done")
    except BaseException:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
        write_json(directory / "worker.json", result)
        raise


def walk_files(root, warnings):
    """Bounded to a requested tree; never follow directory symlinks."""
    if not root.is_dir():
        warnings.append(f"Directory unavailable: {root}")
        return
    for directory, dirs, files in os.walk(root, followlinks=False,
                                          onerror=lambda e: warnings.append(str(e))):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for name in files:
            path = Path(directory) / name
            if path.is_file():
                yield path


def contains_symbol(path):
    needle = OP.encode()
    with path.open("rb") as source:
        tail = b""
        while chunk := source.read(1024 * 1024):
            data = tail + chunk
            if needle in data:
                return True
            tail = data[-len(needle):]
    return False


def build_artifacts(root, warnings):
    for path in walk_files(root, warnings):
        relative = path.relative_to(root)
        # This includes generic aiv.o, auto_gen wrappers, flags, link commands,
        # and logs inside the operator's ExternalProject/CMake target subtree.
        selected = OP in str(relative).lower()
        selected |= path.name in ("CMakeCache.txt", "compile_commands.json")
        if not selected and path.suffix.lower() in TEXT_SUFFIXES:
            # Aggregated generated launch/registration files can have generic names.
            generated = any(s in str(relative).lower() for s in ("auto_gen", "autogen", "generated"))
            if generated and path.stat().st_size <= 4 * 1024 * 1024:
                selected = contains_symbol(path)
        if selected:
            yield path, relative


def copy_artifact(source, destination, output, manifest, kind):
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        record = {"kind": kind, "source": str(source), "resolved_source": str(source.resolve()),
                  "archive_path": str(destination.relative_to(output)),
                  "bytes": destination.stat().st_size, "sha256": sha256(destination)}
        manifest["artifacts"].append(record)
        return record
    except OSError as error:
        manifest["warnings"].append(f"Cannot collect {source}: {error}")
        return None


def find_msobjdump(explicit):
    if explicit:
        return explicit
    found = shutil.which("msobjdump")
    if found:
        return found
    for key in CANN_ENV:
        if os.environ.get(key):
            for relative in ("bin/msobjdump", "tools/msobjdump/msobjdump", "tools/msobjdump/bin/msobjdump"):
                candidate = Path(os.environ[key]) / relative
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
    return None


def inspect_binaries(args, output, manifest):
    """Prefer extraction from the loaded .so over potentially stale build objects."""
    tool = find_msobjdump(args.msobjdump)
    if not tool:
        manifest["warnings"].append("msobjdump unavailable; raw extension/build binaries are retained.")
        return
    directory = output / "binary_inspection"
    directory.mkdir()

    def inspect_run(command, name, timeout=120):
        record = run_logged(command, directory, timeout=timeout, name=name)
        manifest["commands"].append(record)
        if record.get("returncode") != 0:
            manifest["warnings"].append(f"Binary inspection failed: {name}; see binary_inspection/{name}.log")

    inspect_run([tool, "-h"], "help", timeout=60)
    help_text = (directory / "help.log").read_text(encoding="utf-8", errors="replace")
    loaded = [a for a in manifest["artifacts"] if a["kind"] == "loaded_extension"]
    for index, item in enumerate(loaded):
        elf = output / item["archive_path"]
        if "--dump-elf" in help_text:
            command = [tool, "--dump-elf", str(elf)]
            if "--verbose" in help_text:
                command.append("--verbose")
            inspect_run(command, f"loaded_{index}_metadata")
        if "--extract-elf" not in help_text:
            manifest["warnings"].append("Installed msobjdump has no --extract-elf; retaining the original .so.")
            continue
        # Extraction may unpack every operator in the wheel. Retain only sigmoid
        # ELF/metadata here; the exact original .so is also included in the archive.
        with tempfile.TemporaryDirectory(prefix="sigmoid_extract_", dir=output.parent) as temporary:
            command = [tool, "--extract-elf", str(elf), "--out-dir", temporary]
            inspect_run(command, f"loaded_{index}_extract")
            for source in walk_files(Path(temporary), manifest["warnings"]):
                if OP in str(source.relative_to(temporary)) or contains_symbol(source):
                    target = output / "device_from_loaded_extension" / str(index) / source.relative_to(temporary)
                    copy_artifact(source, target, output, manifest, "extracted_device")

    # Collect disassembly if this CANN tool supports it; never require a rebuild
    # with -g. Missing line information does not invalidate address-level diagnosis.
    if "--sass" not in help_text:
        manifest["warnings"].append("Installed msobjdump has no --sass; device binaries remain available for offline analysis.")
        return
    candidates = [a for a in manifest["artifacts"] if a["kind"] == "extracted_device"]
    if not candidates:
        candidates = [a for a in manifest["artifacts"] if a["kind"] == "build"]
    seen = set()
    for item in candidates:
        path = output / item["archive_path"]
        with path.open("rb") as source:
            is_elf = source.read(4) == b"\x7fELF"
        if not is_elf or item["sha256"] in seen:
            continue
        if item["kind"] == "build" and OP not in path.name and not contains_symbol(path):
            continue
        seen.add(item["sha256"])
        name = f"elf_{len(seen):02d}"
        (directory / f"{name}.source.txt").write_text(item["archive_path"] + "\n", encoding="utf-8")
        inspect_run([tool, "--sass", str(path)], name, timeout=60)


def collect(args):
    output = args.output.resolve()
    # A new directory for every incident; do not overwrite/merge old evidence.
    if output.exists() or Path(str(output) + ".tar.gz").exists():
        raise SystemExit(f"Output already exists; choose a new --output: {output}")
    repo = args.repo.resolve()
    builds = [p.resolve() for p in (args.build_dir or [repo / "build"])]
    if any(root == output or root in output.parents for root in builds):
        raise SystemExit("--output must be outside every --build-dir (avoids recursive self-collection).")
    output.mkdir(parents=True, mode=0o700)
    manifest = {
        "created": datetime.now().astimezone().isoformat(), "python": sys.executable,
        "python_version": sys.version, "platform": platform.platform(),
        "repo": str(repo), "build_directories": [str(p) for p in builds],
        "case": {"mode": "eager", "operator": "npu::welm_sigmoid_mul", "batch": args.batch,
                 "heads": args.heads, "head_dim": 256, "device": args.device, "seed": 2026},
        "environment": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
        "artifacts": [], "commands": [], "warnings": [],
        "notes": ["A single functional eager call; no graph, model, Triton, or performance test.",
                  "Build files are candidates; they are not assumed to match the installed wheel.",
                  "Only successful subprocess exit plus worker completion counts as a completed repro."],
    }
    started = time.time()
    repro = output / "repro"
    repro.mkdir()
    try:
        print("[1/4] Snapshotting source and build artifacts", flush=True)
        sources = list((repo / "csrc/welm_sigmoid_mul").glob("**/*"))
        sources += [repo / p for p in ("CMakeLists.txt", "csrc/CMakeLists.txt", "build.sh",
                   "csrc/utils/torch_helper.h", "python/sgl_kernel_npu/sgl_kernel_npu/__init__.py")]
        for source in sources:
            if source.is_file():
                copy_artifact(source, output / "sources" / source.relative_to(repo), output, manifest, "source")
        for name in (Path(__file__).name, "collect_welm_pack_profiles.py"):
            source = Path(__file__).with_name(name)
            copy_artifact(source, output / "collector" / name, output, manifest, "collector")
        for index, root in enumerate(builds):
            for source, relative in build_artifacts(root, manifest["warnings"]):
                copy_artifact(source, output / "build_artifacts" / str(index) / relative, output, manifest, "build")
        for index, source in enumerate(args.build_log):
            copy_artifact(source, output / "build_logs" / f"{index}_{source.name}", output, manifest, "build_log")
        for name, command in (
            ("git_head", ["git", "-C", str(repo), "rev-parse", "HEAD"]),
            ("git_status", ["git", "-C", str(repo), "status", "--short"]),
            ("git_diff", ["git", "-C", str(repo), "diff", "HEAD", "--", "csrc/welm_sigmoid_mul",
                          "csrc/utils/torch_helper.h", "csrc/CMakeLists.txt", "build.sh"]),
        ):
            manifest["commands"].append(run_logged(command, output, name=name))
        manifest["package_versions"] = {}
        for name in ("torch", "torch-npu", "sgl-kernel-npu", "npugraph-ex"):
            try:
                manifest["package_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                manifest["package_versions"][name] = "not found in package metadata"
        for key in CANN_ENV:
            if os.environ.get(key):
                for relative in ("version.info", "compiler/version.info", "runtime/version.info"):
                    source = Path(os.environ[key]) / relative
                    if source.is_file():
                        copy_artifact(source, output / "cann_version" / key / relative, output, manifest, "version")

        print("[2/4] Running one isolated eager call", flush=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--output", str(repro),
                   "--batch", str(args.batch), "--heads", str(args.heads), "--device", str(args.device)]
        overrides = {
            "ASCEND_PROCESS_LOG_PATH": str(output / "runtime_logs"),
            "ASCEND_WORK_PATH": str(output / "runtime_work"),
            "NPU_COLLECT_PATH": str(output / "exception_dump"),
            "ASCEND_GLOBAL_LOG_LEVEL": "0", "ASCEND_SLOG_PRINT_TO_STDOUT": "0",
            "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1",
        }
        for key in ("ASCEND_PROCESS_LOG_PATH", "ASCEND_WORK_PATH", "NPU_COLLECT_PATH"):
            Path(overrides[key]).mkdir()
        manifest["repro_env_overrides"] = overrides
        if args.dry_run:
            manifest["repro"] = {"status": "dry_run", "argv": command}
        else:
            manifest["commands"].append(run_logged(["npu-smi", "info"], output, name="npu_smi"))
            manifest["repro"] = run_logged(command, repro, env={**os.environ, **overrides}, timeout=args.timeout)
        worker_file = repro / "worker.json"
        result = json.loads(worker_file.read_text(encoding="utf-8")) if worker_file.is_file() else {}
        manifest["worker"] = result
        print(f"    exit={manifest['repro'].get('returncode')}, phase={result.get('phase', 'not started')}", flush=True)

        print("[3/4] Collecting loaded extension and device binaries", flush=True)
        libraries = set(result.get("loaded_libraries", []))
        libraries.update(p for p in result.get("mapped_libraries", []) if not p.endswith(" (deleted)"))
        collected_libraries = set()
        for index, value in enumerate(sorted(libraries)):
            source = Path(value)
            if source.resolve() in collected_libraries:
                continue
            collected_libraries.add(source.resolve())
            item = copy_artifact(source, output / "loaded_extension" / str(index) / source.name,
                                 output, manifest, "loaded_extension")
            before = next((digest for p, digest in result.get("loaded_library_sha256", {}).items()
                           if Path(p).resolve() == source.resolve()), None)
            if item and before and item["sha256"] != before:
                manifest["warnings"].append(f"Loaded library changed during collection: {source}")
        if result.get("package_origin"):
            source = Path(result["package_origin"])
            copy_artifact(source, output / "installed_package" / "__init__.py", output, manifest, "installed_source")
            if not libraries:
                source = source.parent / "lib/libsgl_kernel_npu.so"
                if source.is_file():
                    copy_artifact(source, output / "installed_package" / source.name, output, manifest, "unconfirmed_extension")
        # Fallback for CANN versions that ignore the private plog destination.
        # Restrict reads to this child PID and this run, not all server/rank logs.
        if result.get("pid"):
            roots = {Path.home() / "ascend/log"}
            if os.environ.get("ASCEND_PROCESS_LOG_PATH"):
                roots.add(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]))
            for index, root in enumerate(sorted(roots)):
                for category in ("debug/plog", "run/plog", "security/plog", "plog", ""):
                    for source in (root / category).glob(f"plog-{result['pid']}_*"):
                        if source.is_file() and source.stat().st_mtime >= started:
                            copy_artifact(source, output / "fallback_plog" / str(index) / category / source.name,
                                          output, manifest, "plog")
        if not args.dry_run:
            inspect_binaries(args, output, manifest)
    except (Exception, KeyboardInterrupt):
        manifest["collection_error"] = traceback.format_exc()
        print(manifest["collection_error"], file=sys.stderr, flush=True)
    finally:
        kinds = {item["kind"] for item in manifest["artifacts"]}
        if "build" not in kinds:
            manifest["warnings"].append("No build artifacts found. Supply the original --build-dir; do not rebuild before collecting it.")
        if not any(item["kind"] == "build" and Path(item["archive_path"]).name == "aclrtlaunch_welm_sigmoid_mul_kernel.h"
                   for item in manifest["artifacts"]):
            manifest["warnings"].append("Generated aclrtlaunch_welm_sigmoid_mul_kernel.h not found; check the original --build-dir.")
        if "loaded_extension" not in kinds:
            manifest["warnings"].append("No confirmed loaded extension collected; inspect repro/run.log and worker.json.")
        result = manifest.get("worker", {})
        completed = result.get("status") == "completed" and manifest.get("repro", {}).get("returncode") == 0
        manifest["repro_status"] = "dry_run" if args.dry_run else "completed" if completed else "failed_or_incomplete"
        manifest["collection_status"] = "partial" if manifest.get("collection_error") or manifest["warnings"] else "complete"
        write_json(output / "manifest.json", manifest)
        summary = [f"repro_status: {manifest['repro_status']}", f"last_phase: {result.get('phase')}",
                   f"returncode: {manifest.get('repro', {}).get('returncode')}",
                   f"collection_status: {manifest['collection_status']}",
                   "See repro/run.log, repro/worker.json, manifest.json, and binary_inspection/.",
                   "A completed eager call does not validate graph/static/SuperKernel or full accuracy."]
        summary += ["WARNING: " + message for message in manifest["warnings"]]
        (output / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print("[4/4] Packaging collected evidence", flush=True)
        archive = Path(str(output) + ".tar.gz")
        with tarfile.open(archive, "w:gz", compresslevel=1) as target:
            target.add(output, arcname=output.name)
        Path(str(archive) + ".sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
        print(f"Archive: {archive}\nRepro: {manifest['repro_status']}\nCollection: {manifest['collection_status']}", flush=True)
    # Exit 1 reports a repro/collection problem, even though its archive exists.
    return 0 if (args.dry_run and not manifest.get("collection_error")) or (completed and manifest["collection_status"] == "complete") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--build-dir", type=Path, action="append", help="original CMake build directory; repeatable; default: REPO/build")
    parser.add_argument("--build-log", type=Path, action="append", default=[], help="optional original build log; repeatable")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=24, help="local attention heads (TP1=24, TP2=12, TP4=6)")
    parser.add_argument("--device", type=int, default=0, help="logical NPU ID within existing visible-device settings")
    parser.add_argument("--timeout", type=int, default=180, help="maximum seconds for the repro subprocess")
    parser.add_argument("--msobjdump", help="optional path to CANN msobjdump; otherwise auto-detected")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / datetime.now().strftime("welm_sigmoid_crash_%Y%m%d_%H%M%S"))
    parser.add_argument("--dry-run", action="store_true", help="collect sources/build files and package without importing Torch or running NPU/tool commands")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.batch <= 128 and 1 <= args.heads <= 24 and args.device >= 0 and args.timeout > 0):
        parser.error("requires 1<=batch<=128, 1<=heads<=24, device>=0, timeout>0")
    return worker(args) if args.worker else collect(args)


if __name__ == "__main__":
    sys.exit(main())
