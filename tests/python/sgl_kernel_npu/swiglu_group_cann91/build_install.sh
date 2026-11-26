#!/usr/bin/env bash
# Build the original ops-nn operator and its original, standalone Torch subpackage.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash build_install.sh --ops-nn /path/to/ops-nn --prefix /path/to/install [--jobs 16]

Run in the serving Python environment, after sourcing CANN 9.1's set_env.sh.
Prerequisites: git, cmake, g++, python3 with torch/torch_npu, build, setuptools,
wheel and ninja. ops-nn's third_party dependencies must be available locally
or downloadable by its original build.sh. No CANN/PyTorch upgrade is performed.
EOF
}

repo=""; prefix=""; jobs=16
vendor=welm_swiglu91
while (($#)); do
    case "$1" in
        --ops-nn) repo=${2:?missing path}; shift 2 ;;
        --prefix) prefix=${2:?missing path}; shift 2 ;;
        --jobs) jobs=${2:?missing count}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
[[ -n "$repo" && -n "$prefix" && "$jobs" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
[[ $(uname -s) == Linux ]] || { echo 'Run this script on the Linux CANN server.' >&2; exit 2; }
: "${ASCEND_HOME_PATH:?Source CANN 9.1 set_env.sh first}"
for command in python3 git cmake g++ realpath; do command -v "$command" >/dev/null; done
repo=$(realpath "$repo")
prefix=$(realpath -m "$prefix")
# The upstream custom installer does not consistently quote paths containing spaces.
[[ "$repo$prefix" != *[[:space:]]* ]] || { echo 'Use paths without whitespace.' >&2; exit 2; }
[[ -f "$repo/activation/swiglu_group/torch_extension/swiglu_group.py" ]] || {
    echo 'ops-nn must contain SwigluGroup and its torch_extension directory.' >&2; exit 2;
}
export ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH:-$ASCEND_HOME_PATH}
[[ $(realpath "$ASCEND_CANN_PACKAGE_PATH") == "$(realpath "$ASCEND_HOME_PATH")" ]] || {
    echo 'ASCEND_CANN_PACKAGE_PATH and ASCEND_HOME_PATH point to different toolkits.' >&2; exit 2;
}
mkdir -p "$prefix/logs" "$prefix/opp" "$prefix/artifacts"
log="$prefix/logs/build_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee "$log") 2>&1
trap 'status=$?; echo "FAILED at line $LINENO (exit $status). Log: $log"; exit "$status"' ERR

python3 - <<'PY'
import importlib.util
import os
import sys
import torch
import torch_npu

missing = [name for name in ("build", "setuptools", "wheel", "ninja")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing build tools. Install only these packages first: "
                     + sys.executable + " -m pip install " + " ".join(missing))
version = str(torch_npu.npu.utils.get_cann_version())
print("Python:", sys.executable)
print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__)
print("CANN:", version, "ASCEND_HOME_PATH:", os.environ["ASCEND_HOME_PATH"])
if not version.startswith("9.1."):
    raise SystemExit("This script targets CANN 9.1.x; source the intended toolkit first.")
PY
command -v ninja >/dev/null || { echo 'ninja is installed but its executable is not in PATH.'; exit 1; }
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
if command -v bisheng >/dev/null; then command -v bisheng; bisheng --version; fi

marker="$prefix/logs/build_started_$$"
touch "$marker"
cd "$repo"
# --pkg also builds the matching Torch wheel in the inspected ops-nn revision.
# This invokes the upstream build/cleanup in this checkout; do not run concurrent builds here.
bash build.sh --pkg --soc=ascend950 --ops=swiglu_group --vendor_name="$vendor" -j"$jobs"

shopt -s nullglob
runs=("$repo/build_out/cann-ops-nn-${vendor}_linux-"*.run)
wheels=("$repo/torch_extension/dist/cann_ops_nn_${vendor}-"*.whl)
[[ ${#runs[@]} == 1 && ${#wheels[@]} == 1 ]] || {
    echo 'Expected exactly one matching custom .run and Torch wheel; inspect build output.'; exit 1;
}
[[ ${runs[0]} -nt "$marker" && ${wheels[0]} -nt "$marker" ]] || {
    echo 'Refusing to install stale artifacts from an earlier build.'; exit 1;
}
# Check the wheel really contains the selected operator, its C++ wrapper and Meta.
python3 - "${wheels[0]}" <<'PY'
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as wheel:
    files = wheel.namelist()
    for suffix in ("/swiglu_group/swiglu_group.py", "/csrc/swiglu_group.cpp", "/op_builder/builder.py"):
        if not any(name.endswith(suffix) for name in files):
            raise SystemExit("Wheel is missing " + suffix)
PY
cp "${runs[0]}" "${wheels[0]}" "$prefix/artifacts/"
bash "${runs[0]}" --quiet --install-path="$prefix/opp"
vendor_path="$prefix/opp/vendors/${vendor}_nn"
[[ -f "$vendor_path/op_api/lib/libcust_opapi.so" ]] || {
    echo "Missing custom ACLNN library under $vendor_path"; exit 1;
}
python3 -m pip install --no-deps --force-reinstall "${wheels[0]}"

# Avoid the vendor's set_env.bash referencing an unset variable under shell nounset.
{
    printf '# Source CANN 9.1 first, then source this file.\n'
    printf 'export ASCEND_CUSTOM_OPP_PATH=%q${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}\n' "$vendor_path"
    printf 'export WELM_SWIGLU_TORCH_PACKAGE=%q\n' "cann_ops_nn_${vendor}"
    printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$prefix/torch_extensions"
} > "$prefix/env.sh"
source "$prefix/env.sh"
python3 - "$vendor_path/op_api/lib/libcust_opapi.so" <<'PY'
import ctypes
import importlib
import os
import sys
import torch
import torch_npu

lib = ctypes.CDLL(sys.argv[1])
for symbol in ("aclnnSwigluGroupGetWorkspaceSize", "aclnnSwigluGroup"):
    getattr(lib, symbol)
package = importlib.import_module(os.environ["WELM_SWIGLU_TORCH_PACKAGE"])
module = importlib.import_module(package.__name__ + ".ops.activation.swiglu_group.swiglu_group")
extension = module.builder.load()  # Compile the original C++/Torch bridge now, before graph capture.
print("Torch package:", package.__file__)
print("JIT bridge:", extension.__file__)
print("Schema:", torch.ops.cann_ops_nn.swiglu_group.default._schema)
x = torch.empty((4, 4096), dtype=torch.bfloat16, device="meta")
counts = torch.empty((32,), dtype=torch.int64, device="meta")
assert torch.ops.cann_ops_nn.swiglu_group(x, group_index=counts).shape == (4, 2048)
print("INSTALL_AND_META_PASS (device execution and static/SuperKernel are not verified yet)")
PY
echo "Installed. In each new shell: source $prefix/env.sh"
echo "Build log: $log"
