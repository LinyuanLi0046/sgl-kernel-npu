# WeLM Decode input packing

Ascend C direct-launch operator for the BF16 Decode low-latency EP path:

```python
import torch
import torch_npu
import sgl_kernel_npu  # registers the shared library's operators

x_padded, ids_padded, weights_padded = torch.ops.npu.welm_pack_decode_inputs(
    hidden, expert_ids, topk_weights, 128
)
```

## Contract

| Argument | Shape / dtype |
| --- | --- |
| `hidden` | contiguous `[N,2048]`, BF16 |
| `expert_ids` | contiguous `[N,10]`, INT32 |
| `weights` | contiguous `[N,10]`, FP32 |
| `capacity` | `0 < capacity <= 128`, default 128; `0 <= N <= capacity` |

All inputs must be on the same NPU. Outputs have `capacity` rows and the same
dtypes. The first `N` rows are copied bit-for-bit; all remaining rows are zero.
Inputs are unchanged; all three outputs are fresh allocations, including when
`N == capacity`. The existing dispatcher should continue bypassing packing when
`N == capacity`; it need not call this operator just to copy already-full inputs.
The schema and Meta implementation deliberately do not claim output/input aliasing.

The operator is intended for Decode. It cannot infer the model's forward mode
from tensors; a future caller must select it only in its Decode branch. This
change exposes the operator and does not enable it in WeLM or DeepEP automatically.

Expert IDs are copied, not remapped or range-checked on device. For 512 experts,
EP 4/8/16/32 all use this same implementation: EP size and experts-per-rank do not
affect packing. `N=4` means four source tokens and 40 route entries; the expected
number of received expert tokens after Dispatch is a different quantity.

## Kernel and graph behavior

One launch produces all three outputs. Hidden partitioning uses the original
Triton's 1024-BF16-element (2 KiB) units, with adjacent units coalesced into one
DMA per participating AIV. One or two other AIVs pack IDs and weights using two
independent UB buffers. The kernel reads only valid input words and synthesizes
the capacity tail in UB, with no padding input allocation or extra GM reads.
Non-aligned routing reads explicitly pad to 32 bytes with zero; DMA's default
dummy fill is not assumed to be zero.

The small-kernel path uses directly constructed `LocalTensor`s, fixed UB
addresses and fixed event IDs, without `TPipe`/`TBuf` initialization or dynamic
event allocation. `InitSocState()` still initializes the hardware state. Input
DMA fills the valid prefix and its 32-byte alignment padding; vector stores fill
only the disjoint remaining tail. Both routing arrays are submitted before
waiting on their producers, then both outputs are stored. Ordinary kernels end
with `PipeBarrier<PIPE_ALL>()`; SK entries drain MTE3 locally without PIPE_ALL.

The host precomputes the hidden tiles-per-core quotient. The device recovers
the remainder with multiply/subtract; routing divides only by the constant 2
when needed. The partition and number of launch arguments are unchanged.
Each route array has a fixed 640-word UB region (5 KiB for both arrays). For
`C=8` and at least 17 available AIVs, the launch uses 16 hidden cores with 2 KiB
each and one routing core. `N=4` copies eight hidden chunks and zeroes eight;
`N=6` copies twelve and zeroes four. No inter-core synchronization is needed.

Tiling uses scalar launch arguments only: there is no tiling H2D, workspace
initialization, host/device synchronization, `.item()`, sorting or collective.
The host uses the repository's `EXEC_KERNEL_CMD` queue-aware launch helper.
The Torch op has a Meta implementation for npugraph_ex/fullgraph tracing.

## Build

Use the existing kernel-wheel build, specifying the server's exact SoC accepted
by the installed CANN:

```bash
bash build.sh -a kernels YOUR_CANN_SOC_VERSION
```

The default builds the ordinary direct-launch entry without requiring the
SuperKernel compiler features. To also emit the explicit SuperKernel entries:

```bash
SGL_WELM_PACK_SUPERKERNEL=ON bash build.sh -a kernels YOUR_CANN_SOC_VERSION
```

Install the newly generated `output/sgl_kernel_npu-*.whl` into the same Python
environment used for the comparison. The build script packages the wheel but
does not install it. The comparison reports the imported package path.

This is a **build-time** option. The compiler/devkit must provide `__sk__`,
`SK_BIND` and `sk::SkSystemArgs`. Unsupported
toolchains fail compilation rather than silently claiming SuperKernel support.
The SK entry uses the framework-provided block index and the original partition
stored in scalar arguments; it does not repartition data using the enclosing
SK's core count. Both entries share the same packing body. No early-start or
cache-maintenance-disabling flags are requested.

The packing kernel has its own Ascend C build target with
`--cce-auto-sync=off`, because its dependencies are handled explicitly. This
setting does not apply to other operators in the wheel. Both ordinary and SK
builds require the static-LocalTensor APIs, including `InitSocState()`.

On Ascend 950, this target also passes
`-mllvm -cce-aicore-record-overflow=false`. Packing uses raw `uint32_t` copies
and zero stores, with no floating-point arithmetic or conversion, so it does
not need to record numeric overflow status. Input NaNs and negative zero still
travel as unchanged bits. This targets the compiler-generated status-recording
epilogue seen in the profile below; it does not remove `InitSocState()`, DMA
dependencies, cache maintenance or the final ordinary/SK synchronization.
The option is confined to this target and applies to all supported rows and
capacities; other operators and other SoCs keep their existing settings.
The compiler option is also used in the
[official direct-launch build examples](https://github.com/Ascend/samples/blob/fc2aea93e48be8f8cee51d0c6c1b13a08155f83a/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/Add/CMakeLists.txt).
Those examples alone do not validate its effect on the server's 950 compiler.

On Ascend 950 builds, tail filling uses an explicit `__simd_vf__` with
`MicroAPI::Duplicate`, `UpdateMask` and `StoreAlign`. The routing core fills both
IDs and weights in one VF, sharing the zero register, loop and mask; hidden
cores use the same implementation with one destination. Other SoCs retain the
`LocalTensor` `Duplicate` API. The architecture selection is automatic at build
time and adds no runtime environment variable or public Torch entry.

`rows`, `capacity` and the existing partition remain runtime arguments. There
are no capacity-specific or row-specific entry points. For `N=4, C=8`, each
routing tail has 40 uint32 words: one VF iteration writes 40 words to each of
the two disjoint buffers. For `N=6`, input DMA explicitly pads 60 words to 64,
and the VF writes the remaining 16 words per array. A masked final VF store
also handles the general tail sizes up to C=128. Input DMA requests zero
padding only when the valid prefix actually needs alignment padding.

Implementation references (ops-nn revision `267827973`):

- `pooling/pool_3d_common/op_kernel/arch35/copy_pad_impl.h`,
  `CopyPad::CustomDuplicate`: zero register, `UpdateMask`, masked UB stores.
- `index/reverse_v2/op_kernel/arch35/reverse_v2_simd.h`: explicit
  `__simd_vf__` body separate from GM/UB staging. Launch syntax follows the
  [official asc_vf_call API](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/vf_call/asc_vf_call.html).
- cannbot `ascendc-perf-optimize` and `ascendc-regbase-best-practice`: small
  operator scalar overhead, VF boundaries and synchronization review.

The ordinary and SK builds have not been compiled or executed on an NPU in the
Windows development workspace. API presence in Python alone does not establish
that the server's compiler supports these device-side features.

References: [direct-launch SuperKernel adaptation](https://asc.gitcode.com/guide/programming_guide/advanced_programming/super_kernel/kernel_direct_call_adaptation.html),
[SK_BIND](https://asc.gitcode.com/api/Utils-API/SuperKernel/SK_BIND.html),
[static Tensor programming](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html).

## Correctness and latency comparison

Run from the repository root after installing the rebuilt wheel and the same
Triton-Ascend version used by the current implementation:

```bash
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --json pack_graph.json
# Deployed max-tokens-per-rank=8; same graph timing method:
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --capacity 8 --rows 4 --json pack_c8_n4.json
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --capacity 8 --rows 6 --json pack_c8_n6.json
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --mode static --json pack_static.json
# Requires a wheel built with SGL_WELM_PACK_SUPERKERNEL=ON:
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --mode superkernel --json pack_sk.json
```

The default is **N=4, C=128, H=2048, K=10**, with 512-expert IDs. It imports the
actual baseline `python/deep_ep/deep_ep/ops_padding.py`, without initializing EP.
The loader exposes `tl` in that module's globals for Triton versions that do not
resolve closure variables during JIT compilation; it preserves the kernel source.
It checks bitwise preservation (including NaNs and negative zero), zero tails,
empty inputs, non-aligned routing sizes and changing input contents after capture.

Both paths use the same inputs, number of captured calls and replay count.
All captured outputs are kept alive to prevent repeated calls from degenerating
into overwrites of one buffer, and their addresses are checked for reuse. Each
SuperKernel invocation uses a distinct scope name. Warmup/compilation are
excluded from timing, and measurement order alternates. The reported time is
NPU-event graph time divided by call count, **not profiler-only kernel duration**.
Exit code 0 requires the
Ascend C median to be both below 2 us and faster than the measured Triton median;
exit code 2 means the performance target was not met. Correctness failures raise.

For device-task duration and actual SK coverage, additionally collect:

```bash
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --profile-dir pack_profile
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --mode superkernel --profile-dir pack_sk_profile
```

Inspect that a single packing call produces one packing kernel (or participates
in the expected SK), and compare matched device-task durations with the original
2-us observation. The standalone comparison does not establish full-MoE latency
or communication overlap. No sub-2-us result is claimed before device measurement.

The earlier server report was a stable 3.2 us for Ascend C versus 1.7 us for
Triton at `N=4, C=8`, using the default **graph** mode's `median_us`.
The subsequent `welm_pack_profiles_20261127_002834` capture measured the VF
revision (`dfbdcb7`) on `Ascend950DT_9582` with CANN 9.1.0. At `N=4, C=8`,
the median **eager device-task duration** over three samples was 3.649 us for
Ascend C and 1.961 us for Triton. These are different timing methods; do not
compare the graph median directly with the profiler duration.

In that capture, averaging the eight hidden-copy blocks over three samples,
Scalar time was 2.060 us for Ascend C versus 0.579 us for Triton. MTE2/MTE3
times were similar and those blocks did not execute VF. Across all collected
rows (1, 2, 4, 6, 8), each native block also wrote 128 bytes more to main memory.
The available simulator trace shows native `STATUS` read/modify/write in the
epilogue. This motivates disabling numeric overflow recording before changing
the data path further. The simulator logs are incomplete: they cannot establish
the full epilogue duration or prove that this sequence explains the entire gap.
Triton also performs cache maintenance at exit.

After rebuilding the overflow-recording change, the user reported performance
parity with Triton and accepted the result. A post-change JSON/PMU or instruction
trace was not supplied, so no exact new latency or epilogue instruction removal
is independently confirmed here. The two raw profiling packages were deleted at
the user's request;
the baseline measurements above and the reusable comparison/collector remain.

The Triton source is identical between revisions `7c81be8` and `2f45a31`.
Their timing code is also identical, but `2f45a31` added eager correctness cases
before measurement. Consequently, the earlier reported change from 2.x to
1.7 us for Triton is not evidence of a Triton code optimization; the exact
cause requires matched runtime evidence. This compiler change leaves the entire
comparison script unchanged, including those pre-measurement cases. In graph
replay, host metadata checks and output allocations are not repeated and cannot
explain the device-side timing gap.

## Collect and package msprof op / simulator evidence

From the Linux server's repository root, with the rebuilt wheel installed and
the CANN/MindStudio environment sourced:

```bash
python tests/python/sgl_kernel_npu/collect_welm_pack_profiles.py
# Optional: use the exact SoC reported by `import acl; acl.get_soc_name()`:
python tests/python/sgl_kernel_npu/collect_welm_pack_profiles.py --soc-version YOUR_CANN_SOC_VERSION
# The same row sweep at capacity 128:
python tests/python/sgl_kernel_npu/collect_welm_pack_profiles.py --capacity 128
```

Defaults are **rows 1, 2, 4, 6, 8; capacity 8; Ascend C and Triton; both
`msprof op` and `msprof op simulator`**. This produces 20 separate collection
jobs, run sequentially. The SoC defaults to `SOC_VERSION`, or is detected using
`acl.get_soc_name()`. No simulator library is injected into the board environment.
On 950, the profiler selects its simulator through `--soc-version`, as described
in the [official simulator guide](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md).

Each board job launches three target calls; `msprof op` performs ten internal
warmups and collects `Default` metrics. Each simulator job launches **one**
target call, without the benchmark's graph loop or full correctness matrix.
Both reuse its input generator, bitwise checker and original Triton loader.
Kernel-name filtering excludes auxiliary operations. Core filtering is omitted
so that hidden-copy, hidden-zero and routing cores can all be inspected.
Simulator metrics retain their defaults, including synchronization information.
See the [official board guide](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)
for metric and warmup definitions.

**At rows=capacity=8, the production Triton wrapper bypasses packing.** This
collector explicitly launches the same original JIT kernel for that diagnostic
case, with the same block size and grid, and records `full_capacity_diagnostic`.
It does not change production behavior. These are individual **eager kernel**
profiles, not the original NPUGraph `median_us`; simulation time is not hardware
latency. No static-kernel or SuperKernel mode is enabled by this collector.

Results and an adjacent `.tar.gz` default to `output/welm_pack_profiles_<time>`:

```text
welm_pack_profiles_<time>/
  manifest.json                    # configuration, commands, status, artifacts
  summary.tsv                      # one status row per collection job
  *_help.log, npu_smi.log           # installed tool capabilities / device info
  sources/                         # current operator, baseline and scripts
  board/{ascendc,triton}/rows_*/    # profiler/, run.log, worker.json
  simulator/{ascendc,triton}/rows_*/
```

Each `worker.json` records package versions, loaded native library path/SHA256,
baseline SHA256 and the bitwise check. Triton jobs use private `triton_cache/`
directories, retained in the archive with their compiler outputs. Native source
line mapping depends on debug information in the installed wheel; the collector
does not rebuild it or change compilation flags. All profiler outputs, including
CSV, traces, binaries and intermediate data actually generated, are packaged.

Useful options: `--stages board` or `--stages simulator`,
`--implementations ascendc` or `--implementations triton`, `--rows 4 6`,
`--device 0`, `--output /absolute/new_directory`, `--msprof /path/to/msprof`.
`--dry-run` generates commands and an archive without importing Torch or running
profilers. Existing output directories/archives are never overwritten.

Failed/empty captures are marked and the remaining cases continue. Each job has
a 600-second wall-clock limit (`--timeout`); partial outputs and logs are still
packed after a failure or Ctrl+C. Exit code **0** means all requested captures
passed the worker check and produced the expected report files (or a dry run
completed); **1** means the archive contains incomplete/failed captures.
The script has only been checked without NPU hardware in the development
workspace; profiler and simulator execution must be verified on the server.
