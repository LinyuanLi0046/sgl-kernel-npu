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
DMA fills the valid prefix and its 32-byte alignment padding; `Duplicate` fills
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

The previous implementation was reported on the server at 5.x us versus
Triton's 2.x us for `C=8, N=4/6`, using this script's default **graph** mode and
`median_us`; a regression was also reported at C=128. Those observations show
that the previous implementation missed the performance target. The revision
above removes source-level overheads identified using the small-kernel scalar
guidance in `cannbot-skills/ops/ascendc-perf-optimize` and
`ascendc-performance-best-practices`; it has not yet been compiled or timed on
the server. It does not establish which instruction/pipeline accounted for the
reported gap. Timing, baseline selection and the acceptance threshold are
unchanged. In graph replay, host metadata checks and output allocations are
not repeated and cannot explain that device-side timing gap.
