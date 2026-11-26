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

The default builds the ordinary direct-launch entry, preserving compatibility
with older toolchains. To also emit the explicit SuperKernel entries:

```bash
SGL_WELM_PACK_SUPERKERNEL=ON bash build.sh -a kernels YOUR_CANN_SOC_VERSION
```

Install the newly generated `output/sgl_kernel_npu-*.whl` into the same Python
environment used for the comparison. The build script packages the wheel but
does not install it. The comparison reports the imported package path.

This is a **build-time** option. The compiler/devkit must provide `__sk__`,
`SK_BIND`, `sk::SkSystemArgs` and `TPipe::DestroyWithoutPipeAll`. Unsupported
toolchains fail compilation rather than silently claiming SuperKernel support.
The SK entry uses the framework-provided block index and the original partition
stored in scalar arguments; it does not repartition data using the enclosing
SK's core count. Both entries share the same packing body. No early-start or
cache-maintenance-disabling flags are requested.

The ordinary and SK builds have not been compiled or executed on an NPU in the
Windows development workspace. API presence in Python alone does not establish
that the server's compiler supports these device-side features.

References: [direct-launch SuperKernel adaptation](https://asc.gitcode.com/guide/programming_guide/advanced_programming/super_kernel/kernel_direct_call_adaptation.html),
[SK_BIND](https://asc.gitcode.com/api/Utils-API/SuperKernel/SK_BIND.html).

## Correctness and latency comparison

Run from the repository root after installing the rebuilt wheel and the same
Triton-Ascend version used by the current implementation:

```bash
python tests/python/sgl_kernel_npu/bench_welm_pack_decode_inputs.py --json pack_graph.json
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
