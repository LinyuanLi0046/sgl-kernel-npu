# WeLM Decode headwise sigmoid-mul (Ascend 950)

This direct-launch operator implements the actual WeLM attention gate, with
the same two BF16 rounding boundaries as `sigmoid_mul_kernel_npu`:

```python
scale = torch.sigmoid(gate.float()).to(torch.bfloat16)
output = (attention.float() * scale.float()).to(torch.bfloat16)
```

The current WeLM sample has 24 Q heads and head dimension 256. The head count
in this operator is **local attention heads**, not hidden size or EP size:
attention TP1/TP2/TP4 use 24/12/6 heads. DP attention changes the local token
count and attention TP group, while this kernel sees only local tensors.

## Contract and Torch calls

- Ascend 950 (DAV_3510); BF16 gate `[B,H,1]`, BF16 attention `[B,H,256]`.
- Contiguous inputs on the same NPU; `0 <= B <= 128`, `1 <= H <= 24`.
- Attention must start on a 32-byte boundary, as the model's Flash output does.
  Contiguous views with a misaligned storage offset are rejected before launch.
- Decode shapes are supported; the op cannot infer a model's forward mode.
- No model or dispatcher is changed. Enablement in Decode is a later change,
  after the benchmark demonstrates the required speedup.

```python
import torch
import torch_npu
import sgl_kernel_npu

# Functional form for a compiled attention tail; fresh, non-aliasing output.
output = torch.ops.npu.welm_sigmoid_mul(gate, attention)

# In-place form matching the current Triton call. Returns attention itself.
torch.ops.npu.welm_sigmoid_mul_(gate, attention)
```

Both forms share one device kernel and have explicit Torch schemas and Meta
implementations. The in-place form rejects overlap between gate and attention;
otherwise one core could overwrite another core's gate input. Empty batches
do not launch a kernel. Only 950 wheels register these operators.

## Design and performance choices

The design follows cannbot `ascendc-regbase-best-practice` and
`ascendc-perf-optimize`: model tiling first, skip inter-device and inter-core
synchronization analysis (neither is present), then optimize the local
load/compute/store pipeline. The relevant reference is ops-nn's
`activation/sigmoid/op_kernel/arch35/sigmoid_bf16.h`, with FP32 Exp/Div and
explicit BF16 cast traits. Its general-purpose queue framework is not copied.

| Layer | Implementation and reason |
| --- | --- |
| Host | Flatten `B*H` heads; target four heads / 2 KiB attention per core, bounded by available AIVs. Coalesce heads at larger batches; maximum tile is 64 heads. Pass three pointers and three scalar arguments; no tiling tensor, workspace or H2D. |
| Kernel | Three fixed UB buffers and fixed event ID 0, with `InitSocState()`; no TPipe, queue allocation or dynamic event management. Extra SuperKernel cores exit before touching data. |
| UB | Attention input 32 KiB, output 32 KiB, gate 128 B: total 65,664 B. Attention lengths are always multiples of 512 B; use aligned DataCopy. Only the compact gate input needs DataCopyPad, with an explicit zero-aligned tail. |
| VF | One VF per tile. Load up to 64 gates, compute FP32 sigmoid once per gate, round to BF16, and retain the scales in registers. Register Gather broadcasts one scale across each head; four FP32 vectors process its 256 elements. No expanded gate or sigmoid intermediate is written to GM or UB. |
| Dependencies | Submit both input DMAs before one MTE2-to-V wait; wait V-to-MTE3 before output. For the uncommon multi-tile case under limited AIV resources, drain the previous output before reusing UB. No inter-core barrier. |
| Exit | Ordinary entry keeps PIPE_ALL; explicit SK entry drains MTE3 locally. Disable compiler overflow-status recording only for this inference kernel, following the confirmed packing optimization. |

For `B=4,H=24` and at least 24 AIVs, this uses 24 cores, four heads each.
For `B=4,H=12` and at least 12 AIVs it uses 12 cores. The original Triton
wrapper groups 128 heads per program, so both examples launch one Triton
program. This is a reason to evaluate smaller tiles, not a measured speedup.
The tiling rule is shared across batches; there are no B=4/B=8 entry points.

Gate nonlinearities are vectorized across heads before broadcasting. The
four attention-vector iterations are unrolled. This addresses parallelism,
nonlinearity reuse, memory traffic and VF fusion as well as Scalar overhead.
No double buffer is introduced for the usual single-tile-per-core Decode
case. The physical-core-limited multi-tile path is sequential and requires
separate profiling if it becomes a deployment workload.

Numeric overflow recording is deliberately disabled by
`-mllvm -cce-aicore-record-overflow=false` on this target. Unlike packing,
this op performs FP arithmetic: it does not provide overflow-diagnostic
side effects for training. The Exp/Div formula, NaN/Inf outputs and BF16
rounding remain part of the inference contract. No fast-math flag, approximate
reciprocal substitution, or cache-maintenance-disabling option is added.

API references: [register Gather](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/scatter_operations/Gather.html),
[static Tensor initialization and synchronization](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html),
[direct-launch SuperKernel adaptation](https://asc.gitcode.com/guide/programming_guide/advanced_programming/super_kernel/kernel_direct_call_adaptation.html).
The server compiler must provide these DAV_3510 APIs; compiling only the
previous packing op does not establish support for register Gather.

## Build

From the existing sgl-kernel-npu repository, using the server's CANN environment:

```bash
# Includes explicit SK entries by default:
bash build.sh -a kernels Ascend950DT_9582
# To build only the ordinary direct-launch kernel:
SGL_WELM_SIGMOID_MUL_SUPERKERNEL=OFF bash build.sh -a kernels Ascend950DT_9582
```

Install the newly generated `output/sgl_kernel_npu-*.whl` in the Python
environment used below. The build command packages but does not install it.
The SK switch is build-time only and defaults to ON, requiring a compiler/devkit
with `__sk__`, `SK_BIND` and `SkSystemArgs`. It reuses the packing SK_BIND structure
and does not change packing's independent `SGL_WELM_PACK_SUPERKERNEL` setting.

## Correctness and performance comparison

The script loads the **unchanged** original Triton kernel and wrapper from
`welmv4_npu_op.py`, preserving their source locations and default block size.
It imports only those two AST definitions to avoid requiring the complete
SGLang server stack. Pass the actual server source path when auto-discovery
of a sibling SGLang checkout does not find it.

```bash
python tests/python/sgl_kernel_npu/bench_welm_sigmoid_mul.py \
  --triton-source /path/to/sglang/python/sglang/srt/layers/welmv4_npu_op.py \
  --batch-sizes 1 2 4 6 8 --heads 24 --json sigmoid_tp1.json

# Attention TP2; similarly use --heads 6 for attention TP4.
python tests/python/sgl_kernel_npu/bench_welm_sigmoid_mul.py \
  --batch-sizes 1 2 4 6 8 16 32 64 128 --heads 12 --json sigmoid_tp2.json

# After building/installing the SK-enabled wheel:
python tests/python/sgl_kernel_npu/bench_welm_sigmoid_mul.py \
  --batch-sizes 4 --heads 24 --mode superkernel --json sigmoid_sk.json
```

Default: `B=4,H=24`, graph mode, 128 independent in-place calls per capture.
`--mode static` compiles the functional op with npugraph_ex/static kernel,
avoiding AOT functionalization of an alias-returning custom in-place op.
`--mode superkernel` uses that same functional op and additionally
scopes each native call separately and runs graph SuperKernel optimization.
Outputs are kept alive and checked for unique storage. These modes measure
the future functional attention-tail contract; the default graph mode compares
both in-place implementations. Triton retains its original in-place ordinary
NPUGraph timing in all modes. JSON labels the native contract explicitly.
Optional `--profile-dir`
uses the existing graph profiler to inspect actual SK participation; a selected
mode or successful Python call alone does not prove device fusion.

Correctness covers empty/max batches, tail head counts, functional/in-place
contracts, NaN/Inf, and changed inputs after capture. Both implementations must
be within one BF16 ULP of the CPU reference, with matching NaN/Inf locations
and Inf signs. An additional exact multiplication check isolates the required
intermediate BF16 rounding from possible sigmoid approximation differences.

Every timed replay begins with a reset **outside** the timed interval. Calls
within one graph use distinct slices of an attention arena. This prevents
in-place repetition from gradually turning the benchmark into multiplication
by already-zero outputs. Both implementations use the same gate and attention
values, independent-call count and reset procedure. Warmup is excluded and
measurement order alternates. Each reported round is the median of five
single-replay measurements; `median_us` is the median of seven rounds, divided
by the captured call count. Timing/profile helpers are reused from the packing
benchmark without modifying it.

JSON records loaded library path/hash, Triton source path/hash, shape, mode,
native functional/in-place contract,
all round samples and native/Triton speedup. Exit 0 requires native to be
strictly faster for every requested shape (`--min-speedup` can raise the bar);
exit 2 means the performance target was not met. Correctness failures raise.
The default in-place comparison does not establish functional-region or
end-to-end model latency.

This revision has no NPU compilation or timing result from the Windows
workspace. The Decode performance target is **faster than Triton**, and remains
pending a matched server run. Packing reaching parity does not establish a
speedup for this different, arithmetic operator.
