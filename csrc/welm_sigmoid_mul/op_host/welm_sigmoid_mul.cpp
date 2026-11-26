// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include "aclrtlaunch_welm_sigmoid_mul_kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"
#include "../sigmoid_mul_config.h"

#include <algorithm>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>

namespace sglang::npu_kernel {
namespace {
void CheckMetadata(const at::Tensor &gate, const at::Tensor &attention)
{
    TORCH_CHECK(gate.dim() == 3 && attention.dim() == 3 &&
                    gate.size(0) == attention.size(0) && gate.size(1) == attention.size(1) &&
                    gate.size(2) == 1 && attention.size(2) == welm_sigmoid_mul::HEAD_DIM,
                "welm_sigmoid_mul expects gate[B,H,1] and attention[B,H,256]");
    TORCH_CHECK(gate.size(0) <= welm_sigmoid_mul::MAX_TOKENS &&
                    gate.size(1) > 0 && gate.size(1) <= welm_sigmoid_mul::MAX_HEADS,
                "welm_sigmoid_mul supports 0 <= B <= 128 and 1 <= H <= 24");
    TORCH_CHECK(gate.scalar_type() == at::kBFloat16 && attention.scalar_type() == at::kBFloat16,
                "welm_sigmoid_mul requires BF16 gate and attention");
    TORCH_CHECK(gate.is_contiguous() && attention.is_contiguous(),
                "welm_sigmoid_mul requires contiguous inputs");
    TORCH_CHECK(gate.device() == attention.device(), "welm_sigmoid_mul requires inputs on the same device");
}

void Launch(const at::Tensor &gate, const at::Tensor &attention, const at::Tensor &output)
{
    const uint32_t rows = static_cast<uint32_t>(gate.numel());
    if (rows == 0) {
        return;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(attention.data_ptr()) % 32 == 0,
                "welm_sigmoid_mul requires a 32-byte-aligned attention pointer");
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t availableCores = platform->GetCoreNumAiv();
    TORCH_CHECK(availableCores > 0, "welm_sigmoid_mul requires AIV cores");
    const uint32_t requestedCores =
        (rows + welm_sigmoid_mul::TARGET_ROWS_PER_CORE - 1) / welm_sigmoid_mul::TARGET_ROWS_PER_CORE;
    const uint32_t activeCores = std::min(availableCores, requestedCores);
    const uint32_t tileRows = std::min(welm_sigmoid_mul::MAX_TILE_ROWS,
        (rows + activeCores - 1) / activeCores);
    // Recompute the launch count after rounding up rows/core: do not launch
    // empty cores (e.g. six heads are split into two tiles of three heads).
    const uint32_t blockDim = std::min(activeCores, (rows + tileRows - 1) / tileRows);
    EXEC_KERNEL_CMD(welm_sigmoid_mul_kernel, blockDim, gate, attention, output, rows, tileRows, blockDim);
}

at::Tensor SigmoidMulMeta(const at::Tensor &gate, const at::Tensor &attention)
{
    CheckMetadata(gate, attention);
    return at::empty(attention.sizes(), attention.options());
}

at::Tensor &SigmoidMulInplaceMeta(const at::Tensor &gate, at::Tensor &attention)
{
    CheckMetadata(gate, attention);
    return attention;
}

at::Tensor SigmoidMulNpu(const at::Tensor &gate, const at::Tensor &attention)
{
    CheckMetadata(gate, attention);
    TORCH_CHECK(gate.is_privateuseone(), "welm_sigmoid_mul requires NPU tensors");
    const c10::OptionalDeviceGuard guard(gate.device());
    auto output = at::empty(attention.sizes(), attention.options());
    Launch(gate, attention, output);
    return output;
}

at::Tensor &SigmoidMulInplaceNpu(const at::Tensor &gate, at::Tensor &attention)
{
    CheckMetadata(gate, attention);
    TORCH_CHECK(gate.is_privateuseone(), "welm_sigmoid_mul_ requires NPU tensors");
    at::assert_no_overlap(attention, gate);
    const c10::OptionalDeviceGuard guard(gate.device());
    Launch(gate, attention, attention);
    return attention;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("welm_sigmoid_mul(Tensor gate, Tensor attention) -> Tensor");
    m.def("welm_sigmoid_mul_(Tensor gate, Tensor(a!) attention) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("welm_sigmoid_mul", TORCH_FN(SigmoidMulNpu));
    m.impl("welm_sigmoid_mul_", TORCH_FN(SigmoidMulInplaceNpu));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("welm_sigmoid_mul", TORCH_FN(SigmoidMulMeta));
    m.impl("welm_sigmoid_mul_", TORCH_FN(SigmoidMulInplaceMeta));
}
}  // namespace sglang::npu_kernel
