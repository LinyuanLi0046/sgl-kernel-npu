// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include "acl/acl.h"
#include "aclrtlaunch_welm_pack_decode_inputs_kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"
#include "../pack_config.h"

#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <tuple>

namespace sglang::npu_kernel {
namespace {
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

void CheckMetadata(const at::Tensor &x, const at::Tensor &ids, const at::Tensor &weights, int64_t capacity)
{
    TORCH_CHECK(x.dim() == 2 && x.size(1) == welm_pack::HIDDEN,
                "welm_pack_decode_inputs: hidden must have shape [N, 2048]");
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) == x.size(0) && ids.size(1) == welm_pack::TOPK,
                "welm_pack_decode_inputs: expert_ids must have shape [N, 10]");
    TORCH_CHECK(weights.sizes() == ids.sizes(),
                "welm_pack_decode_inputs: weights must have shape [N, 10]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && ids.scalar_type() == at::kInt &&
                    weights.scalar_type() == at::kFloat,
                "welm_pack_decode_inputs requires BF16 hidden, INT32 IDs, FP32 weights");
    TORCH_CHECK(x.is_contiguous() && ids.is_contiguous() && weights.is_contiguous(),
                "welm_pack_decode_inputs requires contiguous inputs");
    TORCH_CHECK(x.device() == ids.device() && x.device() == weights.device(),
                "welm_pack_decode_inputs requires all inputs on the same device");
    TORCH_CHECK(capacity > 0 && capacity <= welm_pack::MAX_CAPACITY && x.size(0) <= capacity,
                "welm_pack_decode_inputs requires 0 <= N <= capacity <= 128 and capacity > 0");
}

Outputs AllocateOutputs(const at::Tensor &x, const at::Tensor &ids, const at::Tensor &weights, int64_t capacity)
{
    return {at::empty({capacity, welm_pack::HIDDEN}, x.options()),
            at::empty({capacity, welm_pack::TOPK}, ids.options()),
            at::empty({capacity, welm_pack::TOPK}, weights.options())};
}

Outputs PackMeta(const at::Tensor &x, const at::Tensor &ids, const at::Tensor &weights, int64_t capacity)
{
    CheckMetadata(x, ids, weights, capacity);
    return AllocateOutputs(x, ids, weights, capacity);
}

Outputs PackNpu(const at::Tensor &x, const at::Tensor &ids, const at::Tensor &weights, int64_t capacity)
{
    CheckMetadata(x, ids, weights, capacity);
    TORCH_CHECK(x.is_privateuseone(), "welm_pack_decode_inputs requires NPU inputs");
    const c10::OptionalDeviceGuard guard(x.device());
    auto outputs = AllocateOutputs(x, ids, weights, capacity);
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t availableCores = platform->GetCoreNumAiv();
    TORCH_CHECK(availableCores >= 4, "welm_pack_decode_inputs requires at least four AIV cores");
    const uint32_t rows = static_cast<uint32_t>(x.size(0));
    const uint32_t cap = static_cast<uint32_t>(capacity);
    const uint32_t routeCores = cap > 64 ? 2 : 1;
    const uint32_t hiddenTiles = cap * welm_pack::HIDDEN_WORDS / welm_pack::TILE_WORDS;
    const uint32_t hiddenCores = std::min(availableCores - routeCores, hiddenTiles);
    const uint32_t blockDim = hiddenCores + routeCores;
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t maxHiddenBytes =
        (hiddenTiles + hiddenCores - 1) / hiddenCores * welm_pack::TILE_WORDS * sizeof(uint32_t);
    TORCH_CHECK(maxHiddenBytes <= ubBytes && 2 * cap * welm_pack::TOPK * sizeof(uint32_t) <= ubBytes,
                "welm_pack_decode_inputs: insufficient UB for the device's AIV core count");
    // All tiling attributes are scalar launch arguments. No tiling tensor,
    // H2D copy, workspace memset, device query of input values, or collective.
    EXEC_KERNEL_CMD(welm_pack_decode_inputs_kernel, blockDim,
                    x, ids, weights, std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs),
                    rows, cap, hiddenCores, routeCores);
    return outputs;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    // Always fresh outputs, including N == capacity: no hidden alias contract.
    // The dispatcher already bypasses packing for N == capacity.
    m.def("welm_pack_decode_inputs(Tensor hidden, Tensor expert_ids, Tensor weights, int capacity=128) "
          "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("welm_pack_decode_inputs", TORCH_FN(PackNpu));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("welm_pack_decode_inputs", TORCH_FN(PackMeta));
}
}  // namespace sglang::npu_kernel
