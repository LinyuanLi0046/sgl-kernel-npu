// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#pragma once

#include <cstddef>
#include <cstdint>

namespace sglang::npu_kernel::welm {
// One argument layout for the host launch and the SK entry. Address is void*
// on the host and GM_ADDR on the device; neither layout contains a tiling tensor
// or the overflow pointer injected by the legacy auto_gen wrapper.
template <typename Address>
struct SigmoidMulArgs {
    Address gate;
    Address attention;
    Address output;
    uint32_t rows;
    uint32_t tileRows;
    uint32_t activeCores;
};

template <typename Address>
struct PackArgs {
    Address hidden;
    Address ids;
    Address weights;
    Address hiddenOut;
    Address idsOut;
    Address weightsOut;
    uint32_t rows;
    uint32_t capacity;
    uint32_t hiddenCores;
    uint32_t hiddenBaseTiles;
};

static_assert(sizeof(SigmoidMulArgs<void *>) == 40);
static_assert(offsetof(SigmoidMulArgs<void *>, rows) == 24);
static_assert(offsetof(SigmoidMulArgs<void *>, activeCores) == 32);
static_assert(sizeof(PackArgs<void *>) == 64);
static_assert(offsetof(PackArgs<void *>, rows) == 48);
static_assert(offsetof(PackArgs<void *>, hiddenBaseTiles) == 60);
}  // namespace sglang::npu_kernel::welm
