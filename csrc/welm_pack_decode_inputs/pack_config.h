// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#ifndef SGL_WELM_PACK_DECODE_CONFIG_H
#define SGL_WELM_PACK_DECODE_CONFIG_H

#include <cstdint>

namespace welm_pack {
constexpr uint32_t HIDDEN = 2048;
constexpr uint32_t TOPK = 10;
constexpr uint32_t MAX_CAPACITY = 128;
// Copy BF16 as raw 32-bit words (two BF16 values); no numeric conversion.
constexpr uint32_t HIDDEN_WORDS = HIDDEN / 2;
// Same 2048-byte hidden tile as the original Triton's 1024 BF16 elements.
constexpr uint32_t TILE_WORDS = 512;
constexpr uint32_t ALIGN_WORDS = 8;  // 32-byte UB alignment
}  // namespace welm_pack

#endif
