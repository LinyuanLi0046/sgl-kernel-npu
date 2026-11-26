// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#ifndef SGL_WELM_SIGMOID_MUL_CONFIG_H
#define SGL_WELM_SIGMOID_MUL_CONFIG_H

#include <cstdint>

namespace welm_sigmoid_mul {
constexpr uint32_t HEAD_DIM = 256;
constexpr uint32_t MAX_TOKENS = 128;
constexpr uint32_t MAX_HEADS = 24;
// Start with 2 KiB of attention per core for small Decode batches. Larger
// batches coalesce adjacent heads, up to one FP32 register of gate values.
constexpr uint32_t TARGET_ROWS_PER_CORE = 4;
constexpr uint32_t MAX_TILE_ROWS = 64;
constexpr uint32_t ATTN_BUFFER_BYTES = MAX_TILE_ROWS * HEAD_DIM * sizeof(uint16_t);
constexpr uint32_t GATE_BUFFER_BYTES = MAX_TILE_ROWS * sizeof(uint16_t);
constexpr uint32_t UB_BYTES = 2 * ATTN_BUFFER_BYTES + GATE_BUFFER_BYTES;
}  // namespace welm_sigmoid_mul

#endif
