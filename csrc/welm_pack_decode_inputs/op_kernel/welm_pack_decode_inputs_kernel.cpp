// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include "kernel_operator.h"
#include "../pack_config.h"

namespace {
using namespace AscendC;

struct PackArgs {
    GM_ADDR hidden;
    GM_ADDR ids;
    GM_ADDR weights;
    GM_ADDR hiddenOut;
    GM_ADDR idsOut;
    GM_ADDR weightsOut;
    uint32_t rows;
    uint32_t capacity;
    uint32_t hiddenCores;
    uint32_t routeCores;
};

__aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t AlignWords(uint32_t n)
{
    return (n + welm_pack::ALIGN_WORDS - 1) / welm_pack::ALIGN_WORDS * welm_pack::ALIGN_WORDS;
}

template <HardEvent event>
__aicore__ inline void Fence(TPipe &pipe)
{
    const int32_t id = static_cast<int32_t>(pipe.FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

// The source is read only over the valid prefix. Padding is generated in UB;
// IDs and FP32 weights are also copied as raw words, preserving every bit.
__aicore__ inline void CopyPadded(TPipe &pipe, LocalTensor<uint32_t> local,
                                 GM_ADDR src, GM_ADDR dst, uint32_t offset,
                                 uint32_t count, uint32_t validTotal)
{
    const uint32_t valid = offset < validTotal ? Min(count, validTotal - offset) : 0;
    if (valid < count) {
        Duplicate(local, uint32_t{0}, AlignWords(count));
        if (valid > 0) {
            Fence<HardEvent::V_MTE2>(pipe);
        } else {
            Fence<HardEvent::V_MTE3>(pipe);
        }
    }
    if (valid > 0) {
        GlobalTensor<uint32_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(src) + offset, valid);
        const DataCopyExtParams copyIn{1, valid * 4U, 0, 0, 0};
        // A partial route chunk need not end on 32 bytes (topk is 10).
        // Specify rightPadding explicitly: with both padding counts zero,
        // DMA alignment bytes may repeat the first element of the input.
        const DataCopyPadExtParams<uint32_t> padding{
            true, 0, static_cast<uint8_t>(AlignWords(valid) - valid), 0};
        DataCopyPad(local, input, copyIn, padding);
        Fence<HardEvent::MTE2_MTE3>(pipe);
    }
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(dst) + offset, count);
    const DataCopyExtParams copyOut{1, count * 4U, 0, 0, 0};
    DataCopyPad(output, local, copyOut);
}

template <bool superKernel>
__aicore__ inline void RunPack(const PackArgs &args, uint32_t blockIdx)
{
    // Do not use the fused SuperKernel's total block count to partition work:
    // the original partition is encoded in scalar args. Extra blocks are idle.
    if (blockIdx >= args.hiddenCores + args.routeCores) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECIN> buffer;
    TBuf<TPosition::VECIN> weightsBuffer;
    if (blockIdx < args.hiddenCores) {
        const uint32_t tiles = args.capacity * welm_pack::HIDDEN_WORDS / welm_pack::TILE_WORDS;
        const uint32_t baseTiles = tiles / args.hiddenCores;
        const uint32_t extraTiles = tiles % args.hiddenCores;
        const uint32_t startTile = blockIdx * baseTiles + Min(blockIdx, extraTiles);
        const uint32_t count = (baseTiles + (blockIdx < extraTiles)) * welm_pack::TILE_WORDS;
        pipe.InitBuffer(buffer, count * sizeof(uint32_t));
        CopyPadded(pipe, buffer.Get<uint32_t>(), args.hidden, args.hiddenOut,
                   startTile * welm_pack::TILE_WORDS, count, args.rows * welm_pack::HIDDEN_WORDS);
    } else {
        const uint32_t routeCore = blockIdx - args.hiddenCores;
        const uint32_t total = args.capacity * welm_pack::TOPK;
        const uint32_t perCore = AlignWords((total + args.routeCores - 1) / args.routeCores);
        const uint32_t offset = routeCore * perCore;
        const uint32_t count = Min(perCore, total - offset);
        pipe.InitBuffer(buffer, AlignWords(count) * sizeof(uint32_t));
        pipe.InitBuffer(weightsBuffer, AlignWords(count) * sizeof(uint32_t));
        CopyPadded(pipe, buffer.Get<uint32_t>(), args.ids, args.idsOut,
                   offset, count, args.rows * welm_pack::TOPK);
        // Separate UB storage allows the preceding MTE3 store to overlap this
        // input copy/zero-fill; no reuse fence between IDs and weights.
        CopyPadded(pipe, weightsBuffer.Get<uint32_t>(), args.weights, args.weightsOut,
                   offset, count, args.rows * welm_pack::TOPK);
    }
    // Drain this core's output DMA before returning to another SK subkernel.
    // This is a local pipeline fence, not an inter-core/global synchronization.
    Fence<HardEvent::MTE3_S>(pipe);
#ifdef SGL_WELM_PACK_SUPERKERNEL
    if constexpr (superKernel) {
        pipe.DestroyWithoutPipeAll();
    }
#endif
}
}  // namespace

#ifdef SGL_WELM_PACK_SUPERKERNEL
extern "C" __global__ __vector__ void welm_pack_decode_inputs_kernel(
#else
extern "C" __global__ __aicore__ void welm_pack_decode_inputs_kernel(
#endif
    GM_ADDR hidden, GM_ADDR ids, GM_ADDR weights,
    GM_ADDR hiddenOut, GM_ADDR idsOut, GM_ADDR weightsOut,
    uint32_t rows, uint32_t capacity, uint32_t hiddenCores, uint32_t routeCores)
{
#ifndef SGL_WELM_PACK_SUPERKERNEL
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    const PackArgs args{hidden, ids, weights, hiddenOut, idsOut, weightsOut,
                        rows, capacity, hiddenCores, routeCores};
    RunPack<false>(args, AscendC::GetBlockIdx());
}

#ifdef SGL_WELM_PACK_SUPERKERNEL
template <uint32_t split>
__sk__ __vector__ void welm_pack_decode_inputs_sk(const PackArgs *args, sk::SkSystemArgs *sysArgs)
{
    RunPack<true>(*args, sysArgs->SkGetBlockIdx());
}

SK_BIND(welm_pack_decode_inputs_kernel, 0,
        welm_pack_decode_inputs_sk<0>, welm_pack_decode_inputs_sk<1>,
        welm_pack_decode_inputs_sk<2>, welm_pack_decode_inputs_sk<3>);
#endif
