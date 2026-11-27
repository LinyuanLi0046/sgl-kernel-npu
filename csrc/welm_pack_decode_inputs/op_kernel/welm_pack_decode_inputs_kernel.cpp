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
    uint32_t hiddenBaseTiles;
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
__aicore__ inline void Fence()
{
    // Each event is consumed immediately. No queue or dynamic event allocation;
    // IDs 6 and 7 are reserved by the Ascend C static-Tensor APIs.
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

#ifdef SGL_WELM_PACK_A5
// Register-store pattern: ops-nn arch35 CopyPad::CustomDuplicate. Keep both
// routing tails in one VF, sharing the zero register and the loop/mask setup.
// The only VF memory accesses are masked UB stores; GM DMA stays outside VF.
template <bool paired>
__simd_vf__ inline void ZeroTailVf(__ubuf__ uint32_t *first, uint32_t words)
{
    constexpr uint32_t regWords = VECTOR_REG_WIDTH / sizeof(uint32_t);
    const uint16_t repeats = (words + regWords - 1) / regWords;
    uint32_t remaining = words;
    MicroAPI::RegTensor<uint32_t> zero;
    MicroAPI::Duplicate(zero, uint32_t{0});
    for (uint16_t i = 0; i < repeats; ++i) {
        // UpdateMask advances remaining; do not decrement it a second time.
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<uint32_t>(remaining);
        MicroAPI::StoreAlign(first + i * regWords, zero, mask);
        if constexpr (paired) {
            // The routing buffers have this fixed separation in RunPack.
            MicroAPI::StoreAlign(first + welm_pack::ROUTE_BUFFER_WORDS + i * regWords, zero, mask);
        }
    }
}
#endif

// The source is read only over the valid prefix. Padding is generated in UB;
// IDs and FP32 weights are also copied as raw words, preserving every bit.
__aicore__ inline void LoadPrefix(LocalTensor<uint32_t> local, GM_ADDR src,
                                 uint32_t offset, uint32_t valid)
{
    if (valid > 0) {
        GlobalTensor<uint32_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(src) + offset, valid);
        const DataCopyExtParams copyIn{1, valid * 4U, 0, 0, 0};
        // A partial route chunk need not end on 32 bytes (topk is 10).
        // Specify rightPadding explicitly: with both padding counts zero,
        // DMA alignment bytes may repeat the first element of the input.
        const uint8_t rightPadding = static_cast<uint8_t>(AlignWords(valid) - valid);
        const DataCopyPadExtParams<uint32_t> padding{rightPadding != 0, 0, rightPadding, 0};
        DataCopyPad(local, input, copyIn, padding);
    }
}

template <bool paired>
__aicore__ inline void ZeroTail(LocalTensor<uint32_t> first, LocalTensor<uint32_t> second,
                               uint32_t count, uint32_t valid)
{
    const uint32_t alignedValid = AlignWords(valid);
    const uint32_t alignedCount = AlignWords(count);
    if (alignedValid < alignedCount) {
        // The DMA (including its right padding) and vector fill write disjoint
        // 32-byte blocks. No V -> MTE2 dependency is needed between them.
#ifdef SGL_WELM_PACK_A5
        auto firstAddr = reinterpret_cast<__ubuf__ uint32_t *>(first.GetPhyAddr()) + alignedValid;
        asc_vf_call<ZeroTailVf<paired>>(firstAddr, alignedCount - alignedValid);
#else
        Duplicate(first[alignedValid], uint32_t{0}, alignedCount - alignedValid);
        if constexpr (paired) {
            Duplicate(second[alignedValid], uint32_t{0}, alignedCount - alignedValid);
        }
#endif
    }
}

__aicore__ inline void WaitPadded(uint32_t count, uint32_t valid)
{
    if (valid > 0) {
        Fence<HardEvent::MTE2_MTE3>();
    }
    if (AlignWords(valid) < AlignWords(count)) {
        Fence<HardEvent::V_MTE3>();
    }
}

__aicore__ inline void StorePadded(LocalTensor<uint32_t> local, GM_ADDR dst,
                                  uint32_t offset, uint32_t count)
{
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(dst) + offset, count);
    const DataCopyExtParams copyOut{1, count * 4U, 0, 0, 0};
    DataCopyPad(output, local, copyOut);
}

template <bool superKernel>
__aicore__ inline void RunPack(const PackArgs &args, uint32_t blockIdx)
{
    const uint32_t capacity = args.capacity;
    const uint32_t hiddenCores = args.hiddenCores;
    const uint32_t routeCores = capacity > 64 ? 2 : 1;
    // Do not use the fused SuperKernel's total block count to partition work:
    // the original partition is encoded in scalar args. Extra blocks are idle.
    if (blockIdx >= hiddenCores + routeCores) {
        return;
    }
    // Static LocalTensor programming requires explicit hardware-state init.
    // TPipe used to perform it; do not assume the preceding SK left defaults.
    InitSocState();
    if (blockIdx < hiddenCores) {
        const uint32_t tiles = capacity * welm_pack::HIDDEN_WORDS / welm_pack::TILE_WORDS;
        const uint32_t baseTiles = args.hiddenBaseTiles;
        const uint32_t extraTiles = tiles - baseTiles * hiddenCores;
        const uint32_t startTile = blockIdx * baseTiles + Min(blockIdx, extraTiles);
        const uint32_t offset = startTile * welm_pack::TILE_WORDS;
        const uint32_t count = (baseTiles + (blockIdx < extraTiles)) * welm_pack::TILE_WORDS;
        const uint32_t validTotal = args.rows * welm_pack::HIDDEN_WORDS;
        const uint32_t valid = offset < validTotal ? Min(count, validTotal - offset) : 0;
        LocalTensor<uint32_t> local(TPosition::VECIN, 0, count);
        LoadPrefix(local, args.hidden, offset, valid);
        ZeroTail<false>(local, local, count, valid);
        WaitPadded(count, valid);
        StorePadded(local, args.hiddenOut, offset, count);
    } else {
        const uint32_t routeCore = blockIdx - hiddenCores;
        const uint32_t total = capacity * welm_pack::TOPK;
        const uint32_t perCore = AlignWords(capacity > 64 ? (total + 1) / 2 : total);
        const uint32_t offset = routeCore * perCore;
        const uint32_t count = Min(perCore, total - offset);
        const uint32_t validTotal = args.rows * welm_pack::TOPK;
        const uint32_t valid = offset < validTotal ? Min(count, validTotal - offset) : 0;
        LocalTensor<uint32_t> idsLocal(TPosition::VECIN, 0, welm_pack::ROUTE_BUFFER_WORDS);
        LocalTensor<uint32_t> weightsLocal(TPosition::VECIN,
            welm_pack::ROUTE_BUFFER_WORDS * sizeof(uint32_t), welm_pack::ROUTE_BUFFER_WORDS);
        // Submit both arrays before waiting. Their input copies and zero fills
        // are independent; one pair of events covers both arrays' producers.
        LoadPrefix(idsLocal, args.ids, offset, valid);
        LoadPrefix(weightsLocal, args.weights, offset, valid);
        ZeroTail<true>(idsLocal, weightsLocal, count, valid);
        WaitPadded(count, valid);
        StorePadded(idsLocal, args.idsOut, offset, count);
        StorePadded(weightsLocal, args.weightsOut, offset, count);
    }
    if constexpr (superKernel) {
        // All input/zero-fill producers have joined MTE3 above. Drain output
        // DMA before UB reuse by the next SK. The direct-launch SK contract
        // omits PIPE_ALL teardown; this fence is local to the current core.
        Fence<HardEvent::MTE3_S>();
    } else {
        // Required at an ordinary static-Tensor kernel's exit. No extra
        // MTE3_S fence followed by TPipe's implicit PIPE_ALL teardown.
        PipeBarrier<PIPE_ALL>();
    }
}
}  // namespace

// Keep the ordinary entry compatible with ascendc_library's auto_gen wrapper.
// Its AIV task type is independent of the separate __sk__ entry below.
extern "C" __global__ __aicore__ void welm_pack_decode_inputs_kernel(
    GM_ADDR hidden, GM_ADDR ids, GM_ADDR weights,
    GM_ADDR hiddenOut, GM_ADDR idsOut, GM_ADDR weightsOut,
    uint32_t rows, uint32_t capacity, uint32_t hiddenCores, uint32_t hiddenBaseTiles)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const PackArgs args{hidden, ids, weights, hiddenOut, idsOut, weightsOut,
                        rows, capacity, hiddenCores, hiddenBaseTiles};
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
