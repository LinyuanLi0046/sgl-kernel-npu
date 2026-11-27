// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include "kernel_operator.h"
#include "../sigmoid_mul_config.h"

namespace {
using namespace AscendC;

struct SigmoidMulArgs {
    GM_ADDR gate;
    GM_ADDR attention;
    GM_ADDR output;
    uint32_t rows;
    uint32_t tileRows;
    uint32_t activeCores;
};

// All gate nonlinearities are computed once per tile, before broadcasting.
// Intermediate sigmoid values stay in registers, including the BF16 rounding.
__simd_vf__ inline void SigmoidMulVf(
    __ubuf__ bfloat16_t *gate, __ubuf__ bfloat16_t *attention,
    __ubuf__ bfloat16_t *output, uint16_t rows)
{
    // CANN 9.1's host-stub pass parses namespace-scope declarations without
    // exposing MicroAPI. Keep register-only declarations inside the VF body.
    namespace Reg = AscendC::MicroAPI;
    // Cast takes its trait by template reference: retain static storage for
    // these compile-time constants, with no runtime initialization.
    static constexpr Reg::CastTrait TO_FLOAT = {
        Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait TO_BFLOAT = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr uint32_t lanes = VECTOR_REG_WIDTH / sizeof(float);
    static_assert(lanes == welm_sigmoid_mul::MAX_TILE_ROWS);
    uint32_t active = rows;
    Reg::MaskReg gateMask = Reg::UpdateMask<float>(active);
    Reg::MaskReg fullMask = Reg::CreateMask<float>();
    Reg::RegTensor<bfloat16_t> gateBf16, scaleBf16;
    Reg::RegTensor<float> gateFp32, negGate, expGate, denominator, ones, sigmoid, scale;
    Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(gateBf16, gate);
    Reg::Cast<float, bfloat16_t, TO_FLOAT>(gateFp32, gateBf16, gateMask);
    Reg::Muls(negGate, gateFp32, -1.0f, gateMask);
    Reg::Exp(expGate, negGate, gateMask);
    Reg::Adds(denominator, expGate, 1.0f, gateMask);
    Reg::Duplicate(ones, 1.0f, gateMask);
    Reg::Div(sigmoid, ones, denominator, gateMask);
    // Match sigmoid_mul_kernel_npu: round sigmoid to BF16 BEFORE multiplying.
    Reg::Cast<bfloat16_t, float, TO_BFLOAT>(scaleBf16, sigmoid, gateMask);
    Reg::Cast<float, bfloat16_t, TO_FLOAT>(scale, scaleBf16, gateMask);

    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<float> headScale, value, product;
    Reg::RegTensor<bfloat16_t> inputBf16, outputBf16;
    for (uint16_t row = 0; row < rows; ++row) {
        Reg::Duplicate(index, static_cast<uint32_t>(row), fullMask);
        Reg::Gather(headScale, scale, index);
#pragma unroll
        for (uint16_t part = 0; part < welm_sigmoid_mul::HEAD_DIM / lanes; ++part) {
            const uint32_t offset = row * welm_sigmoid_mul::HEAD_DIM + part * lanes;
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(inputBf16, attention + offset);
            Reg::Cast<float, bfloat16_t, TO_FLOAT>(value, inputBf16, fullMask);
            Reg::Mul(product, value, headScale, fullMask);
            Reg::Cast<bfloat16_t, float, TO_BFLOAT>(outputBf16, product, fullMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(output + offset, outputBf16, fullMask);
        }
    }
}

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

template <bool superKernel>
__aicore__ inline void RunSigmoidMul(const SigmoidMulArgs &args, uint32_t blockIdx)
{
    // The enclosing SuperKernel can have more cores than this child kernel.
    if (blockIdx >= args.activeCores) {
        return;
    }
    InitSocState();
    GlobalTensor<bfloat16_t> gateGm, attentionGm, outputGm;
    gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.gate));
    attentionGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.attention));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.output));
    LocalTensor<bfloat16_t> attentionUb(TPosition::VECIN, 0,
        welm_sigmoid_mul::ATTN_BUFFER_BYTES / sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> outputUb(TPosition::VECOUT, welm_sigmoid_mul::ATTN_BUFFER_BYTES,
        welm_sigmoid_mul::ATTN_BUFFER_BYTES / sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> gateUb(TPosition::VECIN, 2 * welm_sigmoid_mul::ATTN_BUFFER_BYTES,
        welm_sigmoid_mul::MAX_TILE_ROWS);
    const uint32_t rowStride = args.activeCores * args.tileRows;
    for (uint32_t start = blockIdx * args.tileRows; start < args.rows; start += rowStride) {
        const uint32_t count = args.rows - start < args.tileRows ? args.rows - start : args.tileRows;
        const uint32_t elements = count * welm_sigmoid_mul::HEAD_DIM;
        // Attention is always 512-byte aligned in length. Gate may end on any
        // BF16 element, so only its DMA needs the non-aligned copy interface.
        const DataCopyExtParams gateCopy{1, count * 2U, 0, 0, 0};
        const uint8_t gatePadding = static_cast<uint8_t>((16 - count % 16) % 16);
        const DataCopyPadExtParams<bfloat16_t> padding{gatePadding != 0, 0, gatePadding, 0};
        DataCopyPad(gateUb, gateGm[start], gateCopy, padding);
        DataCopy(attentionUb, attentionGm[start * welm_sigmoid_mul::HEAD_DIM], elements);
        // Both independent input DMAs precede this single producer/consumer join.
        Fence<HardEvent::MTE2_V>();
        asc_vf_call<SigmoidMulVf>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(gateUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(attentionUb.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(outputUb.GetPhyAddr()),
            static_cast<uint16_t>(count));
        Fence<HardEvent::V_MTE3>();
        DataCopy(outputGm[start * welm_sigmoid_mul::HEAD_DIM], outputUb, elements);
        if (start + rowStride < args.rows) {
            // Needed only when a limited-core device processes multiple tiles.
            // MTE3 completion also implies the preceding VF has read its inputs.
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
    if constexpr (superKernel) {
        Fence<HardEvent::MTE3_S>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
}
}  // namespace

// Keep the ordinary entry compatible with ascendc_library's auto_gen wrapper.
// Its AIV task type is independent of the separate __sk__ entry below.
extern "C" __global__ __aicore__ void welm_sigmoid_mul_kernel(
    GM_ADDR gate, GM_ADDR attention, GM_ADDR output,
    uint32_t rows, uint32_t tileRows, uint32_t activeCores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const SigmoidMulArgs args{gate, attention, output, rows, tileRows, activeCores};
    RunSigmoidMul<false>(args, GetBlockIdx());
}

#ifdef SGL_WELM_SIGMOID_MUL_SUPERKERNEL
template <uint32_t split>
__sk__ __vector__ void welm_sigmoid_mul_sk(const SigmoidMulArgs *args, sk::SkSystemArgs *sysArgs)
{
    RunSigmoidMul<true>(*args, sysArgs->SkGetBlockIdx());
}

SK_BIND(welm_sigmoid_mul_kernel, 0,
        welm_sigmoid_mul_sk<0>, welm_sigmoid_mul_sk<1>,
        welm_sigmoid_mul_sk<2>, welm_sigmoid_mul_sk<3>);
#endif
