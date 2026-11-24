"""Fused input packing for the low-latency NPU ops strategy."""

from functools import lru_cache
from typing import Tuple

import torch


@lru_cache(maxsize=1)
def _get_pack_kernel():
    # Keep importing DeepEP and validating inputs independent of Triton/CANN.
    # Only the JIT object is cached; all output tensors belong to their caller.
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["num_rows"])
    def _pack_low_latency_inputs_kernel(
        x_ptr,
        expert_ids_ptr,
        topk_weights_ptr,
        x_out_ptr,
        expert_ids_out_ptr,
        topk_weights_out_ptr,
        num_rows,
        HIDDEN: tl.constexpr,
        TOPK: tl.constexpr,
        CAPACITY: tl.constexpr,
        X_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_SIZE)
        if block_id < X_BLOCKS:
            offsets = block_id.to(tl.int64) * BLOCK_SIZE + lanes
            values = tl.load(
                x_ptr + offsets, mask=offsets < num_rows * HIDDEN, other=0
            )
            tl.store(x_out_ptr + offsets, values, mask=offsets < CAPACITY * HIDDEN)
        else:
            offsets = (block_id - X_BLOCKS).to(tl.int64) * BLOCK_SIZE + lanes
            valid_input = offsets < num_rows * TOPK
            valid_output = offsets < CAPACITY * TOPK
            expert_ids = tl.load(
                expert_ids_ptr + offsets, mask=valid_input, other=0
            )
            topk_weights = tl.load(
                topk_weights_ptr + offsets, mask=valid_input, other=0
            )
            tl.store(expert_ids_out_ptr + offsets, expert_ids, mask=valid_output)
            tl.store(topk_weights_out_ptr + offsets, topk_weights, mask=valid_output)

    return _pack_low_latency_inputs_kernel


def pack_low_latency_inputs(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy the active prefix and zero all padded rows in one device launch.

    Inputs must be contiguous NPU tensors with shapes ``[B, H]``, ``[B, K]``
    and ``[B, K]``. Hidden states support BF16/FP16, expert IDs require INT32,
    and route weights require FP32. ``0 <= B <= capacity`` is supported.
    Inputs are never modified. When ``B == capacity``, return them unchanged;
    otherwise allocate fresh outputs, including for ``B == 0``.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")

    for name, tensor in (
        ("x", x),
        ("expert_ids", expert_ids),
        ("topk_weights", topk_weights),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.ndim != 2:
            raise ValueError(f"{name} must be a strided two-dimensional tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    num_rows, hidden = x.shape
    if hidden <= 0 or expert_ids.shape[1] <= 0:
        raise ValueError("hidden size and top-k must be positive")
    if expert_ids.shape[0] != num_rows or topk_weights.shape != expert_ids.shape:
        raise ValueError("expected x[B, H], expert_ids[B, K], topk_weights[B, K]")
    if num_rows > capacity:
        raise ValueError("capacity must be at least the input row count")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("x must have dtype torch.bfloat16 or torch.float16")
    if expert_ids.dtype != torch.int32:
        raise TypeError("expert_ids must have dtype torch.int32")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must have dtype torch.float32")
    if x.device.type != "npu":
        raise ValueError("inputs must be NPU tensors")
    if expert_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("all inputs must be on the same NPU device")

    if num_rows == capacity:
        return x, expert_ids, topk_weights

    kernel = _get_pack_kernel()
    topk = expert_ids.shape[1]
    x_out = torch.empty((capacity, hidden), dtype=x.dtype, device=x.device)
    expert_ids_out = torch.empty(
        (capacity, topk), dtype=expert_ids.dtype, device=expert_ids.device
    )
    topk_weights_out = torch.empty(
        (capacity, topk), dtype=topk_weights.dtype, device=topk_weights.device
    )
    block_size = 1024
    x_blocks = (capacity * hidden + block_size - 1) // block_size
    route_blocks = (capacity * topk + block_size - 1) // block_size
    kernel[(x_blocks + route_blocks,)](
        x,
        expert_ids,
        topk_weights,
        x_out,
        expert_ids_out,
        topk_weights_out,
        num_rows,
        HIDDEN=hidden,
        TOPK=topk,
        CAPACITY=capacity,
        X_BLOCKS=x_blocks,
        BLOCK_SIZE=block_size,
    )
    return x_out, expert_ids_out, topk_weights_out
