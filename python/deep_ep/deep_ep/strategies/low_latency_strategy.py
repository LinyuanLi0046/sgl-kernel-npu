"""
Low latency mode EP communication strategies.
All low latency mode strategy implementations are in this file.
"""

import os
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from deep_ep_cpp import EventHandle

from ..ep_strategy import LowLatencyEPCommStrategy, register_low_latency_strategy
from ..utils import EventOverlap


@register_low_latency_strategy("default")
class DefaultLowLatencyCommStrategy(LowLatencyEPCommStrategy):
    """
    Low latency mode strategy using Custom operator implementation (deep_ep_cpp).
    This is the default implementation for low latency mode.
    """

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self.runtime = runtime

    def get_name(self) -> str:
        return "default"

    def get_supported_modes(self) -> List[str]:
        return ["low_latency"]

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        use_mxfp4: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        topk_weights: Optional[torch.Tensor] = None,
        quant_mode: Optional[str] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        Tuple,
        EventOverlap,
        Callable,
    ]:
        topk_ids = topk_idx.int()

        valid_quant_modes = {
            None,
            "int8",
            "mx_fp8_e4m3",
            "mx_fp8_e5m2",
            "pertoken_fp8_e4m3",
            "mx_fp4_e2m1",
        }
        if quant_mode not in valid_quant_modes:
            raise ValueError(f"Unsupported quant_mode: {quant_mode}")

        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            event,
            hook,
        ) = self.runtime.low_latency_dispatch(
            x,
            topk_ids,
            cumulative_local_expert_recv_stats,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8,
            round_scale,
            use_ue8m0,
            use_mxfp4,
            async_finish,
            return_recv_hook,
            "none" if quant_mode is None else quant_mode,
        )

        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
            packed_recv_count,
            None,  # expand_scales
        )

        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            cumulative_local_expert_recv_stats,
        )

        return (
            (
                (packed_recv_x, packed_recv_x_scales)
                if quant_mode is not None
                else packed_recv_x
            ),
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        topk_ids = topk_idx.int()

        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            packed_recv_count,
            expand_scales,
        ) = handle

        combined_x, event, hook = self.runtime.low_latency_combine(
            x,
            topk_ids,
            topk_weights,
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            packed_recv_count,
            zero_copy,
            async_finish,
            return_recv_hook,
            out,
        )

        tensors_to_record = (
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )

        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )


@register_low_latency_strategy("ops")
class OpsLowLatencyCommStrategy(LowLatencyEPCommStrategy):
    """Low-latency dispatch and combine using torch_npu MC2 V2 ops.

    Fused padding is opt-in and requires the same, unchanged route weights
    between dispatch and combine (as in the SGLang Ascend runner).
    """

    def __init__(self, runtime, group: dist.ProcessGroup, comm_alg: Optional[str] = None):
        super().__init__(group)
        is_950dt = torch_npu.npu.get_device_name().startswith("Ascend950DT")
        # On 950DT, CANN 9.1 MTE uses fullmesh_v1 dispatch and default combine.
        self._use_950dt_mte = is_950dt and comm_alg in (None, "", "fullmesh_v1")
        if comm_alg is None:
            comm_alg = "fullmesh_v1" if is_950dt else "hierarchy"
        comm_alg_support_list = ["hierarchy", "fullmesh_v1", "fullmesh_v2", "ccu", ""]
        torch._check(
            comm_alg in comm_alg_support_list,
            lambda: (
                f"comm_alg only support {comm_alg_support_list=}, but got {comm_alg=}."
            ),
        )
        self.comm_alg = comm_alg
        self._fused_padding_enabled = os.getenv("DEEPEP_OPS_FUSED_PADDING", "0") == "1"
        self._active_mask_templates = {}

    def get_welm_compile_callables(self, hidden_size, topk, dtype, capacity):
        """Bind tensor-only calls; compilation state belongs to the caller's layer.

        This does not change the process-shared strategy or its padding policy.
        Resolve communication attributes and masks before FX tracing/capture.
        """
        if not self._use_950dt_mte:
            raise ValueError("WeLM full MoE requires Ascend950DT ops MTE dispatch")
        if (hidden_size, topk, dtype) != (2048, 10, torch.bfloat16):
            raise ValueError("WeLM AscendC pack requires BF16 hidden=2048 and topk=10")
        if not 0 < capacity <= 128:
            raise ValueError("WeLM AscendC pack requires dispatch capacity <= 128")
        shared_ranks = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
        counts_type = int(os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1))
        if shared_ranks != 0 or counts_type != 1:
            raise ValueError(
                "WeLM full MoE requires MOE_SHARED_EXPERT_RANK_NUM=0 and "
                "MOE_EXPERT_TOKEN_NUMS_TYPE=1 (per-expert counts)"
            )
        import sgl_kernel_npu  # noqa: F401

        if not hasattr(torch.ops.npu, "welm_pack_decode_inputs"):
            raise RuntimeError("Install sgl-kernel-npu with welm_pack_decode_inputs")
        group = dict(
            group_ep=self.group_name,
            ep_world_size=self.group_size,
            ep_rank_id=self.rank,
            shared_expert_rank_num=shared_ranks,
        )
        if not group["group_ep"] or group["ep_world_size"] <= 1:
            raise ValueError(
                "WeLM full MoE requires an initialized EP communication group"
            )
        self.prepare_low_latency_mask(capacity)
        return dict(
            pack=self._welm_pack_inputs,
            dispatch=partial(
                self._npu_low_latency_dispatch,
                **group,
                expert_token_nums_type=counts_type,
                cast_counts_to_int64=True,
            ),
            combine=partial(self._npu_low_latency_combine, **group),
        )

    @staticmethod
    def _welm_pack_inputs(x, topk_ids, topk_weights, capacity):
        return torch.ops.npu.welm_pack_decode_inputs(x, topk_ids, topk_weights, capacity)

    def _can_fuse_padding(
        self,
        x,
        topk_ids,
        topk_weights,
        capacity,
        use_fp8,
        use_ue8m0,
        use_mxfp4,
        quant_mode,
    ):
        return (
            self._use_950dt_mte
            and not (use_fp8 or use_ue8m0 or use_mxfp4)
            and quant_mode in (None, "bf16")
            and x.device.type == "npu"
            and x.ndim == 2
            and topk_ids.ndim == 2
            and topk_weights is not None
            and topk_weights.ndim == 2
            and isinstance(capacity, int)
            and not isinstance(capacity, bool)
            and 0 <= x.size(0) <= capacity
            and capacity > 0
            and x.size(1) > 0
            and topk_ids.size(1) > 0
            and topk_ids.size(0) == x.size(0)
            and topk_weights.shape == topk_ids.shape
            and x.dtype in (torch.bfloat16, torch.float16)
            and topk_ids.dtype == torch.int32
            and topk_weights.dtype == torch.float32
            and x.device == topk_ids.device == topk_weights.device
            and all(t.is_contiguous() for t in (x, topk_ids, topk_weights))
        )

    def prepare_low_latency_mask(self, capacity: int, device=None) -> torch.Tensor:
        """Prebuild an immutable prefix mask template, normally before capture."""
        device = (
            torch.device("npu", torch.npu.current_device())
            if device is None
            else torch.device(device)
        )
        key = (capacity, device)
        template = self._active_mask_templates.get(key)
        if template is None:
            # Every N <= C uses a contiguous C-element view of [True*C, False*C].
            # Never rewrite this storage: dispatch handles and captured graphs
            # with different N may still reference it at combine time.
            template = torch.empty(2 * capacity, dtype=torch.bool, device=device)
            template[:capacity].fill_(True)
            template[capacity:].zero_()
            self._active_mask_templates[key] = template
        return template

    def get_name(self) -> str:
        return "ops"

    def get_supported_modes(self) -> List[str]:
        return ["low_latency"]

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        use_mxfp4: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        topk_weights: Optional[torch.Tensor] = None,
        quant_mode: Optional[str] = None,
        compile_adapter=None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        Tuple,
        EventOverlap,
        Callable,
    ]:

        topk_ids = topk_idx.int()
        padding_size = num_max_dispatch_tokens_per_rank - x.size(0)
        if padding_size < 0:
            raise ValueError("Input rows exceed the ops low-latency dispatch capacity")
        if compile_adapter is not None and (
            num_max_dispatch_tokens_per_rank > 128
            or not self._can_fuse_padding(
                x,
                topk_ids,
                topk_weights,
                num_max_dispatch_tokens_per_rank,
                use_fp8,
                use_ue8m0,
                use_mxfp4,
                quant_mode,
            )
        ):
            raise ValueError(
                "WeLM full MoE dispatch no longer satisfies its BF16 pack contract"
            )
        if self.comm_alg == "hierarchy":
            assert (
                topk_weights is not None
            ), "When comm_alg='hierarchy', topk_weights can not be None"
        # MTE applies the source rank's route weights at combine time.
        weight_padding = None if self._use_950dt_mte else topk_weights
        if padding_size == 0:
            x_padding = x.contiguous()
            topk_padding = topk_ids.contiguous()
            if weight_padding is not None:
                weight_padding = weight_padding.contiguous()
            x_active_mask = None  # MC2 treats an absent mask as all rows active.
        else:
            template = self.prepare_low_latency_mask(
                num_max_dispatch_tokens_per_rank, x.device
            )
            x_active_mask = template.narrow(
                0, padding_size, num_max_dispatch_tokens_per_rank
            )
            if compile_adapter is not None:
                # Layer-local compiled AscendC pack; never change the shared
                # strategy's DEEPEP_OPS_FUSED_PADDING policy for other calls.
                x_padding, topk_padding, weight_padding = compile_adapter.pack(
                    x, topk_ids, topk_weights, num_max_dispatch_tokens_per_rank
                )
            elif self._fused_padding_enabled and self._can_fuse_padding(
                x,
                topk_ids,
                topk_weights,
                num_max_dispatch_tokens_per_rank,
                use_fp8,
                use_ue8m0,
                use_mxfp4,
                quant_mode,
            ):
                from ..ops_padding import pack_low_latency_inputs

                # Preserve all N input rows, including graph padding. Only the
                # extra capacity tail is zero-filled by the packing kernel.
                x_padding, topk_padding, weight_padding = pack_low_latency_inputs(
                    x, topk_ids, topk_weights, num_max_dispatch_tokens_per_rank
                )
            else:
                x_padding = torch.empty(
                    padding_size,
                    x.size(1),
                    dtype=x.dtype,
                    device=x.device,
                )
                x_padding = torch.cat((x, x_padding), dim=0)
                topk_padding = torch.empty(
                    padding_size,
                    topk_ids.size(1),
                    dtype=topk_ids.dtype,
                    device=topk_ids.device,
                )
                topk_padding = torch.cat((topk_ids, topk_padding), dim=0)
                if not self._use_950dt_mte:
                    weight_padding = topk_weights.new_empty(
                        (padding_size, topk_weights.size(1))
                    )
                    weight_padding = torch.cat((topk_weights, weight_padding), dim=0)

        dispatch = (
            self._npu_low_latency_dispatch
            if compile_adapter is None
            else compile_adapter.dispatch
        )
        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            expand_scales,
        ) = dispatch(
            x=x_padding,
            topk_idx=topk_padding,
            num_experts=num_experts,
            quant_mode=3 if use_ue8m0 else 2 if use_fp8 else 0,
            comm_alg=self.comm_alg,
            # Packed weights belong to the handle; MTE applies them at combine.
            topk_weights=None if self._use_950dt_mte else weight_padding,
            x_active_mask=x_active_mask,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )

        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
            packed_recv_count,
            None if self._use_950dt_mte else expand_scales,
            x_active_mask,
            topk_padding,
            weight_padding,
        )

        event = EventOverlap(EventHandle())
        hook = lambda *args, **kwargs: None

        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            event,
            hook,
        )

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
        compile_adapter=None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:

        src_num = topk_idx.size(0)
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            packed_recv_count,
            expand_scales,
            x_active_mask,
            topk_padding,
            weight_padding,
        ) = handle

        if self._use_950dt_mte:
            if weight_padding is None:
                weight_padding = topk_weights
                if src_num < num_max_dispatch_tokens_per_rank:
                    weight_padding = torch.nn.functional.pad(
                        topk_weights, (0, 0, 0, num_max_dispatch_tokens_per_rank - src_num)
                    )
            else:
                # Keep dispatch-allocated storage alive on the consuming stream.
                # Dispatch->combine execution ordering remains the caller's job.
                weight_padding.record_stream(
                    torch.npu.current_stream(weight_padding.device)
                )
        combine = (
            self._npu_low_latency_combine
            if compile_adapter is None
            else compile_adapter.combine
        )
        combined_x = combine(
            x=x,
            topk_idx=topk_padding,
            topk_weights=weight_padding,
            assist_info_for_combine=src_info,
            ep_send_counts=layout_range,
            num_experts=num_experts,
            comm_alg="" if self._use_950dt_mte else self.comm_alg,
            expand_scales=expand_scales,
            x_active_mask=x_active_mask,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )

        event = EventOverlap(EventHandle())
        hook = lambda *args, **kwargs: None
        combined_x = combined_x[:src_num, :]
        return combined_x, event, hook

    def _npu_low_latency_dispatch(
        self,
        x,
        topk_idx,
        num_experts: int,
        *,
        quant_mode=0,
        comm_alg="",
        x_smooth_scale=None,
        x_active_mask=None,
        topk_weights=None,
        expert_shard_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=None,
        num_max_dispatch_tokens_per_rank=0,
        expert_token_nums_type=None,
        group_ep=None,
        ep_world_size=None,
        ep_rank_id=None,
        cast_counts_to_int64=False,
    ):

        if shared_expert_rank_num is None:
            shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
        if expert_token_nums_type is None:
            expert_token_nums_type = int(os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1))
        if comm_alg == "hierarchy":
            assert (
                shared_expert_num == 0
            ), "When comm_alg='hierarchy', shared_expert_num must be 0."

        (
            expand_x,
            dynamic_scales,
            expand_idx,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=topk_idx,
            group_ep=self.group_name if group_ep is None else group_ep,
            ep_world_size=self.group_size if ep_world_size is None else ep_world_size,
            ep_rank_id=self.rank if ep_rank_id is None else ep_rank_id,
            moe_expert_num=num_experts,
            scales=x_smooth_scale,
            x_active_mask=x_active_mask,
            expert_scales=topk_weights,
            performance_info=None,
            expert_shard_type=expert_shard_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            quant_mode=quant_mode,
            expert_token_nums_type=expert_token_nums_type,
            comm_alg=comm_alg,  # A3: 支持""，"fullmesh_v1"，"fullmesh_v2", "hierarchy"
        )

        return (
            expand_x,
            dynamic_scales,
            expert_token_nums.to(torch.int64)
            if cast_counts_to_int64 else expert_token_nums,
            expand_idx,
            ep_recv_counts,
            expand_scales,
        )

    def _npu_low_latency_combine(
        self,
        x,
        topk_idx,
        topk_weights,
        assist_info_for_combine,
        ep_send_counts,
        *,
        num_experts=0,
        comm_alg="",
        comm_quant_mode=0,
        x_active_mask=None,
        expand_scales=None,
        shared_expert_x=None,
        expert_shared_type=0,
        shared_expert_num=0,
        shared_expert_rank_num=None,
        num_max_dispatch_tokens_per_rank=0,
        group_ep=None,
        ep_world_size=None,
        ep_rank_id=None,
    ):

        if shared_expert_rank_num is None:
            shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
        if comm_alg == "hierarchy":
            assert (
                shared_expert_num == 0
            ), "When comm_alg='hierarchy', shared_expert_num must be 0."

        combine_x = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=x,
            expert_ids=topk_idx,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=topk_weights,
            group_ep=self.group_name if group_ep is None else group_ep,
            ep_world_size=self.group_size if ep_world_size is None else ep_world_size,
            ep_rank_id=self.rank if ep_rank_id is None else ep_rank_id,
            moe_expert_num=num_experts,
            tp_send_counts=None,
            x_active_mask=x_active_mask,
            expand_scales=expand_scales,
            shared_expert_x=shared_expert_x,
            performance_info=None,
            expert_shard_type=expert_shared_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            comm_quant_mode=comm_quant_mode,
            comm_alg=comm_alg,  # A3: 支持""，"hierarchy"两种
        )

        return combine_x


@register_low_latency_strategy("alltoall")
class AllToAllLowLatencyCommStrategy(LowLatencyEPCommStrategy):
    """
    Low latency mode strategy using torch_npu alltoall.
    This strategy uses the alltoall for A3 RoCE.
    """

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)

    def get_name(self) -> str:
        return "alltoall"

    def get_supported_modes(self) -> List[str]:
        return ["low_latency"]

    def low_latency_dispatch(
        self,
        x,
        topk_idx,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        cumulative_local_expert_recv_stats=None,
        use_fp8=True,
        round_scale=False,
        use_ue8m0=False,
        use_mxfp4=False,
        async_finish=False,
        return_recv_hook=False,
        topk_weights: Optional[torch.Tensor] = None,
        quant_mode: Optional[str] = None,
    ):
        group = self.group
        group_size = self.group_size
        num_local_experts = num_experts // group_size
        ep_rank = self.rank
        device = x.device
        hidden = x.size(1)
        aligned_num_tokens = num_max_dispatch_tokens_per_rank
        num_tokens = x.size(0)
        padding_size = aligned_num_tokens - num_tokens
        x_padding = torch.zeros(
            (padding_size, hidden),
            dtype=x.dtype,
            device=x.device,
        )
        topk_padding = torch.zeros(
            padding_size,
            topk_idx.size(1),
            dtype=topk_idx.dtype,
            device=topk_idx.device,
        )
        topk_padding = torch.cat((topk_idx, topk_padding), dim=0)
        x_padding = torch.cat((x, x_padding), dim=0)

        topk_idx_int = topk_padding.to(torch.int32)
        expert_capacity = aligned_num_tokens

        (expanded_x, expanded_row_idx, _, _) = torch_npu.npu_moe_init_routing_v2(
            x_padding,
            topk_idx_int,
            quant_mode=-1,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            drop_pad_mode=1,
            expert_capacity=expert_capacity,
            active_expert_range=[0, num_experts],
        )

        expanded_x_2d = expanded_x.reshape(num_experts * expert_capacity, hidden)
        chunk_size = num_local_experts * expert_capacity

        input_list = [
            expanded_x_2d[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(chunk_size, hidden, dtype=expanded_x_2d.dtype, device=device)
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_x_raw = torch.cat(output_list, dim=0)

        recv_all = recv_x_raw.reshape(
            group_size, num_local_experts, expert_capacity, hidden
        )
        recv_all = recv_all.permute(1, 0, 2, 3).contiguous()
        recv_x = recv_all.reshape(
            num_local_experts * group_size * expert_capacity, hidden
        )
        recv_x_out = (torch_npu.npu_dynamic_quant(recv_x)) if use_fp8 else recv_x

        packed_recv_count = torch.full(
            (num_local_experts,),
            expert_capacity * group_size,
            dtype=torch.int64,
            device=x.device,
        )

        handle_tuple = (
            expanded_row_idx,
            expert_capacity,
            hidden,
            num_tokens,
            num_local_experts,
            group_size,
            packed_recv_count,
        )

        return (
            recv_x_out,
            packed_recv_count,
            handle_tuple,
            EventOverlap(),
            lambda: None,
        )

    def low_latency_combine(
        self,
        x,
        topk_idx,
        topk_weights,
        handle,
        zero_copy=False,
        async_finish=False,
        return_recv_hook=False,
        out=None,
    ):
        expanded_row_idx = handle[0]
        expert_capacity = handle[1]
        hidden = handle[2]
        num_tokens = handle[3]
        num_local_experts = handle[4]
        group_size = handle[5]

        device = x.device
        group = self.group

        x_reordered = x.reshape(num_local_experts, group_size, expert_capacity, hidden)
        x_reordered = x_reordered.permute(1, 0, 2, 3).contiguous()
        x_reordered = x_reordered.reshape(
            group_size * num_local_experts * expert_capacity, hidden
        )

        chunk_size = num_local_experts * expert_capacity
        input_list = [
            x_reordered[r * chunk_size : (r + 1) * chunk_size].contiguous()
            for r in range(group_size)
        ]
        output_list = [
            torch.empty(chunk_size, hidden, dtype=x.dtype, device=device)
            for r in range(group_size)
        ]
        dist.all_to_all(output_list, input_list, group=group)
        recv_all_raw = torch.cat(output_list, dim=0)

        recv_all_raw = recv_all_raw.reshape(
            group_size * num_local_experts, expert_capacity, hidden
        )
        topk_weights_padding = torch.empty(
            expert_capacity,
            topk_weights.size(1),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        topk_weights_padding[:num_tokens].copy_(topk_weights)
        output = torch_npu.npu_moe_finalize_routing(
            expanded_permuted_rows=recv_all_raw,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights_padding,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=3,
        )
        output = output[:num_tokens, :]

        return output, EventOverlap(), lambda: None
