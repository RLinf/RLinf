# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copyreg
import gc
import time
import typing
import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode

if typing.TYPE_CHECKING:
    from vllm.outputs import RequestOutput


COLOR_END = "\033[0m"


def color_print(rank, *args, **kwargs):
    if "end" in kwargs:
        print(f"\033[{31 + rank}m[DP rank {rank}]", *args, **kwargs)
    else:
        print(f"\033[{31 + rank}m[DP rank {rank}]", *args, **kwargs, end="")
    print("\033[0m")


def green(text: str):
    return f"\033[32m{text}\033[0m"


@contextmanager
def sharp_cover(header_text: str, prelen: int = 30, color="\033[32m"):
    len(header_text)
    print("#" * prelen + f" {color}>>> {header_text}{COLOR_END} " + "#" * prelen)

    try:
        yield
    finally:
        print("#" * prelen + f" {color}>>> {header_text}{COLOR_END} " + "#" * prelen)


def print_vllm_outputs(outputs: List["RequestOutput"]):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_ids = output.outputs[0].token_ids
        print(
            f"{green('Prompt')}         : {prompt!r}",
            f"{green('Generated text')} : {generated_text!r}",
            f"{green('Generated ids')}  : {generated_ids}",
            sep="\n",
        )


def print_multi_outputs(resps_all: List[List["RequestOutput"]]):
    for i, resps in enumerate(resps_all):
        with sharp_cover(f"vllm dp {i}"):
            print_vllm_outputs(resps)


def print_sglang_outputs(prompts, outputs: List[Dict], tokenizer):
    output_ids = [output["output_ids"] for output in outputs]
    output_texts = tokenizer.batch_decode(output_ids)
    for p, t, ids in zip(prompts, output_texts, output_ids):
        print(
            f"{green('Prompt')}         : {p!r}",
            f"{green('Generated text')} : {t!r}",
            f"{green('Generated ids')}  : {ids}",
            sep="\n",
        )


def print_multi_sglang_outputs(prompts, outputs: List[List[Dict]], tokenizer):
    for i, resps in enumerate(outputs):
        with sharp_cover(f"sglang dp {i}"):
            print_sglang_outputs(prompts, resps, tokenizer)


def get_module_from_name(module: torch.nn.Module, name: str):
    """
    Args:
        name: str, the name of the module, e.g. model.layers.0.self_attn.qkv_proj
    """
    parts = name.split(".")

    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    return module


def rebind_param_attr(
    model: torch.nn.Module,
    name: str,
    reserved_attr: Dict[str, Dict],
    materialize: bool = False,
):
    """
    here name is already converted to the vLLM format.
    """
    name_paths = name.split(".")
    last_name = name_paths[-1]
    assert last_name in ["weight", "bias"]

    module = get_module_from_name(model, ".".join(name_paths[:-1]))

    param = getattr(module, last_name)

    if materialize and param.device.index != torch.cuda.current_device():
        module.to_empty(device=torch.cuda.current_device())

    param.__dict__.update(reserved_attr[name])
    del reserved_attr[name]


def swap_tensor_pointer(t1: torch.Tensor, t2: torch.Tensor):
    """
    This function swaps the content of the two Tensor objects.
    At a high level, this will make t1 have the content of t2 while preserving
    its identity.

    This will not work if t1 and t2 have different slots.
    """
    # Ensure there are no weakrefs
    if weakref.getweakrefs(t1):
        raise RuntimeError("Cannot swap t1 because it has weakref associated with it")
    if weakref.getweakrefs(t2):
        raise RuntimeError("Cannot swap t2 because it has weakref associated with it")
    t1_slots = set(copyreg._slotnames(t1.__class__))  # type: ignore[attr-defined]
    t2_slots = set(copyreg._slotnames(t2.__class__))  # type: ignore[attr-defined]
    if t1_slots != t2_slots:
        raise RuntimeError("Cannot swap t1 and t2 if they have different slots")

    def swap_attr(name):
        tmp = getattr(t1, name)
        setattr(t1, name, (getattr(t2, name)))
        setattr(t2, name, tmp)

    def error_pre_hook(grad_outputs):
        raise RuntimeError(
            "Trying to execute AccumulateGrad node that was poisoned by swap_tensors "
            "this can happen when you try to run backward on a tensor that was swapped. "
            "For a module m with `torch.__future__.set_swap_module_params_on_conversion(True)` "
            "you should not change the device or dtype of the module (e.g. `m.cpu()` or `m.half()`) "
            "between running forward and backward. To resolve this, please only change the "
            "device/dtype before running forward (or after both forward and backward)."
        )

    def check_use_count(t, name="t1"):
        use_count = t._use_count()
        error_str = (
            f"Expected use_count of {name} to be 1 or 2 with an AccumulateGrad node but got {use_count} "
            f"make sure you are not holding references to the tensor in other places."
        )
        if use_count > 1:
            if use_count == 2 and t.is_leaf:
                accum_grad_node = torch.autograd.graph.get_gradient_edge(t).node
                # Make sure that the accumulate_grad node was not lazy_init-ed by get_gradient_edge
                if t._use_count() == 2:
                    accum_grad_node.register_prehook(error_pre_hook)
                else:
                    raise RuntimeError(error_str)
            else:
                raise RuntimeError(error_str)

    check_use_count(t1, "t1")
    check_use_count(t2, "t2")

    # Swap the types
    # Note that this will fail if there are mismatched slots
    # swap_attr("__class__")

    # Swap the dynamic attributes
    # swap_attr("__dict__")

    # Swap the slots
    # for slot in t1_slots:
    #     if hasattr(t1, slot) and hasattr(t2, slot):
    #         swap_attr(slot)
    #     elif hasattr(t1, slot):
    #         setattr(t2, slot, (getattr(t1, slot)))
    #         delattr(t1, slot)
    #     elif hasattr(t2, slot):
    #         setattr(t1, slot, (getattr(t2, slot)))
    #         delattr(t2, slot)

    # Swap the at::Tensor they point to
    torch._C._swap_tensor_impl(t1, t2)


class VLLMOutputs:
    """
    Accepts a list where each element is the result of vLLM generate, and each result is a list of RequestOutput objects.
    """

    def __init__(self, outputs: List[List["RequestOutput"]]):
        self.outputs = outputs

    def dump_to_json(self, eos: int):
        """
        Merge all lists into a single one, to be returned to a single Megatron DP instance.
        """
        all_reqoutput = [out for outs in self.outputs for out in outs]

        output_ids = [list(output.outputs[0].token_ids) for output in all_reqoutput]
        output_lenghts = [len(out) for out in output_ids]
        response = {
            "token_ids": output_ids,
            "response_lengths": output_lenghts,
            "finished": [ids[-1] == eos for ids in output_ids],
        }

        return response


class CudaMemoryProfiler:
    def __init__(self, device: Optional[torch.types.Device] = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            mem = torch.cuda.max_memory_allocated(self.device)
        return mem

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


class CudaTimeProfiler:
    def __init__(
        self,
        device: Optional[torch.types.Device] = None,
        name: str = "",
        do_print: bool = True,
    ):
        self.device = device
        self.name = name
        self.do_print = do_print

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize()
        self.time_cost = self.start_event.elapsed_time(self.end_event)
        if self.do_print:
            print(green(f"Event {self.name} cost: {self.time_cost:.3f} ms"))


class TimeProfiler:
    def __init__(
        self,
        name: str = "",
        do_print: bool = True,
        do_tb=False,
        writer: SummaryWriter = None,
        tag="",
        step=0,
    ):
        self.name = name
        self.do_print = do_print
        self.do_tb = do_tb
        self.writer = writer
        self.tag = tag
        self.step = step
        if do_tb:
            assert writer is not None

    def __enter__(self):
        self.start_time = time.time()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.time_cost = self.end_time - self.start_time
        if self.do_print:
            print(green(f"Event {self.name} cost: {self.time_cost * 1000:.3f} ms"))
        if self.do_tb:
            self.writer.add_scalar(self.tag, self.time_cost, self.step)


class RankMapper:
    @classmethod
    def get_actor_rank_to_rollout_rank_map(
        cls,
        placement: ModelParallelComponentPlacement,
    ) -> Dict[int, List[Tuple[int, int]]]:
        return cls._get_rank_mapper(
            placement.placement_mode
        ).get_actor_rank_to_rollout_rank_map(
            placement.actor_tp_size,
            placement.actor_pp_size,
            placement.actor_world_size,
            placement.rollout_tp_size,
            placement.rollout_world_size,
        )

    @classmethod
    def get_rollout_rank_to_actor_rank_map(
        cls, placement: ModelParallelComponentPlacement
    ) -> Dict[Tuple[int, int], int]:
        return cls._get_rank_mapper(
            placement.placement_mode
        ).get_rollout_rank_to_actor_rank_map(
            placement.actor_tp_size,
            placement.actor_pp_size,
            placement.actor_world_size,
            placement.rollout_tp_size,
            placement.rollout_world_size,
        )

    @staticmethod
    def _get_rank_mapper(
        placement_mode: PlacementMode,
    ):
        """
        Get the rank mapper class based on the mode.
        """
        if placement_mode == PlacementMode.COLLOCATED:
            return CollocateRankMapper
        elif placement_mode == PlacementMode.DISAGGREGATED:
            return DisaggRankMapper
        else:
            raise ValueError(f"Unsupported mode: {placement_mode}.")


class CollocateRankMapper(RankMapper):
    @classmethod
    def get_actor_rank_to_rollout_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Get the global mapping from actor 1D rank to rollout 2D rank as dict.
        """
        rank_map = {}
        for actor_rank in range(actor_world_size):
            rank_map[actor_rank] = cls._get_actor_rank_to_rollout_rank(
                actor_rank,
                actor_tp_size,
                rollout_tp_size,
            )
        return rank_map

    @classmethod
    def get_rollout_rank_to_actor_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ):
        """
        Get the global mapping from rollout 2D rank to actor 1D rank as dict.
        """
        rank_map = cls.get_actor_rank_to_rollout_rank_map(
            actor_tp_size,
            actor_pp_size,
            actor_world_size,
            rollout_tp_size,
            rollout_world_size,
        )
        return {v: k for k, v in rank_map.items()}

    @staticmethod
    def _get_actor_rank_to_rollout_rank(
        actor_rank: int,
        actor_tp_size: int,
        rollout_tp_size: int,
    ):
        """
        Get the mapping from actor 1D rank to rollout 2D rank.
        """
        num_rollout_dp_ranks_per_actor_tp_group = actor_tp_size // rollout_tp_size

        actor_tp_rank = actor_rank % actor_tp_size

        actor_tp_group_id = actor_rank // actor_tp_size
        rollout_start_dp_rank = (
            actor_tp_group_id * num_rollout_dp_ranks_per_actor_tp_group
        )

        weight_dst_dp_rank_in_rollout = (
            rollout_start_dp_rank
            + actor_tp_rank % num_rollout_dp_ranks_per_actor_tp_group
        )

        weight_dst_tp_rank_in_rollout = (
            actor_tp_rank // num_rollout_dp_ranks_per_actor_tp_group
        )

        return (weight_dst_dp_rank_in_rollout, weight_dst_tp_rank_in_rollout)


class DisaggRankMapper(RankMapper):
    """
    A mapper for disaggregated ranks.
    This is used to map the disaggregated ranks to the actor ranks.

    Assume that actor_tp_size = n * rollout_tp_size
    """

    @classmethod
    def get_actor_rank_to_rollout_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Only ranks in dp=0 actor dp group will send weights to rollout LLM.
        """
        actor_model_parallel_size = actor_tp_size * actor_pp_size
        assert (
            rollout_world_size >= actor_model_parallel_size
            and rollout_world_size % actor_model_parallel_size == 0
        ), (
            f"rollout_world_size ({rollout_world_size}) should be a multiple of actor_model_parallel_size ({actor_model_parallel_size})"
        )

        num_dp_ranks_per_actor_dp_group = (
            rollout_world_size // actor_model_parallel_size
        )
        stride = actor_model_parallel_size // rollout_tp_size

        rank_map = {}
        for actor_rank in range(actor_world_size):
            if actor_rank >= actor_model_parallel_size:
                # dp_rank > 0 will not send weight to any rollout rank
                rank_map[actor_rank] = []
                continue
            gen_dp, gen_tp = cls._get_actor_rank_to_rollout_rank(
                actor_rank,
                actor_tp_size,
                rollout_tp_size,
            )
            rank_map[actor_rank] = [
                (gen_dp + i * stride, gen_tp)
                for i in range(num_dp_ranks_per_actor_dp_group)
            ]

        return rank_map

    @classmethod
    def get_rollout_rank_to_actor_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> Dict[Tuple[int, int], int]:
        rank_map = cls.get_actor_rank_to_rollout_rank_map(
            actor_tp_size,
            actor_pp_size,
            actor_world_size,
            rollout_tp_size,
            rollout_world_size,
        )
        result_map = {}
        for actor_rank, rollout_2d_ranks in rank_map.items():
            for rollout_2d_rank in rollout_2d_ranks:
                result_map[rollout_2d_rank] = actor_rank
        return result_map

    @staticmethod
    def _get_actor_rank_to_rollout_rank(
        actor_rank: int,
        actor_tp_size: int,
        rollout_tp_size: int,
    ) -> Tuple[int, int]:
        assert actor_tp_size % rollout_tp_size == 0, (
            "actor_tp_size must be a multiple of rollout_tp_size"
        )

        num_rollout_dp_ranks_per_actor_tp_group = actor_tp_size // rollout_tp_size
        actor_tp_rank = actor_rank % actor_tp_size
        corresponding_rollout_dp_rank = (
            actor_tp_rank % num_rollout_dp_ranks_per_actor_tp_group
        )
        corresponding_rollout_tp_rank = (
            actor_tp_rank // num_rollout_dp_ranks_per_actor_tp_group
        )

        return (corresponding_rollout_dp_rank, corresponding_rollout_tp_rank)


SUPPORTED_LLM_ROLLOUT_BACKENDS = ["vllm", "sglang"]


def get_rollout_backend_worker(
    cfg: DictConfig, placement: ModelParallelComponentPlacement
) -> Worker:
    rollout_backend = cfg.rollout.get("rollout_backend", None)
    if rollout_backend is None:
        raise ValueError(
            f"rollout_backend must be specified in the config. Support {', '.join(SUPPORTED_LLM_ROLLOUT_BACKENDS)}."
        )
    if rollout_backend not in SUPPORTED_LLM_ROLLOUT_BACKENDS:
        raise ValueError(
            f"rollout_backend {rollout_backend} is not supported. Support {', '.join(SUPPORTED_LLM_ROLLOUT_BACKENDS)}."
        )

    if rollout_backend == "vllm":
        from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

        if placement.placement_mode == PlacementMode.COLLOCATED:
            return VLLMWorker
        elif placement.placement_mode == PlacementMode.DISAGGREGATED:
            raise NotImplementedError(
                "vLLM rollout backend does not support the pipeline mode."
            )
        else:
            raise ValueError(f"Unsupported placement mode: {placement.placement_mode}")
    elif rollout_backend == "sglang":
        from rlinf.workers.rollout.sglang.sglang_worker import (
            AsyncSGLangWorker,
            SGLangWorker,
        )

        if placement.placement_mode == PlacementMode.COLLOCATED:
            return SGLangWorker
        elif placement.placement_mode == PlacementMode.DISAGGREGATED:
            return AsyncSGLangWorker
        else:
            raise ValueError(f"Unsupported placement mode: {placement.placement_mode}")
