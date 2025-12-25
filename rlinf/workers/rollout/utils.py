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

import asyncio
import json
import os
import time
import typing
from contextlib import contextmanager
from dataclasses import dataclass

from omegaconf import DictConfig

from rlinf.data.io_struct import SeqGroupInfo
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode

if typing.TYPE_CHECKING:
    from vllm.outputs import RequestOutput


COLOR_END = "\033[0m"


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


def print_vllm_outputs(outputs: list["RequestOutput"]):
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


def print_multi_outputs(resps_all: list[list["RequestOutput"]]):
    for i, resps in enumerate(resps_all):
        with sharp_cover(f"vllm dp {i}"):
            print_vllm_outputs(resps)


def print_sglang_outputs(prompts, outputs: list[dict], tokenizer):
    output_ids = [output["output_ids"] for output in outputs]
    output_texts = tokenizer.batch_decode(output_ids)
    for p, t, ids in zip(prompts, output_texts, output_ids):
        print(
            f"{green('Prompt')}         : {p!r}",
            f"{green('Generated text')} : {t!r}",
            f"{green('Generated ids')}  : {ids}",
            sep="\n",
        )


def print_multi_sglang_outputs(prompts, outputs: list[list[dict]], tokenizer):
    for i, resps in enumerate(outputs):
        with sharp_cover(f"sglang dp {i}"):
            print_sglang_outputs(prompts, resps, tokenizer)


class RankMapper:
    @classmethod
    def get_actor_rank_to_rollout_rank_map(
        cls,
        placement: ModelParallelComponentPlacement,
    ) -> dict[int, list[tuple[int, int]]]:
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
    ) -> dict[tuple[int, int], int]:
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
        elif placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]:
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
    ) -> dict[int, tuple[int, int]]:
        """
        Get the global mapping from actor 1D rank to rollout 2D rank as dict.
        """
        # rank -> (dp, tp)
        if actor_tp_size == 1:
            return {
                rank: (rank // rollout_tp_size, rank % rollout_tp_size)
                for rank in range(actor_world_size)
            }
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
    ) -> dict[int, list[tuple[int, int]]]:
        """
        Only ranks in dp=0 actor dp group will send weights to rollout LLM.
        """
        actor_model_parallel_size = actor_tp_size
        assert rollout_world_size >= actor_model_parallel_size, (
            f"rollout_world_size ({rollout_world_size}) should more than actor_model_parallel_size ({actor_model_parallel_size})"
        )

        assert rollout_world_size % actor_model_parallel_size == 0, (
            f"rollout_world_size ({rollout_world_size}) should be a multiple of actor_model_parallel_size ({actor_model_parallel_size})"
        )

        actor_dp = actor_world_size // actor_tp_size
        stride = actor_model_parallel_size // rollout_tp_size

        rank_map = {}
        for actor_rank in range(actor_world_size):
            if actor_rank > rollout_world_size:
                rank_map[actor_rank] = []
                continue
            gen_dp, gen_tp = cls._get_actor_rank_to_rollout_rank(
                actor_rank,
                actor_tp_size,
                rollout_tp_size,
            )
            if actor_world_size <= rollout_world_size:
                rank_map[actor_rank] = [
                    (gen_dp + i * stride * actor_dp, gen_tp)
                    for i in range(rollout_world_size // actor_world_size)
                ]
            elif actor_rank < rollout_world_size:
                rank_map[actor_rank] = [(gen_dp, gen_tp)]
            else:
                rank_map[actor_rank] = []

        return rank_map

    @classmethod
    def get_rollout_rank_to_actor_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> dict[tuple[int, int], int]:
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
    ) -> tuple[int, int]:
        assert actor_tp_size % rollout_tp_size == 0, (
            "actor_tp_size must be a multiple of rollout_tp_size"
        )

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


SUPPORTED_LLM_ROLLOUT_BACKENDS = ["vllm", "sglang"]


def get_rollout_backend_worker(cfg: DictConfig) -> Worker:
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

        return VLLMWorker
    elif rollout_backend == "sglang":
        from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

        return SGLangWorker


class RunningStatusManager:
    def __init__(self):
        self._running_seq_group: dict[SeqGroupInfo, asyncio.Task] = {}
        self._aborted_seq_group: list[SeqGroupInfo] = []
        # SeqGroupInfo that have been completed and sent to actor/inference
        # only retained for debugging
        self._done_seq_group: list[SeqGroupInfo] = []

        # asyncio Events
        # set by scheduler coroutine to prevent rollout coroutine from exiting before potential migrations
        self.exit_rollout_iter = asyncio.Event()

    def add_task(self, seq_group: SeqGroupInfo, task: asyncio.Task):
        assert seq_group not in self._running_seq_group, (
            f"Task for sequence group {seq_group.id} is already running."
        )
        self._running_seq_group[seq_group] = task

    def mark_done(self, seq_group: SeqGroupInfo):
        assert seq_group in self._running_seq_group, (
            f"Task for SeqGroup {seq_group.id} not found. "
            "Check whether it has been added correctly or already marked done."
        )
        assert seq_group not in self._done_seq_group
        self._running_seq_group.pop(seq_group)
        self._done_seq_group.append(seq_group)

    def mark_aborted(self, seq_group: SeqGroupInfo):
        assert seq_group in self._running_seq_group, (
            f"Task for SeqGroup {seq_group.id} not found. "
            "Check whether it has been added correctly or already marked aborted."
        )
        assert seq_group not in self._aborted_seq_group
        self._running_seq_group.pop(seq_group)
        self._aborted_seq_group.append(seq_group)

    async def wait_notification(self):
        """
        Wait until the scheduler notifies that it is safe to continue.
        This is used to prevent the rollout coroutine from exiting before potential migrations.
        """
        await self.exit_rollout_iter.wait()
        self.exit_rollout_iter.clear()

    def notify(self):
        """
        Call by scheduler to notify the rollout to continue.
        This is used to prevent the rollout coroutine from exiting before potential migrations.
        """
        self.exit_rollout_iter.set()

    def clear(self):
        self._running_seq_group.clear()
        self._aborted_seq_group.clear()
        self._done_seq_group.clear()
        self.exit_rollout_iter.clear()

    def empty(self) -> bool:
        return len(self._running_seq_group) == 0 and len(self._done_seq_group) == 0

    def get_running_seq_groups(self) -> list[SeqGroupInfo]:
        return list(self._running_seq_group.keys())

    def get_done_seq_groups(self) -> list[SeqGroupInfo]:
        return self._done_seq_group

    def get_aborted_seq_groups(self) -> list[SeqGroupInfo]:
        return self._aborted_seq_group

    def get_running_tasks(self) -> list[asyncio.Task]:
        return list(self._running_seq_group.values())

    @property
    def num_seq_group_running(self) -> int:
        return len(self._running_seq_group)

    @property
    def num_seq_group_done(self) -> int:
        return len(self._done_seq_group)

    @property
    def num_seq_group_aborted(self) -> int:
        return len(self._aborted_seq_group)

    @property
    def num_seq_group(self) -> int:
        return (
            self.num_seq_group_running
            + self.num_seq_group_done
            + self.num_seq_group_aborted
        )

    @property
    def num_seq_running(self) -> int:
        return sum(sg.num_running for sg in self.get_running_seq_groups())

    @property
    def num_seq_returned(self) -> int:
        return self.num_seq - self.num_seq_running

    @property
    def num_seq(self) -> int:
        return sum(sg.group_size for sg in self.get_running_seq_groups()) + sum(
            sg.group_size for sg in self.get_done_seq_groups()
        )


@dataclass
class RolloutEngineStats:
    num_running_reqs: int = 0
    max_running_reqs: int = 0
    num_used_tokens: int = 0
    max_total_num_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0


class MetaInfoStatsCollector:
    """Collector for SGLang meta_info statistics

    This collector is only initialized when enabled via configuration.
    Add the following parameters to your generation config section:

    generation:
      collect_meta_stats: true  # Enable meta_info statistics collection
      meta_stats_file: "custom_meta_stats.jsonl"  # Optional: custom output file
      async_meta_stats_file: "custom_async_meta_stats.jsonl"  # Optional: custom async output file
      schedule_meta_stats_file: "custom_schedule_meta_stats.jsonl"  # Optional: custom schedule output file
    """

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.stats_buffer = []
        self.buffer_size = 100  # Write to file every 100 records

        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(self.output_file)
            if os.path.dirname(self.output_file)
            else ".",
            exist_ok=True,
        )

        # Initialize file with header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w") as f:
                f.write("")  # Create empty file

    def collect_batch_stats(self, outputs: list[dict], batch_id: int) -> None:
        """Collect statistics from a batch of SGLang outputs

        Args:
            outputs: List of SGLang output dictionaries
            batch_id: Unique identifier for this batch
        """
        current_time = time.time()

        for req_idx, output in enumerate(outputs):
            try:
                # Extract meta_info
                meta_info = output.get("meta_info", {})

                # Extract the specific metrics you requested
                stats_record = {
                    "timestamp": current_time,
                    "batch_id": batch_id,
                    "request_id": f"batch_{batch_id}_req_{req_idx}",
                    "prompt_tokens": meta_info.get("prompt_tokens", None),
                    "completion_tokens": meta_info.get("completion_tokens", None),
                    "e2e_latency": meta_info.get("e2e_latency", None),
                    "ttft": meta_info.get("ttft", None),
                    # Additional useful meta_info fields (if available)
                    "finish_reason": meta_info.get("finish_reason", {}).get(
                        "type", None
                    ),
                    "total_tokens": (
                        meta_info.get("prompt_tokens", 0)
                        + meta_info.get("completion_tokens", 0)
                    )
                    if meta_info.get("prompt_tokens") is not None
                    and meta_info.get("completion_tokens") is not None
                    else None,
                    # Add any other meta_info fields that might be useful
                    "meta_info_keys": list(
                        meta_info.keys()
                    ),  # For debugging/inspection
                }

                self.stats_buffer.append(stats_record)

            except Exception as e:
                # Log error but continue processing
                error_record = {
                    "timestamp": current_time,
                    "batch_id": batch_id,
                    "request_id": f"batch_{batch_id}_req_{req_idx}",
                    "error": str(e),
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else "not_dict",
                }
                self.stats_buffer.append(error_record)

        # Write to file if buffer is full
        if len(self.stats_buffer) >= self.buffer_size:
            self._flush_to_file()

    def _flush_to_file(self) -> None:
        """Write buffered statistics to file"""
        if not self.stats_buffer:
            return

        with open(self.output_file, "a") as f:
            for record in self.stats_buffer:
                f.write(json.dumps(record) + "\n")

        print(f"Written {len(self.stats_buffer)} records to {self.output_file}")
        self.stats_buffer = []

    def finalize(self) -> None:
        """Flush any remaining data and close"""
        self._flush_to_file()
        print(f"Finalized stats collection. Data saved to {self.output_file}")
