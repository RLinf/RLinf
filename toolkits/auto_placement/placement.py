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

from abc import ABC
from enum import Enum
from typing import Optional

from node import ComponentNode
from fitter import DataFitter
from util import get_global_config


class ScheduleMode(Enum):
    """Mode of schedule result."""

    SINGLE_NODE = "single_node"
    COLLOCATED = "collocated"
    DISAGGREGATED = "disaggregated"


class PlacementTopology(Enum):
    """Topology derived from the *final* placement GPU overlap pattern.

    This is intentionally separate from ScheduleMode:
    - ScheduleMode is the DP merge operator on the binary recursion tree.
    - PlacementTopology describes the global GPU sharing semantics of the final result.
    """

    SINGLE_NODE = "single_node"
    ALL_SHARED = "all_shared"
    ALL_DISJOINT = "all_disjoint"
    MIXED_SHARED = "mixed_shared"


class ScheduleResult(ABC):
    """Base class for all schedule results."""

    @staticmethod
    def _ranges_overlap(a: range, b: range) -> bool:
        # Treat ranges as half-open intervals [start, stop).
        return max(a.start, b.start) < min(a.stop, b.stop)

    @staticmethod
    def _derive_topology_from_placement(
        total_gpu_num: int, placement: dict[ComponentNode, range]
    ) -> PlacementTopology:
        if len(placement) <= 1:
            return PlacementTopology.SINGLE_NODE

        full = range(total_gpu_num)
        if all(gpu_range == full for gpu_range in placement.values()):
            return PlacementTopology.ALL_SHARED

        ranges = list(placement.values())
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                if ScheduleResult._ranges_overlap(ranges[i], ranges[j]):
                    return PlacementTopology.MIXED_SHARED
        return PlacementTopology.ALL_DISJOINT

    @staticmethod
    def merger_schedule_results(
        total_gpu_num: int,
        source_res: "ScheduleResult",
        sink_res: "ScheduleResult",
        is_collocated: bool,
        warmup_group_num: int = 1,
    ) -> Optional["ScheduleResult"]:
        if source_res is None or sink_res is None:
            return None
        if is_collocated:
            res = CollocatedScheduleResult(total_gpu_num, source_res, sink_res)
        else:
            res = DisaggregatedScheduleResult(
                total_gpu_num, source_res, sink_res, warmup_group_num
            )

        config = get_global_config()

        # In Reasoning task, mixed GPU sharing is not supported.
        if config.task_type == "reasoning" and res.topology == PlacementTopology.MIXED_SHARED:
            return None

        # In Embodiment task, actor should run on all GPUs.
        if config.task_type == "embodied":
            nodes = list(res.placement.keys())
            if (
                nodes[-1].role == "actor"
                and len(res.placement[nodes[-1]]) != res.total_gpu_num
            ):
                print(f"actor now runs on {res.placement[nodes[-1]]} GPUs")
                return None
        return res

    @staticmethod
    def find_best_schedule(
        first: "ScheduleResult", second: "ScheduleResult"
    ) -> Optional["ScheduleResult"]:
        if first is None or second is None:
            return first if first is not None else second
        return first if first.total_cost < second.total_cost else second

    def __init__(
        self,
        mode: ScheduleMode,
        total_gpu_num: int,
        placement: dict[ComponentNode, range],
        cost_per_group_batch: float,
        total_cost: float,
    ):
        self.mode = mode
        self.total_gpu_num = total_gpu_num
        self.placement = placement
        self.cost_per_group_batch = cost_per_group_batch
        self.total_cost = total_cost

    @property
    def topology(self) -> PlacementTopology:
        return self._derive_topology_from_placement(self.total_gpu_num, self.placement)

    @property
    def effective_mode(self) -> str:
        """Unified external mode string for display/metrics.

        For embodied:
        - disaggregated => ALL_DISJOINT (e.g., env/env_rollout/actor all separated)
        - hybrid        => MIXED_SHARED (pipeline + GPU sharing)
        - collocated    => ALL_SHARED
        For reasoning:
        - hybrid (mixed) is filtered out in merger_schedule_results.
        """
        config = get_global_config()
        if config.task_type == "embodied":
            if self.topology == PlacementTopology.ALL_SHARED:
                return "collocated"
            if self.topology == PlacementTopology.ALL_DISJOINT:
                return "disaggregated"
            if self.topology == PlacementTopology.MIXED_SHARED:
                return "hybrid"
            return "single_node"
        # reasoning (and others): keep original DP mode as external name
        return self.mode.value

    def get_cost_per_group_batch(self, *args, **kwargs) -> float:
        return self.cost_per_group_batch

    def is_hybrid(self) -> bool:
        # Kept for backward compatibility with existing call sites.
        # Prefer using `topology`/`effective_mode` for unified semantics.
        return self.topology == PlacementTopology.MIXED_SHARED

    @property
    def placement_str(self) -> str:
        placement_str = ""
        for node, gpu_range in self.placement.items():
            placement_str += f"{node.role} : {gpu_range[0]}-{gpu_range[-1]}\n"
        return placement_str

    def __str__(self):
        return (
            f"ScheduleResult : total_gpu_num={self.total_gpu_num}, total_cost={self.total_cost}, "
            f"mode={self.mode.value}, effective_mode={self.effective_mode}, placement:\n{self.placement_str}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class SingleNodeScheduleResult(ScheduleResult):
    """ScheduleResult for single ComponentNode."""

    def __init__(
        self,
        total_gpu_num: int,
        node: ComponentNode,
        cost_per_group_batch: float,
        total_cost: Optional[float] = None,
    ):
        config = get_global_config()
        if total_cost is None:
            total_cost = cost_per_group_batch * config.rollout_batch_size
        super().__init__(
            mode=ScheduleMode.SINGLE_NODE,
            total_gpu_num=total_gpu_num,
            placement={node: range(total_gpu_num)},
            cost_per_group_batch=cost_per_group_batch,
            total_cost=total_cost,
        )


class CollocatedScheduleResult(ScheduleResult):
    def __init__(
        self, total_gpu_num: int, source_res: ScheduleResult, sink_res: ScheduleResult
    ):
        assert (
            total_gpu_num == source_res.total_gpu_num
            and total_gpu_num == sink_res.total_gpu_num
        )
        self.source_res = source_res
        self.sink_res = sink_res
        config = get_global_config()
        super().__init__(
            mode=ScheduleMode.COLLOCATED,
            total_gpu_num=total_gpu_num,
            placement={
                **source_res.placement,
                **sink_res.placement,
            },
            cost_per_group_batch=None,
            total_cost= self.source_res.total_cost + self.sink_res.total_cost,
        )

    def get_cost_per_group_batch(self, is_source: bool) -> float:
        """get warmup_time and stable_cost and warmup_group_num for the collocated-workflow.

        In Hybird mode, if collocated-workflow is in source, return the self.sink_res attr values. Otherwise, return the self.source_res attr values.
        """
        if is_source:
            return self.sink_res.cost_per_group_batch
        else:
            return self.source_res.cost_per_group_batch


class DisaggregatedScheduleResult(ScheduleResult):
    def __init__(
        self,
        total_gpu_num: int,
        source_res: ScheduleResult,
        sink_res: ScheduleResult,
        warmup_group_num: int = 1,
    ):
        assert total_gpu_num == source_res.total_gpu_num + sink_res.total_gpu_num
        self.source_res = source_res
        self.sink_res = sink_res
        self.warmup_group_num = warmup_group_num

        cost_per_group_batch, total_cost = self._get_disaggregated_time()
        super().__init__(
            mode=ScheduleMode.DISAGGREGATED,
            total_gpu_num=total_gpu_num,
            placement=self._get_disaggregated_placement(),
            cost_per_group_batch=cost_per_group_batch,
            total_cost=total_cost,
        )

    def _get_disaggregated_time(self) -> tuple[float, float]:
        config = get_global_config()

        if config.task_type == "embodied":
            return self._get_disaggregated_time_embodied()
        elif config.task_type == "reasoning":
            return self._get_disaggregated_time_reasoning()
        else:
            raise ValueError(f"Unsupported task type: {config.task_type}")

    def _get_disaggregated_time_embodied(self) -> tuple[float, float]:
        config = get_global_config()

        def _roles(res: ScheduleResult) -> set[str]:
            return {node.role for node in res.placement.keys()}

        def _has_actor_cut(source_roles: set[str], sink_roles: set[str]) -> bool:
            return ("actor" in source_roles) ^ ("actor" in sink_roles)

        def _stage_num() -> int:
            # Default to 2 stages; normalize invalid values to 1.
            n = int(getattr(config, "pipeline_stage_num", 2))
            return n if n > 0 else 1

        def _total_envs() -> int:
            return int(getattr(config, "env_num", 0))

        def _is_env_rollout_cut(source_roles: set[str], sink_roles: set[str]) -> bool:
            # True only when the cut separates env and env_rollout across the two sides.
            source_has_env = "env" in source_roles
            sink_has_env = "env" in sink_roles
            source_has_rollout = "env_rollout" in source_roles
            sink_has_rollout = "env_rollout" in sink_roles
            return (source_has_env ^ sink_has_env) and (source_has_rollout ^ sink_has_rollout)

        def _gpus_for_role(role: str, source_roles: set[str], sink_roles: set[str]) -> int:
            # In this model, each side's `total_gpu_num` is the GPU allocation for
            # that sub-workflow.
            if role in source_roles:
                return int(self.source_res.total_gpu_num)
            if role in sink_roles:
                return int(self.sink_res.total_gpu_num)
            return 0

        def _embodied_generate_pipeline_cost(
            env_gpus: int, rollout_gpus: int, stage_num: int, total_envs: int
        ) -> float:
            # Stage-aware model used only for env <-> env_rollout cuts:
            # - per-stage envs-per-gpu must be integer for both env and rollout sides
            # - stage cost is max(env_stage, rollout_stage) * stage_num
            if total_envs <= 0 or env_gpus <= 0 or rollout_gpus <= 0:
                return float("inf")
            if (total_envs % (env_gpus * stage_num) != 0) or (
                total_envs % (rollout_gpus * stage_num) != 0
            ):
                return float("inf")

            env_num_per_gpu_per_stage = total_envs // env_gpus // stage_num
            rollout_num_per_gpu_per_stage = total_envs // rollout_gpus // stage_num

            env_prof = DataFitter(config.profile_data.env_profile_data)
            rollout_prof = DataFitter(config.profile_data.rollout_profile_data)
            env_stage_cost = float(env_prof.get_value(env_num_per_gpu_per_stage))
            rollout_stage_cost = float(rollout_prof.get_value(rollout_num_per_gpu_per_stage))
            self.warmup_time = env_stage_cost + rollout_stage_cost
            return max(env_stage_cost, rollout_stage_cost) * (stage_num - self.warmup_group_num) + self.warmup_time

        source_roles = _roles(self.source_res)
        sink_roles = _roles(self.sink_res)

        # Warmup/drain for embodied pipeline is treated as negligible in this model.
        self.warmup_time = 0.0

        if _has_actor_cut(source_roles, sink_roles):
            # Runner executes generate and actor sequentially.
            stable_cost = self.source_res.total_cost + self.sink_res.total_cost
            return stable_cost, stable_cost

        # Generate-internal cut: prefer the stage-aware model only for env<->env_rollout cuts.
        if _is_env_rollout_cut(source_roles, sink_roles):
            env_gpus = _gpus_for_role("env", source_roles, sink_roles)
            rollout_gpus = _gpus_for_role("env_rollout", source_roles, sink_roles)
            stable_cost = _embodied_generate_pipeline_cost(
                env_gpus=env_gpus,
                rollout_gpus=rollout_gpus,
                stage_num=_stage_num(),
                total_envs=_total_envs(),
            )
            return stable_cost, stable_cost

        # Other generate cuts: bottleneck dominates (env vs rollout overlap).
        stable_cost = max(self.source_res.total_cost, self.sink_res.total_cost)
        return stable_cost, stable_cost

    def _get_disaggregated_time_reasoning(self) -> tuple[float, float]:
        config = get_global_config()

        source_cost_per_group_batch = self.source_res.get_cost_per_group_batch(
            is_source=True
        )
        sink_cost_per_group_batch = self.sink_res.get_cost_per_group_batch(
            is_source=False
        )

        if self.source_res.mode == ScheduleMode.DISAGGREGATED:
            source_warmup_time = self.source_res.warmup_time
        else:
            source_warmup_time = source_cost_per_group_batch

        if self.sink_res.mode == ScheduleMode.DISAGGREGATED:
            sink_warmup_time = self.sink_res.warmup_time
        else:
            sink_warmup_time = sink_cost_per_group_batch

        self.warmup_time = (
            source_warmup_time + sink_warmup_time
        ) * self.warmup_group_num
        try:
            cost_per_group_batch = max(
                source_cost_per_group_batch, sink_cost_per_group_batch
            )
        except TypeError as e:
            print(f"[debug] TypeError: {e}")
            print(
                f"[debug] {self.source_res=}, source_cost_per_group_batch: {source_cost_per_group_batch}"
            )
            print(
                f"[debug] {self.sink_res=}, sink_cost_per_group_batch: {sink_cost_per_group_batch}"
            )
            raise e

        bottleneck_cost = cost_per_group_batch * (
            config.rollout_batch_size - self.warmup_group_num
        )
        return cost_per_group_batch, self.warmup_time + bottleneck_cost

    def _get_disaggregated_placement(self) -> dict[ComponentNode, int]:
        source_placement: dict[ComponentNode, int] = self.source_res.placement
        sink_placement: dict[ComponentNode, int] = self.sink_res.placement

        pipeline_placement = {**source_placement}
        offset = self.source_res.total_gpu_num
        for node, gpu_range in sink_placement.items():
            pipeline_placement[node] = range(
                gpu_range[0] + offset, gpu_range[-1] + 1 + offset
            )
        return pipeline_placement
