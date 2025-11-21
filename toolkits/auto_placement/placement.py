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
from util import get_global_config


class ScheduleMode(Enum):
    """Mode of schedule result."""

    SINGLE_NODE = "single_node"
    COLLOCATED = "collocated"
    DISAGGREGATED = "disaggregated"
    HYBRID = "hybrid"


class ScheduleResult(ABC):
    """Base class for all schedule results."""

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
            return ScheduleResult.build_collocated_schedule_result(
                total_gpu_num, source_res, sink_res
            )
        return ScheduleResult.build_disaggregated_schedule_result(
            total_gpu_num, source_res, sink_res, warmup_group_num
        )

    @staticmethod
    def build_collocated_schedule_result(
        total_gpu_num: int, source_res: "ScheduleResult", sink_res: "ScheduleResult"
    ) -> Optional["ScheduleResult"]:
        target_modes = [ScheduleMode.COLLOCATED, ScheduleMode.SINGLE_NODE]
        if source_res.mode in target_modes and sink_res.mode in target_modes:
            return CollocatedScheduleResult(total_gpu_num, source_res, sink_res)
        return None
        return HybirdScheduleResult(total_gpu_num, source_res, sink_res)

    @staticmethod
    def build_disaggregated_schedule_result(
        total_gpu_num: int,
        source_res: "ScheduleResult",
        sink_res: "ScheduleResult",
        warmup_group_num,
    ) -> Optional["ScheduleResult"]:
        target_modes = [ScheduleMode.DISAGGREGATED, ScheduleMode.SINGLE_NODE]
        if source_res.mode in target_modes and sink_res.mode in target_modes:
            return DisaggregatedScheduleResult(
                total_gpu_num, source_res, sink_res, warmup_group_num
            )
        return None
        return HybirdScheduleResult(total_gpu_num, source_res, sink_res)

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

    def get_cost_per_group_batch(self, *args, **kwargs) -> float:
        return self.cost_per_group_batch

    @property
    def placement_str(self) -> str:
        if self.mode == ScheduleMode.COLLOCATED:
            gpu_range = list(self.placement.values())[0]
            return (
                ", ".join([f"{node.role}" for node in self.placement])
                + f" : {gpu_range[0]}-{gpu_range[-1]}"
            )
        placement_str = ""
        for node, gpu_range in self.placement.items():
            placement_str += f"{node.role} : {gpu_range[0]}-{gpu_range[-1]}\n"
        return placement_str

    def __str__(self):
        return f"ScheduleResult : total_gpu_num={self.total_gpu_num}, total_cost={self.total_cost}, mode={self.mode.value}, placement:\n{self.placement_str}"

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


class HybirdScheduleResult(ScheduleResult):
    def __init__(
        self, total_gpu_num: int, source_res: ScheduleResult, sink_res: ScheduleResult
    ):
        raise NotImplementedError("HybirdScheduleResult is not implemented")


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
        super().__init__(
            mode=ScheduleMode.COLLOCATED,
            total_gpu_num=total_gpu_num,
            placement={
                **source_res.placement,
                **sink_res.placement,
            },
            cost_per_group_batch=None,
            total_cost=self.source_res.total_cost + self.sink_res.total_cost,
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

        warmup_time, bottleneck_cost = self._get_disaggregated_time()
        super().__init__(
            mode=ScheduleMode.DISAGGREGATED,
            total_gpu_num=total_gpu_num,
            placement=self._get_disaggregated_placement(),
            cost_per_group_batch=warmup_time,
            total_cost=warmup_time + bottleneck_cost,
        )

    def _get_disaggregated_time(self) -> tuple[float, float]:
        config = get_global_config()
        source_cost_per_group_batch = self.source_res.get_cost_per_group_batch(
            is_source=True
        )
        sink_cost_per_group_batch = self.sink_res.get_cost_per_group_batch(
            is_source=False
        )
        warmup_time = (
            source_cost_per_group_batch + sink_cost_per_group_batch
        ) * self.warmup_group_num
        bottleneck_cost = max(
            source_cost_per_group_batch, sink_cost_per_group_batch
        ) * (config.rollout_batch_size - self.warmup_group_num)
        return warmup_time, bottleneck_cost

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
