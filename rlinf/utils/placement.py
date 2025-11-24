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

import logging
from enum import Enum, auto

from omegaconf import DictConfig

from rlinf.scheduler import (
    Cluster,
    ComponentPlacement,
    PackedPlacementStrategy,
)


class PlacementMode(Enum):
    COLLOCATED = auto()
    DISAGGREGATED = auto()
    HYBRID = auto()
    AUTO = auto()


class HybridComponentPlacement(ComponentPlacement):
    """Hybrid component placement that allows components to run on any sets of GPUs."""

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize HybridComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary.
        """
        super().__init__(config, cluster)
        self._placement_mode = PlacementMode.HYBRID


class ModelParallelComponentPlacement(ComponentPlacement):
    """Component placement for model-parallel components.

    The components must be actor, rollout, and optionally inference, whose GPUs must be continuous.

    This placement supports both collocated and disaggregated modes.

    In the collocated mode, all components share the same set of GPUs. In particular, the rollout group is specially placed in a strided manner to enable fast cudaIPC-based weight sync.
    In the disaggregated mode, each component has its own dedicated set of GPUs.

    In the collocated mode, only actor and rollout exist. While in the disaggregated mode, actor, rollout, and inference should all exist.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize ModelParallelComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        super().__init__(config, cluster)

        self._actor_gpus = self._get_component_hardware("actor")
        self._rollout_gpus = self._get_component_hardware("rollout")
        self._inference_gpus = self._get_component_hardware("inference")
        self._reward_gpus = self._get_component_hardware("reward")
        self._cluster_num_gpus = cluster.num_accelerators
        assert self._actor_gpus is not None, (
            "Actor GPUs must be specified in the component_placement config."
        )
        assert self._rollout_gpus is not None, (
            "Rollout GPUs must be specified in the component_placement config."
        )
        assert self._reward_gpus is not None, (
            "Reward GPUs must be specified in the component_placement config."
        )
        assert self._actor_gpus == list(
            range(self._actor_gpus[0], self._actor_gpus[-1] + 1)
        ), f"Actor GPUs {self._actor_gpus} must be continuous."
        assert self._rollout_gpus == list(
            range(self._rollout_gpus[0], self._rollout_gpus[-1] + 1)
        ), f"Rollout GPUs {self._rollout_gpus} must be continuous."
        if self._inference_gpus is not None:
            assert self._inference_gpus == list(
                range(self._inference_gpus[0], self._inference_gpus[-1] + 1)
            ), f"Inference GPUs {self._inference_gpus} must be continuous."

        self._actor_num_gpus = len(self._actor_gpus)
        self._inference_num_gpus = (
            len(self._inference_gpus) if self._inference_gpus else 0
        )
        self._rollout_num_gpus = len(self._rollout_gpus)
        self._reward_num_gpus = len(self._reward_gpus) if self._reward_gpus else 0

        if self._is_auto():
            self._placement_mode = PlacementMode.AUTO
            logging.info("Running in auto mode")
        elif self._is_collocated():
            assert self._inference_gpus is None, (
                "Inference GPUs must not be specified in collocated mode."
            )
            self._placement_mode = PlacementMode.COLLOCATED
            logging.info("Running in collocated mode")
        elif self._is_disaggregated():
            if self._inference_gpus is not None:
                assert self.inference_tp_size <= self.inference_world_size, (
                    f"Inference TP size {self.inference_tp_size} must be less than or equal to Inference world size {self.inference_world_size}."
                )
                assert self._config.algorithm.recompute_logprobs, (
                    f"algorithm.recompute_logprobs has been set to false, which disables inference. So inference GPUs {self._inference_gpus} must not be specified."
                )
            self._placement_mode = PlacementMode.DISAGGREGATED
            logging.info("Running in disaggregated mode")
        else:
            raise ValueError(
                f"The specified placement does not match either the collocated mode (all the components use the same GPUs) or the disaggregated mode (all the components use completely different GPUs), but got {self._component_rank_map}"
            )

        # Sanity checking
        assert self.actor_tp_size <= self.actor_world_size, (
            f"Actor TP size {self.actor_tp_size} must be less than or equal to Actor world size {self.actor_world_size}."
        )
        assert self.rollout_tp_size <= self.rollout_world_size, (
            f"Rollout TP size {self.rollout_tp_size} must be less than or equal to Rollout world size {self.rollout_world_size}."
        )

        self._generate_placements()

    def _is_auto(self):
        if not getattr(self._config.cluster, "auto_scheduler", False):
            return False

        assert self._is_disaggregated(), (
            "AUTO mode is a more advanced version of disaggregated mode, so it must satisfy the requirements of disaggregated mode."
        )

        # Assert components order is : actor -> rollout -> inference
        order_error_msg = "AUTO mode requires components to be placed in the order of actor -> rollout -> inference."
        assert (
            self._actor_gpus[0] == 0
            and self._actor_gpus[-1] == self._rollout_gpus[0] - 1
        ), order_error_msg
        if self._inference_gpus is None:
            assert self._rollout_gpus[-1] == self._cluster_num_gpus - 1, order_error_msg
        else:
            assert self._rollout_gpus[-1] == self._inference_gpus[0] - 1, (
                order_error_msg
            )
            assert self._inference_gpus[-1] == self._cluster_num_gpus - 1, (
                order_error_msg
            )
        return True

    def _is_collocated(self):
        if self._actor_gpus == self._rollout_gpus:
            return True
        return False

    def _is_disaggregated(self):
        actor_gpu_set = set(self._actor_gpus)
        rollout_gpu_set = set(self._rollout_gpus)
        inference_gpu_set = (
            [] if self._inference_gpus is None else set(self._inference_gpus)
        )
        return (
            actor_gpu_set.isdisjoint(rollout_gpu_set)
            and actor_gpu_set.isdisjoint(inference_gpu_set)
            and rollout_gpu_set.isdisjoint(inference_gpu_set)
        )

    def _generate_placements(self):
        if self._placement_mode == PlacementMode.COLLOCATED:
            self._placements["actor"] = PackedPlacementStrategy(
                self._actor_gpus[0], self._actor_gpus[-1]
            )

            if self.actor_tp_size > self.rollout_tp_size:
                assert self.actor_tp_size % self.rollout_tp_size == 0, (
                    f"Actor TP size ({self.actor_tp_size}) must be divisible by Rollout TP size ({self.rollout_tp_size})"
                )
            stride = (
                self.actor_tp_size // self.rollout_tp_size
                if self.actor_tp_size > self.rollout_tp_size
                else 1
            )
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_hardware_per_process=self.rollout_tp_size,
                stride=stride,
            )
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )
        elif self._placement_mode == PlacementMode.DISAGGREGATED:
            num_gpus_per_rollout_dp = len(self._rollout_gpus) // self.rollout_dp_size
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_hardware_per_process=num_gpus_per_rollout_dp,
            )
            if self._inference_gpus is not None:
                self._placements["inference"] = PackedPlacementStrategy(
                    self._inference_gpus[0], self._inference_gpus[-1]
                )
            self._placements["actor"] = PackedPlacementStrategy(
                self._actor_gpus[0], self._actor_gpus[-1]
            )
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )
        elif self._placement_mode == PlacementMode.AUTO:
            # In AUTO mode, actor will be placed on all GPUs
            self._placements["actor"] = PackedPlacementStrategy(
                0, self._cluster_num_gpus - 1
            )

            use_pre_process_policy = getattr(
                self._config.cluster, "use_pre_process_policy", False
            )
            if use_pre_process_policy:
                assert (
                    self._actor_gpus[-1] - self._actor_gpus[0] + 1
                ) % self.rollout_tp_size == 0
                self._rollout_gpus = (
                    list(range(1 + self._actor_gpus[-1])) + self._rollout_gpus
                )
                self._rollout_num_gpus = len(self._rollout_gpus)

            num_gpus_per_rollout_dp = len(self._rollout_gpus) // self.rollout_dp_size
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_hardware_per_process=num_gpus_per_rollout_dp,
            )

            if self._inference_gpus is not None:
                self._placements["inference"] = PackedPlacementStrategy(
                    self._inference_gpus[0], self._inference_gpus[-1]
                )
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )

    @property
    def is_collocated(self):
        return self._placement_mode == PlacementMode.COLLOCATED

    @property
    def is_disaggregated(self):
        return self._placement_mode == PlacementMode.DISAGGREGATED

    @property
    def is_auto(self):
        return self._placement_mode == PlacementMode.AUTO

    @property
    def is_pipeline(self):
        return self.is_disaggregated or self.is_auto

    @property
    def has_dedicated_inference(self):
        return (
            self._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            and self._inference_gpus is not None
        )

    @property
    def actor_dp_size(self) -> int:
        return self._actor_num_gpus // (
            self._config.actor.model.get("tensor_model_parallel_size", 1)
            * self._config.actor.model.get("context_parallel_size", 1)
            * self._config.actor.model.get("pipeline_model_parallel_size", 1)
        )

    @property
    def actor_tp_size(self) -> int:
        return self._config.actor.model.get("tensor_model_parallel_size", 1)

    @property
    def actor_pp_size(self) -> int:
        return self._config.actor.model.get("pipeline_model_parallel_size", 1)

    @property
    def actor_world_size(self) -> int:
        return self._actor_num_gpus

    @property
    def inference_tp_size(self) -> int:
        if (
            hasattr(self._config, "inference")
            and hasattr(self._config.inference, "model")
            and hasattr(self._config.inference.model, "tensor_model_parallel_size")
        ):
            return self._config.inference.model.get("tensor_model_parallel_size", 1)
        else:
            return self.actor_tp_size

    @property
    def inference_pp_size(self) -> int:
        if (
            hasattr(self._config, "inference")
            and hasattr(self._config.inference, "model")
            and hasattr(self._config.inference.model, "pipeline_model_parallel_size")
        ):
            return self._config.inference.model.get("pipeline_model_parallel_size", 1)
        else:
            return self.actor_pp_size

    @property
    def inference_dp_size(self) -> int:
        return self._inference_num_gpus // (
            self.inference_tp_size * self.inference_pp_size
        )

    @property
    def inference_world_size(self) -> int:
        return self._inference_num_gpus

    @property
    def rollout_dp_size(self) -> int:
        return self._rollout_num_gpus // (
            self._config.rollout.get("tensor_parallel_size", 1)
            * self._config.rollout.get("pipeline_parallel_size", 1)
        )

    @property
    def rollout_tp_size(self) -> int:
        return self._config.rollout.get("tensor_parallel_size", 1)

    @property
    def rollout_world_size(self) -> int:
        return self._rollout_num_gpus

    @property
    def reward_world_size(self) -> int:
        return self._reward_num_gpus

    def _get_component_hardware(self, component_name: str):
        if component_name not in self._component_rank_map:
            return None
        return super().get_hardware_ranks(component_name)
