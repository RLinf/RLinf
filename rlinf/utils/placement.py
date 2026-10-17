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
import math
from enum import Enum, auto
from typing import NamedTuple, Optional

from omegaconf import DictConfig

from rlinf.scheduler import (
    Cluster,
    ComponentPlacement,
    FlexiblePlacementStrategy,
    PackedPlacementStrategy,
)
from rlinf.scheduler.cluster.cluster import ClusterEnvVar


class PlacementMode(Enum):
    """
    Component placement mode represents the way to place components on GPUs.

    COLLOCATED: All components share the same set of GPUs.
    DISAGGREGATED: Each component has its own dedicated set of GPUs.
    HYBRID: Hybrid placement mode that allows components to run on any sets of GPUs.
    AUTO: Automatically choose the placement mode based on the component placement.
    """

    COLLOCATED = auto()
    DISAGGREGATED = auto()
    HYBRID = auto()
    AUTO = auto()


class RolloutSyncMode(Enum):
    """
    Rollout sync mode represents the way to synchronize rollout model weights.

    This mode is only used in reasoning scenarios.

    COLLOCATED: Used when rollout and actor components share the same set of GPUs.
        No inter-rank communication is required, and synchronization is typically
        conducted via CUDA IPC for optimal performance.

    DISAGGREGATED: Used when rollout and actor components use different sets of GPUs.
        Inter-rank communication is required, and synchronization is typically
        conducted via collective communication operations, such as NCCL.

    A key difference between modes is the rank mapping data structure:
    - COLLOCATED: rank mapping uses format `dict[int, tuple[int, int]]`
    - DISAGGREGATED: rank mapping uses format `dict[int, list[tuple[int, int]]]`"""

    COLLOCATED = auto()
    DISAGGREGATED = auto()


def placement_mode_to_rollout_sync_mode(
    placement_mode: PlacementMode,
) -> RolloutSyncMode:
    """Map placement mode to rollout sync mode in general cases.

    In special scenarios, the rollout sync mode is not the same as the placement mode. Thus, rollout sync mode should assigned separately, do not use this function in such scenarios.

    Args:
        placement_mode (PlacementMode): The placement mode.

    Returns:
        RolloutSyncMode: The corresponding rollout sync mode.
    """
    return (
        RolloutSyncMode.COLLOCATED
        if placement_mode == PlacementMode.COLLOCATED
        else RolloutSyncMode.DISAGGREGATED
    )


class _RolloutNodeLayout(NamedTuple):
    """Derived multi-node rollout layout.

    Attributes:
        cross_node: True when one model instance spans multiple nodes.
        gpus_per_node: The (equal) rollout GPU share per node. Only defined
            when `cross_node` is True; single-node-per-instance layouts do
            not use this quantity for any derivation.
        nnodes_per_model_instance: Nodes one model instance spans (sglang
            ServerArgs' `nnodes`). Always 1 when not `cross_node`.
        gpus_per_process: GPUs each rollout worker process holds
            (= num_hardware_per_process). `tp * pp` when not `cross_node`.
    """

    cross_node: bool
    gpus_per_node: Optional[int]
    nnodes_per_model_instance: int
    gpus_per_process: int


def _derive_rollout_node_layout(
    rollout_gpu_groups: list[list[int]], tp_pp: int
) -> _RolloutNodeLayout:
    """Decide whether model instances span nodes, and derive the layout.

    Derivation order:
    1. If every node's share is >= tp*pp and divisible by tp*pp, instances
       stay within one node (nnodes=1) and the shares need NOT be equal -
       each node simply hosts whole instances.
    2. Otherwise the layout is treated as cross-node: every node must carry
       an EQUAL share, and tp*pp must be divisible by that share because
       sglang splits an instance's ranks evenly across its nodes.
    """
    shares = [len(group) for group in rollout_gpu_groups]
    assert shares, "Rollout GPUs must cover at least one accelerator."
    if min(shares) >= tp_pp and all(share % tp_pp == 0 for share in shares):
        return _RolloutNodeLayout(
            cross_node=False,
            gpus_per_node=None,
            nnodes_per_model_instance=1,
            gpus_per_process=tp_pp,
        )
    # Cross-node: equal per-node share is required.
    assert all(share == shares[0] for share in shares), (
        f"Rollout requires an equal GPU share on every node carrying it "
        f"(got per-node shares {shares} with tp*pp={tp_pp}); either give "
        f"every node the same share, or make each share a whole multiple "
        f"of tp*pp so model instances stay within one node."
    )
    share = shares[0]
    # sglang splits an instance's tp*pp ranks evenly across its nodes;
    # e.g. share 6 with tp=8 would give 6 GPUs per process where sglang
    # expects 4.
    assert tp_pp % share == 0, (
        f"Cross-node rollout requires tp*pp ({tp_pp}) to be divisible by the "
        f"per-node GPU share ({share}) derived from component_placement; "
        f"either make tp*pp a whole multiple of the share so instances can "
        f"span nodes, or make each node's share a whole multiple of tp*pp "
        f"so model instances stay within one node."
    )
    return _RolloutNodeLayout(
        cross_node=True,
        gpus_per_node=share,
        nnodes_per_model_instance=math.ceil(tp_pp / share),
        gpus_per_process=min(tp_pp, share),
    )


def rollout_backend_supports_cross_node_instances(rollout_cfg: DictConfig) -> bool:
    """Whether the configured rollout backend can host cross-node instances.

    Single definition site of the backend whitelist for rollout model
    instances spanning multiple nodes. This is the
    CURRENT-VERSION capability boundary, not a design verdict: only the
    sglang server backend implements the multi-node path (rendezvous /
    ``dist_init_addr`` / entry-process RPC gating). vLLM cross-node is a
    planned extension - when it lands, extend only this function and drop
    the matching rejection in one place.
    """
    backend = rollout_cfg.get("rollout_backend", None)
    if backend != "sglang":
        return False
    sglang_cfg = rollout_cfg.get("sglang", None)
    backend_type = sglang_cfg.get("backend_type", "engine") if sglang_cfg else "engine"
    return backend_type == "server"


def _assert_cross_node_backend_supported(
    rollout_cfg: DictConfig, layout: _RolloutNodeLayout
) -> None:
    """Readable startup-time rejection of cross-node-incapable backends.

    Without this, e.g. the sglang Engine backend or vLLM would start every
    rollout process with tp*pp GPUs that a single node cannot see and fail
    deep inside the backend with a cryptic error.
    """
    if layout.nnodes_per_model_instance <= 1:
        return
    backend = rollout_cfg.get("rollout_backend", None)
    sglang_cfg = rollout_cfg.get("sglang", None)
    backend_type = sglang_cfg.get("backend_type", "engine") if sglang_cfg else "engine"
    assert rollout_backend_supports_cross_node_instances(rollout_cfg), (
        f"Cross-node rollout model instances "
        f"(nnodes_per_model_instance={layout.nnodes_per_model_instance}) are "
        f"only supported by the sglang server backend in the current version "
        f"(configured: rollout_backend={backend!r}"
        + (f", sglang.backend_type={backend_type!r}" if backend == "sglang" else "")
        + "). vLLM cross-node support is a planned extension."
    )


def _rollout_gpu_groups_by_node(
    rollout_gpus: list[int], node_accelerator_ranks: list[list[int]]
) -> list[list[int]]:
    """Group the rollout hardware ranks by (pseudo-)physical node.

    Takes the per-node accelerator ranks as plain data (captured at
    placement construction), not a live ``Cluster``, so the
    caller stays pickle-safe.

    The per-node share is DERIVED from component_placement - it is neither a
    config key nor a cluster average (`num_accelerators // num_nodes` breaks on
    heterogeneous clusters and clusters with accelerator-less nodes). Nodes
    carrying no rollout GPUs are skipped.

    Test hook: `RLINF_SIMULATED_GPUS_PER_NODE` chunks each physical node's
    accelerators into pseudo-nodes of that size, so one machine can simulate
    a multi-node layout. The hook fakes a node-size (cluster-layer) fact only;
    the rollout layer always derives.

    Returns one list of rollout ranks per (pseudo-)node, in node order. The
    shares may be UNEQUAL: whether that is legal is decided downstream by
    `_derive_rollout_node_layout` (equal shares are only required when a
    model instance spans nodes).
    """
    node_accel_ranks = node_accelerator_ranks
    simulated = Cluster.get_sys_env_var(ClusterEnvVar.SIMULATED_GPUS_PER_NODE)
    if simulated is not None:
        node_size = int(simulated)
        assert node_size > 0, (
            f"{ClusterEnvVar.SIMULATED_GPUS_PER_NODE} must be a positive "
            f"integer, got '{simulated}'."
        )
        flat_ranks = [rank for ranks in node_accel_ranks for rank in ranks]
        node_accel_ranks = [
            flat_ranks[i : i + node_size] for i in range(0, len(flat_ranks), node_size)
        ]
    rollout_set = set(rollout_gpus)
    groups = []
    for ranks in node_accel_ranks:
        share = [rank for rank in ranks if rank in rollout_set]
        if share:
            groups.append(share)
    assert groups, "Rollout GPUs must cover at least one accelerator."
    return groups


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
        if self._inference_gpus is None:  # try 'inference' then 'actor_inference'
            self._inference_gpus = self._get_component_hardware("actor_inference")
        self._critic_inference_gpus = self._get_component_hardware("critic_inference")
        self._reward_gpus = self._get_component_hardware("reward")
        self._critic_gpus = self._get_component_hardware("critic")
        self._cluster_num_gpus = cluster.num_accelerators
        # Store plain data, not the Cluster: placement is pickled into
        # non-Ray processes (sglang server/schedulers), where Ray actor
        # handles cannot be unpickled.
        self._node_accelerator_ranks = cluster.accelerator_ranks
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
        assert (
            self._rollout_gpus
            == list(range(self._rollout_gpus[0], self._rollout_gpus[-1] + 1))
            or self._is_disaggregated()
        ), (
            f"Rollout GPUs {self._rollout_gpus} must be continuous, unless "
            f"the placement is disaggregated."
        )
        if self._inference_gpus is not None:
            assert self._inference_gpus == list(
                range(self._inference_gpus[0], self._inference_gpus[-1] + 1)
            ), f"Inference GPUs {self._inference_gpus} must be continuous."
        if self._critic_inference_gpus is not None:
            assert self._critic_inference_gpus == list(
                range(
                    self._critic_inference_gpus[0], self._critic_inference_gpus[-1] + 1
                )
            ), (
                f"Critic inference GPUs {self._critic_inference_gpus} must be continuous."
            )

        if self._critic_gpus is not None:
            assert self._critic_gpus == list(
                range(self._critic_gpus[0], self._critic_gpus[-1] + 1)
            ), f"Critic GPUs {self._critic_gpus} must be continuous."

        self._actor_num_gpus = len(self._actor_gpus)
        self._inference_num_gpus = (
            len(self._inference_gpus) if self._inference_gpus else 0
        )
        self._critic_inference_num_gpus = (
            len(self._critic_inference_gpus) if self._critic_inference_gpus else 0
        )
        self._rollout_num_gpus = len(self._rollout_gpus)
        self._reward_num_gpus = len(self._reward_gpus) if self._reward_gpus else 0
        self._critic_num_gpus = len(self._critic_gpus) if self._critic_gpus else 0

        if self._is_auto():
            self._placement_mode = PlacementMode.AUTO
            logging.info("Running in auto mode")
        elif self._is_collocated():
            assert self._inference_gpus is None, (
                "Inference GPUs must not be specified in collocated mode."
            )
            assert self._critic_inference_gpus is None, (
                "Critic inference GPUs must not be specified in collocated mode."
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

            if self._critic_inference_gpus is not None:
                assert (
                    self.critic_inference_tp_size <= self.critic_inference_world_size
                ), (
                    f"Inference TP size {self.critic_inference_tp_size} must be less than or equal to Inference world size {self.critic_inference_world_size}."
                )

            self._placement_mode = PlacementMode.DISAGGREGATED
            logging.info("Running in disaggregated mode")
        else:
            raise ValueError(
                f"The specified placement does not match either the collocated mode (all the components use the same GPUs) or the disaggregated mode (all the components use completely different GPUs), but got {self._component_rank_map}"
            )
        self._rollout_sync_mode = placement_mode_to_rollout_sync_mode(
            self._placement_mode
        )
        # Sanity checking
        assert self.actor_tp_size <= self.actor_world_size, (
            f"Actor TP size {self.actor_tp_size} must be less than or equal to Actor world size {self.actor_world_size}."
        )
        assert self.rollout_tp_size <= self.rollout_world_size, (
            f"Rollout TP size {self.rollout_tp_size} must be less than or equal to Rollout world size {self.rollout_world_size}."
        )

        self._recompute_rollout_node_layout()
        # Auto-scheduler migration assumes one node per instance; reject
        # here rather than fail inside the scheduler loop.
        assert not (
            self._placement_mode == PlacementMode.AUTO
            and self.rollout_nnodes_per_model_instance > 1
        ), (
            "Auto scheduler does not support cross-node rollout model "
            "instances (rollout spans "
            f"{self.rollout_nnodes_per_model_instance} nodes per instance). "
            "Disable cluster.auto_scheduler or keep each rollout model "
            "instance on a single node."
        )
        # Cross-node layouts need a backend that actually injects the
        # multi-node parameters (single whitelist).
        _assert_cross_node_backend_supported(self._config.rollout, self._rollout_layout)
        self._generate_placements()

    def _is_auto(self):
        if not getattr(self._config.cluster, "auto_scheduler", False):
            return False

        # TODO for now critic model is not supported in auto scheduling mode
        if self._critic_gpus is not None:
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
        critic_gpu_set = set([] if self._critic_gpus is None else self._critic_gpus)
        rollout_gpu_set = set(self._rollout_gpus)
        inference_gpu_set = (
            [] if self._inference_gpus is None else set(self._inference_gpus)
        )
        critic_inference_gpu_set = set(
            [] if self._critic_inference_gpus is None else self._critic_inference_gpus
        )

        return (
            actor_gpu_set.isdisjoint(rollout_gpu_set)
            and actor_gpu_set.isdisjoint(inference_gpu_set)
            and rollout_gpu_set.isdisjoint(inference_gpu_set)
            and critic_gpu_set.isdisjoint(actor_gpu_set)
            and critic_gpu_set.isdisjoint(rollout_gpu_set)
            and critic_gpu_set.isdisjoint(critic_inference_gpu_set)
            and rollout_gpu_set.isdisjoint(critic_inference_gpu_set)
        )

    def _rollout_placement_strategy(self):
        """Build the rollout placement strategy.

        Continuous rollout intervals (all existing configs) keep using
        PackedPlacementStrategy unchanged. Discontinuous intervals
        (disaggregated-only) use FlexiblePlacementStrategy: each node's GPU
        share is chunked by gpus_per_process and the chunks are laid out in
        node order, so one instance's processes get consecutive ranks -
        matching rollout_model_instance_id(rank) = rank // nnodes.
        """
        gpus = self._rollout_gpus
        if gpus == list(range(gpus[0], gpus[-1] + 1)):
            return PackedPlacementStrategy(
                gpus[0],
                gpus[-1],
                num_hardware_per_process=self.rollout_gpus_per_process,
            )
        assert self.is_disaggregated, (
            f"Discontinuous rollout GPUs {gpus} are only supported in "
            f"disaggregated mode."
        )
        per_process = self.rollout_gpus_per_process
        hardware_ranks_list = [
            group[i : i + per_process]
            for group in self._rollout_gpu_groups
            for i in range(0, len(group), per_process)
        ]
        return FlexiblePlacementStrategy(hardware_ranks_list)

    def _recompute_rollout_node_layout(self) -> None:
        """(Re)derive the per-node rollout GPU groups and node layout.

        Called from `__init__`, and again from the AUTO branch when
        `use_pre_process_policy` extends the rollout GPU interval. Pure
        derivation: legality checks (auto-scheduler, backend whitelist)
        run at the call sites after the layout settles.
        """
        self._rollout_gpu_groups = _rollout_gpu_groups_by_node(
            self._rollout_gpus, self._node_accelerator_ranks
        )
        self._rollout_layout = _derive_rollout_node_layout(
            self._rollout_gpu_groups,
            self.rollout_tp_size
            * self._config.rollout.get("pipeline_parallel_size", 1),
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
            # Dense (stride=1) placement: rollout ranks are packed
            # contiguously. The strided layout used before assumed that rollout
            # rank r shares a device with actor rank r, which no longer holds
            # once sglang assigns its own attention coordinates. send/recv
            # already picks IPC or NCCL by detecting same-device pairs.
            # Cross-node layout legality is asserted by
            # _derive_rollout_node_layout (runs in __init__ unconditionally).
            self._placements["rollout"] = self._rollout_placement_strategy()
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )
            if self._critic_gpus is not None:
                self._placements["critic"] = PackedPlacementStrategy(
                    self._critic_gpus[0], self._critic_gpus[-1]
                )
        elif self._placement_mode == PlacementMode.DISAGGREGATED:
            self._placements["rollout"] = self._rollout_placement_strategy()
            if self._inference_gpus is not None:
                # TODO check the placement name
                self._placements[
                    "inference"
                    if self._critic_inference_gpus is None
                    else "actor_inference"
                ] = PackedPlacementStrategy(
                    self._inference_gpus[0], self._inference_gpus[-1]
                )
            if self._critic_inference_gpus is not None:
                self._placements["critic_inference"] = PackedPlacementStrategy(
                    self._critic_inference_gpus[0], self._critic_inference_gpus[-1]
                )
            self._placements["actor"] = PackedPlacementStrategy(
                self._actor_gpus[0], self._actor_gpus[-1]
            )
            if self._reward_gpus:
                self._placements["reward"] = PackedPlacementStrategy(
                    self._reward_gpus[0], self._reward_gpus[-1]
                )
            if self._critic_gpus is not None:
                self._placements["critic"] = PackedPlacementStrategy(
                    self._critic_gpus[0], self._critic_gpus[-1]
                )
        elif self._placement_mode == PlacementMode.AUTO:
            # In AUTO mode, actor will be placed on all GPUs
            self._placements["actor"] = PackedPlacementStrategy(
                0, self._cluster_num_gpus - 1
            )

            if self._critic_gpus is not None:
                assert False, (
                    "auto placement is not supported when having critic model for now"
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
                self._recompute_rollout_node_layout()
                # Re-check the backend whitelist: the extended rollout
                # interval may have made instances cross-node.
                _assert_cross_node_backend_supported(
                    self._config.rollout, self._rollout_layout
                )

            # Cross-node layout legality is asserted by
            # _derive_rollout_node_layout (runs in __init__ and again from
            # _recompute_rollout_node_layout above).
            self._placements["rollout"] = self._rollout_placement_strategy()

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

    def has_dedicated_inference_for_role(self, role):
        if role == "actor":
            return self.has_dedicated_actor_inference
        elif role == "critic":
            return self.has_dedicated_critic_inference
        else:
            assert False, (
                f"Unknown role {role} while calling has_dedicated_inference_for_role"
            )

    @property
    def has_dedicated_inference(self):
        return (
            self._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            and self._inference_gpus is not None
        )

    @property
    def has_dedicated_actor_inference(self):
        return (
            self._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            and self._inference_gpus is not None
        )

    @property
    def has_dedicated_critic_inference(self):
        return (
            self._placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]
            and self._critic_inference_gpus is not None
        )

    @property
    def actor_dp_size(self) -> int:
        return self._actor_num_gpus // (
            self._config.actor.model.get("tensor_model_parallel_size", 1)
            * self._config.actor.model.get("context_parallel_size", 1)
            * self._config.actor.model.get("pipeline_model_parallel_size", 1)
        )

    @property
    def critic_dp_size(self) -> int:
        return self._critic_num_gpus // (
            self._config.critic.model.get("tensor_model_parallel_size", 1)
            * self._config.critic.model.get("context_parallel_size", 1)
            * self._config.critic.model.get("pipeline_model_parallel_size", 1)
        )

    @property
    def actor_tp_size(self) -> int:
        return self._config.actor.model.get("tensor_model_parallel_size", 1)

    @property
    def critic_tp_size(self) -> int:
        return self._config.critic.model.get("tensor_model_parallel_size", 1)

    @property
    def actor_pp_size(self) -> int:
        return self._config.actor.model.get("pipeline_model_parallel_size", 1)

    @property
    def critic_pp_size(self) -> int:
        return self._config.critic.model.get("pipeline_model_parallel_size", 1)

    @property
    def actor_world_size(self) -> int:
        return self._actor_num_gpus

    @property
    def critic_world_size(self) -> int:
        return self._critic_num_gpus

    @property
    def inference_tp_size(self) -> int:
        if hasattr(self._config, "inference"):
            infer_cfg = self._config.inference
        elif hasattr(self._config, "actor_inference"):
            infer_cfg = self._config.actor_inference
        else:
            return self.actor_tp_size

        return infer_cfg.model.get("tensor_model_parallel_size", 1)

    @property
    def critic_inference_tp_size(self) -> int:
        if (
            hasattr(self._config, "critic_inference")
            and hasattr(self._config.critic_inference, "model")
            and hasattr(
                self._config.critic_inference.model, "tensor_model_parallel_size"
            )
        ):
            return self._config.critic_inference.model.get(
                "tensor_model_parallel_size", 1
            )
        else:
            return self.critic_tp_size

    @property
    def inference_pp_size(self) -> int:
        if hasattr(self._config, "inference"):
            infer_cfg = self._config.inference
        elif hasattr(self._config, "actor_inference"):
            infer_cfg = self._config.actor_inference
        else:
            return self.actor_pp_size
        return infer_cfg.model.get("pipeline_model_parallel_size", self.actor_pp_size)

    @property
    def critic_inference_pp_size(self) -> int:
        if (
            hasattr(self._config, "critic_inference")
            and hasattr(self._config.critic_inference, "model")
            and hasattr(
                self._config.critic_inference.model, "pipeline_model_parallel_size"
            )
        ):
            return self._config.critic_inference.model.get(
                "pipeline_model_parallel_size", 1
            )
        else:
            return self.critic_pp_size

    @property
    def inference_dp_size(self) -> int:
        return self._inference_num_gpus // (
            self.inference_tp_size * self.inference_pp_size
        )

    @property
    def critic_inference_dp_size(self) -> int:
        return self._critic_inference_num_gpus // (
            self.critic_inference_tp_size * self.critic_inference_pp_size
        )

    @property
    def inference_world_size(self) -> int:
        return self._inference_num_gpus

    @property
    def critic_inference_world_size(self) -> int:
        return self._critic_inference_num_gpus

    @property
    def rollout_num_model_instances(self) -> int:
        # Number of model replicas (not sglang's dp_size, which counts
        # attention-DP groups inside one instance).
        return self._rollout_num_gpus // (
            self._config.rollout.get("tensor_parallel_size", 1)
            * self._config.rollout.get("pipeline_parallel_size", 1)
        )

    @property
    def rollout_gpus_per_node(self) -> int:
        # Rollout GPUs per node, derived from component_placement; only
        # defined for cross-node layouts. RLINF_SIMULATED_GPUS_PER_NODE
        # (tests only) overrides the node size.
        if not self._rollout_layout.cross_node:
            raise RuntimeError(
                "rollout_gpus_per_node is only defined when one rollout "
                "model instance spans nodes; this layout keeps every "
                "instance within a single node."
            )
        return self._rollout_layout.gpus_per_node

    @property
    def rollout_nnodes_per_model_instance(self) -> int:
        # Number of physical nodes one model instance spans. Directly
        # corresponds to sglang ServerArgs' nnodes. Always 1 when instances
        # do not span nodes.
        return self._rollout_layout.nnodes_per_model_instance

    @property
    def rollout_gpus_per_process(self) -> int:
        # GPUs per rollout worker process: tp*pp, or the per-node share
        # when an instance spans nodes.
        return self._rollout_layout.gpus_per_process

    @property
    def rollout_num_worker_processes(self) -> int:
        # Rollout WorkerGroup size = instances * nodes per instance. Use
        # this, not rollout_num_model_instances, for worker process ranks.
        return self.rollout_num_model_instances * self.rollout_nnodes_per_model_instance

    def rollout_model_instance_id(self, worker_process_rank: int) -> int:
        """Map a rollout worker process rank to its model instance id."""
        return worker_process_rank // self.rollout_nnodes_per_model_instance

    def rollout_node_rank_in_model_instance(self, worker_process_rank: int) -> int:
        """Rank of a worker process's node within its model instance.

        Directly corresponds to sglang ServerArgs' node_rank.
        """
        return worker_process_rank % self.rollout_nnodes_per_model_instance

    def rollout_model_instance_entry_process_rank(self, model_instance_id: int) -> int:
        """WorkerGroup rank of a model instance's entry process (node_rank==0).

        All execute_on(...) calls targeting a specific model instance must go
        through this - passing model_instance_id directly would hit a
        non-entry process once an instance spans multiple nodes.
        """
        return model_instance_id * self.rollout_nnodes_per_model_instance

    @property
    def rollout_tp_size(self) -> int:
        return self._config.rollout.get("tensor_parallel_size", 1)

    @property
    def rollout_attn_tp_size(self) -> int:
        # Attention TP for the weight-reshard path. Equals rollout_tp_size
        # normally, but under sglang DP-attention the engine's
        # tensor_parallel_size is the world size while attention is sharded by
        # tp // dp. Placement still uses rollout_tp_size; the reshard
        # destination ranks and RankMapper use this one.
        tp = self._config.rollout.get("tensor_parallel_size", 1)
        sglang = self._config.rollout.get("sglang", {})
        if sglang.get("enable_dp_attention", False):
            dp = sglang.get("dp_size", 1)
            assert tp % dp == 0, (
                f"rollout tensor_parallel_size ({tp}) must be divisible by "
                f"sglang dp_size ({dp}) when enable_dp_attention is on"
            )
            return tp // dp
        return tp

    @property
    def rollout_ep_size(self) -> int:
        # Rollout MoE expert-parallel size. Each rollout rank holds
        # num_moe_experts // ep_size experts, which is what dst_ep_rank in
        # MegatronActor is computed against. Defaults to tp when only
        # enable_ep_moe is set. moe_dp replication is a separate TODO.
        sglang = self._config.rollout.get("sglang", {})
        if "ep_size" in sglang:
            return sglang["ep_size"]
        if sglang.get("enable_ep_moe", False):
            return self._config.rollout.get("tensor_parallel_size", 1)
        return 1

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


class ModelParallelEvalComponentPlacement(ComponentPlacement):
    """Component placement for model-parallel components in eval.

    The components must be rollout and reward, whose GPUs must be continuous.

    This placement only supports collocated mode.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize ModelParallelEvalComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        super().__init__(config, cluster)

        self._rollout_gpus = self._get_component_hardware("rollout")
        self._reward_gpus = self._get_component_hardware("reward")
        self._cluster_num_gpus = cluster.num_accelerators
        # Store plain data, not the Cluster: placement is pickled into
        # non-Ray processes (sglang server/schedulers), where Ray actor
        # handles cannot be unpickled.
        self._node_accelerator_ranks = cluster.accelerator_ranks
        assert self._rollout_gpus is not None, (
            "Rollout GPUs must be specified in the component_placement config."
        )
        assert self._reward_gpus is not None, (
            "Reward GPUs must be specified in the component_placement config."
        )
        assert self._rollout_gpus == list(
            range(self._rollout_gpus[0], self._rollout_gpus[-1] + 1)
        ), f"Rollout GPUs {self._rollout_gpus} must be continuous."

        self._rollout_num_gpus = len(self._rollout_gpus)
        self._reward_num_gpus = len(self._reward_gpus) if self._reward_gpus else 0

        self._placement_mode = PlacementMode.COLLOCATED

        # Sanity checking
        assert self.rollout_tp_size <= self.rollout_world_size, (
            f"Rollout TP size {self.rollout_tp_size} must be less than or equal to Rollout world size {self.rollout_world_size}."
        )

        self._rollout_gpu_groups = _rollout_gpu_groups_by_node(
            self._rollout_gpus, self._node_accelerator_ranks
        )
        self._rollout_layout = _derive_rollout_node_layout(
            self._rollout_gpu_groups,
            self.rollout_tp_size
            * self._config.rollout.get("pipeline_parallel_size", 1),
        )
        # Cross-node layouts need a backend that actually injects the
        # multi-node parameters (single whitelist).
        _assert_cross_node_backend_supported(self._config.rollout, self._rollout_layout)
        self._generate_placements()

    def _generate_placements(self):
        assert self._placement_mode == PlacementMode.COLLOCATED
        # Cross-node layout legality is asserted by
        # _derive_rollout_node_layout (runs in __init__ unconditionally).
        self._placements["rollout"] = PackedPlacementStrategy(
            self._rollout_gpus[0],
            self._rollout_gpus[-1],
            num_hardware_per_process=self.rollout_gpus_per_process,
            stride=1,
        )
        if self._reward_gpus:
            self._placements["reward"] = PackedPlacementStrategy(
                self._reward_gpus[0], self._reward_gpus[-1]
            )

    @property
    def is_collocated(self):
        return True

    @property
    def is_disaggregated(self):
        return False

    @property
    def is_auto(self):
        return False

    @property
    def is_pipeline(self):
        return False

    @property
    def has_dedicated_inference(self):
        return False

    @property
    def rollout_num_model_instances(self) -> int:
        # See ModelParallelComponentPlacement.rollout_num_model_instances
        # (renamed from rollout_dp_size; shared semantics).
        return self._rollout_num_gpus // (
            self._config.rollout.get("tensor_parallel_size", 1)
            * self._config.rollout.get("pipeline_parallel_size", 1)
        )

    @property
    def rollout_gpus_per_node(self) -> int:
        # See ModelParallelComponentPlacement.rollout_gpus_per_node:
        # derived from component_placement, never a config key; only
        # defined when one model instance spans nodes.
        if not self._rollout_layout.cross_node:
            raise RuntimeError(
                "rollout_gpus_per_node is only defined when one rollout "
                "model instance spans nodes; this layout keeps every "
                "instance within a single node."
            )
        return self._rollout_layout.gpus_per_node

    @property
    def rollout_nnodes_per_model_instance(self) -> int:
        # See ModelParallelComponentPlacement.rollout_nnodes_per_model_instance.
        return self._rollout_layout.nnodes_per_model_instance

    @property
    def rollout_gpus_per_process(self) -> int:
        # See ModelParallelComponentPlacement.rollout_gpus_per_process.
        return self._rollout_layout.gpus_per_process

    @property
    def rollout_num_worker_processes(self) -> int:
        # See ModelParallelComponentPlacement.rollout_num_worker_processes.
        return self.rollout_num_model_instances * self.rollout_nnodes_per_model_instance

    def rollout_model_instance_id(self, worker_process_rank: int) -> int:
        """Map a rollout worker process rank to its model instance id."""
        return worker_process_rank // self.rollout_nnodes_per_model_instance

    def rollout_node_rank_in_model_instance(self, worker_process_rank: int) -> int:
        """Rank of a worker process's node within its model instance."""
        return worker_process_rank % self.rollout_nnodes_per_model_instance

    def rollout_model_instance_entry_process_rank(self, model_instance_id: int) -> int:
        """WorkerGroup rank of a model instance's entry process (node_rank==0)."""
        return model_instance_id * self.rollout_nnodes_per_model_instance

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


class MultiAgentModelParallelComponentPlacement(ModelParallelComponentPlacement):
    """Component placement for model-parallel components.

    The components must be actor, rollout, and optionally inference, whose GPUs must be continuous.

    This placement supports only collocated mode.

    In the collocated mode, all components share the same set of GPUs. In particular, the rollout group is specially placed in a strided manner to enable fast cudaIPC-based weight sync.
    In the disaggregated mode, each component has its own dedicated set of GPUs.

    In the collocated mode, only actor and rollout exist. While in the disaggregated mode, actor, rollout, and inference should all exist.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize ModelParallelComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        self._cfg = config
        super().__init__(config, cluster)
        self._validate_collocated_placement()
        self._validate_resource_coverage()
        if not self._is_single_engine_placement():
            # use disaggregated rollout sync mode for multi-engine scenario
            self._rollout_sync_mode = RolloutSyncMode.DISAGGREGATED
        else:
            # use collocated rollout sync mode for single-engine scenario
            self._rollout_sync_mode = placement_mode_to_rollout_sync_mode(
                self._placement_mode
            )

    def _is_collocated(self):
        """Check if the placement is collocated for multi-engine scenario.
        This method will override the default behavior of _is_collocated method in ModelParallelComponentPlacement.
        """
        return True

    def _is_single_engine_placement(self):
        if self._actor_gpus == self._rollout_gpus:
            return True
        return False

    def _validate_collocated_placement(self):
        """Check if the placement is collocated for multi-engine scenario.

        This method checks if the placement is collocated for multi-engine scenario.

        In multi-engine scenario, we consider it collocated if:
        1. actor and reward are placed on all GPUs
        2. rollout components are distributed across different GPU ranges
        3. all rollout components' GPUs are subsets of actor/reward's GPUs
        """
        # Get actor and reward GPUs
        actor_gpus = set(self._get_component_hardware("actor"))
        reward_gpus = set(self._get_component_hardware("reward"))

        # Check if actor and reward are on the same GPUs (all GPUs)
        assert actor_gpus == reward_gpus, (
            "Actor and reward must be placed on the same GPUs during multi-agent scenario."
        )

        # Check all rollout components
        rollout_components = [
            key for key in self._component_rank_map if key.startswith("rollout")
        ]

        assert rollout_components, (
            "Rollout components must exist during multi-agent scenario."
        )

        # Check if all rollout components' GPUs are subsets of actor's GPUs
        all_rollout_gpus = set()
        for component in rollout_components:
            component_gpus = set(self._get_component_hardware(component))
            assert component_gpus.issubset(actor_gpus), (
                f"Component {component} GPUs {component_gpus} are not subsets of actor GPUs {actor_gpus}."
            )
            all_rollout_gpus.update(component_gpus)

        # Check if rollout components cover all actor's GPUs
        if all_rollout_gpus != actor_gpus:
            logging.warning(
                f"Rollout components do not cover all actor GPUs. "
                f"Current: {all_rollout_gpus}, Missing: {actor_gpus - all_rollout_gpus}"
            )
        return

    def _validate_resource_coverage(self):
        """
        Validates that components are correctly placed across available GPUs.
        Ensures shared components cover all GPUs and exclusive components do not overlap.
        """
        import torch
        import torch.distributed as dist

        if dist.is_initialized():
            total_world_size = dist.get_world_size()
        else:
            total_world_size = torch.cuda.device_count()

        total_gpus = set(range(total_world_size))

        exclusive_gpu_usage = set()
        shared_names = {"actor", "rollout", "reward", "inference"}
        print(self._component_rank_map)
        for comp_name, node_map in self._component_rank_map.items():
            component_gpus = set()
            for nodes in node_map.keys():
                component_gpus.update(nodes)

            if comp_name in shared_names:
                if component_gpus != total_gpus:
                    logging.warning(
                        f"Shared component '{comp_name}' does not cover all GPUs. "
                        f"Current: {component_gpus}, Missing: {total_gpus - component_gpus}"
                    )
                continue

            overlap = exclusive_gpu_usage.intersection(component_gpus)
            if overlap:
                logging.error(f"Collision detected for component: {comp_name}")
                logging.error(f"Global GPUs for {comp_name}: {component_gpus}")
                logging.error(f"Already occupied GPUs: {exclusive_gpu_usage}")
                raise ValueError(
                    f"Resource Conflict: Component '{comp_name}' global GPU {overlap} "
                    f"is already occupied by another exclusive component."
                )

            exclusive_gpu_usage.update(component_gpus)

        missing = total_gpus - exclusive_gpu_usage
        if missing:
            logging.warning(
                f"Note: Exclusive components (planners/subworkers) did not utilize "
                f"all GPUs. Idle GPUs: {missing}"
            )
        else:
            logging.info(
                "Validation Passed: Exclusive components have perfectly "
                "covered all GPU resources with no overlaps."
            )

    def _validate_component_placement_strategy(
        self, component_name, placement_strategy
    ):
        """
        Validates that component placement strategy are valid.
        )
        """
        assert placement_strategy is not None, (
            f"Placement strategy for component '{component_name}' is None."
        )
        component_cfg = self._cfg.get(component_name, None)
        assert component_cfg is not None, (
            f"component_name {component_name} not found in cfg"
        )
        component_accel_num = (
            placement_strategy._end_hw_rank - placement_strategy._start_hw_rank + 1
        )

        if component_cfg.get("tensor_parallel_size", None) is not None:
            assert component_accel_num % component_cfg.tensor_parallel_size == 0, (
                f"Component '{component_name}' placement strategy must be divisible by tensor_parallel_size."
            )
        if component_cfg.get("pipeline_parallel_size", None) is not None:
            assert component_accel_num % component_cfg.pipeline_parallel_size == 0, (
                f"Component '{component_name}' placement strategy must be divisible by pipeline_parallel_size."
            )
        if component_cfg.get("dp_size", None) is not None:
            assert component_accel_num % component_cfg.dp_size == 0, (
                f"Component '{component_name}' placement strategy must be divisible by dp_size."
            )
        logging.info(
            f"Validation Passed: Placement strategy for component '{component_name}' is valid."
        )

    def get_strategy(
        self, component_name: str, placement_strategy: Optional[type] = None
    ):
        # handling logic for default placement strategies
        if placement_strategy is None:
            component_placement_strategy = super().get_strategy(component_name)
            logging.info(f"{component_name}: {component_placement_strategy}")
            return component_placement_strategy

        if component_name in ("rollout", "actor", "reward", "inference"):
            logging.warning(
                f"Specifying a PlacementStrategy for '{component_name}' in get_strategy() is not allowed.",
                f"Using default PackedPlacementStrategy for '{component_name}' instead.",
            )
            component_placement_strategy = super().get_strategy(component_name)
            logging.info(f"{component_name}: {component_placement_strategy}")
            return component_placement_strategy

        # handling logic for customized placement strategies
        strategy_class = (
            placement_strategy
            if placement_strategy is not None
            else PackedPlacementStrategy
        )
        assert strategy_class in [PackedPlacementStrategy], (
            f"Unsupported strategy class: {strategy_class}. Currently only PackedPlacementStrategy is supported."
        )

        component_placement_strategy = strategy_class(
            self._get_component_hardware(component_name)[0],
            self._get_component_hardware(component_name)[-1],
            num_hardware_per_process=self._placements[
                "rollout"
            ]._num_hardware_per_process,
            stride=self._placements["rollout"]._stride,
        )
        self._validate_component_placement_strategy(
            component_name, component_placement_strategy
        )
        logging.info(f"{component_name}: {component_placement_strategy}")
        return component_placement_strategy


class MultiAgentModelParallelEvalComponentPlacement(
    ModelParallelEvalComponentPlacement
):
    """Component placement for model-parallel components.

    The components must be actor, rollout, and optionally inference, whose GPUs must be continuous.

    This placement supports only collocated mode.

    In the collocated mode, all components share the same set of GPUs. In particular, the rollout group is specially placed in a strided manner to enable fast cudaIPC-based weight sync.
    In the disaggregated mode, each component has its own dedicated set of GPUs.

    In the collocated mode, only actor and rollout exist. While in the disaggregated mode, actor, rollout, and inference should all exist.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize MultiAgentModelParallelEvalComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        self._cfg = config
        super().__init__(config, cluster)

    def _validate_component_placement_strategy(
        self, component_name, placement_strategy
    ):
        """
        Validates that component placement strategy are valid.
        )
        """
        assert placement_strategy is not None, (
            f"Placement strategy for component '{component_name}' is None."
        )
        component_cfg = self._cfg.get(component_name, None)
        assert component_cfg is not None, (
            f"component_name {component_name} not found in cfg"
        )
        component_accel_num = (
            placement_strategy._end_hw_rank - placement_strategy._start_hw_rank + 1
        )

        if component_cfg.get("tensor_parallel_size", None) is not None:
            assert component_accel_num % component_cfg.tensor_parallel_size == 0, (
                f"Component '{component_name}' placement strategy must be divisible by tensor_parallel_size."
            )
        if component_cfg.get("pipeline_parallel_size", None) is not None:
            assert component_accel_num % component_cfg.pipeline_parallel_size == 0, (
                f"Component '{component_name}' placement strategy must be divisible by pipeline_parallel_size."
            )
        if component_cfg.get("dp_size", None) is not None:
            assert component_accel_num % component_cfg.dp_size == 0, (
                f"Component '{component_name}' placement strategy must be divisible by dp_size."
            )
        logging.info(
            f"Validation Passed: Placement strategy for component '{component_name}' is valid."
        )

    def get_strategy(
        self, component_name: str, placement_strategy: Optional[type] = None
    ):
        # handling logic for default placement strategies
        if placement_strategy is None:
            component_placement_strategy = super().get_strategy(component_name)
            logging.info(f"{component_name}: {component_placement_strategy}")
            return component_placement_strategy

        if component_name in ("rollout", "actor", "reward", "inference"):
            logging.warning(
                f"Specifying a PlacementStrategy for '{component_name}' in get_strategy() is not allowed.",
                f"Using default PackedPlacementStrategy for '{component_name}' instead.",
            )
            component_placement_strategy = super().get_strategy(component_name)
            logging.info(f"{component_name}: {component_placement_strategy}")
            return component_placement_strategy

        # handling logic for customized placement strategies
        strategy_class = (
            placement_strategy
            if placement_strategy is not None
            else PackedPlacementStrategy
        )
        assert strategy_class in [PackedPlacementStrategy], (
            f"Unsupported strategy class: {strategy_class}. Currently only PackedPlacementStrategy is supported."
        )

        component_placement_strategy = strategy_class(
            self._get_component_hardware(component_name)[0],
            self._get_component_hardware(component_name)[-1],
            num_hardware_per_process=self._placements[
                "rollout"
            ]._num_hardware_per_process,
            stride=self._placements["rollout"]._stride,
        )
        self._validate_component_placement_strategy(
            component_name, component_placement_strategy
        )
        logging.info(f"{component_name}: {component_placement_strategy}")
        return component_placement_strategy
