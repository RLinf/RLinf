# Copyright 2026 The RLinf Authors.
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

"""Placement view for collocated FSDP to SGLang CUDA-IPC weight sync.

Embodied stage-1 does not launch a reward worker, so it cannot construct
``ModelParallelComponentPlacement``. The scheduler and ``RankMapper`` only
read the fields below. The object is pickled into the SGLang process and
must stay free of Ray handles.

Actor ranks and SGLang TP ranks are not the same number. Collocated mode
puts one actor process on every GPU and packs those GPUs into engines of
``tensor_parallel_size``. An engine stays on one node when the node has
enough GPUs; otherwise it spans nodes and the entry process is the first
of those nodes. CUDA-IPC still lands on the GPU that owns the matching
TP rank, so a cross-node engine does not ship handles across machines.
"""

from rlinf.utils.placement import PlacementMode, RolloutSyncMode


class CollocatedIpcPlacement:
    """Collocated SGLang engines sharing the actor GPUs.

    ``rollout_world_size`` is the actor world, not the TP size. TP only
    decides how those GPUs are grouped into engines.
    """

    def __init__(
        self,
        actor_world_size: int,
        rollout_tp_size: int,
        nnodes_per_model_instance: int = 1,
    ):
        actor_world_size = int(actor_world_size)
        rollout_tp_size = int(rollout_tp_size)
        nnodes_per_model_instance = int(nnodes_per_model_instance)
        if rollout_tp_size < 1 or actor_world_size < 1:
            raise ValueError(
                "actor_world_size and tensor_parallel_size must be positive, "
                f"got {actor_world_size} and {rollout_tp_size}."
            )
        if actor_world_size % rollout_tp_size != 0:
            raise ValueError(
                "Collocated SGLang needs a whole set of TP ranks on the actor "
                f"GPUs: actor_world_size ({actor_world_size}) is not divisible "
                f"by tensor_parallel_size ({rollout_tp_size})."
            )
        if nnodes_per_model_instance < 1 or (
            rollout_tp_size % nnodes_per_model_instance != 0
        ):
            raise ValueError(
                "One SGLang engine must split its TP ranks evenly across "
                f"nodes, got tensor_parallel_size={rollout_tp_size} and "
                f"nnodes_per_model_instance={nnodes_per_model_instance}."
            )
        self.placement_mode = PlacementMode.COLLOCATED
        self._rollout_sync_mode = RolloutSyncMode.COLLOCATED
        self.actor_tp_size = 1
        self.actor_pp_size = 1
        self.actor_world_size = actor_world_size
        self.rollout_tp_size = rollout_tp_size
        self.rollout_world_size = actor_world_size
        self.nnodes_per_model_instance = nnodes_per_model_instance

    @property
    def rollout_nnodes_per_model_instance(self) -> int:
        """Nodes one engine spans. Matches SGLang ``ServerArgs.nnodes``."""
        return self.nnodes_per_model_instance

    def rollout_model_instance_id(self, worker_process_rank: int) -> int:
        """Engine id of a server process. One engine occupies ``nnodes`` processes."""
        return int(worker_process_rank) // self.nnodes_per_model_instance

    def rollout_node_rank_in_model_instance(self, worker_process_rank: int) -> int:
        """``node_rank`` of a server process inside its engine."""
        return int(worker_process_rank) % self.nnodes_per_model_instance

    def rollout_model_instance_entry_process_rank(self, engine_id: int) -> int:
        """Server-group rank of an engine's entry process (``node_rank == 0``)."""
        return int(engine_id) * self.nnodes_per_model_instance


def configure_collocated_sglang(cfg, cluster, component_placement):
    """Stamp the actor world onto the SGLang config and place its processes.

    The returned strategy gives each server process the GPUs it can see.
    When TP fits on a node that is one process per engine. When it does
    not, one engine is several processes that share a ``dist_init_addr``.
    """
    from omegaconf import open_dict

    from rlinf.scheduler import PackedPlacementStrategy
    from rlinf.utils.placement import (
        _derive_rollout_node_layout,
        _rollout_gpu_groups_by_node,
    )

    sglang_cfg = cfg.rollout.sglang
    pipeline_parallel_size = int(sglang_cfg.get("pipeline_parallel_size", 1))
    if pipeline_parallel_size != 1:
        raise ValueError(
            "Embodied SGLang weight sync does not shard pipeline stages, "
            f"got pipeline_parallel_size={pipeline_parallel_size}."
        )
    tp_size = int(sglang_cfg.tensor_parallel_size)
    hardware = [int(rank) for rank in component_placement.get_hardware_ranks("actor")]
    actor_world = int(component_placement.get_world_size("actor"))
    if not hardware:
        raise ValueError("Actor placement has no GPUs for the SGLang servers.")
    if actor_world != len(hardware):
        raise ValueError(
            "Collocated SGLang weight sync needs one actor process per GPU, "
            f"got actor world {actor_world} and {len(hardware)} GPUs."
        )
    if sorted(hardware) != list(range(min(hardware), max(hardware) + 1)):
        raise ValueError(
            "Actor GPUs must be a contiguous range so the SGLang server "
            f"group can be packed, got {sorted(hardware)}."
        )
    groups = _rollout_gpu_groups_by_node(hardware, cluster.accelerator_ranks)
    layout = _derive_rollout_node_layout(groups, tp_size)
    CollocatedIpcPlacement(actor_world, tp_size, layout.nnodes_per_model_instance)
    with open_dict(sglang_cfg):
        sglang_cfg.actor_world_size = actor_world
        sglang_cfg.nnodes_per_model_instance = int(layout.nnodes_per_model_instance)
    return PackedPlacementStrategy(
        start_hardware_rank=min(hardware),
        end_hardware_rank=max(hardware),
        num_hardware_per_process=int(layout.gpus_per_process),
    )
