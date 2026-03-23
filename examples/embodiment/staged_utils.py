# Copyright 2026 Shirui Chen
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

"""Shared helpers for staged embodied entrypoints."""

import ray

from rlinf.scheduler import AcceleratorUtil, Cluster, PackedPlacementStrategy
from rlinf.workers.vlm_planner import VLMPlannerWorker

VLM_PLANNER_NODE_GROUP = "beaker_vlm"


def compute_vlm_gpu_index(cfg) -> int:
    """Return the GPU index to use for VLMPlannerWorker."""
    vlm_cfg = getattr(cfg, "vlm_planner", None)
    if vlm_cfg is not None:
        explicit = getattr(vlm_cfg, "placement", None)
        if explicit is not None:
            return int(explicit)

    vlm_node_ranks: set[int] = set()
    for group in cfg.cluster.node_groups:
        if group.label == VLM_PLANNER_NODE_GROUP:
            node_ranks = group.node_ranks
            if isinstance(node_ranks, int):
                vlm_node_ranks.add(node_ranks)
            else:
                for rank in str(node_ranks).split(","):
                    vlm_node_ranks.add(int(rank.strip()))
            break

    group_ranks: dict[str, set[int]] = {}
    for group in cfg.cluster.node_groups:
        node_ranks = group.node_ranks
        if isinstance(node_ranks, int):
            ranks: set[int] = {node_ranks}
        else:
            ranks = {int(rank.strip()) for rank in str(node_ranks).split(",")}
        group_ranks[group.label] = ranks

    placements_on_shared_node: set[int] = set()
    for component_name in ("actor", "rollout", "env"):
        component = getattr(cfg.cluster.component_placement, component_name, None)
        if component is None:
            continue
        component_group_ranks = group_ranks.get(
            getattr(component, "node_group", ""),
            set(),
        )
        if not (component_group_ranks & vlm_node_ranks):
            continue
        placement_value = str(getattr(component, "placement", 0))
        high = (
            int(placement_value.split("-")[-1])
            if "-" in placement_value
            else int(placement_value)
        )
        placements_on_shared_node.add(high)

    if len(placements_on_shared_node) < 2:
        return 0

    return max(placements_on_shared_node) + 1


def get_vlm_planner_placement(cfg) -> tuple[str, int]:
    """Resolve the node group label and GPU index for the VLM planner."""
    vlm_cfg = getattr(cfg, "vlm_planner", None)
    node_group = str(getattr(vlm_cfg, "node_group", VLM_PLANNER_NODE_GROUP))
    gpu_index = compute_vlm_gpu_index(cfg)
    return node_group, gpu_index


def launch_vlm_planner(cfg, cluster: Cluster):
    """Create a placement-backed VLMPlannerWorker Ray actor."""
    subtask_interval = cfg.env.train.get("subtask_interval", 0)
    top_reward_enabled = cfg.env.train.get("top_reward_enabled", False)
    if subtask_interval <= 0 and not top_reward_enabled:
        return None

    if not hasattr(cfg, "vlm_planner"):
        return None

    node_group_label, vlm_gpu = get_vlm_planner_placement(cfg)
    node_group = cluster.get_node_group(node_group_label)
    if node_group is None or not node_group.nodes:
        raise RuntimeError(
            "VLMPlannerWorker requires a node group labelled "
            f"'{node_group_label}' in cluster.node_groups. Check your YAML config."
        )

    placement_strategy = PackedPlacementStrategy(
        start_hardware_rank=vlm_gpu,
        end_hardware_rank=vlm_gpu,
        node_group=node_group_label,
    )
    placements = placement_strategy.get_placement(cluster, isolate_accelerator=True)
    if len(placements) != 1:
        raise RuntimeError(
            "Expected exactly one placement for VLMPlannerWorker, got "
            f"{len(placements)}."
        )

    placement = placements[0]
    node = cluster.get_node_info(placement.cluster_node_rank)
    env_vars = {
        "VISIBLE_DEVICES": ",".join(placement.visible_accelerators),
        "ACCELERATOR_TYPE": str(node.accelerator_type),
        "ACCELERATOR_MODEL": node.accelerator_model,
        "ISOLATE_ACCELERATOR": "1" if placement.isolate_accelerator else "0",
        "LOCAL_ACCELERATOR_RANK": str(placement.local_accelerator_rank),
        "LOCAL_HARDWARE_RANKS": ",".join(map(str, placement.local_hardware_ranks)),
        "NODE_GROUP_LABEL": placement.node_group_label,
        "NODE_RANK": str(placement.placement_node_rank),
        "CLUSTER_NODE_RANK": str(placement.cluster_node_rank),
        "NODE_LOCAL_RANK": str(placement.local_rank),
        "NODE_LOCAL_WORLD_SIZE": str(placement.local_world_size),
        "RAY_ACTOR": str(1),
    }
    env_vars.update(
        AcceleratorUtil.get_accelerator_env_var(
            node.accelerator_type,
            placement.visible_accelerators,
        )
    )

    worker_name = f"{cfg.vlm_planner.get('group_name', 'VLMPlannerWorker')}_0"
    vlm_actor = cluster.allocate(
        cls=VLMPlannerWorker,
        worker_name=worker_name,
        worker_rank=0,
        node_rank=placement.cluster_node_rank,
        max_concurrency=1,
        env_vars=env_vars,
        node_group_label=placement.node_group_label,
        disable_distributed_log=False,
        cls_args=(cfg,),
        cls_kwargs={},
    )

    ray.get(vlm_actor.get_memory_text.remote())
    return vlm_actor
