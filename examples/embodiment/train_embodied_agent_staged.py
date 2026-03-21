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

"""Entry point for embodied RL with a VLM planner (subtask generation and/or TOPReward).

Extends ``train_embodied_agent.py`` with an additional VLMPlannerWorker Ray
actor that is launched when either ``env.train.subtask_interval > 0`` (subtask
planning) or ``env.train.top_reward_enabled`` is True (dense TOPReward reward
signal). The planner runs Qwen3-VL-8B as a Ray actor.

Usage::

    # Start Ray first (see CLAUDE.md: Multi-Node Setup)
    bash examples/embodiment/run_embodiment.sh yam_ppo_openpi_topreward

Or directly::

    python examples/embodiment/train_embodied_agent_staged.py \
        --config-path examples/embodiment/config/ \
        --config-name yam_ppo_openpi_topreward

The config must contain a ``vlm_planner`` section and a node group labelled
``"beaker_vlm"`` in ``cluster.node_groups``. The VLMPlannerWorker is allocated
through RLinf's placement stack so it inherits the same GPU isolation model as
actor and rollout workers, instead of relying on a standalone Ray ``num_gpus``
reservation.

Configs that use this entry point (auto-selected by run_embodiment.sh /
run_realworld.sh / submit_yam_training.sh):
  - ``yam_ppo_openpi``        — TOPReward only (``subtask_interval: 0``)
  - ``*topreward*``           — TOPReward + optional subtask planning
  - ``*staged*``              — subtask planning + TOPReward (legacy pattern)

Optional remote-desktop simulation:
  - Set ``env.remote_desktop_simulation.enabled: true`` when using
    ``env_type: remote`` to have this script launch a local dummy
    ``RobotServer`` automatically.
  - This simulates the robot desktop input path end to end, so ``RemoteEnv``
    still talks gRPC, but no real desktop machine or SSH tunnel is required.
"""

import json

import hydra
import ray
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.envs.remote.simulated_desktop import (
    launch_simulated_desktop_server,
    stop_process,
)
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import AcceleratorUtil, Cluster, PackedPlacementStrategy
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.vlm_planner import VLMPlannerWorker

mp.set_start_method("spawn", force=True)

_VLM_PLANNER_NODE_GROUP = "beaker_vlm"


def _compute_vlm_gpu_index(cfg) -> int:
    """Return the GPU index to use for VLMPlannerWorker.

    Configs may set ``vlm_planner.placement`` to override the default. When no
    override is provided, the default remains the original heuristic:

    - if actor/rollout use fewer than two distinct GPU indices on the VLM node,
      place the VLM on GPU 0
    - otherwise place it on ``max(actor, rollout, env placements on that node)+1``

    The important difference is that the returned index is now fed into RLinf's
    placement system, which reserves and isolates the selected GPU in the same
    way as actor and rollout workers.
    """
    # Explicit placement override takes precedence over the heuristic.
    vlm_cfg = getattr(cfg, "vlm_planner", None)
    if vlm_cfg is not None:
        explicit = getattr(vlm_cfg, "placement", None)
        if explicit is not None:
            return int(explicit)

    # Find node_ranks of the beaker_vlm group.
    vlm_node_ranks: set[int] = set()
    for g in cfg.cluster.node_groups:
        if g.label == _VLM_PLANNER_NODE_GROUP:
            nr = g.node_ranks
            if isinstance(nr, int):
                vlm_node_ranks.add(nr)
            else:
                for r in str(nr).split(","):
                    vlm_node_ranks.add(int(r.strip()))
            break

    # Build label → node_ranks map for all groups.
    group_ranks: dict[str, set[int]] = {}
    for g in cfg.cluster.node_groups:
        nr = g.node_ranks
        if isinstance(nr, int):
            ranks: set[int] = {nr}
        else:
            ranks = {int(r.strip()) for r in str(nr).split(",")}
        group_ranks[g.label] = ranks

    # Collect distinct GPU placement indices used by other components on the
    # same node as beaker_vlm.
    placements_on_shared_node: set[int] = set()
    for comp_name in ("actor", "rollout", "env"):
        comp = getattr(cfg.cluster.component_placement, comp_name, None)
        if comp is None:
            continue
        comp_group_ranks = group_ranks.get(getattr(comp, "node_group", ""), set())
        if not (comp_group_ranks & vlm_node_ranks):
            continue  # Component is on a different physical node — no conflict.
        placement_val = str(getattr(comp, "placement", 0))
        # Handle range syntax "0-2" — use the high watermark.
        high = (
            int(placement_val.split("-")[-1])
            if "-" in placement_val
            else int(placement_val)
        )
        placements_on_shared_node.add(high)

    # Only one distinct GPU index in use (or dedicated node): VLM shares GPU 0.
    # Two or more distinct indices: every index is occupied — VLM needs max+1.
    if len(placements_on_shared_node) < 2:
        return 0

    return max(placements_on_shared_node) + 1


def _get_vlm_planner_placement(cfg) -> tuple[str, int]:
    """Resolve the node group label and GPU index for the VLM planner."""
    vlm_cfg = getattr(cfg, "vlm_planner", None)
    node_group = str(getattr(vlm_cfg, "node_group", _VLM_PLANNER_NODE_GROUP))
    gpu_index = _compute_vlm_gpu_index(cfg)
    return node_group, gpu_index


def _launch_vlm_planner(cfg, cluster: Cluster):
    """Create a placement-backed VLMPlannerWorker Ray actor.

    Args:
        cfg: Top-level Hydra config.
        cluster: Initialised Cluster object.

    Returns:
        Ray actor handle for VLMPlannerWorker, or None if the ``vlm_planner``
        config section is absent and neither ``env.train.subtask_interval > 0``
        nor ``env.train.top_reward_enabled`` is set.
    """
    subtask_interval = cfg.env.train.get("subtask_interval", 0)
    top_reward_enabled = cfg.env.train.get("top_reward_enabled", False)
    if subtask_interval <= 0 and not top_reward_enabled:
        return None

    if not hasattr(cfg, "vlm_planner"):
        return None

    node_group_label, vlm_gpu = _get_vlm_planner_placement(cfg)
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
            node.accelerator_type, placement.visible_accelerators
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

    # Verify the actor started successfully (will raise if __init__ fails).
    ray.get(vlm_actor.get_memory_text.remote())
    return vlm_actor


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="yam_ppo_openpi_topreward",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    simulated_desktop_server = launch_simulated_desktop_server(cfg)
    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        component_placement = HybridComponentPlacement(cfg, cluster)

        # Create actor worker group (FSDP training on Beaker).
        actor_placement = component_placement.get_strategy("actor")
        actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )

        # Create rollout worker group (inference).
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )

        # Create env worker group (direct YAMEnv or RemoteEnv per config).
        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        runner = EmbodiedRunner(
            cfg=cfg,
            actor=actor_group,
            rollout=rollout_group,
            env=env_group,
        )

        runner.init_workers()

        # Wire the VLM planner into env workers after they have initialised.
        vlm_actor = _launch_vlm_planner(cfg, cluster)
        if vlm_actor is not None:
            env_group.set_vlm_planner(vlm_actor).wait()

        runner.run()
    finally:
        stop_process(simulated_desktop_server)


if __name__ == "__main__":
    main()
