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

"""Entry point for embodied RL with a VLM planner (subtask generation and/or TOPReward).

Extends ``train_embodied_agent.py`` with an additional VLMPlannerWorker Ray
actor that is launched when either ``env.train.subtask_interval > 0`` (subtask
planning) or ``env.train.top_reward_enabled`` is True (dense TOPReward reward
signal).  The planner runs Qwen3-VL-8B as a Ray actor.

Usage::

    # Start Ray first (see CLAUDE.md: Multi-Node Setup)
    bash examples/embodiment/run_embodiment.sh yam_ppo_openpi_topreward

Or directly::

    python examples/embodiment/train_embodied_agent_staged.py \
        --config-path examples/embodiment/config/ \
        --config-name yam_ppo_openpi_topreward

The config must contain a ``vlm_planner`` section and a node group labelled
``"beaker_vlm"`` in ``cluster.node_groups``.  The VLMPlannerWorker is pinned
to the first node in that group via Ray NodeAffinitySchedulingStrategy.
``_compute_vlm_gpu_index`` assigns the VLM to GPU ``max(distinct_actor_rollout_placements) + 1``
to avoid collisions with actor and rollout workers.

Configs that use this entry point (auto-selected by run_embodiment.sh /
run_realworld.sh / submit_yam_training.sh):
  - ``yam_ppo_openpi``        — TOPReward only (``subtask_interval: 0``)
  - ``*topreward*``           — TOPReward + optional subtask planning
  - ``*staged*``              — subtask planning + TOPReward (legacy pattern)
"""

import json

import hydra
import ray
import ray.util.scheduling_strategies
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.vlm_planner import VLMPlannerWorker

mp.set_start_method("spawn", force=True)

_VLM_PLANNER_NODE_GROUP = "beaker_vlm"


def _compute_vlm_gpu_index(cfg) -> int | None:
    """Return the GPU index to pin VLMPlannerWorker to, or ``None`` to let Ray decide.

    Actor and rollout workers bypass Ray's GPU resource pool (they set
    ``CUDA_VISIBLE_DEVICES`` manually and set
    ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1``).  From Ray's
    perspective those GPUs are unclaimed, so a bare ``num_gpus=1`` request
    would assign GPU 0 to VLMPlannerWorker.

    This is the correct behaviour when VLM is meant to share a GPU with the
    actor (e.g. a single-GPU dedicated VLM node where it has the whole device
    to itself).

    However, on single-node Beaker configs with **multiple** GPUs assigned to
    different workers (actor on GPU 0, rollout on GPU 1) the VLM must land on
    yet another GPU (GPU 2) rather than colliding with the actor on GPU 0.

    The heuristic: if two or more *distinct* GPU indices are claimed by other
    components on the same node, those GPUs are all occupied and the VLM needs
    ``max(claimed indices) + 1``.  With only one distinct index (all on GPU 0,
    or a dedicated node), return ``None`` so Ray assigns GPU 0 — which is
    intentional sharing or an isolated node.

    Configs may set ``vlm_planner.placement`` to override this heuristic with
    an explicit GPU index.
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
        high = int(placement_val.split("-")[-1]) if "-" in placement_val else int(placement_val)
        placements_on_shared_node.add(high)

    # Only one distinct GPU index in use (or dedicated node): VLM shares GPU 0.
    # Two or more distinct indices: every index is occupied — VLM needs max+1.
    if len(placements_on_shared_node) < 2:
        return None  # Let Ray assign; GPU 0 is correct.

    return max(placements_on_shared_node) + 1


def _launch_vlm_planner(cfg, cluster: Cluster):
    """Create a VLMPlannerWorker Ray actor pinned to the first Beaker node.

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

    node_group = cluster.get_node_group(_VLM_PLANNER_NODE_GROUP)
    if node_group is None or not node_group.nodes:
        raise RuntimeError(
            f"VLMPlannerWorker requires a node group labelled '{_VLM_PLANNER_NODE_GROUP}' "
            f"in cluster.node_groups.  Check your YAML config."
        )

    # Pin to the first node in the beaker_vlm group.
    beaker_node = node_group.nodes[0]
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=beaker_node.ray_id,
        soft=False,
    )

    # Determine the correct GPU index for VLMPlannerWorker.
    # When actor and rollout occupy distinct GPUs on the same node, Ray would
    # still assign GPU 0 (it doesn't see those GPUs as claimed) and collide
    # with the actor.  _compute_vlm_gpu_index returns the next free GPU in
    # that case, or None when sharing GPU 0 is intentional (single-GPU node
    # or dedicated VLM node).
    vlm_gpu = _compute_vlm_gpu_index(cfg)
    options_kwargs: dict = {
        "num_gpus": 1,
        "scheduling_strategy": scheduling_strategy,
    }
    if vlm_gpu is not None:
        options_kwargs["runtime_env"] = {
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": str(vlm_gpu),
                # Match the actor/rollout pattern: prevent Ray from overwriting.
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        }

    vlm_actor = VLMPlannerWorker.options(**options_kwargs).remote(cfg)

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


if __name__ == "__main__":
    main()
