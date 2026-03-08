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

"""Entry point for stage-aware embodied RL with a VLM subtask planner.

Extends ``train_embodied_agent.py`` with an additional VLMPlannerWorker Ray
actor that runs on the Beaker node.  The planner generates dynamic subtask
instructions and binary rewards by calling a local Qwen3.5 VLM.

Usage::

    # Start Ray on all 3 nodes first (see CLAUDE.md: Multi-Node Setup)
    bash examples/embodiment/run_realworld.sh yam_grpo_openpi_staged

Or directly::

    python examples/embodiment/train_embodied_agent_staged.py \
        --config-path examples/embodiment/config/ \
        --config-name yam_grpo_openpi_staged

The config must contain a ``vlm_planner`` section (see
``examples/embodiment/config/yam_grpo_openpi_staged.yaml``) and a node group
labelled ``"beaker"`` where the VLMPlannerWorker will be placed.
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


def _launch_vlm_planner(cfg, cluster: Cluster):
    """Create a VLMPlannerWorker Ray actor pinned to the first Beaker node.

    Args:
        cfg: Top-level Hydra config.
        cluster: Initialised Cluster object.

    Returns:
        Ray actor handle for VLMPlannerWorker, or None if the ``vlm_planner``
        config section is absent or ``env.train.subtask_interval`` is 0.
    """
    subtask_interval = cfg.env.train.get("subtask_interval", 0)
    if subtask_interval <= 0:
        return None

    if not hasattr(cfg, "vlm_planner"):
        return None

    node_group = cluster.get_node_group(_VLM_PLANNER_NODE_GROUP)
    if node_group is None or not node_group.nodes:
        raise RuntimeError(
            f"VLMPlannerWorker requires a node group labelled '{_VLM_PLANNER_NODE_GROUP}' "
            f"in cluster.node_groups.  Check your YAML config."
        )

    # Pin to the first Beaker node.
    beaker_node = node_group.nodes[0]
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=beaker_node.ray_id,
        soft=False,
    )

    vlm_actor = VLMPlannerWorker.options(
        num_gpus=1,
        scheduling_strategy=scheduling_strategy,
    ).remote(cfg)

    # Verify the actor started successfully (will raise if __init__ fails).
    ray.get(vlm_actor.get_memory_text.remote())
    return vlm_actor


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="yam_grpo_openpi_staged",
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

    # Create rollout worker group (π₀.5 inference on desktop GPU).
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group (YAM robot on robot controller node).
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
