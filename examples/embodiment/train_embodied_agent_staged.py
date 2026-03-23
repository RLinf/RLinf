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
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from staged_utils import launch_vlm_planner
from rlinf.config import validate_cfg
from rlinf.envs.remote.simulated_desktop import (
    launch_simulated_desktop_server,
    stop_process,
)
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


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
        vlm_actor = launch_vlm_planner(cfg, cluster)
        if vlm_actor is not None:
            env_group.set_vlm_planner(vlm_actor).wait()

        runner.run()
    finally:
        stop_process(simulated_desktop_server)


if __name__ == "__main__":
    main()
