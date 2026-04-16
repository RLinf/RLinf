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

import json

import hydra
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


@hydra.main(version_base="1.1", config_path="config", config_name="d4rl_iql_mujoco")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    if (
        cfg.algorithm.loss_type == "offline_iql"
        and cfg.actor.model.model_type == "mlp_policy"
    ):
        dataset_type = str(cfg.data.get("dataset_type", "")).lower()
        if dataset_type == "d4rl":
            from rlinf.data.datasets.d4rl import D4RLDataset

            task_name = str(cfg.data.task_name)
            obs_dim, action_dim = D4RLDataset.infer_obs_action_dims_from_env(task_name)
            with open_dict(cfg):
                cfg.actor.model.obs_dim = int(obs_dim)
                cfg.actor.model.action_dim = int(action_dim)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    if cfg.algorithm.loss_type == "offline_iql":
        from rlinf.workers.actor.fsdp_iql_policy_worker import EmbodiedIQLFSDPPolicy

        actor_worker_cls = EmbodiedIQLFSDPPolicy
    else:
        raise NotImplementedError(
            f"Unsupported offline algorithm.loss_type={cfg.algorithm.loss_type!r}. "
            "Current train_offline_rl entry only supports 'offline_iql'."
        )
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval
    env_group = None
    rollout_group = None
    if enable_eval:
        # Create env worker group
        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        # Create rollout worker group
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )

    runner = OfflineRunner(
        cfg=cfg,
        actor=actor_group,
        env=env_group,
        rollout=rollout_group,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
