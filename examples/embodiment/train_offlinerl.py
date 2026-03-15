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
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.dataset import DatasetWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


@hydra.main(version_base="1.1", config_path="config", config_name="d4rl_iql_mujoco")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create dataset worker group
    dataset_placement = component_placement.get_strategy("dataset")
    dataset_group = DatasetWorker.create_group(cfg).launch(
        cluster, name=cfg.dataset.group_name, placement_strategy=dataset_placement
    )
    dataset_group.init_worker().wait()
    obs_action_dims = dataset_group.get_obs_action_dims().wait()
    if isinstance(obs_action_dims, list):
        obs_action_dims = next(
            (dims for dims in obs_action_dims if dims is not None), None
        )
    if not (isinstance(obs_action_dims, (tuple, list)) and len(obs_action_dims) == 2):
        raise TypeError(
            "DatasetWorker.get_obs_action_dims() should return (obs_dim, action_dim), "
            f"got {obs_action_dims!r}."
        )
    cfg.actor.model.obs_dim = int(obs_action_dims[0])
    cfg.actor.model.action_dim = int(obs_action_dims[1])

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    if cfg.algorithm.loss_type == "offline_iql":
        from rlinf.workers.actor.fsdp_iql_policy_worker import EmbodiedIQLFSDPPolicy

        actor_worker_cls = EmbodiedIQLFSDPPolicy
    else:
        raise NotImplementedError(
            f"Unsupported offline algorithm.loss_type={cfg.algorithm.loss_type!r}. "
            "Current train_offlinerl entry only supports 'offline_iql'."
        )
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

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
        dataset=dataset_group,
        env=env_group,
        rollout=rollout_group,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
