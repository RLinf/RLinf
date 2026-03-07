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
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_sac_mlp_async"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # If the env uses a remote classifier, start the server on a GPU node.
    clf_cfg = cfg.env.train.get("classifier_reward_wrapper", None)
    server_handle = None
    if clf_cfg is not None and clf_cfg.get("remote", False):
        import ray
        from rlinf.workers.reward.classifier_reward_server import (
            ClassifierRewardServer,
        )

        server_name = clf_cfg.get("server_name", "ClassifierRewardServer")
        server_handle = ClassifierRewardServer.options(
            name=server_name,
            num_gpus=0.05,
        ).remote(
            checkpoint_path=clf_cfg.checkpoint_path,
            image_keys=clf_cfg.get("image_keys", None),
            device=clf_cfg.get("device", "cuda"),
        )
        ray.get(server_handle.ready.remote())

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
            AsyncEmbodiedSACFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "decoupled_actor_critic":
        from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
        from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor

        runner_cls = AsyncPPOEmbodiedRunner
        actor_worker_cls = AsyncPPOEmbodiedFSDPActor
    else:
        raise ValueError(
            f"Unsupported loss type {cfg.algorithm.loss_type} for async embodied runner"
        )

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = AsyncEnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
