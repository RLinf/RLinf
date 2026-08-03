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
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.reward import EmbodiedAPIRewardWorker, EmbodiedRewardWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)

_REWARD_SERVER_COMPONENT_NAME = "reward_server"


def should_launch_managed_sglang_reward_api(cfg) -> bool:
    reward_cfg = cfg.get("reward", {})
    if not reward_cfg.get("use_reward_model", False):
        return False
    if str(reward_cfg.get("worker_type", "model")).lower() != "api":
        return False

    api_cfg = reward_cfg.get("api", {})
    api_base = str(
        api_cfg.get("api_base") or api_cfg.get("_runtime_api_base") or ""
    ).strip()
    if api_base:
        return False
    if "router_server_args" not in cfg:
        raise ValueError(
            "reward.worker_type='api' requires either reward.api.api_base or the "
            "standard top-level router_server_args block for Ray-managed SGLang."
        )
    return True


def _resolve_reward_api_base_url(server_group, router_group) -> str:
    if router_group is not None:
        return router_group.get_router_url().wait()[0].rstrip("/")
    if server_group is not None:
        server_urls = server_group.get_server_url().wait()
        if server_urls:
            return str(server_urls[0]).rstrip("/")
    raise RuntimeError(
        "Unable to resolve reward.api._runtime_api_base from managed SGLang reward API."
    )


def launch_managed_sglang_reward_api(cfg, cluster, component_placement):
    if not should_launch_managed_sglang_reward_api(cfg):
        return None
    from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server

    server_group = None
    router_group = None
    try:
        server_group, router_group = launch_sglang_router_and_server(
            config=cfg,
            cluster=cluster,
            rollout_hardware_ranks=None,
            router_server_args=cfg.router_server_args,
            placement_strategy=component_placement.get_strategy(
                _REWARD_SERVER_COMPONENT_NAME
            ),
        )
        api_base = _resolve_reward_api_base_url(server_group, router_group)
        with open_dict(cfg.reward):
            if "api" not in cfg.reward:
                cfg.reward.api = {}
        with open_dict(cfg.reward.api):
            cfg.reward.api._runtime_api_base = api_base
        return server_group, router_group
    except Exception:
        stop_managed_sglang_reward_api((server_group, router_group))
        raise


def stop_managed_sglang_reward_api(managed_reward_api) -> None:
    if managed_reward_api is None:
        return
    server_group, router_group = managed_reward_api
    try:
        if router_group is not None:
            router_group.shutdown().wait()
    finally:
        if server_group is not None:
            server_group.shutdown().wait()


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

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
            AsyncEmbodiedSACFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "rlt_ac":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import AsyncRLTACFSDPPolicy

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncRLTACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
            AsyncEmbodiedDAGGERFSDPPolicy,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedDAGGERFSDPPolicy
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

    reward_group = None
    managed_sglang_reward_api = None
    try:
        managed_sglang_reward_api = launch_managed_sglang_reward_api(
            cfg, cluster, component_placement
        )
        if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
            "reward", {}
        ).get("standalone_realworld", False):
            # Create reward worker group
            reward_placement = component_placement.get_strategy("reward")
            reward_worker_cls = (
                EmbodiedAPIRewardWorker
                if str(cfg.reward.get("worker_type", "model")).lower() == "api"
                else EmbodiedRewardWorker
            )
            reward_group = reward_worker_cls.create_group(cfg).launch(
                cluster,
                name=cfg.reward.group_name,
                placement_strategy=reward_placement,
            )

        runner = runner_cls(
            cfg=cfg,
            actor=actor_group,
            rollout=rollout_group,
            env=env_group,
            reward=reward_group,
        )

        runner.init_workers()
        runner.run()
    finally:
        try:
            if reward_group is not None:
                reward_group.stop().wait()
        finally:
            stop_managed_sglang_reward_api(managed_sglang_reward_api)


if __name__ == "__main__":
    main()
