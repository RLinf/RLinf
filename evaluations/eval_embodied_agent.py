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
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="libero",
    config_name="libero_spatial_starvla_eval",
)
def main(cfg) -> None:
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker group. Select the worker by ``rollout_backend``:
    # ``apxinf`` is an eval-only in-process backend. vLLM is intentionally not
    # wired into embodied evaluation.
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_backend = cfg.rollout.get("rollout_backend", "huggingface")
    phyai_serving_mode = cfg.rollout.get("phyai", {}).get("serving_mode", "engine")
    # Default env worker; RTC on the huggingface path overrides it below.
    env_worker_cls = EnvWorker
    if rollout_backend == "apxinf":
        from rlinf.workers.rollout.apxinf import ApxInfRolloutWorker

        rollout_group = ApxInfRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement,
        )
    elif rollout_backend == "sglang":
        from rlinf.workers.rollout.utils import get_rollout_backend_worker

        rollout_group = (
            get_rollout_backend_worker(cfg)
            .create_group(cfg, component_placement)
            .launch(
                cluster,
                name=cfg.rollout.group_name,
                placement_strategy=rollout_placement,
            )
        )
    elif rollout_backend == "huggingface":
        if cfg.runner.get("rtc", {}).get("enabled", False):
            from rlinf.workers.env.rtc_env_worker import RTCEnvWorker
            from rlinf.workers.rollout.hf.rtc_huggingface_worker import (
                RTCMultiStepRolloutWorker,
            )

            env_worker_cls = RTCEnvWorker
            rollout_worker_cls = RTCMultiStepRolloutWorker
        else:
            rollout_worker_cls = MultiStepRolloutWorker

        # Create rollout worker group
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = rollout_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )
    elif rollout_backend == "phyai":
        if phyai_serving_mode == "engine":
            from rlinf.workers.rollout.phyai import PhyAIRolloutWorker

            rollout_group = PhyAIRolloutWorker.create_group(cfg)
        elif phyai_serving_mode == "gateway":
            from rlinf.workers.rollout.phyai import PhyAIGatewayRolloutWorker

            rollout_group = PhyAIGatewayRolloutWorker.create_group(cfg)
        else:
            raise ValueError(
                "Expected phyai.serving_mode in ['engine', 'gateway'], "
                f"got {phyai_serving_mode!r}."
            )
        rollout_group = rollout_group.launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement,
        )
    else:
        raise ValueError(f"Unsupported rollout backend: {rollout_backend}")
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = env_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # launch the sglang server
    if rollout_backend == "sglang":
        from rlinf.workers.rollout.sglang import (
            launch_sglang_router_and_server,
        )

        server_group, _ = launch_sglang_router_and_server(
            cfg,
            cluster,
            rollout_hardware_ranks=component_placement.get_hardware_ranks("rollout"),
            router_server_args=cfg.rollout.sglang,
        )
        _server_urls = list(server_group.get_server_url().wait())
        get_logger().info(
            f"[eval] launched {len(_server_urls)} sglang server(s): {_server_urls}"
        )
        rollout_group.set_sglang_server_urls(_server_urls).wait()

    service_group = None
    try:
        if rollout_backend == "phyai" and phyai_serving_mode == "gateway":
            gateway_url = cfg.rollout.phyai.get("gateway_url")
            if isinstance(gateway_url, str) and gateway_url.strip():
                gateway_urls = [gateway_url.strip().rstrip("/")]
                get_logger().info(
                    f"[eval] using external PhyAI Gateway: {gateway_urls[0]}"
                )
            else:
                from rlinf.workers.rollout.phyai.launcher import (
                    launch_phyai_gateway_and_server,
                )

                if "phyai_server" in component_placement.components:
                    server_placement = component_placement.get_strategy("phyai_server")
                else:
                    # Synchronous evaluation needs a single Gateway/model-server
                    # pair; rollout workers round-robin their HTTP requests onto
                    # it. Launch one single-GPU service worker on the first
                    # accelerator the rollout placement already uses, so the
                    # service follows the configured placement instead of
                    # hard-coding GPU 0.
                    from rlinf.scheduler.placement import PackedPlacementStrategy

                    rollout_ranks = sorted(
                        int(rank)
                        for rank in component_placement.get_hardware_ranks("rollout")
                    )
                    server_rank = rollout_ranks[0] if rollout_ranks else 0
                    server_placement = PackedPlacementStrategy(
                        start_hardware_rank=server_rank,
                        end_hardware_rank=server_rank,
                    )
                gateway_urls, service_group = launch_phyai_gateway_and_server(
                    cfg,
                    cluster,
                    placement_strategy=server_placement,
                    phyai_cfg=cfg.rollout.phyai,
                )
                get_logger().info(f"[eval] launched PhyAI Gateways: {gateway_urls}")

            rollout_group.set_phyai_server_urls(gateway_urls).wait()
        runner = EmbodiedEvalRunner(
            cfg=cfg,
            rollout=rollout_group,
            env=env_group,
        )
        runner.init_workers()
        runner.run()
    finally:
        if service_group is not None:
            service_group.stop().wait()


if __name__ == "__main__":
    main()
