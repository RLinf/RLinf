# Copyright 2026 The RLinf Authors.
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

"""Launch synchronous GRPO training for RPent embodied manipulation."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rlinf.agents.rpent import (
    RPentAgentLoopWorker,
    mixed_simulators,
    mixed_training_dataset,
)
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.runners.rpent_runner import RPentGRPORunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.rollout.utils import get_rollout_backend_worker


def _env_vars(section) -> dict[str, str] | None:
    raw = section.get("env_vars")
    return OmegaConf.to_container(raw, resolve=True) if raw else None


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="train_qwen3.5_4b_rpent_grpo",
)
@output_redirector
def main(cfg) -> None:
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    cfg = validate_cfg(cfg)
    log_dir = Path(cfg.runner.logger.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml", resolve=True)

    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    simulators = mixed_simulators(cfg)
    if simulators:
        train_ds, val_ds = mixed_training_dataset(cfg, tokenizer, simulators)
    else:
        train_ds, val_ds = create_reasoning_datasets(cfg, tokenizer)

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement)
        .launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
            env_vars=_env_vars(cfg.rollout),
        )
    )
    agent_loop = RPentAgentLoopWorker.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=placement.get_strategy("agentloop"),
        env_vars=_env_vars(cfg.agentloop),
    )
    actor = MAMegatronActor.create_group(cfg, placement).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
        env_vars=_env_vars(cfg.actor),
    )

    runner = RPentGRPORunner(
        simulators=simulators,
        cfg=cfg,
        placement=placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout,
        inference=None,
        actor=actor,
        reward=None,
        agent_loop=agent_loop,
        critic=None,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
