# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the four-GPU PhyAI PPO end-to-end contract test."""

import math
import os
import time
from pathlib import Path
from typing import Any

import ray
import torch
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.phyai import PhyAIWorker


class ContractCheckingActor(EmbodiedFSDPActor):
    """Assert that real PPO data and optimizer updates remain finite."""

    def run_training(self) -> dict[str, float]:
        """Run the production training method with E2E contract assertions."""
        for key in (
            "rewards",
            "prev_logprobs",
            "prev_values",
            "advantages",
            "returns",
        ):
            value = self.rollout_batch.get(key)
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all().item(), (
                    f"PhyAI E2E received non-finite {key}."
                )

        optimizer_steps_before = self.optimizer_steps
        completed_optimizer_steps: list[int] = []

        def record_optimizer_step(
            optimizer: torch.optim.Optimizer,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            del optimizer, args, kwargs
            completed_optimizer_steps.append(self.optimizer_steps)

        hook = self.optimizer.register_step_post_hook(record_optimizer_step)
        try:
            metrics = super().run_training()
        finally:
            hook.remove()

        assert completed_optimizer_steps, "PhyAI E2E did not execute an optimizer step."
        assert self.optimizer_steps > optimizer_steps_before, (
            "PhyAI E2E optimizer step counter did not advance."
        )
        for key, value in metrics.items():
            assert math.isfinite(float(value)), (
                f"PhyAI E2E produced non-finite metric {key}={value}."
            )
        self.log_info(
            "PHYAI_E2E_TRAIN_PASS "
            f"optimizer_steps={optimizer_steps_before}->{self.optimizer_steps}"
        )
        return metrics


class ContractCheckingRunner(EmbodiedRunner):
    """Verify a rollout after the final trained weight version is applied."""

    def _finish_run(self) -> None:
        assert self.global_step == self.max_steps, (
            "PhyAI E2E stopped before completing all PPO steps: "
            f"{self.global_step}/{self.max_steps}."
        )
        self.actor.set_global_step(self.global_step).wait()
        self.rollout.set_global_step(self.global_step).wait()
        self.env.set_global_step(self.global_step).wait()
        self.update_rollout_weights()

        env_handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
        )
        rollout_handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            actor_channel=self.actor_channel,
        )
        self.actor.recv_rollout_trajectories(input_channel=self.actor_channel).wait()
        rollout_handle.wait()
        env_handle.wait()
        self.actor.compute_advantages_and_returns().wait()
        self.logger.info("PHYAI_E2E_UPDATED_ROLLOUT_PASS version=%s", self.global_step)
        super()._finish_run()


def _load_config() -> Any:
    repo_path = Path(os.environ["REPO_PATH"]).resolve()
    embodied_path = repo_path / "examples" / "embodiment"
    os.environ.setdefault("EMBODIED_PATH", str(embodied_path))
    config_dir = embodied_path / "config"
    overlay_path = Path(__file__).with_name(
        "libero_spatial_ppo_openpi_pi05_phyai_ci.yaml"
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")
    OmegaConf.set_struct(cfg, False)
    return validate_cfg(OmegaConf.merge(cfg, OmegaConf.load(overlay_path)))


def main() -> None:
    """Launch real RLinf workers and run three complete PPO steps."""
    cfg = _load_config()
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    placement = HybridComponentPlacement(cfg, cluster)
    actor = ContractCheckingActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    rollout = PhyAIWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = ContractCheckingRunner(
        cfg=cfg,
        actor=actor,
        rollout=rollout,
        env=env,
    )
    started = time.monotonic()
    try:
        runner.init_workers()
        runner.run()
        print(
            "PHYAI_E2E_COMPLETE_PASS "
            f"ppo_steps={runner.global_step} "
            f"elapsed={time.monotonic() - started:.3f}s",
            flush=True,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
