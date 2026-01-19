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

"""Runner for training ResNet Reward Model using FSDPRewardWorker."""

import logging
import os
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rlinf.scheduler import Cluster
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.reward.reward_worker import FSDPRewardWorker

logger = logging.getLogger(__name__)


class RewardTrainingRunner:
    """Runner for training reward models with FSDP."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: FSDPRewardWorker,
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor
        self.run_timer = run_timer

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

        # Early stopping state
        early_stop_cfg = cfg.runner.get("early_stop", {})
        self.early_stop_enabled = early_stop_cfg.get("enabled", False)
        self.early_stop_patience = early_stop_cfg.get("patience", 5)
        self.early_stop_min_delta = early_stop_cfg.get("min_delta", 0.001)
        self.early_stop_monitor = early_stop_cfg.get("monitor", "val_loss")
        self.early_stop_counter = 0

        # Compute max_steps from config
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self) -> None:
        """Initialize worker and optionally resume from checkpoint."""
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        if os.path.exists(actor_checkpoint_path):
            self.actor.load_checkpoint(actor_checkpoint_path).wait()
            self.global_step = int(resume_dir.split("global_step_")[-1])
            logger.info(f"Resumed from step {self.global_step}")

    def run(self) -> None:
        """Main training loop."""
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Training Reward Model",
            ncols=120,
        )

        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)

            with self.timer("step"):
                # Run training step
                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

                # Check if we should save
                _, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.get("val_check_interval", 100),
                    self.cfg.runner.get("save_interval", 500),
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

                # Track best validation metrics and early stopping
                improved = False
                if "val_loss" in actor_metrics[0]:
                    val_loss = actor_metrics[0]["val_loss"]
                    if val_loss < self.best_val_loss - self.early_stop_min_delta:
                        self.best_val_loss = val_loss
                        if self.early_stop_monitor == "val_loss":
                            improved = True

                if "val_accuracy" in actor_metrics[0]:
                    val_acc = actor_metrics[0]["val_accuracy"]
                    if val_acc > self.best_val_acc + self.early_stop_min_delta:
                        self.best_val_acc = val_acc
                        self._save_checkpoint(is_best=True)
                        if self.early_stop_monitor == "val_accuracy":
                            improved = True

                # Early stopping check
                if self.early_stop_enabled and (
                    "val_loss" in actor_metrics[0] or "val_accuracy" in actor_metrics[0]
                ):
                    if improved:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        logger.info(
                            f"Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}"
                        )

                    if self.early_stop_counter >= self.early_stop_patience:
                        logger.info(
                            f"Early stopping triggered! No improvement for {self.early_stop_patience} checks."
                        )
                        break

            # Log metrics
            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {f"train/{k}": v for k, v in actor_metrics[0].items()}

            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            # Update progress bar
            display_metrics = {
                "loss": actor_metrics[0].get("loss", 0),
                "acc": actor_metrics[0].get("accuracy", 0),
            }
            if "val_accuracy" in actor_metrics[0]:
                display_metrics["val_acc"] = actor_metrics[0]["val_accuracy"]

            global_pbar.set_postfix(display_metrics, refresh=False)
            global_pbar.update(1)

        logger.info(f"Training complete! Best val_acc: {self.best_val_acc:.4f}")
        self.metric_logger.finish()

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        log_dir = self.cfg.runner.logger.log_path
        if is_best:
            save_dir = os.path.join(
                log_dir,
                "checkpoints/best_model",
            )
        else:
            save_dir = os.path.join(
                log_dir,
                f"checkpoints/global_step_{self.global_step}",
            )

        actor_save_path = os.path.join(save_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

        if is_best:
            logger.info(
                f"Saved best model (val_acc={self.best_val_acc:.4f}) to {save_dir}"
            )
        else:
            logger.info(f"Saved checkpoint at step {self.global_step}")

    def set_max_steps(self) -> None:
        """Set maximum training steps from config.

        If max_steps is -1, use max_epochs as max_steps (1 epoch = 1 step for reward training).
        """
        max_steps = self.cfg.runner.get("max_steps", -1)
        if max_steps <= 0:
            # Use max_epochs as max_steps
            self.max_steps = self.cfg.runner.get("max_epochs", 800)
        else:
            self.max_steps = max_steps


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for reward model training."""
    logger.info("Starting Reward Model Training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create cluster and worker
    cluster = Cluster()

    # Create worker with config
    actor_cfg = OmegaConf.create(
        {
            "actor": cfg.reward,
            "data": cfg.data,
            "runner": cfg.runner,
        }
    )
    actor = cluster.create_worker_group(
        FSDPRewardWorker,
        args=(actor_cfg,),
        num_workers=1,
    )

    # Create runner
    runner = RewardTrainingRunner(cfg, actor)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
