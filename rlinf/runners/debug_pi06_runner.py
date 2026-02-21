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

"""
Debug Pi06 Runner: Loads data directly from LeRobot datasets for CFG training.

Differences from Pi06Runner:
- Does not perform rollout to collect data; loads data directly from LeRobot datasets
- Uses offline_training_step as the main loop
- Logs detailed metrics for each offline step
- Retains periodic eval functionality (requires rollout and env workers)
"""

import os

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.debug_fsdp_actor_worker_cfg import DebugCFGFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class DebugPi06Runner:
    """
    Debug version of pi06 training pipeline (trains from LeRobot datasets):
    1) Loads LeRobot datasets from the data.datasets configuration
    2) Uses offline_training_step as the main loop, logging detailed metrics per step
    3) Periodically evaluates model performance in the environment
    """

    def __init__(
        self,
        cfg: DictConfig,
        actor: DebugCFGFSDPActor,
        rollout: MultiStepRolloutWorker = None,
        env: EnvWorker = None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env

        # Data channels for evaluation
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        self.consumed_samples = 0
        self.global_step = 0

        # compute `max_steps` - use offline_training_step as max_steps
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        """Initialize all workers and build CFG DataLoader."""
        # Initialize actor
        actor_future = self.actor.init_worker()

        # Initialize rollout and env workers (for evaluation)
        rollout_future = None
        env_future = None
        if self.rollout is not None:
            rollout_future = self.rollout.init_worker()
        if self.env is not None:
            env_future = self.env.init_worker()

        # Wait for all workers
        actor_future.wait()
        if rollout_future is not None:
            rollout_future.wait()
        if env_future is not None:
            env_future.wait()

        # Build CFG DataLoader from LeRobot datasets
        self.actor.build_cfg_dataloader(self.cfg).wait()

        # Resume checkpoint if specified
        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        if os.path.exists(actor_checkpoint_path):
            self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        """Sync actor weights to rollout worker for evaluation."""
        if self.rollout is None:
            return
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        """Run evaluation in environment."""
        if self.rollout is None or self.env is None:
            return {}

        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        """Main training loop."""
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Offline Training Step",
            ncols=800,
        )

        # set global step
        self.actor.set_global_step(self.global_step)
        if self.rollout is not None:
            self.rollout.set_global_step(self.global_step)

        # Initial eval (step 0)
        eval_metrics = {}
        if (
            self.cfg.runner.val_check_interval > 0
            and self.rollout is not None
            and self.env is not None
        ):
            with self.timer("eval"):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=0)

        # Main loop: each offline step corresponds to one _step
        for _step in range(start_step, self.max_steps):
            with self.timer("step"):
                # Periodic evaluation
                eval_metrics = {}
                if (
                    self.cfg.runner.val_check_interval > 0
                    and self.global_step > 0
                    and self.global_step % self.cfg.runner.val_check_interval == 0
                    and self.rollout is not None
                    and self.env is not None
                ):
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                # Single training step
                with self.timer("actor_training"):
                    actor_training_metrics = self.actor.run_training().wait()

                self.global_step += 1
                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            actor_metrics = {
                f"actor/{k}": v for k, v in actor_training_metrics[0].items()
            }
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(actor_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(actor_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        # Use offline_training_step as max_steps
        self.max_steps = self.cfg.runner.get("offline_training_step", 10000)

    @property
    def epoch(self):
        return self.global_step
