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

import logging
import os
from typing import Optional

import pandas as pd
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.rollout.server.online_router_worker import OnlineRouterWorker
from rlinf.workers.rollout.server.server_rollout_worker import ServerRolloutWorker
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

logging.getLogger().setLevel(logging.INFO)


class CodingOnlineRLRunner:
    """Runner for online coding model training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        rollout: SGLangWorker,
        inference: Optional[MegatronInference],
        actor: MegatronActor,
        online_router: OnlineRouterWorker,
        server_rollout: ServerRolloutWorker,
    ):
        """"""
        self.cfg = cfg
        self.component_placement = placement
        self.is_pipeline = self.component_placement.is_disaggregated
        self.has_dedicated_inference = inference is not None

        # Workers
        self.rollout = rollout
        self.actor = actor
        self.online_router = online_router
        self.server_rollout = server_rollout
        # Collocated mode uses actor as inference
        self.inference = inference if self.has_dedicated_inference else self.actor

        # Data channels
        self.dataloader_channel = Channel.create("DataLoader")
        # Create a local channel (i.e., a channel that is different in every process)
        # if inference is not a dedicated worker
        self.inference_channel = Channel.create(
            "Inference", local=not self.has_dedicated_inference
        )
        self.actor_channel = Channel.create("Actor", local=True)

        # Configurations
        self.compute_ref_logprobs = self.cfg.algorithm.kl_beta > 0
        self.recompute_logprobs = self.cfg.algorithm.recompute_logprobs
        assert self.recompute_logprobs, "online rl must recompute logprobs"
        self.consumed_samples = 0
        self.global_steps = 0

        # Build dataloader and compute `max_steps`
        self.max_steps = self.cfg.runner.get("max_steps", -self.global_steps)

        # Wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        # Timers
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        # Must be done before actor init
        if self.cfg.runner.resume_dir is None:
            logging.info("Training from scratch")
            if (
                self.cfg.actor.training_backend == "megatron"
                and self.cfg.actor.megatron.use_hf_ckpt
            ):
                from toolkits.ckpt_convertor.convert_hf_to_mg import convert_hf_to_mg

                convert_hf_to_mg(
                    self.cfg.actor.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.actor.megatron.ckpt_convertor,
                )

        # Init workers
        self.rollout.init_worker().wait()
        self.actor.init_worker().wait()
        self.online_router.init_worker(self.rollout).wait()
        self.server_rollout.init_worker().wait()
        if self.has_dedicated_inference:
            self.inference.init_worker().wait()

        if self.cfg.runner.resume_dir is None:
            return

        # Checkpoint loading
        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
        logging.info(f"Setting global step to {self.global_steps}")
        print(f"Setting global step to {self.global_steps}")

        actor_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "actor")
        self.actor.load_checkpoint(actor_checkpoint_path).wait()

    def _compute_flops_metrics(self, time_metrics, act_rollout_metrics) -> dict:
        rollout_time = time_metrics.get("rollout")
        inference_time = time_metrics.get("inference", -1)
        training_time = time_metrics.get("training")

        num_gpus_actor = self.component_placement.actor_world_size
        num_gpus_rollout = self.component_placement.rollout_world_size

        rollout_tflops = act_rollout_metrics["rollout_tflops"]
        inference_tflops = act_rollout_metrics["inference_tflops"]
        training_tflops = act_rollout_metrics["training_tflops"]

        flops_metrics = {
            "rollout_tflops_per_gpu": 0.0,
            "inference_tflops_per_gpu": 0.0,
            "training_tflops_per_gpu": 0.0,
        }
        if rollout_time > 0 and rollout_tflops > 0:
            flops_metrics["rollout_tflops_per_gpu"] = (
                rollout_tflops / rollout_time / num_gpus_rollout
            )

        if inference_time > 0 and inference_tflops > 0:
            num_gpus_inference = self.component_placement.inference_world_size
            if num_gpus_inference == 0:
                num_gpus_inference = self.component_placement.actor_world_size
            flops_metrics["inference_tflops_per_gpu"] = (
                inference_tflops / inference_time / num_gpus_inference
            )

        if training_time > 0 and training_tflops > 0:
            flops_metrics["training_tflops_per_gpu"] = (
                training_tflops / training_time / num_gpus_actor
            )

        return flops_metrics

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name,
            f"checkpoints/global_step_{self.global_steps}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")

        # actor
        self.actor.save_checkpoint(actor_save_path, self.global_steps).wait()

    def _sync_weights(self):
        self.online_router.sync_model_start()
        self.actor.sync_model_to_rollout()
        self.rollout.sync_model_from_actor().wait()
        self.actor.del_reshard_state_dict().wait()

        if self.has_dedicated_inference:
            self.actor.sync_model_to_inference()
            self.inference.sync_model_from_actor().wait()
        self.online_router.sync_model_end()

    def run(self):
        global_pbar = tqdm(
            initial=0,
            total=self.cfg.runner.max_epochs,
            desc="Global Step",
            ncols=620,
        )

        self.online_router.server_start()
        self.server_rollout.server_start()
        self.run_timer.start_time()
        for _ in range(self.cfg.runner.max_epochs):
            with self.timer("step"):
                with self.timer("sync_weights"):
                    self._sync_weights()

                rollout_handle: Handle = self.server_rollout.rollout(
                    output_channel=self.dataloader_channel,
                )

                if self.recompute_logprobs:
                    # Inference prev/ref logprobs
                    infer_handle: Handle = self.inference.run_inference(
                        input_channel=self.dataloader_channel,
                        output_channel=self.inference_channel,
                        compute_ref_logprobs=self.compute_ref_logprobs,
                    )
                    inference_channel = self.inference_channel
                else:
                    infer_handle = None
                    inference_channel = self.dataloader_channel

                # Actor training
                actor_handle: Handle = self.actor.run_training(
                    input_channel=inference_channel,
                )

                metrics = actor_handle.wait()
                actor_rollout_metrics = metrics[0][0]
                actor_training_metrics = metrics[0][1]
                self.global_steps += 1

                run_time_exceeded = self.run_timer.is_finished()
                _, save_model, is_train_end = check_progress(
                    self.global_steps,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=run_time_exceeded,
                )

                if save_model:
                    self._save_checkpoint()

                if is_train_end:
                    logging.info(
                        f"Step limit given by max_steps={self.max_steps} reached. Stopping run"
                    )
                    return

                if run_time_exceeded:
                    logging.info(
                        f"Time limit given by run_timer={self.run_timer} reached. Stopping run"
                    )
                    return

                # To ensure the router server is paused (old requests are finished and new requests are paused).
                # So it's safe to do weight sync on sglang.
                rollout_handle.wait()

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            if infer_handle is not None:
                # Inference time should be the min time across ranks, because different DP receive the rollout results differently
                # But at the begin of the pp schedule, there is a timer barrier
                # This makes all DP end at the same time, while they start at differnt times, and thus only the min time is correct
                time_metrics["inference"] = infer_handle.consume_duration(
                    reduction_type="min"
                )

            logging_steps = (self.global_steps - 1) * self.cfg.algorithm.n_minibatches
            # add prefix to the metrics
            log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
            }

            self.metric_logger.log(log_time_metrics, logging_steps)
            self.metric_logger.log(rollout_metrics, logging_steps)
            for i in range(self.cfg.algorithm.n_minibatches):
                training_metrics = {
                    f"train/{k}": v for k, v in actor_training_metrics[i].items()
                }
                self.metric_logger.log(training_metrics, logging_steps + i)

            logging_metrics = time_metrics

            if self.cfg.actor.get("calculate_flops", False):
                flops_metrics = self._compute_flops_metrics(
                    time_metrics, actor_rollout_metrics
                )
                flops_metrics = {f"flops/{k}": v for k, v in flops_metrics.items()}
                self.metric_logger.log(flops_metrics, logging_steps)
                logging_metrics.update(flops_metrics)

            logging_metrics.update(actor_rollout_metrics)
            logging_metrics.update(actor_training_metrics[-1])

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.server_rollout.shutdown()
        self.online_router.server_stop()
        self.server_rollout.server_stop()
        # No need to wait for rollout_handle since rollout service runs continuously
        self.metric_logger.finish()
