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

import time
import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # gRPC evaluation exchanges small CPU observations/actions. Route those
        # channels through Ray so the first rollout does not lazily initialize
        # torch.distributed process groups in every worker.
        channel_transport = (
            "ray" if cfg.rollout.get("rollout_backend") == "grpc" else "collective"
        )
        self.env_channel = Channel.create("Env", transport=channel_transport)
        self.rollout_channel = Channel.create("Rollout", transport=channel_transport)

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        if self.cfg.rollout.get("rollout_backend") == "grpc":
            # Real environments can move during construction/reset. A failed
            # remote handshake must be observed before initializing hardware.
            rollout_handle.wait()
            self.env.init_worker().wait()
            return
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        # Channel direction convention (names follow the receiver, not the sender):
        #   rollout_channel: env -> rollout  (env sends obs/RTC requests, rollout receives)
        #   env_channel:     rollout -> env  (rollout sends actions/RTC responses, env receives)
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )

        try:
            env_results = env_handle.wait()
        except BaseException:
            # The environment can terminate first when an operator aborts or
            # a hardware fault is raised. Wake a rollout worker that may still
            # be waiting for its next observation before stack cleanup tries
            # to call shutdown on the actor.
            try:
                self.rollout.request_evaluation_stop().wait()
            except BaseException as stop_error:  # noqa: BLE001 - preserve root error
                self.logger.warning(
                    "Could not request rollout evaluation stop: %s", stop_error
                )
            try:
                rollout_handle.wait()
            except BaseException:
                pass
            raise
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            try:
                rollout_results = rollout_handle.wait()
            except BaseException:
                try:
                    self.env.request_evaluation_stop().wait()
                except BaseException as stop_error:  # noqa: BLE001 - preserve root error
                    self.logger.warning(
                        "Could not request environment evaluation stop: %s", stop_error
                    )
                raise
            rollout_metrics_list = [
                results for results in rollout_results if results is not None
            ]
            rollout_metrics = compute_evaluate_metrics(rollout_metrics_list)
            rollout_metrics.pop("num_trajectories", None)
        else:
            rollout_metrics = {}

        env_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(env_metrics_list)
        eval_metrics.update(rollout_metrics)
        return eval_metrics

    def run(self):
        start_time = time.time()
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)
        print_metrics_table(
            step=0,
            total_steps=1,
            start_time=start_time,
            metrics=eval_metrics,
            log_path=self.metric_logger.log_path,
        )

        self.metric_logger.finish()
