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

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Worker
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: DictConfig,
        rollout: Worker,
        env: Worker,
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_loger = MetricLogger(cfg)

        self.logger = get_logger()

    def _load_eval_policy(self):
        self.rollout.load_checkpoint(self.cfg.runner.eval_policy_path).wait()

    def init_workers(self):
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        if self.cfg.runner.eval_policy_path is not None:
            self.logger.info(
                f"Using checkpoint for evaluation (from runner.eval_policy_path): {self.cfg.runner.eval_policy_path}"
            )
            self._load_eval_policy()
        else:
            self.logger.info(
                f"Using checkpoint for evaluation (from rollout.model.model_path): {self.cfg.rollout.model.model_path}"
            )

    def evaluate(self):
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_results = env_futures.wait()
        rollout_futures.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_loger.finish()
