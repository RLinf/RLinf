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

"""Evaluation of G1 motion-imitation policies and export of result tables."""

from __future__ import annotations

from omegaconf import DictConfig

from rlinf.envs.sim.g1_mimic.eval_metrics import finalize_eval_results
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner


def evaluate_g1_mimic(
    cfg: DictConfig,
    env,
    rollout,
    env_channel,
    rollout_channel,
    step: int | None = None,
    logger=None,
) -> dict[str, float]:
    """Run G1 evaluation across env and rollout workers and summarize results."""
    meta = env.eval_metadata().wait()[0]
    env_handle = env.evaluate(
        input_channel=env_channel, rollout_channel=rollout_channel
    )
    rollout_handle = rollout.evaluate(
        input_channel=rollout_channel, output_channel=env_channel
    )
    env_results = env_handle.wait()
    rollout_handle.wait()
    metrics, out_dir = finalize_eval_results(cfg, env_results, meta, step=step)
    if out_dir is not None and logger is not None:
        logger.info(f"Wrote SuperMimic evaluation results to {out_dir}")
    return metrics


class SuperMimicEvalRunner(EmbodiedEvalRunner):
    """Run G1 evaluation tasks and summarize the per-job rows across env ranks."""

    def evaluate(self) -> dict[str, float]:
        return evaluate_g1_mimic(
            self.cfg,
            self.env,
            self.rollout,
            self.env_channel,
            self.rollout_channel,
            logger=self.logger,
        )
