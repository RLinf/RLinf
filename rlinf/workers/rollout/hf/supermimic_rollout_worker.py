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

"""Rollout worker that acts for the SuperMimic G1 env one control step at a time."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.supermimic.interface import (
    check_interface,
    find_interface,
    interface_record,
)
from rlinf.scheduler import Worker
from rlinf.utils.precision import torch_dtype_from_precision
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class SuperMimicRolloutWorker(MultiStepRolloutWorker):
    """Policy and motion critic of one rank, served to the env worker of that rank.

    Training acts through :meth:`serve`; evaluation uses
    :meth:`MultiStepRolloutWorker.evaluate` with the packed ``states``
    observation of the evaluation env.
    """

    def init_worker(self) -> None:
        ckpt_path = self.cfg.runner.get("ckpt_path", None)
        if ckpt_path:
            model_cfg = OmegaConf.merge(self.model_cfg, self.cfg.rollout.model)
            expected = interface_record(model_cfg, self.cfg.algorithm)
            check_interface(find_interface(ckpt_path), expected, ckpt_path)
        super().init_worker()
        amp = self.cfg.rollout.get("amp_autocast", None)
        dtype = None
        if amp is not None and amp.enabled:
            dtype = torch_dtype_from_precision(amp.precision)
        self.hf_model.set_autocast(dtype)

    @Worker.timer("serve")
    def serve(self, num_steps: int) -> None:
        """Answer ``num_steps`` control steps of the env worker of this rank.

        Each step folds the env's merged input statistics into the policy
        norms, returns the action and, when motion episodes ended, their
        motion-critic values.
        """
        env, rank = self.cfg.env.group_name, self._rank
        model = self.hf_model
        for _ in range(num_steps):
            obs, counts = self.recv(env, rank)
            if "obs_stats" in obs:
                model.fold_policy_stats(
                    obs["obs_stats"],
                    counts["obs_count"],
                    obs["goal_stats"],
                    counts["goal_count"],
                )
            with torch.no_grad():
                action = model.act(
                    obs["history"], obs["goal"], obs.get("command"), obs.get("task")
                )
            self.send(action, env, rank, piggyback_payload={})
            features, _ = self.recv(env, rank)
            if features is not None:
                self.send(model.critic_value(features), env, rank, piggyback_payload={})
