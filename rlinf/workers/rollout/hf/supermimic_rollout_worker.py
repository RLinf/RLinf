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

from rlinf.scheduler import Worker
from rlinf.utils.precision import torch_dtype_from_precision
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class SuperMimicRolloutWorker(MultiStepRolloutWorker):
    """Motion policy of one rank, served to the env worker of that rank.

    Training acts through :meth:`serve`; evaluation uses
    :meth:`MultiStepRolloutWorker.evaluate` with the packed ``states``
    observation of the evaluation env.
    """

    def init_worker(self) -> None:
        super().init_worker()
        amp = self.cfg.rollout.get("amp_autocast", None)
        dtype = None
        if amp is not None and amp.enabled:
            dtype = torch_dtype_from_precision(amp.precision)
        self.hf_model.set_autocast(dtype)

    @Worker.timer("serve")
    def serve(self, num_steps: int) -> None:
        """Answer ``num_steps`` control steps of the env worker of this rank.

        Each step acts with the input norms of the previous steps, then folds
        the step's merged input moments into them.
        """
        env, rank = self.cfg.env.group_name, self._rank
        model = self.hf_model
        for _ in range(num_steps):
            obs = self.recv(env, rank)
            with torch.no_grad():
                action = model.act(obs["enc"], obs["goal"])
            self.send(action, env, rank)
            model.fold_policy_stats(obs["enc_stats"], obs["goal_stats"])
