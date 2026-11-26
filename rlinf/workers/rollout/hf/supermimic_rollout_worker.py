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

from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM
from rlinf.models.embodiment.supermimic.ipc import (
    FlatDictBuffer,
    FlatTensorPacker,
    obs_packet_schema,
    use_direct_weight_sync,
)
from rlinf.models.embodiment.supermimic.supermimic_policy import sync_param_names
from rlinf.scheduler import Worker
from rlinf.utils.compile_cache import CompileCache, run_compiled
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
        self._compile_cache = CompileCache(
            enabled=bool(self.hf_model.compile_cache.enabled), dynamic=False
        )
        self._use_direct_weight_sync = not self.only_eval and use_direct_weight_sync(
            self.cfg, self.placement.get_world_size("actor"), self._world_size
        )
        self._sync_packer = (
            FlatTensorPacker(self.hf_model, sync_param_names(self.hf_model))
            if self._use_direct_weight_sync
            else None
        )
        if self.enable_train:
            model_device = next(self.hf_model.parameters()).device
            self._obs_buf = FlatDictBuffer(
                obs_packet_schema(
                    self.per_node_train_batch_size,
                    self.hf_model.representation.encoding_dim,
                ),
                model_device,
            )
            self._action_buf = torch.empty(
                (self.per_node_train_batch_size, ACTION_DIM),
                dtype=torch.float32,
                device=model_device,
            )

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self):
        if not self._use_direct_weight_sync:
            return await super().sync_model_from_actor()
        self.recv_tensor(
            self._sync_packer.buffer,
            src_group_name=self.actor_group_name,
            src_rank=self._rank,
        )
        self._sync_packer.unpack()
        self.version += 1

    @Worker.timer("serve")
    def serve(self, num_steps: int) -> None:
        """Answer ``num_steps`` control steps of the env worker of this rank.

        Each step acts with the input norms of the previous steps, then folds
        the step's merged input moments into them.
        """
        env, rank = self.cfg.env.group_name, self._rank
        model = self.hf_model
        views = self._obs_buf.views
        for _ in range(num_steps):
            self.recv_tensor(self._obs_buf.buffer, env, rank)
            with torch.no_grad():
                action = run_compiled(
                    "rollout-act",
                    model.act,
                    views["enc"],
                    views["goal"],
                    cache=self._compile_cache,
                )
            self._action_buf.copy_(action)
            if self.torch_platform is not None and self.torch_platform.is_available():
                self.torch_platform.current_stream().synchronize()
            self.send_tensor(self._action_buf, env, rank)
            model.fold_policy_stats(views["enc_stats"], views["goal_stats"])
