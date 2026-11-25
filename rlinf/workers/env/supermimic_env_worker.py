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

"""Env worker that steps the G1 motion-imitation env for SuperMimic training.

Training talks to the rollout and actor workers of the same rank once per
control step: the observation goes to the rollout worker, its action comes
back, and the transition goes to the actor's replay. Both messages carry the
merged moments of the step's encodings and goals, which both workers fold
into their copies of the policy input norms. Evaluation uses
:meth:`EnvWorker.evaluate`.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from rlinf.algorithms.supermimic.reset_sampler import ReturnResetSampler
from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM
from rlinf.models.embodiment.supermimic.ipc import (
    FlatDictBuffer,
    obs_packet_schema,
    transition_packet_schema,
)
from rlinf.models.embodiment.supermimic.normalization import (
    merged_moments,
    split_moments,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.env.env_worker import EnvWorker

_STATE_FILE = "env_rank_{rank}.pt"


class SuperMimicEnvWorker(EnvWorker):
    """Drive one rank's share of ``env.train.total_num_envs`` one step at a time."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._actor_group_name = cfg.actor.group_name if not self.only_eval else None
        self._env_group: dist.ProcessGroup | None = None

    def init_worker(self) -> None:
        # The reset statistics and the input moments are merged over every
        # training env rank, so those ranks share one process group.
        distributed = self.enable_train and self._world_size > 1
        if distributed and not dist.is_initialized():
            local_rank = os.environ.get("LOCAL_RANK")
            if (
                local_rank is not None
                and self.torch_platform is not None
                and self.torch_platform.is_available()
            ):
                self.torch_platform.set_device(int(local_rank))
            dist.init_process_group(timeout=Cluster.get_collective_timeout())
        super().init_worker()
        self._env_group = dist.group.WORLD if distributed else None
        if self.enable_train:
            for env in self.env_list:
                env.set_reset_sampler(
                    ReturnResetSampler(
                        env.motion.ends,
                        env.num_envs,
                        env.device,
                        group=self._env_group,
                    )
                )
            env = self.train_env
            self._obs_buf = FlatDictBuffer(
                obs_packet_schema(env.num_envs, env.representation.encoding_dim),
                env.device,
            )
            self._action_buf = torch.empty(
                (env.num_envs, ACTION_DIM), dtype=torch.float32, device=env.device
            )
            self._trans_buf = FlatDictBuffer(
                transition_packet_schema(
                    env.num_envs,
                    env.representation.state_dim,
                    env.representation.encoding_dim,
                ),
                env.device,
            )

    @property
    def train_env(self):
        return self.env_list[0]

    def _sync_stream(self) -> None:
        if self.torch_platform is not None and self.torch_platform.is_available():
            self.torch_platform.current_stream().synchronize()

    @Worker.timer("collect")
    def collect(self, num_steps: int) -> None:
        """Run ``num_steps`` control steps with the rollout and actor of this rank."""
        env = self.train_env
        rollout, actor, rank = (
            self.cfg.rollout.group_name,
            self._actor_group_name,
            self._rank,
        )
        for _ in range(num_steps):
            obs = env.observe()
            enc_width = int(obs["enc"].shape[-1])
            combined = merged_moments(
                torch.cat((obs["enc"], obs["goal"]), dim=-1), self._env_group
            )
            enc_stats, goal_stats = split_moments(combined, enc_width)
            self._obs_buf.pack(
                {**obs, "enc_stats": enc_stats, "goal_stats": goal_stats}
            )
            self._sync_stream()
            self.send_tensor(self._obs_buf.buffer, rollout, rank)
            self.recv_tensor(self._action_buf, rollout, rank)
            transition = env.step(self._action_buf)
            pd_target = transition.pop("pd_target")
            transition.pop("reward", None)
            transition["target_stats"] = merged_moments(pd_target, self._env_group)
            transition["enc_stats"] = enc_stats
            transition["goal_stats"] = goal_stats
            self._trans_buf.pack(transition)
            self._sync_stream()
            self.send_tensor(self._trans_buf.buffer, actor, rank)

    def consume_metrics(self) -> dict[str, float]:
        """Simulator and reset statistics of this rank since the last call."""
        return self.train_env.consume_metrics()

    def save_checkpoint(self, save_path: str) -> None:
        """Reset statistics and RNG of this rank; the simulator state is not saved."""
        os.makedirs(save_path, exist_ok=True)
        state = {"env": self.train_env.state_dict(), "rng": get_rng_state()}
        torch.save(state, os.path.join(save_path, _STATE_FILE.format(rank=self._rank)))

    def load_checkpoint(self, load_path: str) -> None:
        """Restore the reset statistics and, when this rank saved one, its RNG.

        The statistics are pooled over the ranks, so a run resumed on more
        ranks takes them from rank 0 and keeps its freshly seeded generators.
        """
        path = os.path.join(load_path, _STATE_FILE.format(rank=self._rank))
        own = os.path.exists(path)
        if not own:
            path = os.path.join(load_path, _STATE_FILE.format(rank=0))
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.train_env.load_state_dict(state["env"])
        if own:
            set_rng_state(state["rng"])
