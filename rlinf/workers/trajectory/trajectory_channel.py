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

from typing import Any

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler.collective import AsyncWork
from rlinf.scheduler.worker import WorkerGroup

from .storage import (
    Actions,
    EnvBootstrap,
    Intervention,
    Observations,
    RewardRequest,
    Rewards,
    RolloutBootstrap,
)
from .utils import assign_trajectory_rank, assign_trajectory_ranks


def trajectory_queue_key(direction: str, rank: int) -> str:
    """Build the queue key shared by a trajectory channel endpoint and worker."""
    return f"{direction}:{rank}"


class TrajectoryCommEndpoint:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.channel = TrajectoryStorageChannel.connect(
            worker,
            worker_type=self.worker_type,
        )

    worker_type = "unknown"

    @property
    def trajectory_group_name(self) -> str:
        return self.worker.cfg.trajectory.group_name

    def assigned_trajectory_rank(
        self, peer_world_size: int, trajectory_world_size: int
    ) -> int:
        return assign_trajectory_rank(
            self.worker._rank,
            peer_world_size,
            trajectory_world_size,
        )

    def assigned_trajectory_ranks(
        self, peer_world_size: int, trajectory_world_size: int
    ) -> list[int]:
        return assign_trajectory_ranks(
            self.worker._rank,
            peer_world_size,
            trajectory_world_size,
        )


class TrajectoryStorageChannel:
    """Typed view over the trajectory storage worker group.

    Each view preserves the ordinary ``put``/``get`` shape while selecting a
    direction-specific queue and an explicit trajectory replica.  Explicit
    replica selection is required because trajectory ownership is determined
    by component rank mapping, not by the generic channel's node-local hash.
    """

    _WORKER_TYPES = {"env", "rollout", "reward", "actor"}

    def __init__(self, channel: Channel, worker: Worker, worker_type: str):
        if worker_type not in self._WORKER_TYPES:
            raise ValueError(f"Unknown trajectory worker type: {worker_type}")
        self._channel = channel
        self._worker = worker
        self._worker_type = worker_type

    @classmethod
    def connect(cls, worker: Worker, worker_type: str) -> "TrajectoryStorageChannel":
        from .trajectory_worker import TrajectoryChannelWorker

        group_name = worker.cfg.trajectory.group_name
        worker_group = WorkerGroup.from_group_name(
            TrajectoryChannelWorker,
            group_name,
        )
        channel_actors = {
            info.rank: info.worker for info in worker_group.worker_info_list
        }
        maxsize = worker_group.execute_on(0).maxsize().wait()[0]
        channel = Channel()
        channel._initialize(
            channel_name=group_name,
            channel_worker_group=worker_group,
            channel_worker_actor=channel_actors[0],
            current_worker=worker,
            maxsize=maxsize,
            channel_actors=channel_actors,
        )
        return cls(channel, worker, worker_type)

    @property
    def worker_type(self) -> str:
        return self._worker_type

    @property
    def worker(self) -> Worker:
        return self._worker

    def _queue_key(self, direction: str) -> str:
        return trajectory_queue_key(direction, self._worker._rank)

    def put(self, item: Any, trajectory_rank: int, *, async_op: bool = True):
        direction = {
            "env": "env_to_trajectory",
            "rollout": "rollout_to_trajectory",
            "reward": "reward_to_trajectory",
        }.get(self._worker_type)
        if direction is None:
            raise RuntimeError("Actor workers cannot put into TrajectoryChannel")
        return self._channel.put(
            item,
            key=self._queue_key(direction),
            async_op=async_op,
            channel_rank=trajectory_rank,
        )

    def get(self, trajectory_rank: int, *, async_op: bool = True):
        direction = {
            "env": "trajectory_to_env",
            "rollout": "trajectory_to_rollout",
            "reward": "trajectory_to_reward",
            "actor": "trajectory_to_actor",
        }[self._worker_type]
        return self._channel.get(
            key=self._queue_key(direction),
            async_op=async_op,
            channel_rank=trajectory_rank,
        )


class _CompositeChannelWork(AsyncWork):
    """Wait for transport and storage writes as one channel operation."""

    def __init__(self, works: list[AsyncWork]):
        self._works = works

    def wait(self):
        for work in self._works:
            work.wait()

    async def async_wait(self):
        for work in self._works:
            await work.async_wait()

    def then(self, func, *args, **kwargs):
        self.wait()
        return func(*args, **kwargs)

    def done(self):
        return all(work.done() for work in self._works)

    def get_next_work(self):
        return None


class TrajectoryChannel:
    """Drop-in Channel that persists trajectory protocol messages internally.

    Application workers only use :meth:`put` through ``send_to``.  The channel
    forwards the protocol object to its peer and mirrors its storage-specific
    representation to the assigned ``TrajectoryChannelWorker``.
    """

    _WORKER_TYPES = {"env", "rollout", "reward", "actor"}

    def __init__(self, transport: Channel, trajectory_group_name: str):
        self._transport = transport
        self._trajectory_group_name = trajectory_group_name
        self._storage_views: dict[str, TrajectoryStorageChannel] = {}

    @classmethod
    def create(
        cls,
        name: str,
        trajectory_group_name: str,
        maxsize: int = 0,
        distributed: bool = False,
        node_rank: int = 0,
    ) -> "TrajectoryChannel":
        return cls(
            Channel.create(
                name=name,
                maxsize=maxsize,
                distributed=distributed,
                node_rank=node_rank,
            ),
            trajectory_group_name,
        )

    def _worker_type(self) -> str | None:
        worker = Worker.current_worker
        if worker is None:
            return None
        group_names = {
            "env": worker.cfg.env.group_name,
            "rollout": worker.cfg.rollout.group_name,
            "reward": worker.cfg.reward.group_name,
            "actor": worker.cfg.actor.group_name,
        }
        for worker_type, group_name in group_names.items():
            if worker.worker_address.root_group_name == group_name:
                return worker_type
        return None

    def _storage_view(self) -> TrajectoryStorageChannel | None:
        worker_type = self._worker_type()
        if worker_type is None or worker_type == "actor":
            return None
        if (
            Worker.current_worker.cfg.trajectory.group_name
            != self._trajectory_group_name
        ):
            raise ValueError(
                "TrajectoryChannel group name does not match the worker configuration."
            )
        if worker_type not in self._storage_views:
            self._storage_views[worker_type] = TrajectoryStorageChannel.connect(
                Worker.current_worker,
                worker_type,
            )
        return self._storage_views[worker_type]

    def _storage_records(self, item: Any) -> list[Any]:
        """Derive storage records from a regular env/rollout protocol message."""
        if isinstance(item, Observations):
            if item.mode != "train" or item.storage_obs is None:
                return []
            if item.current_step <= 0:
                raise ValueError("Transition observations require current_step > 0.")
            records: list[Any] = [
                Observations(
                    global_step=item.global_step,
                    rank=item.rank,
                    current_step=item.current_step - 1,
                    current_epoch=item.current_epoch,
                    mode=item.mode,
                    stage_id=item.stage_id,
                    obs=item.storage_obs,
                    next_obs=item.storage_next_obs or {},
                    reward_obs=item.storage_reward_obs or {},
                    env_infos=item.storage_env_infos,
                    has_final_obs=item.storage_has_final_obs,
                    rewards=item.storage_rewards,
                    dones=item.storage_dones,
                    terminations=item.storage_terminations,
                    truncations=item.storage_truncations,
                )
            ]
            if item.intervene_actions is not None and item.intervene_flags is not None:
                records.append(
                    Intervention(
                        global_step=item.global_step,
                        rank=item.rank,
                        current_step=item.current_step - 1,
                        current_epoch=item.current_epoch,
                        intervene_actions=item.intervene_actions,
                        intervene_flags=item.intervene_flags,
                    )
                )
            worker = Worker.current_worker
            max_steps = (
                worker.cfg.env.train.max_steps_per_rollout_epoch
                // worker.cfg.actor.model.num_action_chunks
            )
            if item.current_step == max_steps:
                records.append(
                    EnvBootstrap(
                        global_step=item.global_step,
                        rank=item.rank,
                        current_epoch=item.current_epoch,
                        observations=item.next_obs if item.has_final_obs else item.obs,
                        env_infos=item.env_infos,
                        has_final_obs=item.has_final_obs,
                        rewards=item.rewards,
                        dones=item.dones,
                        terminations=item.terminations,
                        truncations=item.truncations,
                    )
                )
            return records
        if isinstance(item, Actions):
            if item.mode != "train":
                return []
            if item.is_bootstrap:
                return [
                    RolloutBootstrap(
                        global_step=item.global_step,
                        rank=item.rank,
                        current_epoch=item.current_epoch,
                        prev_values=item.prev_values,
                    )
                ]
            return [item]
        if isinstance(item, EnvBootstrap):
            # This is the initial rollout input.  The final environment state
            # is derived from the last post-step Observations message above.
            return []
        if isinstance(item, (Intervention, RolloutBootstrap, Rewards)):
            return [item]
        return []

    def put(
        self,
        item: Any,
        weight: int = 0,
        key: Any = "default_queue",
        async_op: bool = False,
        channel_rank: int | None = None,
    ):
        storage_view = self._storage_view()
        storage_works = []
        if storage_view is not None:
            worker_type = storage_view.worker_type
            component_world_size = (
                storage_view.worker._component_placement.get_world_size(worker_type)
            )
            trajectory_world_size = (
                storage_view.worker._component_placement.get_world_size("trajectory")
            )
            trajectory_rank = assign_trajectory_rank(
                storage_view.worker._rank,
                component_world_size,
                trajectory_world_size,
            )
            storage_item = item.get("batch") if isinstance(item, dict) else item
            for storage_record in self._storage_records(storage_item):
                storage_work = storage_view.put(
                    storage_record,
                    trajectory_rank,
                    async_op=async_op,
                )
                if storage_work is not None:
                    storage_works.append(storage_work)

        transport_work = self._transport.put(
            item=item,
            weight=weight,
            key=key,
            async_op=async_op,
            channel_rank=channel_rank,
        )
        if not async_op:
            return None
        works = [*storage_works, *([transport_work] if transport_work else [])]
        return _CompositeChannelWork(works) if works else None

    def get(self, *args, **kwargs):
        return self._transport.get(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._transport, name)


class EnvTrajectoryComm(TrajectoryCommEndpoint):
    worker_type = "env"

    def send_observations(
        self,
        observations: Observations,
        trajectory_rank: int,
        *,
        async_op: bool = True,
    ):
        return self.channel.put(observations, trajectory_rank, async_op=async_op)

    def send_intervention(
        self,
        intervention: Intervention,
        trajectory_rank: int,
        *,
        async_op: bool = True,
    ):
        return self.channel.put(intervention, trajectory_rank, async_op=async_op)

    def send_bootstrap(
        self,
        bootstrap: EnvBootstrap,
        trajectory_rank: int,
        *,
        async_op: bool = True,
    ):
        return self.channel.put(bootstrap, trajectory_rank, async_op=async_op)

    def recv_actions(self, trajectory_rank: int):
        return self.channel.get(trajectory_rank, async_op=True)


class RolloutTrajectoryComm(TrajectoryCommEndpoint):
    worker_type = "rollout"

    async def recv_observations(self, trajectory_rank: int) -> Observations:
        return await self.channel.get(trajectory_rank, async_op=True).async_wait()

    def send_actions(
        self,
        actions: Actions,
        trajectory_rank: int,
        *,
        async_op: bool = True,
    ):
        return self.channel.put(actions, trajectory_rank, async_op=async_op)

    def send_bootstrap(
        self,
        bootstrap: RolloutBootstrap,
        trajectory_rank: int,
        *,
        async_op: bool = True,
    ):
        return self.channel.put(bootstrap, trajectory_rank, async_op=async_op)


class RewardTrajectoryComm(TrajectoryCommEndpoint):
    worker_type = "reward"

    async def recv_request(self, trajectory_rank: int) -> RewardRequest:
        return await self.channel.get(trajectory_rank, async_op=True).async_wait()

    def send_rewards(
        self,
        rewards: Rewards,
        trajectory_rank: int,
        *,
        async_op: bool = True,
    ):
        return self.channel.put(rewards, trajectory_rank, async_op=async_op)


class ActorTrajectoryComm(TrajectoryCommEndpoint):
    worker_type = "actor"

    def recv_trajectory_sync(self, trajectory_rank: int) -> Trajectory:
        return self.channel.get(trajectory_rank, async_op=False)

    async def recv_trajectory(self, trajectory_rank: int) -> Trajectory:
        return await self.channel.get(trajectory_rank, async_op=True).async_wait()
