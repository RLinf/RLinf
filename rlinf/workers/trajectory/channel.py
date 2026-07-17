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

"""Trajectory-specific channel transport.

This module deliberately does not inherit from :class:`Channel`.  A trajectory
channel owns a shared ``TrajectoryChannelWorker`` group, whereas a generic
channel owns one worker group per channel name.
"""

import asyncio
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping

import ray
import ray.actor

from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.cluster import Cluster
from rlinf.scheduler.collective.async_work import (
    AsyncChannelCommWork,
    AsyncChannelWork,
    AsyncRayWork,
    AsyncRouteWork,
    AsyncWork,
)
from rlinf.scheduler.placement import NodePlacementStrategy, PlacementStrategy
from rlinf.scheduler.worker import Worker
from rlinf.scheduler.worker.worker_group import WorkerGroup
from rlinf.utils.placement import HybridComponentPlacement

from .async_work import AsyncTrajectoryCommWork
from .compression import TrajectoryCompressionConfig
from .data import (
    Actions,
    EnvBootstrap,
    Observations,
    Rewards,
    RolloutBootstrap,
    TrajectoryData,
    TrajectoryEnvelope,
    merge_trajectory_data,
    merge_trajectory_envelopes,
)
from .route_plan import TrajectoryRoute, TrajectoryRoutePlan
from .storage import TrajectoryStorageConfig
from .transport import TransportEndpoint

if TYPE_CHECKING:
    from .trajectory_worker import TrajectoryChannelWorker


class TrajectoryChannel:
    """A channel backed by a shared group of trajectory workers.

    With a route plan, source records are split by global environment slot,
    assembled on their owning trajectory workers, then merged back into the
    actor rank's contiguous slot shard on ``get``.
    """

    @classmethod
    def create(
        cls,
        name: str,
        max_size: int = 0,
        node_ranks: list[int] | None = None,
        placement_strategy: PlacementStrategy | None = None,
        channel_worker_group: WorkerGroup["TrajectoryChannelWorker"] | None = None,
        mode: str = "relay",
        actor_channel_name: str = "Actor",
        route_plan: TrajectoryRoutePlan | None = None,
        cfg: Any | None = None,
        component_worker_groups: Mapping[str, WorkerGroup] | None = None,
        disable_distributed_log: bool = True,
    ) -> "TrajectoryChannel":
        """Create a channel or attach to an existing trajectory worker group.

        Args:
            name: Physical name of the trajectory worker group.
            max_size: Maximum size of each logical queue.
            node_ranks: Compatibility shorthand for one worker on each listed
                cluster node. Cannot be combined with ``placement_strategy``.
            placement_strategy: Physical placement of trajectory workers. Pass
                ``component_placement.get_strategy("trajectory")`` to use the
                same placement semantics as other components. When omitted,
                ``node_ranks`` (or all cluster nodes) is used.
            channel_worker_group: An already launched trajectory worker group.
            mode: ``"relay"`` preserves normal channel queue semantics;
                ``"trajectory_output"`` retrieves complete actor trajectories.
            actor_channel_name: Logical channel name used for complete trajectory
                output. All shared logical channels must use the same value.
            route_plan: Optional placement-aware trajectory shard plan.
            cfg: Training config used to derive the number of environment slots.
            component_worker_groups: Launched env, rollout, reward, and actor
                worker groups used to derive actual component world sizes.
            disable_distributed_log: Whether to disable distributed worker logs.
        """
        from .trajectory_worker import TrajectoryChannelWorker

        if channel_worker_group is None:
            cluster = Cluster()
            placement = cls._resolve_placement_strategy(
                cluster,
                node_ranks=node_ranks,
                placement_strategy=placement_strategy,
            )
            channel_worker_group = TrajectoryChannelWorker.create_group(
                maxsize=max_size
            ).launch(
                cluster=cluster,
                name=name,
                placement_strategy=placement,
                max_concurrency=2**31 - 1,
                disable_distributed_log=disable_distributed_log,
            )

        return cls._from_worker_group(
            name,
            channel_worker_group,
            route_plan=route_plan,
            cfg=cfg,
            component_worker_groups=component_worker_groups,
            mode=mode,
            actor_channel_name=actor_channel_name,
        )

    @property
    def channel_worker_group(self) -> WorkerGroup["TrajectoryChannelWorker"]:
        """Return the shared worker group backing this logical channel."""
        return self._channel_worker_group

    @property
    def route_plan(self) -> TrajectoryRoutePlan | None:
        """Return the route plan used by this logical channel."""
        return self._route_plan

    @classmethod
    def connect(
        cls,
        name: str,
        route_plan: TrajectoryRoutePlan | None = None,
        cfg: Any | None = None,
        component_worker_groups: Mapping[str, WorkerGroup] | None = None,
        mode: str = "relay",
        actor_channel_name: str = "Actor",
    ) -> "TrajectoryChannel":
        """Connect to a previously launched trajectory worker group."""
        from .trajectory_worker import TrajectoryChannelWorker

        channel_worker_group = WorkerGroup.from_group_name(
            TrajectoryChannelWorker,
            name,
        )
        return cls._from_worker_group(
            name,
            channel_worker_group,
            route_plan=route_plan,
            cfg=cfg,
            component_worker_groups=component_worker_groups,
            mode=mode,
            actor_channel_name=actor_channel_name,
        )

    @classmethod
    def _from_worker_group(
        cls,
        name: str,
        channel_worker_group: WorkerGroup["TrajectoryChannelWorker"],
        *,
        route_plan: TrajectoryRoutePlan | None,
        cfg: Any | None,
        component_worker_groups: Mapping[str, WorkerGroup] | None,
        mode: str,
        actor_channel_name: str,
    ) -> "TrajectoryChannel":
        """Build a logical channel from an existing worker group."""
        if route_plan is None and (
            cfg is not None or component_worker_groups is not None
        ):
            if cfg is None or component_worker_groups is None:
                raise ValueError(
                    "cfg and component_worker_groups must be provided together "
                    "when constructing a trajectory route plan."
                )
            route_plan = TrajectoryRoutePlan.from_worker_groups(
                cfg,
                trajectory_worker_group=channel_worker_group,
                component_worker_groups=component_worker_groups,
            )

        channel = cls()
        channel._initialize(
            name, channel_worker_group, route_plan, mode, actor_channel_name
        )
        if route_plan is not None and cfg is not None:
            channel._configure_worker_storage(cfg, route_plan, actor_channel_name)
        return channel

    def _initialize(
        self,
        channel_name: str,
        channel_worker_group: WorkerGroup["TrajectoryChannelWorker"],
        route_plan: TrajectoryRoutePlan | None,
        mode: str,
        actor_channel_name: str,
    ) -> None:
        self._channel_name = channel_name
        self._channel_worker_group_name = channel_worker_group.worker_group_name
        self._channel_worker_group = channel_worker_group
        self._current_worker = Worker.current_worker
        self._channel_workers_by_rank: dict[int, ray.actor.ActorHandle] = {
            worker.rank: worker.worker
            for worker in channel_worker_group.worker_info_list
        }
        if not self._channel_workers_by_rank:
            raise ValueError("Trajectory channel worker group must not be empty.")
        self._route_plan = route_plan
        if mode not in {"relay", "trajectory_output"}:
            raise ValueError(f"Unsupported trajectory channel mode: {mode!r}.")
        self._mode = mode
        self._actor_channel_name = actor_channel_name
        self._transport_endpoints: dict[tuple[int, str], TransportEndpoint] = {}
        self._task_descriptions_by_slot: dict[int, str] = {}
        if route_plan is not None and route_plan.trajectory_world_size != len(
            self._channel_workers_by_rank
        ):
            raise ValueError(
                "Trajectory route plan world size does not match the trajectory "
                "worker group."
            )

    def __setstate__(self, state_dict: dict[str, Any]) -> None:
        """Bind this channel copy to the worker that receives it over Ray."""
        self.__dict__.update(state_dict)
        if self._current_worker is None:
            self._current_worker = Worker.current_worker

    @staticmethod
    def _resolve_placement_strategy(
        cluster: Cluster,
        *,
        node_ranks: list[int] | None,
        placement_strategy: PlacementStrategy | None,
    ) -> PlacementStrategy:
        """Resolve the compatibility node shorthand to a placement strategy."""
        if placement_strategy is not None:
            if node_ranks is not None:
                raise ValueError(
                    "node_ranks and placement_strategy cannot be provided together."
                )
            return placement_strategy
        return NodePlacementStrategy(
            node_ranks if node_ranks is not None else list(range(cluster.num_nodes))
        )

    def _route_channel_workers(self, item: TrajectoryData) -> list[TrajectoryRoute]:
        """Return trajectory worker routes for one source record.

        Without a route plan, the source rank selects one worker by modulo.  A
        route plan partitions the source rank's global slot range into one or
        more trajectory worker shards.
        """
        if route_plan := getattr(self, "_route_plan", None):
            if item.slot_ids is not None:
                routes: dict[int, tuple[list[int], list[int]]] = {}
                for local_index, slot_id in enumerate(item.slot_ids):
                    worker_rank = route_plan.channel_worker_for_slot(slot_id)
                    indices, slot_ids = routes.setdefault(worker_rank, ([], []))
                    indices.append(local_index)
                    slot_ids.append(slot_id)
                return [
                    TrajectoryRoute(
                        channel_worker_rank=worker_rank,
                        local_indices=tuple(indices),
                        slot_ids=tuple(slot_ids),
                    )
                    for worker_rank, (indices, slot_ids) in routes.items()
                ]
            return route_plan.routes_for(self._component_for(item), item.rank)

        ranks = sorted(self._channel_workers_by_rank)
        return [
            TrajectoryRoute(
                channel_worker_rank=ranks[item.rank % len(ranks)],
                local_indices=(),
                slot_ids=(),
            )
        ]

    def _route_item(
        self,
        item: Any,
    ) -> list[tuple[int, TrajectoryData | TrajectoryEnvelope]]:
        """Split ``item`` into its trajectory-worker-owned shards."""
        envelope = TrajectoryEnvelope.from_channel_item(item)
        routes = self._route_channel_workers(envelope.record)
        if getattr(self, "_route_plan", None) is None:
            return [(routes[0].channel_worker_rank, item)]
        is_decoupled = envelope.batch_index is not None
        return [
            (
                route.channel_worker_rank,
                (
                    envelope.select(route.local_indices, slot_ids=route.slot_ids)
                    if is_decoupled
                    else envelope.record.select(
                        route.local_indices, slot_ids=route.slot_ids
                    )
                ),
            )
            for route in routes
        ]

    def prepare_outbound(self, item: Any) -> Any:
        """Attach stable global slot ids before generic worker routing splits it.

        ``Worker.send_to`` splits one source batch for peer workers before it
        calls ``put``.  Recording the source slot identity here keeps each such
        shard routable to the same trajectory worker afterwards.
        """
        envelope = TrajectoryEnvelope.from_channel_item(item)
        record = envelope.record
        if self._route_plan is None or record.slot_ids is not None:
            record = self._cache_outbound_task_descriptions(record)
            if record is envelope.record:
                return item
            return TrajectoryEnvelope(
                record=record, batch_index=envelope.batch_index
            ).to_channel_item()
        component = self._component_for(record)
        start, end = self._route_plan.component_slot_range(component, record.rank)
        if record.batch_size != end - start:
            raise ValueError(
                f"{type(record).__name__} batch size ({record.batch_size}) does not "
                f"match {component!r} rank {record.rank}'s slot range ({end - start})."
            )
        record = record.select(range(record.batch_size), slot_ids=range(start, end))
        record = self._cache_outbound_task_descriptions(record)
        if envelope.batch_index is None:
            return record
        return TrajectoryEnvelope(
            record=record, batch_index=envelope.batch_index
        ).to_channel_item()

    def _cache_outbound_task_descriptions(
        self, record: TrajectoryData
    ) -> TrajectoryData:
        """Emit task descriptions only when a rollout slot's value changes."""
        if (
            self._channel_name != "Rollout"
            or not isinstance(record, Observations)
            or record.task_descriptions is None
        ):
            return record
        if record.slot_ids is None:
            raise ValueError("Task descriptions require routed trajectory slot_ids.")
        if len(record.task_descriptions) != len(record.slot_ids):
            raise ValueError("Task descriptions must align with trajectory slot_ids.")
        changed = any(
            self._task_descriptions_by_slot.get(slot_id) != description
            for slot_id, description in zip(
                record.slot_ids, record.task_descriptions, strict=True
            )
        )
        if not changed:
            return replace(record, task_descriptions=None)
        self._task_descriptions_by_slot.update(
            zip(record.slot_ids, record.task_descriptions, strict=True)
        )
        return record

    def _get_channel_worker(self, rank: int) -> ray.actor.ActorHandle:
        """Return the trajectory worker actor for ``rank``."""
        try:
            return self._channel_workers_by_rank[rank]
        except KeyError as error:
            raise ValueError(
                f"Invalid trajectory channel worker rank: {rank}."
            ) from error

    def _configure_worker_storage(
        self,
        cfg: Any,
        route_plan: TrajectoryRoutePlan,
        actor_channel_name: str,
    ) -> None:
        """Install the local slot shard and rollout shape on every worker."""
        num_action_chunks = int(cfg.actor.model.get("num_action_chunks", 1))
        max_steps = int(cfg.env.train.max_steps_per_rollout_epoch)
        if max_steps % num_action_chunks:
            raise ValueError(
                "env.train.max_steps_per_rollout_epoch must be divisible by "
                "actor.model.num_action_chunks."
            )
        config = TrajectoryStorageConfig(
            num_envs=0,
            rollout_epoch=int(cfg.env.train.rollout_epoch),
            max_steps_per_rollout_epoch=max_steps // num_action_chunks,
            max_episode_length=int(cfg.env.train.max_episode_steps),
            requires_values=bool(
                cfg.actor.model.get("add_value_head", False)
                or cfg.critic.get("use_critic_model", False)
            ),
            action_chunk_shape=(
                num_action_chunks,
                int(cfg.actor.model.action_dim),
            ),
            requires_external_rewards=(
                bool(cfg.reward.get("use_reward_model", False))
                and cfg.reward.get("reward_mode", "per_step") == "per_step"
                and "reward" in route_plan.component_world_sizes
            ),
            env_reward_weight=float(cfg.reward.get("env_reward_weight", 1.0)),
            reward_weight=float(cfg.reward.get("reward_weight", 1.0)),
            gamma=float(cfg.algorithm.gamma),
            auto_reset=bool(cfg.env.train.auto_reset),
            bootstrap_type=str(cfg.algorithm.get("bootstrap_type", "standard")),
        )
        compression_cfg = cfg.trajectory.get("actor_transfer_compression", {})
        compression_config = TrajectoryCompressionConfig(
            enabled=bool(compression_cfg.get("enabled", False)),
            level=int(compression_cfg.get("level", 1)),
            chunk_steps=int(compression_cfg.get("chunk_steps", 16)),
        )
        works = []
        for worker_rank, worker in self._channel_workers_by_rank.items():
            slot_start, slot_end = route_plan.channel_worker_slot_range(worker_rank)
            slot_ids = tuple(range(slot_start, slot_end))
            if not slot_ids:
                continue
            actor_slot_indices: dict[int, tuple[int, ...]] = {}
            if "actor" in route_plan.component_world_sizes:
                actor_world_size = route_plan.component_world_sizes["actor"]
                for actor_rank in range(actor_world_size):
                    local_range = route_plan.component_local_range_on_channel_worker(
                        "actor", actor_rank, worker_rank
                    )
                    if local_range is not None:
                        start, end = local_range
                        actor_slot_indices[actor_rank] = tuple(range(start, end))
            else:
                actor_slot_indices[0] = tuple(range(len(slot_ids)))
            works.append(
                worker.configure_storage.remote(
                    TrajectoryStorageConfig(
                        num_envs=len(slot_ids),
                        rollout_epoch=config.rollout_epoch,
                        max_steps_per_rollout_epoch=config.max_steps_per_rollout_epoch,
                        max_episode_length=config.max_episode_length,
                        requires_values=config.requires_values,
                        action_chunk_shape=config.action_chunk_shape,
                        requires_external_rewards=config.requires_external_rewards,
                        env_reward_weight=config.env_reward_weight,
                        reward_weight=config.reward_weight,
                        gamma=config.gamma,
                        auto_reset=config.auto_reset,
                        bootstrap_type=config.bootstrap_type,
                    ),
                    slot_ids,
                    actor_slot_indices,
                    actor_channel_name,
                    compression_config,
                )
            )
        ray.get(works)

    def put(
        self,
        item: Any,
        weight: int = 0,
        key: Any = DEFAULT_KEY,
        async_op: bool = False,
    ) -> AsyncWork | None:
        """Put one record, splitting it across trajectory workers if required."""
        works = [
            self._put_shard(worker_rank, shard, weight, key, nowait=False)
            for worker_rank, shard in self._route_item(item)
        ]
        if async_op:
            if len(works) == 1:
                return works[0]
            return AsyncRouteWork(works, lambda _: None)
        for work in works:
            work.wait()
        return None

    def put_nowait(
        self,
        item: TrajectoryData,
        weight: int = 0,
        key: Any = DEFAULT_KEY,
    ) -> None:
        """Put all record shards without waiting for queue capacity."""
        for worker_rank, shard in self._route_item(item):
            self._put_shard(worker_rank, shard, weight, key, nowait=True).wait()

    def record(self, item: TrajectoryData, async_op: bool = False) -> AsyncWork | None:
        """Write a routed record to trajectory storage without live relay."""
        works = [
            self._record_shard(worker_rank, shard)
            for worker_rank, shard in self._route_item(item)
        ]
        if async_op:
            if len(works) == 1:
                return works[0]
            return AsyncRouteWork(works, lambda _: None)
        for work in works:
            work.wait()
        return None

    def get(
        self,
        key: Any = DEFAULT_KEY,
        async_op: bool = False,
    ) -> AsyncWork | Any:
        """Get a live relay item or a complete actor trajectory."""
        if self._mode == "relay":
            return self._get_live(key, async_op)

        return self._get_trajectory(key, async_op)

    def _get_trajectory(
        self,
        key: Any,
        async_op: bool,
    ) -> AsyncWork | Any:
        """Get the complete trajectory owned by the current actor rank."""
        actor_rank, worker_ranks = self._actor_routes()
        worker_key = (self._actor_channel_name, key, actor_rank)

        if self._current_worker is not None:
            works = []
            for worker_rank in worker_ranks:
                channel_worker = self._get_channel_worker(worker_rank)
                query_id = uuid.uuid4().int
                channel_worker.get.remote(
                    dst_addr=self._current_worker.worker_address,
                    query_id=query_id,
                    key=worker_key,
                )
                async_comm_work = self._current_worker.recv(
                    self._channel_worker_group_name,
                    worker_rank,
                    async_op=True,
                )
                works.append(
                    AsyncChannelCommWork(
                        async_comm_work=async_comm_work,
                        query_id=query_id,
                        channel_actor=channel_worker,
                    )
                )
        else:
            works = [
                AsyncRayWork(
                    self._get_channel_worker(worker_rank).get_via_ray.remote(
                        key=worker_key
                    )
                )
                for worker_rank in worker_ranks
            ]

        work: AsyncWork = AsyncTrajectoryCommWork(works)
        if async_op:
            return work
        return work.wait()

    def _get_live(self, key: Any, async_op: bool) -> AsyncWork | Any:
        """Get one normal channel item from this logical channel namespace."""
        if self._route_plan is None:
            worker_rank = (
                0
                if self._current_worker is None
                else self._route_channel_worker_rank(self._current_worker._rank)
            )
            worker_ranks = [worker_rank]
        else:
            component = self._consumer_component()
            consumer_rank = (
                0 if self._current_worker is None else self._current_worker._rank
            )
            worker_ranks = [
                route.channel_worker_rank
                for route in self._route_plan.routes_for(component, consumer_rank)
            ]
        worker_key = self._wire_key(key)

        if self._current_worker is not None:
            works = []
            for worker_rank in worker_ranks:
                channel_worker = self._get_channel_worker(worker_rank)
                query_id = uuid.uuid4().int
                channel_worker.get.remote(
                    dst_addr=self._current_worker.worker_address,
                    query_id=query_id,
                    key=worker_key,
                )
                async_comm_work = self._current_worker.recv(
                    self._channel_worker_group_name,
                    worker_rank,
                    async_op=True,
                )
                works.append(
                    AsyncChannelCommWork(
                        async_comm_work=async_comm_work,
                        query_id=query_id,
                        channel_actor=channel_worker,
                    )
                )
        else:
            works = [
                AsyncRayWork(
                    self._get_channel_worker(worker_rank).get_via_ray.remote(
                        key=worker_key
                    )
                )
                for worker_rank in worker_ranks
            ]
        work: AsyncWork = AsyncRouteWork(works, self._merge_live_items)
        if async_op:
            return work
        return work.wait()

    def get_nowait(self, key: Any = DEFAULT_KEY) -> Any:
        """Get one record without waiting for queue data."""
        if self._mode == "trajectory_output":
            actor_rank, worker_ranks = self._actor_routes()
            worker_key = (key, actor_rank)
            works = [
                AsyncRayWork(
                    self._get_channel_worker(worker_rank).get_via_ray.remote(
                        key=worker_key, nowait=True
                    )
                )
                for worker_rank in worker_ranks
            ]
            return AsyncTrajectoryCommWork(works).wait()
        if self._current_worker is None:
            channel_worker = self._get_channel_worker(0)
            return ray.get(
                channel_worker.get_via_ray.remote(key=self._wire_key(key), nowait=True)
            )

        worker_rank = self._route_channel_worker_rank(self._current_worker._rank)
        channel_worker = self._get_channel_worker(worker_rank)
        query_id = uuid.uuid4().int
        channel_worker.get.remote(
            dst_addr=self._current_worker.worker_address,
            query_id=query_id,
            key=self._wire_key(key),
            nowait=True,
        )
        data, result_query_id = self._current_worker.recv(
            self._channel_worker_group_name,
            worker_rank,
        )
        if result_query_id is asyncio.QueueEmpty:
            raise asyncio.QueueEmpty
        return data

    def _route_channel_worker_rank(self, rank: int) -> int:
        """Map a consumer rank to its trajectory worker rank."""
        worker_ranks = sorted(self._channel_workers_by_rank)
        return worker_ranks[rank % len(worker_ranks)]

    def _actor_routes(self) -> tuple[int, list[int]]:
        """Return the actor rank and trajectory workers needed by its slots."""
        if self._route_plan is None:
            rank = 0 if self._current_worker is None else self._current_worker._rank
            return rank, [self._route_channel_worker_rank(rank)]
        actor_rank = 0 if self._current_worker is None else self._current_worker._rank
        routes = self._route_plan.routes_for("actor", actor_rank)
        return actor_rank, [route.channel_worker_rank for route in routes]

    def _consumer_component(self) -> str:
        """Map a live logical channel to the component consuming its queue."""
        component_by_channel = {
            "Env": "env",
            "Rollout": "rollout",
            "Reward": "reward",
        }
        try:
            return component_by_channel[self._channel_name]
        except KeyError as error:
            raise ValueError(
                "A routed relay channel must be named Env, Rollout, or Reward; "
                f"got {self._channel_name!r}."
            ) from error

    def _put_shard(
        self,
        worker_rank: int,
        item: TrajectoryData | TrajectoryEnvelope,
        weight: int,
        key: Any,
        *,
        nowait: bool,
    ) -> AsyncWork:
        channel_worker = self._get_channel_worker(worker_rank)
        worker_key = self._wire_key(key)
        if self._current_worker is not None:
            frame = None
            transport_key = None
            if isinstance(item, TrajectoryData):
                transport_key = self._transport_key(item)
                endpoint = self._transport_endpoint(worker_rank, transport_key)
                frame = endpoint.encode(item)
            if frame is not None:
                async_channel_work = AsyncChannelWork(
                    channel_name="TrajectoryP2P",
                    channel_key=self._transport_channel_key(worker_rank),
                    channel_actor=channel_worker,
                    method="put_frame",
                    src_addr=self._current_worker.worker_address,
                    key=worker_key,
                    weight=weight,
                    transport_key=transport_key,
                    nowait=nowait,
                )
                self._current_worker.send_tensor_frame(
                    (frame.header, *frame.tensors),
                    self._channel_worker_group_name,
                    worker_rank,
                    async_op=True,
                )
                return async_channel_work
            async_channel_work = AsyncChannelWork(
                channel_name="TrajectoryP2P",
                channel_key=self._transport_channel_key(worker_rank),
                channel_actor=channel_worker,
                method="put",
                src_addr=self._current_worker.worker_address,
                nowait=nowait,
                transport_key=transport_key,
            )
            self._current_worker.send(
                item,
                self._channel_worker_group_name,
                worker_rank,
                async_op=True,
                piggyback_payload=(worker_key, weight),
            )
            if isinstance(item, TrajectoryData):
                endpoint.bootstrap(item)
            return async_channel_work

        return AsyncRayWork(
            channel_worker.put_via_ray.remote(
                item=item,
                weight=weight,
                key=worker_key,
                nowait=nowait,
            )
        )

    def _record_shard(
        self,
        worker_rank: int,
        item: TrajectoryData | TrajectoryEnvelope,
    ) -> AsyncWork:
        """Send one storage-only shard through P2P or driver-side Ray transport."""
        channel_worker = self._get_channel_worker(worker_rank)
        if self._current_worker is not None:
            frame = None
            transport_key = None
            if isinstance(item, TrajectoryData):
                transport_key = self._transport_key(item)
                endpoint = self._transport_endpoint(worker_rank, transport_key)
                frame = endpoint.encode(item)
            if frame is not None:
                async_channel_work = AsyncChannelWork(
                    channel_name="TrajectoryP2P",
                    channel_key=self._transport_channel_key(worker_rank),
                    channel_actor=channel_worker,
                    method="record_frame",
                    src_addr=self._current_worker.worker_address,
                    transport_key=transport_key,
                )
                self._current_worker.send_tensor_frame(
                    (frame.header, *frame.tensors),
                    self._channel_worker_group_name,
                    worker_rank,
                    async_op=True,
                )
                return async_channel_work
            async_channel_work = AsyncChannelWork(
                channel_name="TrajectoryP2P",
                channel_key=self._transport_channel_key(worker_rank),
                channel_actor=channel_worker,
                method="record",
                src_addr=self._current_worker.worker_address,
                transport_key=transport_key,
            )
            self._current_worker.send(
                item,
                self._channel_worker_group_name,
                worker_rank,
                async_op=True,
            )
            if isinstance(item, TrajectoryData):
                endpoint.bootstrap(item)
            return async_channel_work

        return AsyncRayWork(channel_worker.record_via_ray.remote(item))

    def _wire_key(self, key: Any) -> tuple[str, Any]:
        """Namespace a logical channel key inside the shared worker group."""
        return self._channel_name, key

    def _transport_endpoint(
        self, worker_rank: int, transport_key: str
    ) -> TransportEndpoint:
        endpoint_key = worker_rank, transport_key
        endpoint = self._transport_endpoints.get(endpoint_key)
        if endpoint is None:
            endpoint = TransportEndpoint()
            self._transport_endpoints[endpoint_key] = endpoint
        return endpoint

    def _transport_channel_key(self, worker_rank: int) -> tuple[str, int]:
        """Serialize all P2P frames sent to one trajectory worker.

        ``Actions`` and ``RolloutBootstrap`` use different logical queues but
        the same collective source/destination pair.  Their frame send/recv
        order is therefore a transport invariant, not a logical-key property.
        """
        return self._channel_worker_group_name, worker_rank

    def _transport_key(self, item: TrajectoryData | TrajectoryEnvelope) -> str:
        """Return the fixed-frame identity for one routed trajectory payload."""
        if not isinstance(item, TrajectoryData):
            raise ValueError("Only TrajectoryData payloads support tensor frames.")
        return f"{self._channel_name}:{type(item).__qualname__}"

    @staticmethod
    def _merge_live_items(items: list[Any]) -> Any:
        """Merge routed relay shards while retaining the caller's payload shape."""
        if len(items) == 1:
            return items[0]
        if all(isinstance(item, TrajectoryData) for item in items):
            return merge_trajectory_data(items)
        envelopes = [TrajectoryEnvelope.from_channel_item(item) for item in items]
        return merge_trajectory_envelopes(envelopes).to_channel_item()

    @staticmethod
    def _component_for(item: TrajectoryData) -> str:
        if isinstance(item, (Observations, EnvBootstrap)):
            return "env"
        if isinstance(item, (Actions, RolloutBootstrap)):
            return "rollout"
        if isinstance(item, Rewards):
            return "reward"
        raise ValueError(f"Unsupported trajectory data type: {type(item).__name__}.")


@dataclass(frozen=True)
class TrajectoryChannels:
    """The logical embodied channels backed by one trajectory worker group."""

    env: TrajectoryChannel
    rollout: TrajectoryChannel
    reward: TrajectoryChannel
    actor: TrajectoryChannel

    @classmethod
    def create(
        cls,
        cfg: Any,
        *,
        component_worker_groups: Mapping[str, WorkerGroup],
    ) -> "TrajectoryChannels":
        """Create the shared worker group, route plan, and logical channels."""
        placement = HybridComponentPlacement(cfg, Cluster())
        root_channel = TrajectoryChannel.create(
            cfg.trajectory.group_name,
            placement_strategy=placement.get_strategy("trajectory"),
            cfg=cfg,
            component_worker_groups=component_worker_groups,
        )
        channel_args = {
            "channel_worker_group": root_channel.channel_worker_group,
            "route_plan": root_channel.route_plan,
            "actor_channel_name": "Actor",
        }
        return cls(
            env=TrajectoryChannel.create("Env", mode="relay", **channel_args),
            rollout=TrajectoryChannel.create("Rollout", mode="relay", **channel_args),
            reward=TrajectoryChannel.create("Reward", mode="relay", **channel_args),
            actor=TrajectoryChannel.create(
                "Actor", mode="trajectory_output", **channel_args
            ),
        )
