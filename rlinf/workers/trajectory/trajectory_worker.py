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

import asyncio
from collections import OrderedDict, defaultdict
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.scheduler import (
    ChannelWorker,
    Cluster,
    Worker,
    WorkerAddress,
    infer_batch_size,
    merge_batches,
    split_batch,
)
from rlinf.utils.nested_dict_process import clone_nested_to_cpu
from rlinf.utils.placement import HybridComponentPlacement

from .history_manager import HistoryManager
from .storage import (
    Actions,
    EnvBootstrap,
    Intervention,
    Observations,
    RewardRequest,
    Rewards,
    RolloutBootstrap,
    TrajectoryData,
    TrajectoryStorage,
)
from .trajectory_channel import trajectory_queue_key
from .utils import assign_peer_ranks


class TrajectoryChannelWorker(ChannelWorker):
    def __init__(self, cfg: DictConfig):
        # Keep the channel transport and trajectory state in the same actor.
        # The protocol adapter is introduced separately; existing direct
        # communication paths remain valid during the migration.
        ChannelWorker.__init__(
            self,
            maxsize=int(cfg.trajectory.get("channel_maxsize", 0)),
        )
        self.cfg = cfg
        self.trajectory_cfg = cfg.trajectory

        self._stopping = False
        self._trajectory_group_name: str = cfg.trajectory.group_name
        self.require_reward_worker: bool = bool(
            self.trajectory_cfg.get(
                "require_reward_worker",
                cfg.reward.use_reward_model,
            )
        )
        self.env_group_name: str = cfg.env.group_name
        self.rollout_group_name: str = cfg.rollout.group_name
        self.actor_group_name: str = cfg.actor.group_name
        self.reward_group_name: str = (
            cfg.reward.group_name if self.require_reward_worker else ""
        )
        self.reward_weight = float(cfg.reward.get("reward_weight", 1.0))
        self.env_reward_weight = float(cfg.reward.get("env_reward_weight", 0.0))
        self.reward_mode = cfg.reward.get("reward_mode", "per_step")
        self.history_reward_assign = bool(
            cfg.reward.get("history_reward_assign", False)
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")

        self.rollout_epoch = int(cfg.env.train.rollout_epoch)
        self.max_steps_per_rollout_epoch = int(
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.max_episode_length = int(cfg.env.train.max_episode_steps)
        self.steps_held = int(self.trajectory_cfg.get("steps_held", 2))
        self.env_status_check_interval = float(
            self.trajectory_cfg.get("env_status_check_interval", 5.0)
        )
        self.data_device: torch.device | str = self.trajectory_cfg.get(
            "data_device", "cpu"
        )

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.env_ws = int(self._component_placement.get_world_size("env"))
        self.rollout_ws = int(self._component_placement.get_world_size("rollout"))
        self.actor_ws = int(self._component_placement.get_world_size("actor"))
        self.reward_ws = (
            int(self._component_placement.get_world_size("reward"))
            if self.require_reward_worker
            else 0
        )
        self.env_ranks: list[int] = []
        self.rollout_ranks: list[int] = []
        self.reward_ranks: list[int] = []
        self.actor_ranks: list[int] = []
        self.num_envs_per_env_rank = 0
        self.local_num_envs = 0
        self.env_rank_to_slots: dict[int, list[int]] = {}
        self.rollout_rank_to_slots: dict[int, list[int]] = {}
        self.reward_rank_to_slots: dict[int, list[int]] = {}
        self.history_managers: dict[int, HistoryManager] = {}
        self.failed_env_ranks: set[int] = set()

    def init_worker(self):
        self.handler_queue: asyncio.Queue[TrajectoryData] = asyncio.Queue()
        self._handler_error: asyncio.Future[None] | None = None
        self.received_counts: dict[str, int] = defaultdict(int)
        self.tasks: list[asyncio.Task] = []
        self.observation_tasks: dict[int, asyncio.Task] = {}
        self.handler_tasks: set[asyncio.Task] = set()
        self.trajectory_storages: OrderedDict[int, TrajectoryStorage] = OrderedDict()
        self.eval_observations: dict[
            tuple[int, int, int, int, int], dict[int, Observations]
        ] = {}
        self.eval_routes: dict[
            tuple[int, int, int, int, int], list[tuple[int, int]]
        ] = {}
        self.ready_trajectories: dict[int, OrderedDict[int, Trajectory]] = {}
        self.storage_lock = asyncio.Lock()
        self.trajectory_storages.clear()
        self.ready_trajectories.clear()
        self._init_comm_mapping()
        self.ready_trajectories = {
            actor_rank: OrderedDict() for actor_rank in self.actor_ranks
        }
        if self.reward_mode == "history_buffer":
            self.history_managers = {
                env_rank: HistoryManager(self.cfg.reward, len(slots))
                for env_rank, slots in self.env_rank_to_slots.items()
            }

    @property
    def trajectory_group_name(self) -> str:
        return self._trajectory_group_name

    async def run_loop(self):
        self._handler_error = asyncio.get_running_loop().create_future()
        self.observation_tasks = {
            rank: asyncio.create_task(self._recv_data_loop(self.env_group_name, rank))
            for rank in self.env_ranks
        }
        self.tasks.extend(
            [
                *self.observation_tasks.values(),
                *[
                    asyncio.create_task(
                        self._recv_data_loop(self.rollout_group_name, rank)
                    )
                    for rank in self.rollout_ranks
                ],
                *[
                    asyncio.create_task(self._send_trajectory_loop(rank))
                    for rank in self.actor_ranks
                ],
                asyncio.create_task(self._monitor_env_worker_status_loop()),
                asyncio.create_task(self._data_handler_loop()),
            ]
        )
        if self.require_reward_worker:
            self.tasks.extend(
                [
                    *[
                        asyncio.create_task(
                            self._recv_data_loop(self.reward_group_name, rank)
                        )
                        for rank in self.reward_ranks
                    ],
                ]
            )
        try:
            await asyncio.gather(*self.tasks, self._handler_error)
        except asyncio.CancelledError:
            if not self._stopping:
                raise
        finally:
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()
            self.observation_tasks.clear()
            self.handler_tasks.clear()
            self.eval_observations.clear()
            self.eval_routes.clear()

    async def stop(self):
        if self._stopping:
            return
        self._stopping = True
        self._mem_cleaner_task.cancel()
        for handler_task in self.handler_tasks:
            handler_task.cancel()
        for task in self.tasks:
            task.cancel()
        if self._handler_error is not None and not self._handler_error.done():
            self._handler_error.cancel()

    def _init_comm_mapping(self):
        min_peer_ws = min(
            self.env_ws,
            self.rollout_ws,
            self.actor_ws,
            self.reward_ws if self.require_reward_worker else self.actor_ws,
        )
        if self._world_size > min_peer_ws:
            raise ValueError(
                "trajectory world size must not exceed any participating component "
                "world size; otherwise some trajectory ranks cannot receive data."
            )
        self.env_ranks = assign_peer_ranks(
            self._rank,
            self.env_ws,
            self._world_size,
        )
        self.rollout_ranks = assign_peer_ranks(
            self._rank,
            self.rollout_ws,
            self._world_size,
        )
        self.actor_ranks = assign_peer_ranks(
            self._rank,
            self.actor_ws,
            self._world_size,
        )
        self.reward_ranks = (
            assign_peer_ranks(
                self._rank,
                self.reward_ws,
                self._world_size,
            )
            if self.require_reward_worker
            else []
        )

        total_num_envs = int(self.cfg.env.train.total_num_envs)
        pipeline_stage_num = int(self.cfg.rollout.pipeline_stage_num)
        logical_env_ws = self.env_ws * pipeline_stage_num
        self.num_envs_per_env_rank = total_num_envs // logical_env_ws
        self.local_num_envs = self.num_envs_per_env_rank * len(self.env_ranks)

        self.env_rank_to_slots = {}
        for local_rank_idx, env_rank in enumerate(self.env_ranks):
            start = local_rank_idx * self.num_envs_per_env_rank
            end = start + self.num_envs_per_env_rank
            self.env_rank_to_slots[env_rank] = list(range(start, end))

        self.rollout_rank_to_slots = {}
        for local_rank_idx, rollout_rank in enumerate(self.rollout_ranks):
            start = self.local_num_envs * local_rank_idx // len(self.rollout_ranks)
            end = self.local_num_envs * (local_rank_idx + 1) // len(self.rollout_ranks)
            self.rollout_rank_to_slots[rollout_rank] = list(range(start, end))

        self.reward_rank_to_slots = {}
        for local_rank_idx, reward_rank in enumerate(self.reward_ranks):
            start = self.local_num_envs * local_rank_idx // len(self.reward_ranks)
            end = self.local_num_envs * (local_rank_idx + 1) // len(self.reward_ranks)
            self.reward_rank_to_slots[reward_rank] = list(range(start, end))

    async def _recv_data_loop(self, src_group_name: str, rank: int) -> None:
        direction = {
            self.env_group_name: "env_to_trajectory",
            self.rollout_group_name: "rollout_to_trajectory",
            self.reward_group_name: "reward_to_trajectory",
        }[src_group_name]
        while not self._stopping:
            try:
                data: TrajectoryData = await self.get_via_ray(
                    key=trajectory_queue_key(direction, rank)
                )
            except asyncio.CancelledError:
                return

            await self.handler_queue.put(data)

    async def _send_trajectory_loop(self, rank: int):
        while not self._stopping:
            trajectory = None
            async with self.storage_lock:
                if ready_queue := self.ready_trajectories[rank]:
                    _, trajectory = ready_queue.popitem(last=False)

            if trajectory is None:
                await asyncio.sleep(0.01)
                continue

            await self.put_via_ray(
                trajectory,
                weight=0,
                key=trajectory_queue_key("trajectory_to_actor", rank),
            )

    async def _handle_observations(self, data: Observations):
        if data.mode == "eval":
            await self._handle_eval_observations(data)
            return

        reward_request = None
        async with self.storage_lock:
            storage = self._get_or_create_storage(data.global_step)
            slots = self._slots_for_rank(
                rank=data.rank,
                rank_to_slots=self.env_rank_to_slots,
                source="env",
            )
            reward_request = self._build_observation_reward_request(data)
            storage.write_observations(
                data,
                slots,
                mark_rewards=reward_request is None,
            )
            if data.intervene_actions is not None and data.intervene_flags is not None:
                storage.apply_intervention(
                    current_epoch=data.current_epoch,
                    current_step=data.current_step,
                    slot_indices=slots,
                    intervene_actions=data.intervene_actions,
                    intervene_flags=data.intervene_flags,
                )
            self._seal_if_complete(data.global_step, storage)

        if reward_request is not None:
            asyncio.create_task(self._send_reward_request(reward_request))

    async def _handle_actions(self, data: Actions):
        if data.mode == "eval":
            await self._handle_eval_actions(data)
            return

        async with self.storage_lock:
            storage = self._get_or_create_storage(data.global_step)
            slots = self._slots_for_rank(
                rank=data.rank,
                rank_to_slots=self.rollout_rank_to_slots,
                source="rollout",
            )
            storage.write_actions(data, slots)
            self._seal_if_complete(data.global_step, storage)

    async def _handle_eval_observations(self, data: Observations) -> None:
        rollout_rank = self._rollout_rank_for_env_rank(data.rank)
        key = self._eval_route_key(data, rollout_rank)
        merged_data = None
        route: list[tuple[int, int]] | None = None

        async with self.storage_lock:
            observations_by_rank = self.eval_observations.setdefault(key, {})
            observations_by_rank[data.rank] = data
            env_ranks = self._env_ranks_for_rollout_rank(rollout_rank)
            if all(env_rank in observations_by_rank for env_rank in env_ranks):
                ordered_observations = [
                    observations_by_rank.pop(env_rank) for env_rank in env_ranks
                ]
                if not observations_by_rank:
                    del self.eval_observations[key]
                split_sizes = [
                    infer_batch_size(observations.obs)
                    for observations in ordered_observations
                ]
                route = list(zip(env_ranks, split_sizes))
                self.eval_routes[key] = route
                merged_obs = merge_batches(
                    [observations.obs for observations in ordered_observations]
                )
                has_final_obs = any(
                    observations.has_final_obs for observations in ordered_observations
                )
                merged_final_obs = {}
                if has_final_obs:
                    merged_final_obs = merge_batches(
                        [
                            observations.next_obs
                            if observations.has_final_obs
                            else observations.obs
                            for observations in ordered_observations
                        ]
                    )
                rlt_switch_flags = None
                if any(
                    observations.rlt_switch_flags is not None
                    for observations in ordered_observations
                ):
                    ref_flags = next(
                        observations.rlt_switch_flags
                        for observations in ordered_observations
                        if observations.rlt_switch_flags is not None
                    )
                    filled_flags = []
                    for observations, split_size in zip(
                        ordered_observations, split_sizes
                    ):
                        if observations.rlt_switch_flags is None:
                            fill_shape = (split_size, *ref_flags.shape[1:])
                            filled_flags.append(
                                torch.zeros(fill_shape, dtype=ref_flags.dtype)
                            )
                        else:
                            filled_flags.append(observations.rlt_switch_flags)
                    rlt_switch_flags = merge_batches(filled_flags)
                merged_data = Observations(
                    mode="eval",
                    global_step=data.global_step,
                    rank=self._rank,
                    current_epoch=data.current_epoch,
                    current_step=data.current_step,
                    stage_id=data.stage_id,
                    obs=merged_obs,
                    next_obs=merged_final_obs,
                    rlt_switch_flags=rlt_switch_flags,
                    has_final_obs=has_final_obs,
                )

        if merged_data is None:
            return
        await self.put_via_ray(
            merged_data,
            weight=0,
            key=trajectory_queue_key("trajectory_to_rollout", rollout_rank),
        )

    async def _handle_eval_actions(self, data: Actions) -> None:
        key = self._eval_route_key(data, data.rank)
        async with self.storage_lock:
            route = self.eval_routes.pop(key, None)
        if route is None:
            raise RuntimeError(f"Missing eval route for actions with key={key}.")

        split_actions = split_batch(data.actions, [size for _, size in route])
        for (env_rank, _), actions in zip(route, split_actions):
            await self.put_via_ray(
                Actions(
                    mode="eval",
                    global_step=data.global_step,
                    rank=self._rank,
                    current_epoch=data.current_epoch,
                    current_step=data.current_step,
                    stage_id=data.stage_id,
                    actions=actions,
                ),
                weight=0,
                key=trajectory_queue_key("trajectory_to_env", env_rank),
            )

    async def _handle_intervention(self, data: Intervention):
        async with self.storage_lock:
            storage = self._get_or_create_storage(data.global_step)
            slots = self._slots_for_rank(
                rank=data.rank,
                rank_to_slots=self.env_rank_to_slots,
                source="env",
            )
            storage.apply_intervention(
                current_epoch=data.current_epoch,
                current_step=data.current_step,
                slot_indices=slots,
                intervene_actions=data.intervene_actions,
                intervene_flags=data.intervene_flags,
            )

    async def _handle_env_bootstrap(self, data: EnvBootstrap):
        reward_request = None
        async with self.storage_lock:
            storage = self._get_or_create_storage(data.global_step)
            slots = self._slots_for_rank(
                rank=data.rank,
                rank_to_slots=self.env_rank_to_slots,
                source="env",
            )
            reward_request = self._build_bootstrap_reward_request(data)
            storage.write_env_bootstrap(
                data,
                slots,
                mark_rewards=reward_request is None,
            )
            if reward_request is None:
                storage.apply_terminal_bootstrap_reward(
                    current_epoch=data.current_epoch,
                    gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
                    auto_reset=bool(self.cfg.env.train.auto_reset),
                    bootstrap_type=self.cfg.algorithm.get("bootstrap_type", "standard"),
                )
            self._seal_if_complete(data.global_step, storage)

        if reward_request is not None:
            asyncio.create_task(self._send_reward_request(reward_request))

    async def _handle_rollout_bootstrap(self, data: RolloutBootstrap):
        async with self.storage_lock:
            storage = self._get_or_create_storage(data.global_step)
            slots = self._slots_for_rank(
                rank=data.rank,
                rank_to_slots=self.rollout_rank_to_slots,
                source="rollout",
            )
            storage.write_rollout_bootstrap(data, slots)
            if not self.require_reward_worker:
                storage.apply_terminal_bootstrap_reward(
                    current_epoch=data.current_epoch,
                    gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
                    auto_reset=bool(self.cfg.env.train.auto_reset),
                    bootstrap_type=self.cfg.algorithm.get("bootstrap_type", "standard"),
                )
            self._seal_if_complete(data.global_step, storage)

    async def _handle_rewards(self, data: Rewards):
        async with self.storage_lock:
            storage = self._get_or_create_storage(data.global_step)
            slots = self._slots_for_rank(
                rank=data.rank,
                rank_to_slots=self.env_rank_to_slots,
                source="env",
            )
            if data.reward_mode == "terminal":
                storage.write_terminal_rewards(
                    data,
                    slots,
                    reward_weight=self.reward_weight,
                    env_reward_weight=self.env_reward_weight,
                )
            else:
                storage.write_rewards(
                    data,
                    slots,
                    reward_weight=self.reward_weight,
                    env_reward_weight=self.env_reward_weight,
                )
            if data.reward_mode == "history_buffer" and self.history_reward_assign:
                storage.assign_history_rewards(
                    current_epoch=data.current_epoch,
                    current_step=data.current_step,
                    slot_indices=slots,
                    rewards=data.rewards,
                    history_lengths=data.history_lengths,
                    reward_weight=self.reward_weight,
                )
            if data.current_step == self.max_steps_per_rollout_epoch:
                storage.apply_terminal_bootstrap_reward(
                    current_epoch=data.current_epoch,
                    gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
                    auto_reset=bool(self.cfg.env.train.auto_reset),
                    bootstrap_type=self.cfg.algorithm.get("bootstrap_type", "standard"),
                )
            self._seal_if_complete(data.global_step, storage)

    def _build_observation_reward_request(
        self, data: Observations
    ) -> RewardRequest | None:
        if not self.require_reward_worker:
            return None
        if data.current_step == 0:
            return None
        if self.reward_mode == "terminal" and not data.has_final_obs:
            return None
        if self.reward_mode not in {"per_step", "history_buffer", "terminal"}:
            return None

        observations, history_lengths = self._build_reward_observations(data)
        return RewardRequest(
            global_step=data.global_step,
            rank=self._rank,
            source_rank=data.rank,
            current_epoch=data.current_epoch,
            current_step=data.current_step,
            reward_mode=self.reward_mode,
            observations=observations,
            history_lengths=history_lengths,
        )

    def _build_bootstrap_reward_request(
        self, data: EnvBootstrap
    ) -> RewardRequest | None:
        if not self.require_reward_worker:
            return None
        if self.reward_mode == "terminal" and not data.has_final_obs:
            return None
        if self.reward_mode not in {"per_step", "history_buffer", "terminal"}:
            return None

        observations, history_lengths = self._build_bootstrap_reward_observations(data)
        if data.current_epoch == self.rollout_epoch - 1:
            observations["last_run"] = torch.ones(
                (len(self.env_rank_to_slots[data.rank]), 1), dtype=torch.bool
            )
        return RewardRequest(
            global_step=data.global_step,
            rank=self._rank,
            source_rank=data.rank,
            current_epoch=data.current_epoch,
            current_step=self.max_steps_per_rollout_epoch,
            reward_mode=self.reward_mode,
            observations=observations,
            history_lengths=history_lengths,
        )

    def _build_reward_observations(
        self, data: Observations
    ) -> tuple[dict[str, Any], dict[str, list[int]]]:
        observations = dict(data.reward_obs or data.obs)
        history_lengths = self._maybe_attach_history_input(
            env_rank=data.rank,
            observations=observations,
            dones=data.dones,
        )
        self._attach_reward_env_infos(observations, data.env_infos)
        if data.dones is not None:
            dones = data.dones
            if getattr(dones, "ndim", 0) > 1:
                dones = dones[:, -1]
            observations["dones"] = dones
        return observations, history_lengths

    def _build_bootstrap_reward_observations(
        self, data: EnvBootstrap
    ) -> tuple[dict[str, Any], dict[str, list[int]]]:
        observations = dict(data.observations)
        history_lengths = self._maybe_attach_history_input(
            env_rank=data.rank,
            observations=observations,
            dones=data.dones,
        )
        self._attach_reward_env_infos(observations, data.env_infos)
        if data.dones is not None:
            dones = data.dones
            if getattr(dones, "ndim", 0) > 1:
                dones = dones[:, -1]
            observations["dones"] = dones
        return observations, history_lengths

    def _maybe_attach_history_input(
        self,
        *,
        env_rank: int,
        observations: dict[str, torch.Tensor],
        dones: torch.Tensor | None,
    ) -> dict[str, list[int]]:
        if self.reward_mode != "history_buffer":
            return {}
        history_manager = self.history_managers[env_rank]
        history_manager.append_to_history_entries(observations)
        if dones is None:
            return {}
        if getattr(dones, "ndim", 0) > 1:
            dones = dones[:, -1]
        history_input, history_lengths = history_manager.build_history_input(
            dones=dones
        )
        observations["history_input"] = history_input
        return history_lengths

    def _attach_reward_env_infos(
        self,
        observations: dict[str, torch.Tensor],
        env_infos: dict | None,
    ) -> None:
        if env_infos is None:
            return
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key in env_infos:
                reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        if reward_env_infos:
            observations["env_infos"] = reward_env_infos

    async def _send_reward_request(self, request: RewardRequest) -> None:
        if not self.reward_ranks:
            raise RuntimeError(
                "Reward worker is required but no reward ranks are assigned."
            )
        reward_rank = self.reward_ranks[request.current_step % len(self.reward_ranks)]
        await self.put_via_ray(
            request,
            weight=0,
            key=trajectory_queue_key("trajectory_to_reward", reward_rank),
        )

    def get_env_worker_status(self) -> dict[int, bool]:
        status = {}
        for env_rank in self.env_ranks:
            worker_name = WorkerAddress(self.env_group_name, env_rank).get_name()
            status[env_rank] = Worker.check_worker_alive(worker_name)
        return status

    async def _monitor_env_worker_status_loop(self) -> None:
        while not self._stopping:
            status = self.get_env_worker_status()
            failed_ranks = [rank for rank, alive in status.items() if not alive]
            await self._mark_env_ranks_failed(failed_ranks)
            await asyncio.sleep(self.env_status_check_interval)

    async def _data_handler_loop(self):
        while not self._stopping:
            try:
                data: TrajectoryData = await self.handler_queue.get()
            except asyncio.CancelledError:
                break

            self.received_counts[type(data).__name__] += 1
            if isinstance(data, Observations):
                task = asyncio.create_task(self._handle_observations(data))
            elif isinstance(data, Intervention):
                task = asyncio.create_task(self._handle_intervention(data))
            elif isinstance(data, Actions):
                task = asyncio.create_task(self._handle_actions(data))
            elif isinstance(data, EnvBootstrap):
                task = asyncio.create_task(self._handle_env_bootstrap(data))
            elif isinstance(data, RolloutBootstrap):
                task = asyncio.create_task(self._handle_rollout_bootstrap(data))
            elif isinstance(data, Rewards):
                task = asyncio.create_task(self._handle_rewards(data))
            else:
                raise ValueError(f"Unknown data type: {type(data)}")

            self.handler_tasks.add(task)
            task.add_done_callback(self._on_handler_task_done)

    def _on_handler_task_done(self, task: asyncio.Task) -> None:
        self.handler_tasks.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is None:
            return
        self.log_error(f"Trajectory handler task failed: {exception!r}")
        if self._handler_error is not None and not self._handler_error.done():
            self._handler_error.set_exception(exception)

    async def debug_state(self) -> dict:
        async with self.storage_lock:
            storages = {}
            for global_step, storage in self.trajectory_storages.items():
                storages[int(global_step)] = {
                    "observation_ready": self._mask_count(storage.observation_ready),
                    "action_ready": self._mask_count(storage.action_ready),
                    "reward_ready": self._mask_count(storage.reward_ready),
                    "value_ready": self._mask_count(storage.value_ready),
                    "missing_reward_indices": self._missing_indices(
                        storage.reward_ready
                    ),
                    "failed_slots": int(storage.failed_slot_mask.sum().item()),
                    "complete": storage.complete(),
                }
            return {
                "rank": self._rank,
                "env_ranks": list(self.env_ranks),
                "rollout_ranks": list(self.rollout_ranks),
                "actor_ranks": list(self.actor_ranks),
                "reward_ranks": list(self.reward_ranks),
                "handler_queue_size": self.handler_queue.qsize(),
                "handler_tasks": len(self.handler_tasks),
                "received_counts": dict(self.received_counts),
                "ready_trajectories": {
                    actor_rank: len(queue)
                    for actor_rank, queue in self.ready_trajectories.items()
                },
                "storages": storages,
            }

    @staticmethod
    def _mask_count(mask: torch.Tensor) -> dict[str, int]:
        return {
            "ready": int(mask.sum().item()),
            "total": int(mask.numel()),
        }

    @staticmethod
    def _missing_indices(mask: torch.Tensor, limit: int = 16) -> list[list[int]]:
        missing = (~mask).nonzero(as_tuple=False)
        return missing[:limit].to(dtype=torch.int64).cpu().tolist()

    def _get_or_create_storage(self, global_step: int) -> TrajectoryStorage:
        global_step = int(global_step)
        if global_step in self.trajectory_storages:
            return self.trajectory_storages[global_step]
        if len(self.trajectory_storages) >= self.steps_held:
            oldest_step = next(iter(self.trajectory_storages))
            raise RuntimeError(
                f"Trajectory storage window is full: steps_held={self.steps_held}, "
                f"oldest_unfinished_global_step={oldest_step}, new_global_step={global_step}."
            )
        storage = TrajectoryStorage(
            num_envs=self.local_num_envs,
            rollout_epoch=self.rollout_epoch,
            max_steps_per_rollout_epoch=self.max_steps_per_rollout_epoch,
            max_episode_length=self.max_episode_length,
            data_device=self.data_device,
            requires_values=self.cfg.algorithm.loss_type == "actor_critic",
        )
        for env_rank in self.failed_env_ranks:
            storage.mark_failed_slots(self.env_rank_to_slots[env_rank])
        self.trajectory_storages[global_step] = storage
        return storage

    def _seal_if_complete(self, global_step: int, storage: TrajectoryStorage) -> None:
        if not storage.complete():
            return
        good_slots = [
            slot
            for slot in range(storage.num_envs)
            if not bool(storage.failed_slot_mask[slot].item())
        ]
        if good_slots:
            trajectory = storage.to_trajectory(good_slots)
            for actor_rank, shard in zip(
                self.actor_ranks,
                self._split_trajectory_for_actors(trajectory),
            ):
                self.ready_trajectories[actor_rank][int(global_step)] = shard
        del self.trajectory_storages[int(global_step)]

    def _split_trajectory_for_actors(self, trajectory: Trajectory) -> list[Trajectory]:
        split_count = len(self.actor_ranks)
        if split_count == 1:
            return [trajectory]
        split_sizes = self._split_sizes(trajectory.actions.shape[1], split_count)
        shards = [
            Trajectory(
                max_episode_length=trajectory.max_episode_length,
                model_weights_id=trajectory.model_weights_id,
            )
            for _ in split_sizes
        ]

        def split_tensor(value: torch.Tensor | None) -> list[torch.Tensor | None]:
            if value is None:
                return [None for _ in split_sizes]
            return list(torch.split(value, split_sizes, dim=1))

        for field_name in trajectory.__dataclass_fields__:
            value = getattr(trajectory, field_name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                for shard, split_value in zip(shards, split_tensor(value)):
                    setattr(shard, field_name, split_value.contiguous())
            elif isinstance(value, dict):
                split_dicts = [{} for _ in split_sizes]
                for key, tensor in value.items():
                    for split_dict, split_value in zip(
                        split_dicts, split_tensor(tensor)
                    ):
                        split_dict[key] = split_value.contiguous()
                for shard, split_dict in zip(shards, split_dicts):
                    setattr(shard, field_name, split_dict)
            elif isinstance(value, (int, str)):
                for shard in shards:
                    setattr(shard, field_name, value)
        return shards

    @staticmethod
    def _split_sizes(total_size: int, split_count: int) -> list[int]:
        return [
            total_size * (idx + 1) // split_count - total_size * idx // split_count
            for idx in range(split_count)
        ]

    def _rollout_rank_for_env_rank(self, env_rank: int) -> int:
        env_slots = set(
            self._slots_for_rank(
                rank=env_rank,
                rank_to_slots=self.env_rank_to_slots,
                source="env",
            )
        )
        rollout_ranks = [
            rollout_rank
            for rollout_rank, rollout_slots in self.rollout_rank_to_slots.items()
            if env_slots.intersection(rollout_slots)
        ]
        if len(rollout_ranks) != 1:
            raise ValueError(
                f"Eval routing expects env rank {env_rank} to map to one rollout "
                f"rank, got {rollout_ranks}."
            )
        return rollout_ranks[0]

    def _env_ranks_for_rollout_rank(self, rollout_rank: int) -> list[int]:
        rollout_slots = set(
            self._slots_for_rank(
                rank=rollout_rank,
                rank_to_slots=self.rollout_rank_to_slots,
                source="rollout",
            )
        )
        env_ranks = [
            env_rank
            for env_rank, env_slots in self.env_rank_to_slots.items()
            if rollout_slots.intersection(env_slots)
        ]
        return sorted(env_ranks, key=lambda rank: self.env_rank_to_slots[rank][0])

    @staticmethod
    def _eval_route_key(
        data: Observations | Actions,
        rollout_rank: int,
    ) -> tuple[int, int, int, int, int]:
        return (
            int(data.global_step),
            int(data.current_epoch),
            int(data.current_step),
            int(data.stage_id),
            int(rollout_rank),
        )

    def _mark_env_rank_failed(self, env_rank: int) -> None:
        if env_rank not in self.env_rank_to_slots:
            raise ValueError(f"Env rank {env_rank} is not assigned to this worker.")
        if env_rank in self.failed_env_ranks:
            return
        self.failed_env_ranks.add(env_rank)
        slots = self.env_rank_to_slots[env_rank]
        for storage in self.trajectory_storages.values():
            storage.mark_failed_slots(slots)

    async def _mark_env_ranks_failed(self, env_ranks: list[int]) -> None:
        if not env_ranks:
            return

        cancelled_tasks = []
        async with self.storage_lock:
            for env_rank in env_ranks:
                already_failed = env_rank in self.failed_env_ranks
                self._mark_env_rank_failed(env_rank)
                if already_failed:
                    continue
                task = self.observation_tasks.get(env_rank)
                if task is not None and not task.done():
                    task.cancel()
                    cancelled_tasks.append(task)

            for global_step, storage in list(self.trajectory_storages.items()):
                self._seal_if_complete(global_step, storage)

        if cancelled_tasks:
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)

    def _slots_for_rank(
        self,
        *,
        rank: int,
        rank_to_slots: dict[int, list[int]],
        source: str,
    ) -> list[int]:
        if rank not in rank_to_slots:
            raise ValueError(f"{source} rank {rank} is not assigned to this worker.")
        return rank_to_slots[rank]


# Compatibility name for callers that still use the original worker class.
TrajectoryWorker = TrajectoryChannelWorker
