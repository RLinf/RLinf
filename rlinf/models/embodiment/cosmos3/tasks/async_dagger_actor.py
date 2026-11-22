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
import threading
import time

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
    AsyncEmbodiedDAGGERFSDPPolicy,
)

from .dagger_actor import Cosmos3DAggerActor


class Cosmos3AsyncDAggerActor(AsyncEmbodiedDAGGERFSDPPolicy, Cosmos3DAggerActor):
    """Train the Cosmos student while the upstream receiver collects episodes."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.should_stop = False
        self._lerobot_ingest_lock = threading.RLock()
        self._lerobot_receiver_drained = threading.Event()
        self._recv_error: Exception | None = None

    def _check_receiver(self):
        if self._recv_error is not None:
            raise RuntimeError(
                "The DAgger trajectory receiver failed."
            ) from self._recv_error

    async def recv_rollout_trajectories(self, input_channel):
        self._check_receiver()
        if not self.should_stop and input_channel is not None:
            await super().recv_rollout_trajectories(input_channel)

    def _recv_lerobot_thread_main(self, input_channel):
        try:
            try:
                super()._recv_lerobot_thread_main(input_channel)
            finally:
                self._lerobot_receiver_drained.set()
                self._archive_pending_lerobot_episodes()
        except Exception as error:
            self._recv_error = error

    def _append_lerobot_episode(self, ep_frames):
        # Checkpointing must not rotate the shard while an episode is being appended.
        with self._lerobot_ingest_lock:
            return super()._append_lerobot_episode(ep_frames)

    def _archive_pending_lerobot_episodes(self):
        with self._lerobot_ingest_lock:
            return super()._archive_pending_lerobot_episodes()

    def _recv_rollout_thread_main(self, input_channel):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        try:
            while not self.should_stop:
                received_any = False
                for _ in range(split_num):
                    if self.should_stop:
                        break
                    try:
                        trajectory = input_channel.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    self._recv_queue.put(trajectory)
                    received_any = True
                if not received_any:
                    time.sleep(0.1)
        except Exception as error:
            self._recv_error = error

    async def _wait_for_replay_buffer_ready(self, min_buffer_size: int):
        while not self.should_stop:
            self._check_receiver()
            self._drain_received_trajectories(
                max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 1024)
            )
            if await self.replay_buffer.is_ready_async(min_buffer_size):
                return
            await asyncio.sleep(1)

    @Worker.timer("run_training")
    async def run_training(self):
        self._check_receiver()
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        if not self.enable_online_lerobot:
            await self._wait_for_replay_buffer_ready(
                self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
            )
        elif self._skip_lerobot_training():
            return {}

        if self.should_stop:
            return {}
        self._begin_sft_updates()
        metrics = {}
        for _ in range(self.cfg.algorithm.get("update_epoch", 1)):
            await asyncio.sleep(0)
            append_to_dict(metrics, self.update_one_epoch())
        return self._finish_sft_updates(metrics)

    def save_checkpoint(self, save_base_path, step):
        self._check_receiver()
        return super().save_checkpoint(save_base_path, step)

    async def stop(self):
        await super().stop()
        if self.enable_online_lerobot:
            receiver = getattr(self, "_recv_lerobot_thread", None)
            if receiver is None:
                await asyncio.to_thread(self._archive_pending_lerobot_episodes)
            elif receiver.is_alive():
                if not self._lerobot_receiver_drained.is_set():
                    raise TimeoutError(
                        "The LeRobot receiver is still processing episodes. "
                        "It will finalize the archive when it exits."
                    )
                # The input has drained; allow the final disk write to finish.
                await asyncio.to_thread(receiver.join)
        self._check_receiver()
