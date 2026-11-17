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
import queue
import threading
import time

from rlinf.utils.metric_utils import compute_split_num
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


class AsyncEmbodiedDAGGERFSDPPolicy(EmbodiedDAGGERFSDPPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.should_stop = False
        self._model_lock = asyncio.Lock()
        self._lerobot_ingest_lock = threading.RLock()
        self._recv_error: Exception | None = None

    def _check_receiver(self):
        if self._recv_error is not None:
            raise RuntimeError(
                "The DAgger trajectory receiver failed."
            ) from self._recv_error

    async def recv_rollout_trajectories(self, input_channel):
        self._check_receiver()
        if self.should_stop or input_channel is None:
            return
        if self.enable_online_lerobot:
            if (
                getattr(self, "_recv_lerobot_thread", None) is None
                or not self._recv_lerobot_thread.is_alive()
            ):
                self._recv_lerobot_thread = threading.Thread(
                    target=self._recv_lerobot_thread_main,
                    args=(input_channel,),
                    daemon=True,
                )
                self._recv_lerobot_thread.start()
            return
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_lerobot_thread_main(self, input_channel):
        """Background thread: receive episode batches from EnvWorker.

        Each received episode is appended to the actor's live in-memory
        LeRobot dataset immediately. Archive writes are flushed separately by
        ``algorithm.dagger.online_lerobot.finalize_interval`` and do not gate
        training readiness.
        """
        try:
            while not self.should_stop:
                received_any = self._recv_lerobot_episodes_from_channel(input_channel)
                if self.dataset.is_ready():
                    self._ensure_lerobot_loader()
                if not received_any:
                    time.sleep(0.1)
        except Exception as error:
            self._recv_error = error

    def _append_lerobot_episode(self, ep_frames):
        # A checkpoint must not rotate the archive path during a memory append.
        with self._lerobot_ingest_lock:
            return super()._append_lerobot_episode(ep_frames)

    def _archive_pending_lerobot_episodes(self):
        with self._lerobot_ingest_lock:
            # Publish training readiness before the archive's potentially slow I/O.
            if not self.should_stop and self.dataset.is_ready():
                self._ensure_lerobot_loader()
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

    def _drain_received_trajectories(self, max_trajectories: int | None = None):
        if getattr(self, "_recv_queue", None) is None:
            return

        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return

        self.recv_buffer_rollout_trajectories(recv_list)

    async def _wait_for_replay_buffer_ready(self, min_buffer_size: int):
        while not self.should_stop:
            self._check_receiver()
            self._drain_received_trajectories(
                max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 1024)
            )
            if await self.replay_buffer.is_ready_async(min_buffer_size):
                return
            await asyncio.sleep(0.1)

    async def run_training(self):
        """Use the synchronous DAgger update contract after async reception."""
        self._check_receiver()
        if not self.enable_online_lerobot:
            min_buffer_size = self.cfg.algorithm.replay_buffer.get(
                "min_buffer_size", 100
            )
            await self._wait_for_replay_buffer_ready(min_buffer_size)

        async with self._model_lock:
            self._check_receiver()
            if self.should_stop:
                return {}
            return super().run_training()

    async def sync_model_to_rollout(self):
        # State dict tensors can alias model storage throughout the async send.
        async with self._model_lock:
            return await super().sync_model_to_rollout()

    async def save_checkpoint(self, save_base_path, step):
        async with self._model_lock:
            self._check_receiver()
            return super().save_checkpoint(save_base_path, step)

    async def load_checkpoint(self, load_base_path):
        async with self._model_lock:
            return super().load_checkpoint(load_base_path)

    async def set_global_step(self, global_step):
        async with self._model_lock:
            return super().set_global_step(global_step)

    async def stop(self):
        self.should_stop = True
        recv_thread = getattr(self, "_recv_rollout_thread", None)
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
        lerobot_thread = getattr(self, "_recv_lerobot_thread", None)
        if lerobot_thread is not None and lerobot_thread.is_alive():
            await asyncio.to_thread(lerobot_thread.join, 5)
        if self.enable_online_lerobot:
            await asyncio.to_thread(self._archive_pending_lerobot_episodes)
        self._check_receiver()
