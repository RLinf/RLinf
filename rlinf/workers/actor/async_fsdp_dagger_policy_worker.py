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

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


class AsyncEmbodiedDAGGERFSDPPolicy(EmbodiedDAGGERFSDPPolicy):
    should_stop = False

    def __init__(self, cfg):
        super().__init__(cfg)
        if self._cosmos_training is not None:
            self.should_stop = False
            self._lerobot_ingest_lock = threading.RLock()
            self._lerobot_receiver_drained = threading.Event()
            self._recv_error: Exception | None = None

    def _check_receiver(self):
        if self._cosmos_training is not None and self._recv_error is not None:
            raise RuntimeError(
                "The DAgger trajectory receiver failed."
            ) from self._recv_error

    async def recv_rollout_trajectories(self, input_channel):
        if self._cosmos_training is not None:
            self._check_receiver()
            if self.should_stop or input_channel is None:
                return
        if self.enable_online_lerobot:
            if (
                getattr(self, "_recv_lerobot_thread", None) is None
                or not self._recv_lerobot_thread.is_alive()
            ):
                if input_channel is not None:
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
        if self._cosmos_training is None:
            return self._receive_lerobot_episodes(input_channel)
        try:
            try:
                self._receive_lerobot_episodes(input_channel)
            finally:
                self._lerobot_receiver_drained.set()
                self._archive_pending_lerobot_episodes()
        except Exception as error:
            self._recv_error = error

    def _receive_lerobot_episodes(self, input_channel):
        """Background thread: receive episode batches from EnvWorker.

        Each received episode is appended to the actor's live in-memory
        LeRobot dataset immediately. Archive writes are flushed separately by
        ``algorithm.dagger.online_lerobot.finalize_interval`` and are not part of training
        readiness.
        """
        while not self.should_stop:
            received_any = self._recv_lerobot_episodes_from_channel(input_channel)
            if self.dataset.is_ready():
                self._ensure_lerobot_loader()
            if not received_any:
                time.sleep(0.1)

    def _append_lerobot_episode(self, ep_frames):
        if self._cosmos_training is None:
            return super()._append_lerobot_episode(ep_frames)
        # Checkpointing must not rotate the shard while an episode is being appended.
        with self._lerobot_ingest_lock:
            return super()._append_lerobot_episode(ep_frames)

    def _archive_pending_lerobot_episodes(self):
        if self._cosmos_training is None:
            return super()._archive_pending_lerobot_episodes()
        with self._lerobot_ingest_lock:
            return super()._archive_pending_lerobot_episodes()

    def _recv_rollout_thread_main(self, input_channel):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        if self._cosmos_training is None:
            while not self.should_stop:
                for _ in range(split_num):
                    trajectory = input_channel.get()
                    self._recv_queue.put(trajectory)
            return

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

        intervene_traj_list = []
        for traj in recv_list:
            intervene_trajs = traj.extract_intervene_traj(mode="all")
            if intervene_trajs is not None:
                intervene_traj_list.extend(intervene_trajs)
        if intervene_traj_list:
            self.replay_buffer.add_trajectories(intervene_traj_list)

    async def _wait_for_replay_buffer_ready(self, min_buffer_size: int):
        while self._cosmos_training is None or not self.should_stop:
            self._check_receiver()
            self._drain_received_trajectories(
                max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 1024)
            )
            if await self.replay_buffer.is_ready_async(min_buffer_size):
                return
            await asyncio.sleep(1)

    async def _prepare_training(self) -> bool:
        self._check_receiver()
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        if not self.enable_online_lerobot:
            min_buffer_size = self.cfg.algorithm.replay_buffer.get(
                "min_buffer_size", 100
            )
            await self._wait_for_replay_buffer_ready(min_buffer_size)
        elif self._skip_lerobot_training():
            return False
        return self._cosmos_training is None or not self.should_stop

    @Worker.timer("run_training")
    async def run_training(self):
        """Run async DAgger updates with replay-buffer samples."""
        if not await self._prepare_training():
            return {}

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            await asyncio.sleep(0)
            self._record_sft_update(metrics, self.update_one_epoch())

        return self._finish_sft_updates(metrics)

    def save_checkpoint(self, save_base_path, step):
        self._check_receiver()
        return super().save_checkpoint(save_base_path, step)

    async def stop(self):
        self.should_stop = True
        recv_thread = getattr(self, "_recv_rollout_thread", None)
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
        lerobot_thread = getattr(self, "_recv_lerobot_thread", None)
        if lerobot_thread is not None and lerobot_thread.is_alive():
            await asyncio.to_thread(lerobot_thread.join, 5)
        if self._cosmos_training is None:
            return
        if self.enable_online_lerobot:
            if lerobot_thread is None:
                await asyncio.to_thread(self._archive_pending_lerobot_episodes)
            elif lerobot_thread.is_alive():
                if not self._lerobot_receiver_drained.is_set():
                    raise TimeoutError(
                        "The LeRobot receiver is still processing episodes. "
                        "It will finalize the archive when it exits."
                    )
                # The input has drained; allow the final disk write to finish.
                await asyncio.to_thread(lerobot_thread.join)
        self._check_receiver()
