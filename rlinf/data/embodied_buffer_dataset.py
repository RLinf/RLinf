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

import queue
import threading
import time

from torch.utils.data import IterableDataset

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import concat_batch

logger = get_logger()


class ReplayBufferDataset(IterableDataset):
    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: TrajectoryReplayBuffer,
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        **kwargs,
    ):
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size

    def __iter__(self):
        while True:
            is_ready = True
            if not self.replay_buffer.is_ready(self.min_replay_buffer_size):
                is_ready = False
            if self.demo_buffer is not None and not self.demo_buffer.is_ready(
                self.min_demo_buffer_size
            ):
                is_ready = False

            if is_ready:
                if self.demo_buffer is not None:
                    replay_batch = self.replay_buffer.sample(self.batch_size // 2)
                    demo_batch = self.demo_buffer.sample(self.batch_size // 2)
                    batch = concat_batch(replay_batch, demo_batch)
                else:
                    batch = self.replay_buffer.sample(self.batch_size)
                yield batch

    def close(self):
        del self.replay_buffer
        del self.demo_buffer

    def __del__(self):
        self.close()


class PreloadReplayBufferDataset(ReplayBufferDataset):
    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: TrajectoryReplayBuffer,
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        prefetch_size: int = 10,
    ):
        self._stop_event = threading.Event()

        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

        self.preload_queue = queue.Queue(maxsize=prefetch_size)
        self.sample_thread = None

    def _sample_buffer(self):
        while not self._stop_event.is_set():
            is_ready = True
            if not self.replay_buffer.is_ready(self.min_replay_buffer_size):
                is_ready = False
            if self.demo_buffer is not None and not self.demo_buffer.is_ready(
                self.min_demo_buffer_size
            ):
                is_ready = False

            if is_ready:
                if self.demo_buffer is not None:
                    replay_batch = self.replay_buffer.sample(self.batch_size // 2)
                    demo_batch = self.demo_buffer.sample(self.batch_size // 2)
                    batch = concat_batch(replay_batch, demo_batch)
                else:
                    batch = self.replay_buffer.sample(self.batch_size)
            else:
                time.sleep(3)
                continue

            try:
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                logger.info("Queue is full, skipping sample")
                time.sleep(0.5)
                continue
            except Exception as e:
                logger.error(f"Error in ReplayBufferDataset: {e}")
                break

    def __iter__(self):
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(
                target=self._sample_buffer, daemon=True
            )
            self.sample_thread.start()

        while not self._stop_event.is_set():
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

    def close(self):
        self._stop_event.set()

        thread_timeout = 10
        if self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    f"Sample thread is still alive after {thread_timeout} seconds, force killing"
                )

    def __del__(self):
        if not self._stop_event.is_set():
            self.close()


def replay_buffer_collate_fn(batch):
    return batch[0]
