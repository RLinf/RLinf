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
    """Dataset that samples batches from replay and demonstration buffers.

    This dataset provides an infinite iterator that yields batches sampled from
    a replay buffer and optionally a demonstration buffer. When both buffers are
    provided, batches are composed of half replay samples and half demonstration
    samples.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing offline demonstration trajectories
            and online human-in-the-loop trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: TrajectoryReplayBuffer,
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        **kwargs,
    ):
        """Initializes the ReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch. When demo_buffer is
                provided, batch_size // 2 samples come from each buffer.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            **kwargs: Additional keyword arguments (unused, for compatibility).
        """
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size

    def __iter__(self):
        """Returns an infinite iterator that yields batches.

        Waits until both buffers (if demo_buffer is provided) reach their
        minimum size requirements before yielding batches. When ready, samples
        from replay buffer only or from both replay and demo buffers.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
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
        """Releases references to replay and demo buffers."""
        del self.replay_buffer
        del self.demo_buffer

    def __del__(self):
        """Destructor that ensures buffers are cleaned up."""
        self.close()


class PreloadReplayBufferDataset(ReplayBufferDataset):
    """Dataset that prefetches batches from replay and demo buffers in background.

    This dataset extends ReplayBufferDataset by prefetching batches in a
    background thread, which can improve throughput by overlapping sampling
    with training. Batches are stored in a queue of configurable size.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing demonstration trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
        prefetch_size: Maximum number of batches to prefetch and store in queue.
        preload_queue: Queue holding prefetched batches.
        sample_thread: Background thread that samples batches.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: TrajectoryReplayBuffer,
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        prefetch_size: int = 10,
    ):
        """Initializes the PreloadReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch. When demo_buffer is
                provided, batch_size // 2 samples come from each buffer.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            prefetch_size: Maximum number of batches to prefetch and store in
                the queue. Defaults to 10.
        """
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
        """Background thread target that continuously samples batches.

        Runs in a loop until stop event is set. Waits for buffers to be ready,
        samples batches, and puts them in the preload queue. If the queue is
        full, skips the sample and retries. Sleeps when buffers are not ready
        or when errors occur.
        """
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
        """Returns an iterator that yields prefetched batches.

        Starts the background sampling thread on first call. Retrieves batches
        from the preload queue and yields them. Stops when the stop event is set.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
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
        """Stops the background sampling thread and cleans up resources.

        Sets the stop event and waits up to 10 seconds for the sampling thread
        to terminate. Logs a warning if the thread does not terminate in time.
        """
        self._stop_event.set()

        thread_timeout = 10
        if self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    f"Sample thread is still alive after {thread_timeout} seconds, force killing"
                )

    def __del__(self):
        """Destructor that ensures the sampling thread is stopped."""
        if not self._stop_event.is_set():
            self.close()


def replay_buffer_collate_fn(batch):
    """Collate function for DataLoader that returns the first batch element.

    Since the dataset already yields complete batches, this function simply
    extracts the batch from the list wrapper added by DataLoader.

    Args:
        batch: List containing a single batch dictionary.

    Returns:
        The unwrapped batch dictionary.
    """
    return batch[0]
