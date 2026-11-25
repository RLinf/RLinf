Replay Buffer Interface
=======================

``TrajectoryReplayBuffer`` stores batched ``[T, B, ...]`` trajectories with windowed chunk-step sampling, optional FIFO caching, and distributed checkpointing. For practical usage, CLI utilities, and visualization tools, see :doc:`../../concepts/replay_buffer`.

TrajectoryReplayBuffer
----------------------

.. autoclass:: rlinf.data.storage.replay.TrajectoryReplayBuffer
   :members: add_trajectories, sample, save_checkpoint, load_checkpoint, clear, clear_cache, close, is_ready
   :member-order: bysource

TrajectoryCache
---------------

.. autoclass:: rlinf.data.storage.replay.TrajectoryCache
   :members:
   :member-order: bysource

Replay Buffer Datasets
----------------------

.. autoclass:: rlinf.data.storage.replay.ReplayBufferDataset
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.storage.replay.PreloadReplayBufferDataset
   :members:
   :member-order: bysource
