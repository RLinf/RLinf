Replay Buffer 接口
==================

``TrajectoryReplayBuffer`` 以 ``[T, B, ...]`` 形状存储批量轨迹，支持滑动窗口 chunk 采样、FIFO 内存缓存与分布式 checkpoint 加载。日常使用示例、CLI 工具与可视化脚本见 :doc:`../../concepts/replay_buffer`。

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
