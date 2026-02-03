Replay Buffer 使用教程
==============================

本教程聚焦 `TrajectoryReplayBuffer` 的 **实际使用** 与 **配置建议**。
更完整的设计说明与数据流细节见 API 文档：:doc:`../../apis/replay_buffer`。

快速开始
--------

.. code-block:: python

   from rlinf.data.replay_buffer import TrajectoryReplayBuffer

   buffer = TrajectoryReplayBuffer(
       seed=1234,
       storage_dir="/path/to/buffer",
       enable_cache=True,
       cache_size=5,
       storage_format="pt",
       sample_window_size=100,
       save_trajectories=True,
   )

常用参数
--------

- `storage_dir`：轨迹存储目录；不传则使用日志目录。
- `storage_format`：`pt`（默认）或 `pkl`。
- `enable_cache` / `cache_size`：启用并控制缓存数量，用于提升采样吞吐。
- `sample_window_size`：仅在最近 N 条轨迹内采样；0 表示全量。
- `save_trajectories`：是否落盘保存；为 `False` 时需开启缓存，且不支持 checkpoint。

写入轨迹
--------

.. code-block:: python

   # trajectories 为 List[Trajectory]
   buffer.add_trajectories(trajectories)

写入阶段的关键行为：

- 为每条轨迹生成 `uuid` 与 `trajectory_id`
- 更新 `_trajectory_index` 与计数器
- 在后台线程异步保存轨迹文件（若 `save_trajectories=True`）

采样训练
--------

.. code-block:: python

   batch = buffer.sample(num_chunks=256)
   # batch 形状: [num_chunks, ...]

采样在滑动窗口内随机抽取 transition，并返回与 rollout 对齐的 batch 字典。

保存与加载
----------

.. code-block:: python

   buffer.save_checkpoint("/path/to/ckpt")

   buffer.load_checkpoint(
       "/path/to/ckpt",
       is_distributed=True,
       local_rank=0,
       world_size=4,
   )

`save_trajectories=False` 时不支持 checkpoint 保存。
加载时会从 checkpoint 的 metadata 中恢复 `storage_dir`。
轨迹数据保存格式为 `trajectory_{trajectory_id}_{model_weights_uuid}_{model_update_count}.{storage_format}`。

命令行测试
--------------

.. code-block:: bash

   python rlinf/data/replay_buffer.py \
     --load-path /path/to/buffer \
     --num-chunks 1024 \
     --cache-size 10 \
     --enable-cache

该命令会加载 buffer checkpoint 并进行一次采样，输出 batch 的 key 与 shape。

合并 / 拆分工具
-----------------

脚本位置：`toolkits/replay_buffer/merge_or_split_replay_buffer.py`

.. code-block:: bash

   # 合并多个 rank（按原 trajectory_id 交错）
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/merged \
     --copy

.. code-block:: bash

   # 拆分单个 buffer，取前 N 条轨迹
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/split \
     --split-count 30 \
     --copy

资源释放与重置
--------------

.. code-block:: python

   buffer.close()        # 关闭异步保存线程
   buffer.clear()        # 清空索引与计数
   buffer.clear_cache()  # 清空缓存并关闭线程

实践建议
--------

- **吞吐优先**：开启 `enable_cache`，`cache_size` 设为近期活跃轨迹数。
- **数据新鲜度**：使用 `sample_window_size` 限制采样窗口。
