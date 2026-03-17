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
       enable_cache=True,
       cache_size=5,
       sample_window_size=100,
       auto_save=True,
       auto_save_path="/path/to/buffer",
       trajectory_format="pt",
   )

常用参数
--------

- `enable_cache` / `cache_size`：启用并控制缓存数量，用于提升采样吞吐。
- `sample_window_size`：仅在最近 N 条轨迹内采样；0 表示全量。
- `auto_save`：是否自动落盘；为 `False` 时仅缓存并在保存 checkpoint 时落盘。
- `auto_save_path`：开启 auto_save 时的轨迹存储目录。
- `trajectory_format`：`pt`（默认）或 `pkl`。

写入轨迹
--------

.. code-block:: python

   # trajectories 为 List[Trajectory]
   buffer.add_trajectories(trajectories)

写入阶段的关键行为：

- 为每条轨迹生成 `uuid` 与 `trajectory_id`
- 更新 `_trajectory_index` 与计数器
- 在后台线程异步保存轨迹文件（若 `auto_save=True`）

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
       load_path="/path/to/ckpt",
       is_distributed=True,
       local_rank=0,
       world_size=4,
   )

保存 checkpoint 时会把缓存轨迹与 metadata 一并写入 checkpoint 路径。
加载时需要设置 `load_path` 指向包含 metadata 和轨迹文件的 checkpoint 目录。
轨迹数据保存格式为 `trajectory_{trajectory_id}_{model_weights_uuid}_{model_update_count}.{trajectory_format}`。

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

可视化工具
----------

RLinf 提供了交互式可视化工具，用于检查 replay buffer 保存的轨迹数据。

功能特性
~~~~~~~~

- **延迟加载**：使用 `TrajectoryReplayBuffer` 按需加载轨迹，避免将所有数据加载到内存
- **自动切换**：到达最后一帧时自动前进到下一条轨迹
- **跳转轨迹**：在文本框中输入轨迹 ID 直接跳转
- **多相机支持**：查看主相机、腕部相机或额外视角相机
- **批次导航**：在 B > 1 时可在批次索引间导航
- **SSH/无显示器模式支持**：保存图像以便查看

交互模式（本地机器有显示）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0

键盘导航：

- ``←`` / ``→`` (或 ``p`` / ``n``)：上一步/下一步（在边界自动切换轨迹）
- ``↑`` / ``↓``：上一条/下一条轨迹
- ``b`` / ``v``：在批次索引间切换（如果 B > 1）
- ``s``：保存当前视图到图像文件
- ``Home`` / ``End``：跳转到第一步/最后一步
- ``q`` / ``Esc``：退出
- 在文本框中输入轨迹 ID 直接跳转

SSH/无显示器模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用无显示器交互脚本：

.. code-block:: bash

   python toolkits/replay_buffer/visualize_headless.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --output viz.png

然后在 VSCode 中：

1. 在编辑器中打开 ``viz.png``
2. 使用命令行提示进行导航
3. 图像自动更新 - VSCode 会显示变化

**命令：**

- ``n`` / ``next``：下一步（在末尾自动切换到下一条轨迹）
- ``p`` / ``prev``：上一步
- ``nt`` / ``nexttraj``：下一条轨迹
- ``pt`` / ``prevtraj``：上一条轨迹
- ``j <id>``：跳转到轨迹 ID（例如 ``j 42``）
- ``info``：显示当前位置
- ``q`` / ``quit``：退出

带 X11 转发的自动保存
~~~~~~~~~~~~~~~~~~~~~

如果启用了 X11 转发：

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --save_image --output viz.png

使用键盘导航，图像文件会自动更新。在 VSCode 中打开 ``viz.png`` 查看当前视图。

静态图像导出
~~~~~~~~~~~~

保存单帧而不进行交互：

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --save_image --output viz.png --no_display

显示信息
~~~~~~~~

可视化工具显示：

- **当前观察** （左面板）
- **下一个观察** （右面板）
- **轨迹 ID** 和索引位置
- **步骤** 和 **批次** 索引
- 每个转换的 **动作**、 **奖励** 和 **完成** 标志

查看不同相机角度
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 主相机（默认）
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera main_images

   # 腕部相机
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera wrist_images

   # 额外视角相机
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera extra_view_images

注意事项
~~~~~~~~

- 工具使用 ``TrajectoryReplayBuffer.load_checkpoint()`` 读取元数据和索引文件
- 轨迹使用 ``_load_trajectory()`` 按需延迟加载
- 缓存大小设置为 5 条轨迹以平衡内存和性能
- 当在轨迹 i 的最后一帧按 ``→`` 时，会自动跳转到轨迹 i+1 的第 0 帧
- 当在轨迹 i 的第一帧按 ``←`` 时，会自动跳转到轨迹 i-1 的最后一帧
- 图像文件以 150 DPI 保存，在保持良好质量的同时控制文件大小
