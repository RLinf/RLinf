Replay Buffer
======================

本节介绍 RLinf 中的 `TrajectoryReplayBuffer` 设计思路、核心数据结构与采样流程。
该实现是 **轨迹级（trajectory-based）** 的 replay buffer，直接存储形状为 `[T, B, ...]`
的批量轨迹，不做 episode 级拆分，每个 trajectory 会包含多个 episode；采样时按 chunk step 粒度从 trajectory 中抽取。

设计目标
-------------

- **低内存消耗**：buffer 维护轨迹数据的索引信息，避免内存占用过高。
- **低 I/O 压力**：轨迹以文件保存，索引与元信息分离，异步写盘降低训练阻塞。
- **高吞吐采样**：按窗口采样最新轨迹，向量化映射全局样本索引到轨迹内位置。
- **易扩展**：支持缓存策略、不同存储格式与分布式加载。
- **训练友好**：采样输出与 rollout batch 对齐（字典形式，键与张量结构保持一致）。

核心数据结构
-----------------

轨迹索引
^^^^^^^^^^

每条轨迹在索引中保存：

- `uuid`：唯一标识
- `trajectory_id`：递增 ID
- `num_samples`：轨迹内样本总数（`T * B`）
- `shape`：轨迹张量形状
- `max_episode_length`：最大 episode 长度

索引与元数据分别写入：

- `trajectory_index.json`：轨迹索引
- `metadata.json`：buffer 元数据（总样本数、格式、seed 等）

轨迹缓存
^^^^^^^^^^

`TrajectoryCache` 为 FIFO 缓存，缓存 **flatten 后** 的轨迹字典：

- 轨迹以 `[T, B, ...]` 存储
- flatten 后视为 `[T*B, ...]`，便于按 transition 抽样
- cache 仅保留最近若干条轨迹，避免频繁读盘

采样流程
--------

采样接口为 `sample`，参数为 `num_chunks`，返回值为 `[num_chunks, ...]` 的 batch。

1. **滑动窗口**：只在最近 `sample_window_size` 条轨迹内采样（可设为 0 表示全量）。
2. **均匀采样**：在窗口总样本数范围内随机采样索引。
3. **索引映射**：将全局索引映射到具体轨迹与轨迹内位置（`bucketize`）。
4. **批量采样**：一次加载轨迹并批量取样，构建 `[num_chunks, ...]` 的 batch。

这一设计避免频繁加载同一轨迹，同时保证采样分布平滑。

数据流程说明
------------------

模型和环境交互的多轮数据会通过 Rollout Worker 收集，并转换为 `Trajectory`，然后通过 Channel 发送给 Actor Worker。
Actor Worker 将 `Trajectory` 存入 `TrajectoryReplayBuffer`，并进行采样训练。


Rollout Worker 收集轨迹数据流程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rollout worker按时间累积 step 结果，收集轨迹的过程。其中
`[obs₀, obs₁]` 为一个transition, `[act₀, r₀, dones, ... ]` 为一个 **ChunkStepResult**：

.. code-block:: text

   time →
   [ obs₀, obs₁, act₀, r₀, dones₀, ... ] ──┐
   [ obs₁, obs₂, act₁, r₁, dones₁, ... ] ──┼── EmbodiedRolloutResult── Trajectory 
   [ obs₂, obs₃, act₂, r₂, dones₂, ... ] ──┤                                │
   [ obs₃, obs₄, act₃, r₃, dones₃, ... ] ──┘                           Channel(put)   

Actor Worker 保存轨迹数据流程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Actor worker 从 Channel 接收 `Trajectory`，存入 `TrajectoryReplayBuffer`，并进行采样训练：

.. code-block:: text

   Channel(get) ── Trajectory ── ReplayBuffer
                                     │
                                     ├─ add_trajectories(trajectories)
                                     │     ├─ 生成 uuid + trajectory_id
                                     │     ├─ 更新 _trajectory_index / 计数器
                                     │     └─ 线程池异步保存轨迹文件
                                     │
                                     └─ sample(num_chunks) ──▶ 训练 batch

存储与异步写盘
--------------

轨迹保存支持两种格式：

- `pt`：`torch.save` 方式，默认格式
- `pkl`：`pickle` 方式

写盘在单线程 `ThreadPoolExecutor` 中异步执行，写完后再更新元数据与索引。
这样可以降低训练过程中 I/O 对吞吐的影响。

检查点与分布式加载
----------------------

支持将 buffer 元信息保存为 checkpoint，并在分布式场景下按 rank 分片加载：

- `load_path`：包含 metadata 和轨迹文件的 checkpoint 目录
- `is_distributed`：是否启用分片加载
- `local_rank`：当前 rank 只加载自己的那一份轨迹（0-based）
- `world_size`：总 rank 数（分片数）
- 维持各分片的 `size`、`total_samples`、`trajectory_counter` 一致性

使用建议
------------

- **长轨迹场景**：优先使用窗口采样，减少旧数据干扰。
- **高并发训练**：启用缓存提升采样吞吐，避免频繁读盘。
- **无持久化需求**：可设置 `auto_save=False`，保存 checkpoint 时会把缓存轨迹与 metadata 一并写入保存路径。
