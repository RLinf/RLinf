Trajectory Channel
==================

使用 ``TrajectoryChannel`` 将具身 rollout 数据从 rollout worker 传给 actor
worker，而不在 environment worker 中组装完整 trajectory。它由一个独立的
``TrajectoryWorker`` 保存 trajectory 状态，并通过 worker 间通信传输数据。

RLinf 何时使用它
-----------------

具身 runner 为 Env/Rollout 到 Actor 的数据流创建 ``TrajectoryChannel``。
Environment 和 Rollout 通过普通 :doc:`Channel <channel>` 交换推理请求。两者将
各自产生的训练数据直接发布给 TrajectoryWorker，后者按 action chunk 合并两条数据流。

各组件的职责如下：

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 组件
     - 职责
   * - ``EnvWorker``
     - 发送 ``PolicyInput``、执行 action，并发布 reward、done flag 和算法相关的
       环境数据。
   * - ``RolloutWorker``
     - 执行策略推理、返回 ``PolicyOutput`` action、发布模型输出，并在环境仿真期间
       处理 terminal observation。
   * - ``TrajectoryWorker``
     - 重组路由分片、合并环境与模型结果、应用滞后更新，并输出 Actor 可消费的数据。
   * - ``ActorWorker``
     - 订阅完成的数据并执行算法对应的更新。

数据流
------

下面展示一个 rollout epoch。``Channel`` 和 ``TrajectoryChannel`` 使用独立的
数据流，因此 trajectory 组装不会向策略响应通路增加 payload。

.. code-block:: text

   EnvWorker              RolloutWorker         TrajectoryWorker       ActorWorker
       | PolicyInput            |                       |                    |
       |----------------------->|                       |                    |
       |                        | PolicyStep            |                    |
       |                        |---------------------->|                    |
       |     PolicyOutput       |                       |                    |
       |<-----------------------|                       |                    |
       | simulate action        |                       |                    |
       | PolicyInput + completion                      |                    |
       |----------------------->| EnvStepResult         |                    |
       |                        |---------------------->| append when ready  |
       |          ... repeat for each action chunk ... |                    |
       | PolicyInput + final completion                |                    |
       |----------------------->| EnvStepResult         |                    |
       |                        |---------------------->| append final chunk |
       |                        |                       | infer completion   |
       |                        |                       | from received keys |
       |                        |                       | trajectory / batch |
       |                        |                       |------------------->|

每个 policy input 都可以携带上一个 action 的 ``PolicyCompletion``。普通 transition
直接复用当前推理的 model input；需要 terminal 处理时，RolloutWorker 会额外执行一次
terminal inference，然后发布完整的 ``EnvStepResult``。静态路由使用 final policy
input 完成最后一个 action，且不再生成新 action；decoupled mode 则由下一条 trajectory
的 bootstrap policy input 携带该 completion。RolloutWorker 不保存跨请求的
trajectory 状态。

事件与结果
----------

``TrajectoryChannel.publish()`` 接受以下内部事件：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 事件
     - 含义
   * - ``TrajectoryStart``
     - 一个逻辑 environment source 的初始 done 和 termination 状态。
   * - ``PolicyStep``
     - Rollout 推理产生的 observation、action、log-probability、value、version 和
       model forward input。
   * - ``EnvStepResult``
     - ``PolicyCompletion`` 携带的 environment outcome，以及 RolloutWorker 补齐的
       next observation 和 next model input。用于 truncation reward 的标量
       bootstrap value 与 trajectory 末尾追加的完整 value boundary 分开保存。

TrajectoryWorker 根据 ``TrajectoryKey`` 自行推导完成状态，不再接收单独的 epoch-end
或 trajectory-end 事件。Pipeline 模式下，同一 ``(step_id, epoch_id)`` 收齐
``source_count * chunk_count`` 个完整 key 后 flush 一个 epoch；其他模式下，同一
``step_id`` 收齐 ``source_count * rollout_epoch * chunk_count`` 个完整 key 后 flush
一个训练 step。

普通训练中，``subscribe()`` 返回完整 ``Trajectory``。异步 Actor 使用
``try_subscribe()``；没有已组装数据时，它在启动 P2P receive 前抛出
``asyncio.QueueEmpty``。Online LeRobot DAgger 中，channel 返回已完成的 episode
字典。启用 training pipeline 后，Actor 从 ``actor:<rank>`` 队列订阅准备好的
micro-batch。

路由与顺序
----------

``TrajectoryKey`` 使用 ``step_id``、``epoch_id``、``env_rank``、``stage_id`` 和
``chunk_id`` 标识一个 action chunk。``TrajectorySource`` 额外记录 source shard 的 batch
size 与 offset。Channel 路由可以将一个 environment batch 拆给多个 RolloutWorker，
也可以合并无关 source。TrajectoryWorker 先按 offset 恢复原始 batch，再合并具有相同
key 的事件。

Decoupled mode 下，policy input 可以交给任意空闲 RolloutWorker。上一个 action 的
completion 与下一个请求一起传递，因此不依赖 RolloutWorker affinity 或本地 pending
状态。普通 policy request 会记录 action 的返回路由。在 trajectory 边界，最后一个
completion 会挂到下一条 bootstrap request：它既补全上一个 key，也请求下一个 key
的第一个 action。TrajectoryWorker 按 key 合并 ``PolicyStep`` 和完整的
``EnvStepResult``。

完成的数据按照 actor world size 分片。这带来两个性质：

* 一个逻辑 source 的 segment 始终进入同一个 collector，不会跨 worker 组装同一条
  trajectory。
* 输出 shard 数可以被逻辑 source 数整除，因此每个 source 产生相同数量的 Actor
  可消费数据项。

Pipeline 模式使用 ``CommMapper`` 将每个逻辑 source 映射到 actor rank。它在 epoch
完成后计算 advantage，按配置在 Actor 对应的 batch 上执行 normalization，然后向
Actor 队列输出 micro-batch。

Placement
---------

Trajectory 组装目前只使用一个物理 ``TrajectoryWorker``。通过
``cluster.trajectory_node_rank`` 选择其节点：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     trajectory_node_rank: 1

默认使用 node rank ``0``。取值必须位于 ``[0, cluster.num_nodes - 1]``。当节点间
网络较慢时，将该 worker 放在主要 trajectory 流量附近。这个配置不会改变
``ChannelWorker`` 的 placement 或普通 ``Channel`` 的路由。

通信语义
--------

``publish()``、``subscribe()`` 和 ``try_subscribe()`` 是仅供 worker 内部使用的
接口。每次操作使用小型 Ray RPC 协调，并使用 RLinf P2P ``send``/``recv`` 传输
payload。在 RLinf ``Worker`` 外调用这些接口都会抛出 ``RuntimeError``。

两个接口都支持同步和异步完成。传入 ``async_op=True`` 可获得 ``AsyncWork``
句柄，再调用 ``wait()`` 或 ``async_wait()``。完成状态同时覆盖控制操作与 payload
传输；任一侧的错误都会传递给调用者。

算法行为
--------

Worker 根据配置选择保存字段，而不是为每个算法暴露不同的 channel 接口：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 模式
     - 额外行为
   * - PPO / GRPO / NFT
     - 组装配置的 loss 和 advantage 函数所需的 reward、done flag、policy input、
       log-probability 与 value。
   * - SAC / CrossQ
     - 当 ``rollout.collect_transitions`` 启用时保留 observation transition，并为
       replay buffer 输出 trajectory。
   * - DAgger
     - 应用 intervention action 和 flag；online LeRobot 模式输出完成的 episode，
       而非标准 trajectory。
   * - RLT
     - 从模型 forward input 中提取并保存 RLT transition observation。
   * - Training pipeline
     - 完成每个 epoch、计算 advantage、路由 Actor 专属 batch，并按照配置拆分
       micro-batch。

Training pipeline 当前不支持 online LeRobot 数据或 decoupled environment 模式。
无效组合会在 ``TrajectoryWorker`` 初始化时直接失败。

运行检查
--------

如果训练停止推进，先确定阻塞边界，再调整队列或 placement 配置：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 检查项
   * - Rollout 等待 policy input
     - 检查普通 Env 到 Rollout 的 ``Channel`` 通路；此时还没有产生 trajectory
       事件。
   * - Rollout 完成后 Actor 仍等待
     - 确认 Env 为每个 key 发送了 ``PolicyCompletion``，Rollout 发布了匹配的
       ``PolicyStep`` 和完整的 ``EnvStepResult``。
   * - Pipeline Actor 等待
     - 确认 Actor 使用 ``actor:<rank>`` 订阅，并且配置的 batch size 可以被
       ``actor.micro_batch_size`` 整除。
   * - 启动时 placement 失败
     - 对照 ``cluster.num_nodes`` 检查 ``cluster.trajectory_node_rank``，并确认 Ray
       能识别该 node rank。

普通队列通信参见 :doc:`Channel <channel>`，集群资源模型参见
:doc:`Placement <placement>`。
