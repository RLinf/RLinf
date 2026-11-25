M2Flow 编程流程
===============

RLinf 采用宏观到微观流（M2Flow）编程模型：你在顶层只用 worker 组与队列描述 RL 算法流程，把设备映射、流水线切分和集合通信交给运行时处理。本页讲解如何在 Python 中组合 worker 组、队列与 runner；四阶段运行生命周期见 :doc:`execution_flow`，Hydra YAML 配置项见 :doc:`../reference/configuration`。

宏观逻辑与微观执行分离
----------------------

在传统分布式 RL 代码里，算法流程往往与硬件部署耦合在一起。一旦把 rollout 移到独立 GPU，或者让环境步进与模型推理流水线重叠，训练主循环就得跟着大改。

M2Flow 将这两层职责分开：

- **宏观流程（Macro）**：入口脚本和 runner 只声明有哪些角色以及各阶段的先后顺序，即创建 worker 组、用异步队列连接它们，并按阶段发起组调用。
- **微观执行（Micro）**：placement 策略、队列缓冲与集合通信组决定每个 rank 落在哪张卡上、批次如何切分以及张量何时跨设备传输。

宏观代码只面向 ``WorkerGroup`` 与 ``Channel`` 句柄编程，因此在共享式、分离式与混合式部署之间切换时，只需修改 YAML 中的 ``cluster.component_placement``，无需改动 runner 代码。

在入口脚本中编排 Worker 组
--------------------------

入口脚本先根据配置构建集群视图，再按 placement 策略拉起每个组件，最后把各组句柄交给 runner。下面节选自 ``examples/embodiment/train_embodied_agent.py`` 的代码展示了具身训练的标准入口结构：

.. code-block:: python

   cluster = Cluster(cluster_cfg=cfg.cluster)
   component_placement = HybridComponentPlacement(cfg, cluster)

   actor_placement = component_placement.get_strategy("actor")
   actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
       cluster, placement_strategy=actor_placement
   )

   rollout_placement = component_placement.get_strategy("rollout")
   rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
       cluster, placement_strategy=rollout_placement
   )

   env_placement = component_placement.get_strategy("env")
   env_group = EnvWorker.create_group(cfg).launch(
       cluster, placement_strategy=env_placement
   )

   runner = EmbodiedRunner(
       cfg=cfg,
       actor=actor_group,
       rollout=rollout_group,
       env=env_group,
   )
   runner.init_workers()
   runner.run()

这段代码体现出两个特点：无论运行在单卡还是多节点上，所有 worker 组都通过统一的 ``create_group(...).launch(...)`` 启动；runner 只接收抽象的组句柄，不感知具体 GPU 编号，因此同一套 runner 代码适用于任意合法的 placement 配置（配置语法见 :doc:`placement`）。

用异步句柄与 Channel 驱动训练循环
---------------------------------

调用 ``WorkerGroup`` 上的方法会立即返回异步 future 句柄，不会阻塞主控进程。Runner 借助这一特性并发启动生产者与消费者阶段，用 ``Channel`` 传递数据，只在阶段边界调用 ``.wait()`` 同步。

在具身训练循环中，每一步迭代包含四个宏观动作：

.. code-block:: python

   for step in range(max_steps):
       update_rollout_weights()
       generate_rollouts()
       actor_group.compute_advantages_and_returns().wait()
       actor_group.run_training().wait()

其中权重同步与 rollout 采样借助异步句柄和队列实现微观并发：

1. **成对发起权重同步**：同时启动 rollout 接收端与 actor 发送端，传输结束后统一等待。

   .. code-block:: python

      def update_rollout_weights():
          rollout_futures = rollout_group.sync_model_from_actor()
          actor_futures = actor_group.sync_model_to_rollout()
          actor_futures.wait()
          rollout_futures.wait()

2. **队列驱动的细粒度流水线**：并行拉起环境交互、rollout 推理与 actor 轨迹接收三个阶段，通过共享队列持续交换观测、动作和轨迹，直到本轮采样结束。

   .. code-block:: python

      def generate_rollouts():
          env_futures = env_group.interact(
              input_channel=env_channel, output_channel=rollout_channel
          )
          rollout_futures = rollout_group.generate(
              input_channel=rollout_channel,
              output_channel=env_channel,
              actor_channel=actor_channel,
          )
          actor_futures = actor_group.recv_rollout_batch(
              input_channel=actor_channel
          )
          env_futures.wait()
          rollout_futures.wait()
          actor_futures.wait()

将 ``rollout.pipeline_stage_num`` 设为大于 1 且把环境与 rollout 分到不同 GPU 后，上述三个并发调用会自动按流水线分片重叠执行环境步进与推理，Python 主循环本身无需修改。

延伸阅读
--------

- :doc:`execution_flow`：从启动脚本到 checkpoint 保存的四阶段运行生命周期。
- :doc:`execution_modes`：共享式、分离式与混合式模式在显存与吞吐上的权衡。
- :doc:`worker` 与 :doc:`channel`：worker 组、异步句柄与分布式队列的完整机制。
- :doc:`../reference/api/runners`：内置 runner 的 Python API 参考。
