高层次编程流程概览
===========================

本节将带你了解 RLinf 的顶层编程逻辑。  
我们不会涉及底层细节，而是聚焦于最高层的 API，帮助你理解整体控制流程，  
并能够定制你自己的算法或项目。

我们的示例突出展示了 RLinf 的核心能力：  
**混合式模式下的细粒度流水线训练**，用于训练具身智能环境中的 VLA 模型。

YAML 配置
-----------------------

在启动任何训练脚本之前，最重要的步骤是准备好配置文件（YAML）。例如：

- 针对 **VLA** agent 的具身任务训练配置在  
  ``examples/embodiment/config``  
- 针对数学推理的 **LLM** 模型训练配置在  
  ``examples/reasoning/config/math``  

建议你先熟悉这些示例 YAML 的结构，然后逐步迭代以适配你的任务。关键选项包括（但不限于）：

**1. 执行模式，以及使用的节点/GPU 数量**

.. code:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-7
       env: 0-3
       rollout: 4-7

**2. 模型路径，tokenizer路径和输出路径**

- ``rollout.model.model_path``  
- ``actor.tokenizer.tokenizer_model``  
- ``actor.model.model_path``  
- ``runner.logger.log_path``  

**3. 训练超参数，例如最大训练步数、批大小等**

- ``runner.max_epochs``  
- ``runner.max_steps``  
- ``actor.global_batch_size``  
- ``actor.micro_batch_size``  

第一次运行建议使用默认参数，然后逐步调整。完整参数说明请参考 :doc:`yaml`。

Worker 启动调度流程
---------------------------

以下 Python 代码节选自  
``examples/embodiment/train_embodied_agent.py``，  
它代表了所有 RLinf 项目 ``main`` 入口的标准模式：

.. code:: python

   cluster = Cluster(num_nodes)
   component_placement = HybridComponentPlacement(cfg, cluster)

   # 创建 actor worker 组
   actor_placement = component_placement.get_strategy("actor")
   actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
       cluster, placement_strategy=actor_placement
   )

   # 创建 rollout worker 组
   rollout_placement = component_placement.get_strategy("rollout")
   rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
       cluster, placement_strategy=rollout_placement
   )

   # 创建 env worker 组
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

该入口流程主要完成以下三件事：

1. 从配置文件初始化 ``Cluster`` （集群资源视图）和  
   ``HybridComponentPlacement`` （所有 RL worker 的 GPU 分布策略）  
2. 创建 **actor**、**rollout** 和 **env** 的 WorkerGroup，并统一管理  
3. 构建 ``EmbodiedRunner``，并通过 ``runner.run()`` 启动主训练循环

训练循环概览
------------------------

``runner.run()`` 的高层逻辑（定义于  
``rlinf/runners/embodied_runner.py``）大致如下：

.. code:: python

   for step in range(training_step):
       update_rollout_weights()
       generate_rollouts()

       actor_group.compute_advantages_and_returns()

       actor_group.run_training()

这个训练循环包含四个核心阶段：

1. **actor 和 rollout 模型同步**，调用 ``update_rollout_weights()``：

   .. code:: python

      def update_rollout_weights():
          rollout_futures = rollout_group.sync_model_from_actor()
          actor_futures = actor_group.sync_model_to_rollout()
          actor_futures.wait()
          rollout_futures.wait()

2. **混合式模式下的细粒度 rollout 流水线**，调用 ``generate_rollouts()``：

   .. code:: python

      def generate_rollouts(self):
          env_futures = env_group.interact()
          rollout_futures = rollout_group.generate()
          actor_futures = actor_group.recv_rollout_batch()
          env_futures.wait()
          actor_futures.wait()
          rollout_futures.wait()

   这里最关键的两步是 ``env_group.interact()`` 和  
   ``rollout_group.generate()``，它们通过两个生产者-消费者队列连接，  
   实现了 **细粒度流水线加速 rollout** 的能力。  
   详见 :doc:`../mode/hybrid`。

3. **优势值与回报计算**，通过  
   ``actor_group.compute_advantages_and_returns()``，  
   基于上一步收集的 rollout 数据进行处理。

4. **策略更新**，通过  
   ``actor_group.run_training()``，使用 rollout 数据和计算好的 advantage/return，执行训练。
