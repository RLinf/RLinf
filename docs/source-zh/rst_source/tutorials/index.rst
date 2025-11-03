教程
=========

本节将对 **RLinf** 进行深入讲解。我们提供了一系列实操教程，涵盖该库的所有核心组件与功能。首先，我们将从整体上介绍 RLinf 的执行流程，帮助用户理解其如何完成一个强化学习（RL）训练过程。

RLinf 执行流程概览
--------------------------------

下图展示了 RLinf 的整体执行流程，包括三部分内容：左侧是主要代码流程，中间是与代码流程对应的主进程执行逻辑，右侧是 Worker、WorkerGroup 和 Channel 等核心概念的说明。

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf_exec_flow.jpg
   :alt: RLinf execution flow
   :width: 95%
   :align: center

- **代码流程概览。** 我们先来看图中左侧的主要代码流程。`run.sh` 脚本会运行 `main_grpo.py`，这个py文件是整个训练的入口点。在 `main_grpo.py` 的主函数中，首先会根据 YAML 配置文件（如 `cluster/component_placement`）确定各类 Worker（例如 actor、rollout）的放置位置，即每个Worker跑在哪些GPU上。每个 Worker 可以通过 YAML 配置灵活地分配到任意数量的 GPU 或其他加速设备上。确定放置后，脚本会创建 WorkerGroup，每个 WorkerGroup 包含一个或多个相同类型的 Worker 进程。这些 WorkerGroup 会被传递给 Runner，在 Runner 的 `run()` 函数中封装了完整的 RL 训练流程。

- **主进程执行逻辑。** 图中间部分展示了主函数更详细的执行流程。在这里，Worker 的放置由 Worker Placement Strategy 决定，随后通过 `create_group()` 创建各个 WorkerGroup（如 ActorGroup、RolloutGroup）。Runner（位于 `rlinf/runners/` 中）接收这些 WorkerGroup，并通过 `Channel.Create()` 建立各组之间的数据通信通道。每个 WorkerGroup 执行其核心任务（如 train 或 rollout 函数），并通过分配的 Channel 进行数据交换。在每次迭代结束后，ActorGroup 会调用 `sync_model` 将模型权重同步到 RolloutGroup。

- **关键概念与特性。** 图右侧突出展示了 RLinf 的三大核心特性：(i) 灵活的 Worker 放置机制，任意 WorkerGroup 可以弹性部署在任意节点或 GPU 上；(ii) 易用的通信接口，用户只需通过 WorkerGroup 名称即可发送或接收数据；(iii) 分布式数据通道（Channel）， Worker 之间可以使用 channel.put 与 channel.get 轻松交换数据。

RLinf 采用模块化设计，通过 Worker、WorkerGroup 与 Channel 抽象了分布式系统的复杂性。这种设计使用户能够以极低的分布式编程成本构建大规模的强化学习训练流水线，尤其适用于具身智能与智能体系统。


.. toctree::
   :hidden:
   :maxdepth: 4

   user/index
   mode/index
   scheduler/index
   communication/index
   advance/index
   rlalg/index
   extend/index
