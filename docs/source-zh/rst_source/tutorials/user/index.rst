统一编程接口
===============================

本章节将从用户视角介绍 RLinf 的基础 **静态组件**，  
这些组件构成了配置和启动可扩展 RL 任务的核心框架。

- :doc:`yaml`  
   详细介绍 RLinf 所用的 YAML 配置参数。  
   教你如何组织配置文件，使其更清晰、灵活、易于复现。

- :doc:`worker`  
   介绍 *Worker* 的概念：这是 RLinf 中的模块化执行单元，每个 Worker 负责强化学习流程中的某个具体任务。  
   多个相同类型的 Worker 组成 *WorkerGroup*，方便实现分布式执行并提升扩展性。

- :doc:`placement`  
   解释 RLinf 如何在不同任务与 Worker 之间合理分配硬件资源，  
   以实现硬件资源的高效利用与执行负载的平衡。这不仅包括加速硬件（如 GPU、NPU），还包括机器人硬件以及 CPU 节点。

- :doc:`cluster`  
   描述全局唯一的 *Cluster* 对象，它负责协调训练任务中所有节点的角色、进程和通信操作。

- :doc:`flow`  
   结合 WorkerGroup、Placement 和 Cluster 的概念，  
   展示 RLinf 的完整编程流程，帮助你理解其整体运行机制。

.. toctree::
   :hidden:
   :maxdepth: 1

   yaml
   worker
   placement
   cluster
   flow
