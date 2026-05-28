使用与编程教程
===============

本节介绍 RLinf 的核心编程模型和部署模式。
您将学习基本概念——Worker、WorkerGroup、放置策略和通信机制——
以及如何从单节点扩展到多节点集群，并灵活配置执行模式。

- :doc:`worker`
   介绍 *Worker*，即 RLinf 中的模块化执行单元。多个相似的 Worker
   组成 *WorkerGroup*，简化分布式执行。

- :doc:`placement`
   介绍 RLinf 如何在任务和 Worker 之间策略性地分配硬件资源，
   确保在 GPU、NPU、机器人硬件和纯 CPU 节点上的高效利用。

- :doc:`flow`
   整合 WorkerGroup、Placement 和 Cluster 的概念，
   展示 RLinf 的完整编程流程。

- :doc:`channel`
   介绍 *Channel* 抽象，用于 Worker 之间异步的生产者-消费者通信，
   是实现跨 RL 阶段细粒度流水线的关键。

- :doc:`collective`
   介绍 Worker 之间底层、高性能的 Python 对象交换，
   使用 CUDA IPC 和 NCCL 等优化的点对点后端以降低通信开销。

- :doc:`multi_node`
   启动多机 Ray 集群，配置环境变量和代码同步，
   并通过 Ray 集群启动 RLinf 训练任务。

- :doc:`collocated`
   所有主要组件共享同一组 GPU 或节点。

- :doc:`disaggregated`
   各组件分配到不同的 GPU 或节点，实现资源专用。

- :doc:`hybrid`
   灵活组合共享式和分离式的放置与执行策略，
   支持细粒度流水线和最优 GPU 利用率。


.. toctree::
   :hidden:
   :maxdepth: 1

   worker
   placement
   flow
   channel
   collective
   multi_node
   collocated
   disaggregated
   hybrid
