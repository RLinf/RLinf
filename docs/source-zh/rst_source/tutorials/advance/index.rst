高级特性
==============================

本章将逐步深入讲解 RLinf 如何实现 **高效执行**，
并提供实用指南，帮助你充分优化 RL 后训练工作流。

- :doc:`lora`
   展示如何在 RLinf 中集成低秩适配 (LoRA)，
   以极小的计算开销实现参数高效的微调。

- :doc:`5D`
   解释 RLinf 如何支持 Megatron 风格的 5D 并行，包括：
   张量并行 (TP)、数据并行 (DP)、流水线并行 (PP)、
   序列并行 (SP) 和上下文并行 (CP)。
   学习如何配置和组合这些维度，以高效扩展大模型。

- :doc:`cluster`
   介绍全局唯一的 *Cluster* 对象，负责协调分布式训练中所有角色、
   进程和跨节点通信。涵盖 Ray 初始化、节点发现和 Worker 分配。

- :doc:`collective`
   介绍 Worker 之间底层、高性能的 Python 对象交换，
   使用 CUDA IPC 和 NCCL 等优化的点对点后端以降低通信开销。

- :doc:`version`
   描述如何在不同的 SGLang 版本之间动态切换，
   以满足不同的兼容性需求或实验要求。

- :doc:`nsight`
   介绍基于 Hydra 的 ``cluster.nsight`` 配置，用于通过 ``nsys profile``
   包装指定的 Ray worker group，并说明如何启用、关闭以及选择需要采样的 worker。

- :doc:`dynamic_scheduling`
   涵盖 RLinf 的在线扩缩与动态调度机制：如何在训练过程中对资源进行
   秒级弹性扩缩与组件间迁移，以最大化吞吐和利用率，
   包括前置依赖、配置示例和可选调度策略。

- :doc:`auto_placement`
   详细介绍 RLinf 中自动放置的具体实现，
   包括如何正确配置以启用自动放置功能。

- :doc:`distributed_tracing`
   解释如何使用基于 HTTP 的分布式追踪系统，通过 Chrome Trace Viewer
   或 Perfetto 可视化多进程执行过程并定位性能瓶颈。

.. toctree::
   :hidden:
   :maxdepth: 2

   lora
   5D
   cluster
   collective
   version
   nsight
   dynamic_scheduling
   auto_placement
   distributed_tracing
