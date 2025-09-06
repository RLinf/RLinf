教程
=========

本章节将深入讲解 **RLinf** 的设计原理与使用方法。  
你将通过一系列动手实践的教程，了解该库的核心组件与功能模块。

- :doc:`user/index`  
   从用户视角出发，本教程介绍 RLinf 的基本组成部分，包括如何使用 YAML 配置任务、如何为每个 RL 子任务分配 Worker，以及如何从全局集群层面管理 GPU 资源。  
   最后，我们提供一套完整的编程流程，帮助你理解 RLinf 的整体运行逻辑。

- :doc:`mode/index`  
   了解 RLinf 支持的多种执行模式，包括共享式模式、分离式模式以及支持细粒度流水线的混合式模式。

- :doc:`scheduler/index`  
   学习 RLinf 的自动调度机制，主要包括在线动态扩缩机制和自动调度策略，用于根据任务负载实时调整资源分配。

- :doc:`communication/index`  
   探索 RLinf 弹性通信系统的底层逻辑，涵盖点对点通信模型以及基于生产者/消费者机制的通道实现方式。

- :doc:`advance/index`  
   深入了解 RLinf 的高级特性，例如 5D 并行配置与 LoRA 集成设计，帮助你最大化训练效率与性能表现。

- :doc:`rlalg/index`  
   提供对每种支持的 RL 算法（如 PPO、GRPO 等）的完整教程，包括可直接使用的配置文件与实用调参技巧。

- :doc:`extend/index`  
   学习如何扩展 RLinf，集成你自定义的训练算法、模拟环境与模型架构，以满足具体研究需求。

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
