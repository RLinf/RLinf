API手册
==========

带你了解 RLinf 中最核心的 API 接口及其用法。  
这些关键 API 是暴露给用户的，用来简化 RL 中复杂的数据流，让用户只需关注高层抽象，而无需关心底层的具体实现。

本 API 文档采用自底向上的方式展开，首先介绍 RLinf 的基础 API，包括：

.. list-table::
   :header-rows: 1

   * - API
     - 内容
   * - :doc:`Worker <worker>`
     - Worker 与 Worker 组的统一接口。
   * - :doc:`Placement <placement>`
     - RLinf 的 GPU Placement 策略介绍。
   * - :doc:`Cluster <cluster>`
     - 通过集群支持分布式训练。
   * - :doc:`Channel <channel>`
     - 底层通信原语，包括生产者-消费者队列抽象。

随后我们介绍上层 API，用于实现 RL 的不同阶段：

.. list-table::
   :header-rows: 1

   * - API
     - 内容
   * - :doc:`Actor <actor>`
     - 基于 FSDP 与 Megatron 的 Actor 封装。
   * - :doc:`Rollout <rollout>`
     - 基于 Hugging Face 与 SGLang 的 Rollout 封装。
   * - :doc:`Env <env>`
     - 面向具身智能场景的环境封装。
   * - :doc:`Data <data>`
     - 不同 Worker 间传输的数据结构。
   * - :doc:`Embodied Data <embodied_data>`
     - 具身场景的 Env/Rollout 数据结构。
   * - :doc:`Replay Buffer <replay_buffer>`
     - 轨迹级 Replay Buffer 的设计与采样机制。

.. toctree::
   :hidden:
   :maxdepth: 1

   worker
   placement
   cluster
   channel

   actor
   rollout
   env
   data
   embodied_data
   replay_buffer
