API手册
==========

本节为你详细介绍 RLinf 中最核心的 API 接口，帮助用户深入理解我们的 API 设计和使用方式。  
这些关键 API 是暴露给用户的，用来简化 RL 中复杂的数据流，让用户只需关注高层抽象，而无需关心底层的具体实现。

本 API 文档采用自底向上的方式展开，首先介绍 RLinf 的基础 API，包括：

- :doc:`worker` — Worker 与 Worker 组的统一接口。  
- :doc:`placement` — RLinf 的 GPU Placement 策略介绍。  
- :doc:`cluster` — 通过集群支持分布式训练。  
- :doc:`channel` — 底层通信原语，包括生产者–消费者队列抽象。  

随后我们介绍上层 API，用于实现 RL 的不同阶段：

- :doc:`actor` — 基于 FSDP 与 Megatron 的 Actor 封装。  
- :doc:`rollout` — 基于 Huggingface 与 SGLang 的 Rollout 封装。  
- :doc:`env` — 面向具身智能场景的环境封装。  
- :doc:`data` — 不同 Worker 间数据传输的数据结构封装。  
- :doc:`embodied_data` — 具身场景的 Env/Rollout 数据结构。
- :doc:`replay_buffer` — 轨迹级 Replay Buffer 的设计与采样机制。  

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
