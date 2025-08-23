APIs
==========

本文档将为用户详细的展开RLinf中最关键的api接口，旨在帮助用户深入了解我们的API设计和使用方法。

本api文档从底层逐步往上，首先展开RLinf的基石api，
包括：

- 对worker以及workergroup的统一接口

- 对RLinf的GPU placement策略的介绍

- 对RLinf的分布式训练的支持cluster

- 在通信上的底层实现，包括基于生产者-消费者的队列抽象

在这之后，我们将介绍RLinf的上侧用于具体实现RL不同阶段的api
包括：

- 基于FSDP和Megatron的actor封装（TODO: 包括hybrid engines manager）

- 基于huggingface和sglang的用于rollout的封装

- 在embody intelligence场景下对env的封装

- 在megatron backend下对于inference stage的封装


.. toctree::
   :hidden:
   :maxdepth: 1

   worker



