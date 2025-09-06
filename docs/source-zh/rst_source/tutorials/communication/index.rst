弹性通信
=====================

本章解释 RLinf 如何在 Worker 之间实现通信，  
以支持高效、可扩展和具备弹性的分布式执行。

- :doc:`collective`  
   介绍 Worker 之间的低层次、高性能 Python 对象交换，  
   使用优化过的点对点后端（如 CUDA IPC 和 NCCL）来减少通信开销。  

- :doc:`channel`  
   介绍一种更高层次的异步通信抽象——  
   *Channel*，它充当生产者-消费者队列。  
   这种抽象对于在不同 RL 阶段间实现细粒度流水线至关重要。  

.. toctree::
   :hidden:
   :maxdepth: 1

   collective
   channel
