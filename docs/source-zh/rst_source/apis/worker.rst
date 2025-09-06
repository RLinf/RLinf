Worker 接口
===================================

本节将详细介绍 RLinf 中 **Worker** 与 **WorkerGroup** 的统一接口设计。  
**Worker** 是 RLinf 中最基本的执行单元。RL 训练的不同阶段都会继承自 Worker，从而实现统一的通信与调度。  
**WorkerGroup** 则是多个 Worker 的集合，它让用户无需直接处理分布式训练的复杂性。  
通过 WorkerGroup，用户可以更方便地管理和调度多个 Worker，从而实现更高效的分布式训练。  

Worker
-------

.. autoclass:: rlinf.scheduler.worker.worker.Worker
   :members: worker_address, create_group, send, recv, send_tensor, recv_tensor, create_channel, connect_channel, broadcast
   :class-doc-from: init
   :exclude-members: __init__, __new__

WorkerGroup
-----------

.. autoclass:: rlinf.scheduler.worker.worker_group.WorkerGroup
   :members:
