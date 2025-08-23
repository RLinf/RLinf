Worker and WorkerGroup 
=========================

这一部分文档将详细介绍RLinf中对Worker和WorkerGroup的统一接口设计。
Worker是RLinf中执行任务的基本单元，后续的RL 训练中的不同阶段都会继承它，以实现统一的通信与调度
而WorkerGroup则是多个Worker的集合，让用户不必处理分布式训练中的复杂性。
通过WorkerGroup，用户可以更方便地管理和调度多个Worker，从而实现更高效的分布式训练。

.. autoclass:: rlinf.workers.actor.megatron_actor_worker.MegatronActor
   :show-inheritance:
   :members: 

.. autoclass:: rlinf.hybrid_engines.megatron.megatron_model_manager.MegatronModelManager
   :members: 

.. autoclass:: rlinf.utils.placement.ComponentPlacement
   :members: