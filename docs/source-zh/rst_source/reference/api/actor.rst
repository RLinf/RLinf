Actor 接口
=================

本节介绍 RLinf 框架中 **Actor** 类的关键 API。  
其实现包括基于 **Megatron** 和 **FSDP** 两种后端。  

此外，还提供了关于 `ModelManager` 的信息。  
`ModelManager` 作为 Actor 类的父类，负责管理底层模型，并提供参数加载 / 卸载等关键 API。  

MegatronActor
---------------

.. autoclass:: rlinf.workers.actor.megatron_actor_worker.MegatronActor
   :show-inheritance:
   :members: 
   :member-order: bysource

MegatronModelManager
-----------------------

.. autoclass:: rlinf.hybrid_engines.megatron.megatron_model_manager.MegatronModelManager
   :members: 
   :member-order: bysource


FSDPActor
--------------

.. autoclass:: rlinf.workers.actor.fsdp_actor_worker.EmbodiedFSDPActor
   :show-inheritance:
   :members: 
   :member-order: bysource

FSDPModelManager
------------------

.. autoclass:: rlinf.hybrid_engines.fsdp.fsdp_model_manager.FSDPModelManager
   :members: 
   :member-order: bysource