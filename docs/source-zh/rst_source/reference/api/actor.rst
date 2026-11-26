Actor 接口
==========

查阅 Megatron 与 FSDP 后端上的 actor、SFT 与模型管理类，用于扩展策略训练、参数卸载与权重同步。

MegatronActor
-------------

.. autoclass:: rlinf.workers.actor.megatron_actor_worker.MegatronActor
   :show-inheritance:
   :members:
   :member-order: bysource

MegatronModelManager
--------------------

.. autoclass:: rlinf.hybrid_engines.megatron.megatron_model_manager.MegatronModelManager
   :members:
   :member-order: bysource

EmbodiedFSDPActor
-----------------

.. autoclass:: rlinf.workers.actor.embodied_fsdp_actor_worker.EmbodiedFSDPActor
   :show-inheritance:
   :members:
   :member-order: bysource

FSDPActor
---------

.. autoclass:: rlinf.workers.actor.fsdp_actor_worker.FSDPActor
   :show-inheritance:
   :members:
   :member-order: bysource

FSDPSftWorker
-------------

.. autoclass:: rlinf.workers.sft.fsdp_sft_worker.FSDPSftWorker
   :show-inheritance:
   :members:
   :member-order: bysource

FSDPModelManager
----------------

.. autoclass:: rlinf.hybrid_engines.fsdp.fsdp_model_manager.FSDPModelManager
   :members:
   :member-order: bysource
