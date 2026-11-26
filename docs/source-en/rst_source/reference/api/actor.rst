Actor Interface
===============

Use these actor, SFT, and model-manager APIs when inspecting or extending policy training, parameter offloading, and weight synchronization on Megatron or FSDP backends.

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
