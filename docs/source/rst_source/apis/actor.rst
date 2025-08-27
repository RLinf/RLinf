Actor Interface
=================

This section provides the key APIs of the **Actor** classes in the RLinf framework.  
It includes implementations based on both **Megatron** and **FSDP** backends.  
In addition, information about the `ModelManager` is provided. As the parent class of the Actor classes, it manages the underlying model as well as critical APIs for parameter onload/offload.

MegatronActor
---------------

.. autoclass:: rlinf.workers.actor.megatron_actor_worker.MegatronActor
   :show-inheritance:
   :members: 

MegatronModelManager
-----------------------

.. autoclass:: rlinf.hybrid_engines.megatron.megatron_model_manager.MegatronModelManager
   :members: 


FSDPActor
--------------

.. autoclass:: rlinf.workers.actor.fsdp_actor_worker.EmbodiedFSDPActor
   :show-inheritance:
   :members: 

FSDPModelManager
------------------

.. autoclass:: rlinf.hybrid_engines.fsdp.fsdp_model_manager.FSDPModelManager
   :members: 