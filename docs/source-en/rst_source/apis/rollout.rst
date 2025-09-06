Rollout Interface
=================

This section provides the key APIs of the **Rollout** classes in the RLinf framework.  
It includes implementations based on both **SGLang** and **Hugging Face** backends.  
For SGLang, we provide two designs: one based on synchronous execution and another based on asynchronous execution.


SGLang
---------

.. autoclass:: rlinf.workers.rollout.sglang.sglang_worker.SGLangWorker
   :members: 

.. autoclass:: rlinf.workers.rollout.sglang.sglang_worker.AsyncSGLangWorker
   :members: 

Huggingface
------------

.. autoclass:: rlinf.workers.rollout.hf.huggingface_worker.MultiStepRolloutWorker
   :members: 