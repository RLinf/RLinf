Rollout 接口
=================

本节介绍 RLinf 框架中 **Rollout** 类的关键 API。  
它包含基于 **SGLang** 和 **Hugging Face** 后端的实现。  
对于 SGLang，我们提供了两种设计：一种基于同步执行，另一种基于异步执行。  

SGLang
---------

.. autoclass:: rlinf.workers.rollout.sglang.sglang_worker.SGLangWorker
   :members: 
   :member-order: bysource

.. autoclass:: rlinf.workers.rollout.sglang.sglang_worker.AsyncSGLangWorker
   :members: 
   :member-order: bysource

Huggingface
------------

.. autoclass:: rlinf.workers.rollout.hf.huggingface_worker.MultiStepRolloutWorker
   :members: 
   :member-order: bysource