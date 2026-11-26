Rollout Interface
=================

Use these rollout worker APIs when wiring SGLang, vLLM, or Hugging Face inference engines into reasoning, agentic, or embodied RL and evaluation loops.

SGLangWorker
------------

.. autoclass:: rlinf.workers.rollout.sglang.sglang_worker.SGLangWorker
   :show-inheritance:
   :members:
   :member-order: bysource

VLLMWorker
----------

.. autoclass:: rlinf.workers.rollout.vllm.vllm_worker.VLLMWorker
   :show-inheritance:
   :members:
   :member-order: bysource

MultiStepRolloutWorker
----------------------

.. autoclass:: rlinf.workers.rollout.hf.huggingface_worker.MultiStepRolloutWorker
   :show-inheritance:
   :members:
   :member-order: bysource

SGLangEmbodiedWorker
--------------------

.. autoclass:: rlinf.workers.rollout.sglang.sglang_embodied_worker.SGLangEmbodiedWorker
   :show-inheritance:
   :members:
   :member-order: bysource
