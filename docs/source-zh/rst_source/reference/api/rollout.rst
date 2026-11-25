Rollout 接口
============

查阅基于 SGLang、vLLM 与 Hugging Face 后端的 rollout worker 接口，用于推理、智能体或具身任务的采样与评估。

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
