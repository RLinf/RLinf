Multi-Turn Visual Agent Training (Qwen3.5)
==========================================

Turn on multi-turn image alignment so SGLang rollout and Megatron-Bridge training agree on every processed image and visual token boundary in Qwen3.5 Agent RL, preventing generated marker tokens from corrupting earlier turns.

**Workflow:** Check scope and environment -> enable alignment in your training YAML -> wire your visual agent loop if custom -> verify with the standalone script.

Scope and Requirements
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Item
     - Details
   * - **Supported setup**
     - Local dense Qwen3.5 checkpoints (``Qwen3_5ForConditionalGeneration``) with the default SGLang rollout worker and Megatron-Bridge training (compatible with ``PP`` and ``CP``; ``Qwen/Qwen3.5-4B`` verified on 1, 2, and 8 NVIDIA A100 80 GB GPUs).
   * - **Not supported**
     - Video or audio inputs, FSDP training, ``session_server.enable: true``, or dedicated ``inference`` / ``actor_inference`` / ``critic_inference`` deployments (logprobs and values must be recomputed on the training nodes).
   * - **Requirements**
     - Python 3.12 ``agentic`` environment with SGLang ``0.5.12.post1`` through ``0.5.22+`` (see :doc:`version`), Megatron Core 0.18.x, Megatron-Bridge 0.5.0 (:doc:`../extending/mbridge`), and ``fla-core``.

.. dropdown:: Install the agentic environment

   From the repository root, install the Python 3.12 agentic environment (see :doc:`../start/installation` for details):

   .. code-block:: bash

      bash requirements/install.sh agentic --sglang 0.5.12.post1 && source .venv/bin/activate

Training Configuration
----------------------

Multi-turn image alignment is off by default. Enable ``rollout.sglang.align_multimodal_inputs: true`` together with dynamic rollout batching and Megatron-Bridge:

.. code-block:: yaml

   agentloop:
     is_dynamic_rollout_batch: true
   session_server:
     enable: false
   rollout:
     model:
       model_type: qwen3_5
     sglang:
       align_multimodal_inputs: true
   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
     megatron:
       mbridge: true
       use_hf_ckpt: true

If you are implementing a custom multi-turn visual agent, follow :ref:`multi-turn-visual-agents` in :doc:`../extending/new_agent`.

.. dropdown:: Optional: Train a Critic

   When ``critic.use_critic_model: true``, configure ``critic`` with Megatron-Bridge as well:

   .. code-block:: yaml

      critic:
        use_critic_model: true
        training_backend: megatron
        model:
          model_type: qwen3_5
        megatron:
          mbridge: true
          use_hf_ckpt: true

.. dropdown:: Optional: Logprob Mode

   Training recomputes logprobs by default (``algorithm.recompute_logprobs: true``), or can reuse SGLang rollout logprobs directly:

   .. code-block:: yaml

      algorithm:
        recompute_logprobs: false
      rollout:
        return_logprobs: true

Quick Verification (Standalone Script)
--------------------------------------

Run the bundled synthetic two-turn check to verify that SGLang and Megatron-Bridge work together on your machine (this standalone check does not read your custom training YAML).

Download ``Qwen/Qwen3.5-4B`` and run the 2-GPU check from the repository root. The check passes when both rollout turns report passed and the printed JSON lines report finite loss, positive grad norm, and positive ``changed`` parameter counts:

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
   bash examples/agent/qwen35/run_image_alignment.sh \
       rollout.model.model_path=./Qwen3.5-4B

.. dropdown:: View observed output on 2xA100 80 GB

   .. code-block:: json

      {"role": "actor", "ranks": [{"rank": 0, "loss": -0.960578441619873, "grad_norm": 290.3096923828125, "changed": 294963}, {"rank": 1, "loss": -0.960578441619873, "grad_norm": 290.3096923828125, "changed": 272910}]}
      {"role": "critic", "ranks": [{"rank": 0, "loss": 0.5, "grad_norm": 1544.4698486328125, "changed": 49}, {"rank": 1, "loss": 0.5, "grad_norm": 1544.4698486328125, "changed": 49}]}

.. dropdown:: 1-GPU Rollout Check and 8-GPU Check

   To verify two-turn SGLang image isolation on 1 GPU only (look for ``Turn 2 passed``):

   .. code-block:: bash

      python examples/agent/qwen35/check_image_alignment.py --model ./Qwen3.5-4B

   To override GPU indices or run across 8 idle GPUs using ``examples/agent/qwen35/config/image_alignment.yaml``:

   .. code-block:: bash

      bash examples/agent/qwen35/run_image_alignment.sh \
          rollout.model.model_path=./Qwen3.5-4B \
          cluster.component_placement.actor=0-7

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Error
     - Fix
   * - ``non-empty image_spans and matching image_data``
     - Pass image arguments only when images exist and match ``image_spans`` in length; use the state helper in :ref:`multi-turn-visual-agents`.
   * - ``SGLang image patch width ... differs``
     - Ensure your agent expands tokens using ``rollout.model.model_path`` and shares the same image pixel-limit env vars as SGLang.
   * - ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` conflict
     - Unset any custom value of this variable so RLinf can register its processor automatically.

How It Works
------------

Each training turn stays aligned across rollout and training in three steps:

1. **Agent preprocessing & expansion**: Before calling generation, the agent preprocesses each image with the aligned processor, expands visual placeholders to exact patch width, and records ``image_spans``.
2. **Span-isolated SGLang rollout**: SGLang encodes images only inside ``image_spans`` and treats any literal marker tokens in generated text as ordinary text tokens.
3. **Mask-aware Megatron-Bridge forward**: During training, ``mm_token_type_ids`` masks literal marker tokens during visual scattering and M-RoPE indexing while keeping full language embeddings and native pipeline/context parallel packing intact (real-model verified on ``TP=2, PP=1, CP=1``; PP and CP hooks covered by unit tests).

.. dropdown:: Cross-backend logprob differences

   Multi-turn image alignment ensures identical processed images and visual token boundaries between SGLang and Megatron-Bridge, though small numerical logprob differences across inference and training kernels remain normal. The verification scripts check input alignment and optimizer updates rather than downstream convergence.

Related Guides
--------------

- :doc:`../extending/new_agent` for custom multi-turn visual agents (:ref:`multi-turn-visual-agents`).
- :doc:`../extending/mbridge` for Megatron-Bridge prerequisites and model setup.
- :doc:`agentic_config` for agentic training YAML config keys.
- :doc:`version` for SGLang and vLLM version compatibility.
