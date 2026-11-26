Qwen3.5 Multi-Turn Image Alignment
==================================

Keep SGLang rollout and Megatron-Bridge training on the same processed images and M-RoPE positions in multi-turn Qwen3.5 visual Agent RL. Enabling image capture tracks explicit image spans across turns, so generated text containing image marker tokens never corrupts earlier turns.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Dimension
     - Details
   * - **When to use**
     - Multi-turn visual Agent RL with a local dense Qwen3.5 checkpoint (``Qwen3_5ForConditionalGeneration``), SGLang rollout, and Megatron-Bridge training.
   * - **Keep default**
     - Text-only runs, VLM SFT, FSDP training, or workflows that rely on :doc:`Session Server <session_server>`.
   * - **Requirements**
     - Python 3.12 agentic environment with SGLang ``>= 0.5.12.post1``, Megatron Core 0.18.x, Megatron-Bridge 0.5.0 (:doc:`../extending/mbridge`), and ``fla-core``; verified on 1, 2, or 8 NVIDIA A100 80 GB GPUs for ``Qwen/Qwen3.5-4B``.

.. dropdown:: Install the agentic environment

   From the repository root, install the Python 3.12 agentic environment (includes SGLang, Megatron Core, Megatron-Bridge, and ``fla-core``; see :doc:`../start/installation` for general setup):

   .. code-block:: bash

      bash requirements/install.sh agentic --sglang 0.5.12.post1 && source .venv/bin/activate

Enable in Config
----------------

Image capture is off by default. Turn on ``rollout.sglang.capture_multi_modal_inputs: true`` together with dynamic rollout batching, Megatron-Bridge, ``PP=1``, and ``CP=1``; RLinf registers the SGLang multimodal processor automatically on startup:

.. code-block:: yaml

   agentloop:
     is_dynamic_rollout_batch: true
   session_server:
     enable: false
   rollout:
     model:
       model_type: qwen3_5
     sglang:
       capture_multi_modal_inputs: true
   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       pipeline_model_parallel_size: 1
       context_parallel_size: 1
     megatron:
       mbridge: true
       use_hf_ckpt: true

To pass per-turn images and spans from a custom visual agent loop, follow :ref:`multi-turn-visual-agents` in :doc:`../extending/new_agent`.

.. dropdown:: Optional: Train a Critic

   When ``critic.use_critic_model: true``, configure ``critic`` with Megatron-Bridge, ``PP=1``, and ``CP=1``:

   .. code-block:: yaml

      critic:
        use_critic_model: true
        training_backend: megatron
        model:
          model_type: qwen3_5
          pipeline_model_parallel_size: 1
          context_parallel_size: 1
        megatron:
          mbridge: true
          use_hf_ckpt: true

.. dropdown:: Optional: CPU Prefetch and Logprob Mode

   To overlap CPU loading of multimodal micro-batches with training forward passes, set per-rank CPU prefetch budgets under ``actor.multi_modal`` (and ``critic.multi_modal`` when training a critic); see :doc:`agentic_config` for details:

   .. code-block:: yaml

      actor:
        multi_modal:
          prefetch_batches: 2
          prefetch_bytes: 536870912  # 512 MiB

   Both Megatron-Bridge logprob recomputation (``algorithm.recompute_logprobs: true``, default) and rollout logprobs (``algorithm.recompute_logprobs: false`` with ``rollout.return_logprobs: true``) work with image capture.

.. dropdown:: Optional: Offload Images to Shared Disk

   By default, captured pixel tensors ride inline inside each rollout batch. When multi-turn episodes carry large or repeated images across turns, configure ``image_store_dir`` in your agent loop to point to a shared directory mounted on every training node (see :ref:`multi-turn-visual-agents`).

   Each distinct image tensor is saved once as a BF16 content-addressed ``.safetensors`` file and verified with SHA-256 on load. Keep that directory mounted and readable while in-flight batches or saved rollout files reference it, then remove it manually after the run finishes.

Verify Your Setup
-----------------

Download ``Qwen/Qwen3.5-4B`` (or substitute your local checkpoint directory), then run the two quick checks from the repository root (verified on 80 GB A100 GPUs):

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B

1. **Check two-turn capture on 1 GPU.** Look for ``Turn 2 passed`` in the logs:

   .. code-block:: bash

      python examples/agent/qwen35/replay_images.py --model ./Qwen3.5-4B

2. **Check an actor and critic update on 2 GPUs.** Replace ``0,1`` with two contiguous free GPU indices. The script starts an isolated local Ray instance, runs synthetic two-turn episodes (with vision weights frozen in ``examples/agent/qwen35/config/replay_update.yaml``), and prints one JSON summary line for actor and one for critic; confirm every rank reports finite ``loss``, positive ``grad_norm``, and ``changed > 0`` (updated parameter elements):

   .. code-block:: bash

      bash examples/agent/qwen35/run_replay_update.sh ./Qwen3.5-4B 0,1

   Observed output on 2×A100 80 GB:

   .. code-block:: json

      {"role": "actor", "ranks": [{"rank": 0, "loss": -1.0, "grad_norm": 519.580078125, "changed": 303411}, {"rank": 1, "loss": -1.0, "grad_norm": 519.580078125, "changed": 280264}]}
      {"role": "critic", "ranks": [{"rank": 0, "loss": 0.5, "grad_norm": 1670.570556640625, "changed": 49}, {"rank": 1, "loss": 0.5, "grad_norm": 1670.570556640625, "changed": 49}]}

.. dropdown:: Optional: 8-GPU Check (TP=2, DP=4)

   Pass 8 contiguous free GPU indices to run the same check with ``TP=2, DP=4`` using ``examples/agent/qwen35/config/replay_update.yaml``:

   .. code-block:: bash

      bash examples/agent/qwen35/run_replay_update.sh ./Qwen3.5-4B 0,1,2,3,4,5,6,7

How It Works
------------

Each training step aligns rollout and training in three stages:

1. **Capture during rollout**: SGLang runs vision preprocessing only on the marked spans, records the expanded token IDs, pixel tensors, image grids, and M-RoPE positions, and verifies image history across turns.
2. **Carry per-turn payloads**: Each turn becomes one dynamic-batch row with its own captured image inputs (empty before any image arrives), so images stay aligned with each turn through batching and load balancing.
3. **Verify during training**: Before each Megatron-Bridge forward pass, RLinf checks that the prompt prefix and M-RoPE positions match SGLang's capture, drives the vision encoder only on marked spans, and keeps the sampled token IDs for the language model.

.. warning::

   Image alignment guarantees identical image inputs and M-RoPE positions between rollout and training, but SGLang and Megatron-Bridge logprobs still differ numerically across backends. The verification scripts check input alignment and optimizer updates, not downstream task convergence.

.. dropdown:: Current Limitations

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Dimension
        - Limitation
      * - Media types
        - Image inputs only; video and audio are unsupported.
      * - Training backend
        - Requires Megatron-Bridge on ``actor`` (and ``critic`` when enabled) with ``PP=1``, ``CP=1``, and ``agentloop.is_dynamic_rollout_batch: true``.
      * - Rollout serving
        - Requires the default SGLang rollout worker with ``session_server.enable: false`` and ``rollout.sglang.serving_mode`` unset.
      * - Local handoff
        - The rollout worker and its local SGLang tokenizer process must share ``TMPDIR`` (default ``/tmp`` works when unchanged across processes).

.. dropdown:: Troubleshooting

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - Error
        - Fix
      * - ``non-empty image_spans and images``
        - Pass ``image_spans`` only when ``image_data`` is non-empty and matches the span count (see :ref:`multi-turn-visual-agents`).
      * - ``sampled token history`` / ``positions changed``
        - Advance each turn from the expanded prompt plus generated tokens, and forward copied image spans and image history unchanged (see :ref:`multi-turn-visual-agents`).
      * - ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` conflict
        - Unset any custom value of ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` so RLinf can register its processor automatically.
      * - ``SGLang did not write image capture file``
        - Check that the checkpoint is Qwen3.5, SGLang is ``>= 0.5.12.post1``, and the rollout worker shares ``TMPDIR`` with the SGLang tokenizer process.
      * - Missing pixels or checksum mismatch
        - Ensure every training node mounts the same ``image_store_dir`` and does not clean it while batches still reference it.

Related Guides
--------------

- :doc:`../extending/new_agent` for implementing a custom multi-turn visual agent (:ref:`multi-turn-visual-agents`).
- :doc:`../extending/mbridge` for Megatron-Bridge prerequisites and model configuration.
- :doc:`agentic_config` for rollout, actor, and multimodal prefetch config keys.
- :doc:`version` for SGLang and vLLM version compatibility.
