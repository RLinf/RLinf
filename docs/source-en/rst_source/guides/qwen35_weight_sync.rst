Configure Qwen3.5 Weight Synchronization
================================================================================

Use a Megatron actor and SGLang rollout to synchronize dense Qwen3.5 weights, including the vision encoder, merger, language model, and optional multi-token prediction (MTP) branch. Use these settings to add weight loading and synchronization to an existing task. This guide does not provide a multimodal RL training task.

Prepare the Environment
----------------------------------------------

Use a checkpoint whose configuration declares ``Qwen3_5ForConditionalGeneration``. The checkpoint directory name does not determine the architecture. Both actor and rollout must be able to read the complete checkpoint, including its tokenizer and processor files.

Install the agentic SGLang environment from the repository root:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

The agentic environment selects SGLang 0.5.12.post1 and installs Megatron Bridge and FLA. FLA supplies the gated delta network kernels used by Qwen3.5. The weight synchronization checks used BF16, Megatron Bridge 0.5.0, Transformers 5.6.1, and fla-core 0.6.0 on NVIDIA A100 GPUs. See :doc:`version` for engine version selection.

Configure the Model and MTP
------------------------------------------------------

Add these settings to your Megatron/SGLang task configuration. Replace ``<checkpoint>`` with the same local checkpoint path in all three fields. This fragment contains the model settings; retain the task's data, algorithm, placement, and memory settings.

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       precision: bf16
       load_mtp: true
     tokenizer:
       tokenizer_model: <checkpoint>
     megatron:
       mbridge: true
       use_hf_ckpt: true
       ckpt_convertor:
         hf_model_path: <checkpoint>
   rollout:
     rollout_backend: sglang
     model:
       model_type: qwen3_5
       model_path: <checkpoint>
       precision: bf16
     sglang:
       mm_enable_dp_encoder: true
       disable_radix_cache: true

``actor.model.load_mtp`` defaults to ``true``. It loads MTP into the actor and enables SGLang's NEXTN speculative decoding with the same checkpoint as the draft model. SGLang normalizes NEXTN to EAGLE internally. Let ``actor.model.load_mtp`` select the speculative decoding algorithm and draft checkpoint. The supported MTP structure has one prediction layer and shares embeddings with the main model.

Set ``actor.model.load_mtp: false`` to load and synchronize only the vision and language branches. The main model still includes its vision encoder and merger. Keep ``rollout.sglang.disable_radix_cache: true`` for the Qwen3.5 NEXTN setup; this is the integration's default for this model.

Choose the Vision Layout
------------------------------------------------

Set ``rollout.sglang.mm_enable_dp_encoder: true`` to replicate the vision encoder on rollout ranks. Set it to ``false`` to use tensor parallelism for the vision encoder; ``false`` is the default. Language attention, feed-forward layers, and the vision encoder use their respective target layouts during synchronization.

For example, with rollout engine TP 4 and attention DP 2, attention TP is 2. With the vision option enabled, each vision replica has TP 1; language feed-forward layers still use engine TP 4. Actor TP can differ from these rollout layouts.

.. list-table:: Example Target Layout
   :header-rows: 1

   * - Component
     - Tensor parallel size
   * - Language attention
     - 2
   * - Language feed-forward layers
     - 4
   * - Vision encoder with vision DP enabled
     - 1

Use ``rollout.tensor_parallel_size: 4``, ``rollout.sglang.enable_dp_attention: true``, and ``rollout.sglang.dp_size: 2`` for this language layout. Set ``rollout.pipeline_parallel_size: 1``: the RLinf SGLang launch integration does not yet forward rollout PP to the engine. This restriction does not apply to actor pipeline or virtual pipeline parallelism; both have passed full-model weight synchronization checks.

Verify Weight Loading
------------------------------------------

Run the component tests in the agentic environment from the repository root:

.. code-block:: bash

   PYTHONPATH=.:tests/unit_tests python -m pytest tests/unit_tests/test_ckpt_convertor.py -q

With the tested dependencies installed, all 123 tests should pass. Inspect skipped tests if you use a different environment: if SGLang or Megatron dependencies are missing, passing tests do not verify loading weights with those libraries.

Full-model checks compare every destination parameter's shape, dtype, and raw bytes with an independent reference loaded through SGLang's full-checkpoint loader. They cover initial weights and deliberately changed weights, colocated and disaggregated placement, unequal actor/rollout TP, actor PP/VP, attention DP, and vision DP. The 4B checks also cover shared output embeddings and MTP disabled. These checks establish weight loading correctness, not generated-output equality, training convergence, or speculative decoding speedup.

Asynchronous weight updates while inference requests are running, context parallelism, and multi-node synchronization still require integration tests. Use the existing task's pause and resume lifecycle when integrating asynchronous training.
