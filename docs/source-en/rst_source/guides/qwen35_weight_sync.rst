Configure Qwen3.5 Weight Synchronization
================================================================================

Use a Megatron actor and SGLang rollout to synchronize dense Qwen3.5 weights, including the vision encoder, merger, language model, and optional multi-token prediction (MTP) branch. Use these settings to add weight loading and synchronization to an existing task. This guide does not provide a multimodal RL training task.

Prepare the Environment
----------------------------------------------

Use a checkpoint whose configuration declares ``Qwen3_5ForConditionalGeneration``. The checkpoint directory name does not determine the architecture. Both actor and rollout must be able to read the complete checkpoint, including its tokenizer and processor files.

Install the agentic SGLang environment from the repository root:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

The agentic environment selects SGLang 0.5.12.post1 and installs Megatron Bridge and FLA. FLA supplies the gated delta network kernels used by Qwen3.5. Weight synchronization was tested on NVIDIA A100 GPUs with BF16, Megatron Bridge 0.5.0, Transformers 5.6.1, and fla-core 0.6.0. See :doc:`version` for engine version selection.

Configure the Model and MTP
------------------------------------------------------

Add these settings to your Megatron/SGLang task configuration. Replace ``<checkpoint>`` with the same local checkpoint path in all three fields. This fragment contains the model settings; retain the task's data, algorithm, placement, and memory settings.

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       precision: bf16
       load_mtp: false
       mtp_loss_scaling_factor: 0.1
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

``actor.model.load_mtp`` defaults to ``false``. This loads and synchronizes the language model, vision encoder, and merger without constructing MTP layers or enabling speculative decoding.

Set ``actor.model.load_mtp: true`` to train and synchronize the MTP branch and enable NEXTN speculative decoding in SGLang. The draft uses the same checkpoint as the rollout model; external draft checkpoints are not supported for this Megatron/SGLang integration. Explicit NEXTN or EAGLE settings and decoding parameters take precedence. SGLang treats NEXTN as EAGLE. The supported MTP architecture has one prediction layer and shares embeddings with the main model.

With MTP enabled, RL actor updates include an auxiliary multi-token prediction loss on response tokens. ``actor.model.mtp_loss_scaling_factor`` controls its weight and defaults to ``0.1``. MTP gradients also update the language model backbone. Check ``actor/mtp_loss`` in the training metrics to monitor the unscaled auxiliary loss. Logprob recomputation and dedicated inference do not add this loss.

With actor MTP enabled, the default for ``rollout.sglang.disable_radix_cache`` depends on the Mamba scheduling mode: ``true`` for ``no_buffer``, or ``false`` for ``extra_buffer``. Explicit values take precedence, and SGLang checks compatibility with the scheduling mode. With actor MTP disabled, SGLang's default applies unless you set the value explicitly.

When the actor supplies all main-model and draft weights, SGLang uses dummy initialization unless validation against the checkpoint is enabled. It receives actor weights before rollout. Enabling ``rollout.validate_weight_first_sync`` loads the checkpoint first and checks both the main model and the enabled MTP draft after synchronization. This built-in check compares position-aware norms with a tolerance; use byte comparisons for bitwise verification. Resuming training disables the first-sync checkpoint comparison because the actor's resumed weights can differ from the initial checkpoint.

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

Use ``rollout.tensor_parallel_size: 4``, ``rollout.sglang.enable_dp_attention: true``, and ``rollout.sglang.dp_size: 2`` for this language layout. Set ``rollout.pipeline_parallel_size: 1``: the RLinf SGLang launch integration does not forward rollout PP to the engine. This restriction does not apply to actor pipeline or virtual pipeline parallelism; both have passed full-model weight synchronization checks.

Verify Weight Loading
------------------------------------------

Activate the agentic environment, then run the component tests from the repository root. Preserve the environment's existing Python paths:

.. code-block:: bash

   export PYTHONPATH="$PWD:$PWD/tests/unit_tests:$PYTHONPATH"
   CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit_tests/test_ckpt_convertor.py -q
   CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit_tests/test_megatron.py -q

With the required dependencies installed, all tests should pass without skips. A missing SGLang or Megatron dependency can skip the corresponding tests; install it before treating the run as validation. These CPU checks do not establish full-model GPU training convergence or speculative decoding performance.
