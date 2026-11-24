Configure Qwen3.5 Weight Synchronization
================================================================================

Use these settings in an existing Megatron/SGLang task to synchronize dense Qwen3.5 weights: the language model, vision encoder, merger, and optional multi-token prediction (MTP) branch.

Prepare the Environment
----------------------------------------------

Use a checkpoint whose configuration declares ``Qwen3_5ForConditionalGeneration``. Both actor and rollout must be able to read the complete checkpoint, including its tokenizer and processor files.

Install the agentic SGLang environment from the repository root:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

This installs SGLang 0.5.12.post1, Megatron Bridge, and FLA. See :doc:`version` for engine version selection.

Configure the Model
------------------------------------------------------

Add this fragment to your task configuration. Replace ``<checkpoint>`` with the same local checkpoint path in all three fields.

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       precision: bf16
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

Verify the First Synchronization
------------------------------------------

For a fresh run, enable comparison against the checkpoint:

.. code-block:: yaml

   rollout:
     validate_weight_first_sync: true

Launch your task with its usual command. SGLang loads the checkpoint and compares it with the weights received from the actor on the first synchronization. Look for ``sglang: validate_weight success`` in each rollout rank's log. A mismatch raises an error that lists the affected parameters.

This check compares position-aware weight norms with a tolerance; it is not a bitwise comparison. It is disabled when resuming from a training checkpoint, whose weights can differ from the initial model. After checking a fresh run, remove the setting to avoid loading rollout weights from disk before synchronization.

Enable MTP Training and Decoding
------------------------------------------------------

To train and synchronize MTP, add:

.. code-block:: yaml

   actor:
     model:
       load_mtp: true

This also enables NEXTN speculative decoding using the MTP branch from the same checkpoint. External draft checkpoints are not supported.

Training adds an auxiliary multi-token prediction loss on response tokens. Its gradients also update the language model backbone. Monitor ``actor/mtp_loss`` for the unscaled loss; adjust its contribution with ``actor.model.mtp_loss_scaling_factor`` if needed. The first-sync check above also validates the enabled draft.

Choose the Vision Layout
------------------------------------------------

Set ``rollout.sglang.mm_enable_dp_encoder: true`` to replicate the vision encoder on rollout ranks. Otherwise, it uses tensor parallelism. For example, this configuration uses attention TP 2, feed-forward TP 4, and vision TP 1:

.. code-block:: yaml

   rollout:
     tensor_parallel_size: 4
     sglang:
       enable_dp_attention: true
       dp_size: 2
       mm_enable_dp_encoder: true

Keep ``rollout.pipeline_parallel_size: 1`` for SGLang. Actor TP, pipeline parallelism, and virtual pipeline parallelism can differ from rollout placement. See :doc:`../concepts/placement` for placement configuration.
