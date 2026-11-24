LoRA Fine-Tuning
================

Use LoRA (Low-Rank Adaptation) in RLinf to fine-tune large policies and diffusion backbones with lower GPU memory by training low-rank adapter matrices while keeping base weights frozen.

Supported Backends
------------------

LoRA configuration differs by model family:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Model family
     - Config key
     - Notes
   * - Embodied FSDP VLAs (OpenVLA, OpenVLA-OFT, etc.)
     - ``actor.model.is_lora: True``
     - Supports ``lora_rank``, ``lora_path``, ``lora_target_scope``, and pre-merged adapters via ``merged_loras``.
   * - Diffusion models (SD3.5 Medium, Wan2.2 TI2V 5B)
     - ``actor.model.<model>.use_lora: True``
     - Configured under the model block in :doc:`Diffusion-NFT <../examples/diffusion_nft>` and :doc:`DiffusionART <../examples/diffusion_art>` YAMLs (for example ``actor.model.wan22_ti2v_5b.use_lora: True``).

Configure Embodied FSDP LoRA
----------------------------

Enable LoRA in the ``actor.model`` section of an embodied FSDP YAML:

.. code-block:: yaml

   actor:
     model:
       is_lora: True
       lora_rank: 32
       lora_path: null            # Set to an existing adapter path to continue training
       lora_target_scope: null    # Optional: set to "all_linear" to adapt all linear layers

- ``is_lora``: enables (``True``) or disables (``False``) LoRA fine-tuning. Set ``is_lora: False`` for full-model fine-tuning.
- ``lora_rank``: rank of the low-rank matrices (typically ``8``–``64``). Ignored when ``is_lora: False``.
- ``lora_path``: path to an existing LoRA adapter to resume training from; leave ``null`` to initialize a new adapter.
- ``lora_target_scope``: when unset, RLinf injects LoRA into standard vision, projection, and LLM attention/MLP modules (``proj``, ``qkv``, ``fc1``–``fc3``, ``out_proj``, ``q_proj``/``k_proj``/``v_proj``/``o_proj``, ``gate_proj``/``up_proj``/``down_proj``, ``lm_head``). Set ``lora_target_scope: "all_linear"`` to target all linear layers.

Pre-Merge Existing LoRA Adapters
--------------------------------

To fold one or more existing LoRA adapters into the base model weights before initializing a fresh trainable adapter on top, list them under ``actor.model.merged_loras``. Each entry takes a ``path`` and an optional ``scale`` (defaults to ``1.0``):

.. code-block:: yaml

   actor:
     model:
       model_path: /path/to/base_model
       merged_loras:
         - path: /path/to/pretrained_lora
           scale: 0.5
       is_lora: True
       lora_rank: 32

RLinf merges every entry in ``merged_loras`` in order at model load time and then applies the trainable LoRA configured by ``is_lora``. For a working example that blends a ManiSkill adapter into a LIBERO checkpoint before multi-env RL, see ``examples/embodiment/config/maniskill_libero_grpo_openvlaoft.yaml`` and :doc:`Heterogeneous Environments <hetero_envs>`.

Export and Merge Trained LoRA Checkpoints
-----------------------------------------

After FSDP training finishes, export the ``full_weights.pt`` checkpoint to Hugging Face ``.safetensors`` format using :doc:`Checkpoint Conversion <convertor>`. Set ``convertor.merge_lora_weighs: True`` (default) to bake the trained LoRA weights into the exported model, or ``False`` to save only the adapter weights.
