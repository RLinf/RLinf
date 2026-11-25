Add a Model
===========

Pick the guide that matches your model family and training or rollout backend.

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Guide
     - Best For
     - Backend
   * - :doc:`Embodied Policy (FSDP) <../new_model_fsdp>`
     - New VLA or embodied RL policy
     - FSDP + Hugging Face
   * - :doc:`SFT Model & Dataset <../new_model_sft>`
     - Supervised fine-tuning with custom datasets
     - FSDP SFT Worker
   * - :doc:`Embodied Rollout (SGLang) <../sglang_embodied_model>`
     - Fast embodied evaluation / world-action rollouts
     - SGLang Embodied Worker
   * - :doc:`LLM via Megatron-Bridge <../mbridge>`
     - Hugging Face LLMs on Megatron + SGLang (recommended)
     - Megatron-Bridge
   * - :doc:`LLM via Manual Megatron <../new_model_megatron>`
     - Custom architectures unsupported by Megatron-Bridge
     - Manual Megatron + SGLang Resharding
   * - :doc:`Weight Synchronization <../weight_syncer>`
     - Optimize or customize actor-to-rollout model weight transfer
     - ``WeightSyncer`` (FSDP + HF Rollout)

.. toctree::
   :hidden:

   Embodied Policy (FSDP) <../new_model_fsdp>
   SFT Model & Dataset <../new_model_sft>
   Embodied Rollout (SGLang) <../sglang_embodied_model>
   LLM via Megatron-Bridge <../mbridge>
   LLM via Manual Megatron <../new_model_megatron>
   Weight Synchronization <../weight_syncer>
