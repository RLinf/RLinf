Advanced Integrations
=====================

Use these guides when you need low-level backend integration—manual Megatron weight resharding (when :doc:`Megatron-Bridge <../mbridge>` does not cover your model) or custom actor-to-rollout weight transport.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`Manual Megatron Model <../new_model_megatron>`
     - Hand-written Megatron actor and SGLang weight resharding integration.
   * - :doc:`Weight Synchronization <../weight_syncer>`
     - Optimize actor-to-rollout weight sync in embodied training.

.. toctree::
   :hidden:

   Manual Megatron Model <../new_model_megatron>
   Weight Synchronization <../weight_syncer>
