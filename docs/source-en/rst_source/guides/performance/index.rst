Performance
===========

Use these guides when latency, throughput, memory, placement, or large-model
training efficiency becomes the bottleneck.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`RTC <../rtc>`
     - Hide policy inference latency by overlapping it with action-chunk execution, in simulation and on real robots.
   * - :doc:`Env Decoupled <../env_decoupled_mode>`
     - Decouple Env Workers from Rollout Workers for dynamic embodied rollout scheduling.
   * - :doc:`Training Pipeline Mode <../training_pipeline>`
     - Stream packed actor micro-batches from env workers to overlap embodied rollout and FSDP training.
   * - :doc:`Observation Compression <../obs_compression>`
     - Losslessly compress image observations on the Env to Rollout channel to save bandwidth.
   * - :doc:`LoRA <../lora>`
     - Train with LoRA adapters.
   * - :doc:`Auto Placement <../auto_placement>`
     - Auto-select the best placement for a workload.
   * - :doc:`Dynamic Scheduling <../dynamic_scheduling>`
     - Dynamically schedule resources during training.
   * - :doc:`Profiling <../profile>`
     - System-level profiling of Ray worker processes.
   * - :doc:`5D Parallelism <../5D>`
     - Configure 5D parallelism for large models.
   * - :doc:`Async Agentic RL <../async_agentic>`
     - Overlap multi-turn agent rollout with training; two admission limits bound how stale the training data gets.
   * - :doc:`FUSCO <../../examples/system/fusco>`
     - Fuse data transformation and All-to-All communication to accelerate Megatron MoE training.

.. toctree::
   :hidden:

   RTC <../rtc>
   Env Decoupled <../env_decoupled_mode>
   Training Pipeline Mode <../training_pipeline>
   Observation Compression <../obs_compression>
   LoRA <../lora>
   Auto Placement <../auto_placement>
   Dynamic Scheduling <../dynamic_scheduling>
   Profiling <../profile>
   5D Parallelism <../5D>
   Async Agentic RL <../async_agentic>
   FUSCO <../../examples/system/fusco>
