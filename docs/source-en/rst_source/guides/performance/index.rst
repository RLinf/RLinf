Performance & Parallelism
=========================

Use these guides when latency, throughput, GPU memory, placement, or large-model training efficiency becomes the bottleneck.

Universal
---------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Guide
     - Pick this when (symptom)
     - What you get
   * - :doc:`LoRA <../lora>`
     - Full-parameter fine-tuning runs out of GPU memory.
     - Train low-rank adapters on embodied FSDP VLAs (``is_lora`` / ``merged_loras``) or diffusion models (``use_lora``).
   * - :doc:`Profiling <../profile>`
     - You do not know whether env, rollout, communication, or actor training is the bottleneck.
     - System-level tracing and timeline profiling across Ray worker processes.

Embodied Throughput
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Guide
     - Pick this when (symptom)
     - What you get
   * - :doc:`RTC <../rtc>`
     - Policy inference latency causes pauses between action chunks on robots or simulators.
     - Overlap policy inference with action-chunk execution in simulation and on real robots.
   * - :doc:`Env Decoupled <../env_decoupled_mode>`
     - Slow or variable simulator steps leave rollout GPUs waiting on stragglers.
     - Decouple env workers from rollout workers for dynamic embodied rollout scheduling.
   * - :doc:`Training Pipeline Mode <../training_pipeline>`
     - Actor GPUs sit idle until a full rollout epoch finishes, or large trajectory transfers stall steps.
     - Split rollout into pipeline stages and stream packed micro-batches from env workers to FSDP actors.
   * - :doc:`Observation Compression <../obs_compression>`
     - High-resolution image frames saturate network or IPC bandwidth between env and rollout.
     - Losslessly compress image observations on the env-to-rollout channel with ``lz4`` or ``zstd``.

Large-Model & Agentic
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Guide
     - Pick this when (symptom)
     - What you get
   * - :doc:`5D Parallelism <../5D>`
     - A large LLM or MoE model does not fit on one GPU or slows down on long sequences.
     - Configure Megatron TP, DP, PP, SP, CP, and MoE Expert Parallelism (EP) with GPU sizing presets.
   * - :doc:`Async Agentic RL <../async_agentic>`
     - Multi-turn agent rollout leaves training GPUs idle in synchronous steps.
     - Overlap multi-turn rollout with Megatron training using bounded in-flight and buffer staleness limits.
   * - :doc:`Auto Placement <../auto_placement>`
     - You want the fastest collocated vs disaggregated GPU split for a Megatron reasoning run.
     - Search optimal ``cluster.component_placement`` layouts from measured per-iteration timings.
   * - :doc:`Dynamic Scheduling <../dynamic_scheduling>`
     - Rollout finishes long before actor updates on disaggregated Megatron-LM 0.11 pipelines.
     - Reallocate GPUs across rollout, inference, and actor at runtime (legacy Megatron-LM 0.11 setup).
   * - :doc:`FUSCO <../../examples/system/fusco>`
     - MoE All-to-All communication dominates Megatron actor step time.
     - Fuse token permutation and All-to-All communication to accelerate Megatron MoE training.

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
