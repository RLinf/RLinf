Performance
===========

Use these guides to improve latency, throughput, memory use, and placement for
embodied RL and SFT workloads.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`RTC <../rtc>`
     - Hide policy inference latency by overlapping it with action-chunk execution, in simulation and on real robots.
   * - :doc:`Env Decoupled Mode <../env_decoupled_mode>`
     - Decouple Env Workers from Rollout Workers for dynamic embodied rollout scheduling.
   * - :doc:`Observation Compression <../obs_compression>`
     - Losslessly compress image observations on the Env to Rollout channel to save bandwidth.
   * - :doc:`LoRA <../lora>`
     - Train with LoRA adapters.
   * - :doc:`FSDP Practical Tips <../fsdp_tips>`
     - Configure hybrid sharding for multi-node embodied training.
   * - :doc:`Auto Placement <../auto_placement>`
     - Auto-select the best placement for a workload.
   * - :doc:`Profiling <../profile>`
     - System-level profiling of Ray worker processes.

.. toctree::
   :hidden:

   RTC <../rtc>
   Env Decoupled Mode <../env_decoupled_mode>
   Observation Compression <../obs_compression>
   LoRA <../lora>
   FSDP Practical Tips <../fsdp_tips>
   Auto Placement <../auto_placement>
   Profiling <../profile>
