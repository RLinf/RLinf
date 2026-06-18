Performance
===========

Use these guides when throughput, memory, placement, or large-model training
efficiency becomes the bottleneck.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`Auto Placement <../auto_placement>`
     - Auto-select the best placement for a workload.
   * - :doc:`Dynamic Scheduling <../dynamic_scheduling>`
     - Dynamically schedule resources during training.
   * - :doc:`Profiling <../profile>`
     - System-level profiling of Ray worker processes.
   * - :doc:`5D Parallelism <../5D>`
     - Configure 5D parallelism for large models.
   * - :doc:`LoRA <../lora>`
     - Train with LoRA adapters.

.. toctree::
   :hidden:

   Auto Placement <../auto_placement>
   Dynamic Scheduling <../dynamic_scheduling>
   Profiling <../profile>
   5D Parallelism <../5D>
   LoRA <../lora>
