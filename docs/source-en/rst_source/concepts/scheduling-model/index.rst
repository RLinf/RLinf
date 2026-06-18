Scheduling Model
================

Use these concepts when you need to reason about where work runs and how RLinf
stores trajectory data.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Concept
     - What you get
   * - :doc:`Placement <../placement>`
     - How workers map onto nodes and GPUs.
   * - :doc:`Execution Modes <../execution_modes>`
     - Collocated, disaggregated, and hybrid placement trade-offs.
   * - :doc:`Replay Buffer <../replay_buffer>`
     - Trajectory replay buffer design and sampling.

.. toctree::
   :hidden:

   Placement <../placement>
   Execution Modes <../execution_modes>
   Replay Buffer <../replay_buffer>
