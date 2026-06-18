APIs
==========


Walk through RLinf's most essential API interfaces and how to use them.
These key APIs are exposed to users to simplify the complex data flows of RL, allowing them to focus on higher-level abstractions without needing to worry about the underlying implementations.

This API documentation proceeds bottom-up, starting with the foundational APIs of RLinf, including:

.. list-table::
   :header-rows: 1

   * - API
     - What you get
   * - :doc:`Worker <worker>`
     - A unified interface for workers and worker groups.
   * - :doc:`Placement <placement>`
     - An introduction to RLinf’s GPU placement strategies.
   * - :doc:`Cluster <cluster>`
     - Support for distributed training via clusters.
   * - :doc:`Channel <channel>`
     - Low-level communication primitives, including a producer-consumer queue abstraction.

After that, we introduce the upper-layer APIs used to implement different stages of RL:

.. list-table::
   :header-rows: 1

   * - API
     - What you get
   * - :doc:`Actor <actor>`
     - Actor wrappers based on FSDP and Megatron.
   * - :doc:`Rollout <rollout>`
     - Rollout wrappers built on Hugging Face and SGLang.
   * - :doc:`Env <env>`
     - Environment wrappers for embodied intelligence scenarios.
   * - :doc:`Data <data>`
     - Data structures transferred between different workers.
   * - :doc:`Embodied Data <embodied_data>`
     - Embodied Env/Rollout data structures.
   * - :doc:`Replay Buffer <replay_buffer>`
     - Trajectory replay buffer design and sampling.

.. toctree::
   :hidden:
   :maxdepth: 1

   worker
   placement
   cluster
   channel

   actor
   rollout
   env
   data
   embodied_data
   replay_buffer
