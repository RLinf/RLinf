APIs
====

Look up RLinf's public Python interfaces, organized from foundational scheduling primitives up to workers, models, robotics, algorithms, and data containers.

Foundational Scheduling APIs
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - API
     - What you get
   * - :doc:`Worker <worker>`
     - ``Worker`` and ``WorkerGroup`` abstractions for distributed execution and communication.
   * - :doc:`Placement <placement>`
     - Placement strategies mapping worker groups onto nodes and accelerators.
   * - :doc:`Cluster <cluster>`
     - Ray cluster discovery, node groups, and worker allocation.
   * - :doc:`Channel <channel>`
     - Asynchronous producer-consumer queues with optional collectors and dispatchers.

RL Components, Models, Robotics, and Data
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - API
     - What you get
   * - :doc:`Actor <actor>`
     - FSDP and Megatron actor workers.
   * - :doc:`Rollout <rollout>`
     - Hugging Face and SGLang rollout workers.
   * - :doc:`Env <env>`
     - Environment worker wrappers for embodied tasks.
   * - :doc:`Models & Policies <models>`
     - Model registry (``register_model``, ``get_model``) and embodied ``BasePolicy`` contract.
   * - :doc:`Robotics <robotics>`
     - Hardware ``Connection``, ``RobotPart``, ``Robot``, and discovery APIs.
   * - :doc:`Algorithms & Registries <algorithms>`
     - Advantage, policy-loss, loss-scale, and tool-call parser registries.
   * - :doc:`Agentic Data <data>`
     - Agentic rollout requests and result structures.
   * - :doc:`Embodied Data <embodied_data>`
     - Embodied environment and rollout data structures.
   * - :doc:`Replay Buffer <replay_buffer>`
     - ``TrajectoryReplayBuffer``, ``TrajectoryCache``, and preload datasets.

.. toctree::
   :hidden:
   :maxdepth: 1

   Worker <worker>
   Placement <placement>
   Cluster <cluster>
   Channel <channel>
   Actor <actor>
   Rollout <rollout>
   Environment <env>
   Models & Policies <models>
   Robotics <robotics>
   Algorithms & Registries <algorithms>
   Data <data>
   Embodied Data <embodied_data>
   Replay Buffer <replay_buffer>
