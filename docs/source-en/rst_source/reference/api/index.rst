Python API
==========

Look up RLinf's public Python interfaces, listed alphabetically across scheduling primitives, workers, models, robotics, algorithms, and data containers.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - API
     - What you get
   * - :doc:`Actor <actor>`
     - Megatron and FSDP actor and SFT workers plus model managers.
   * - :doc:`Agentic Data <data>`
     - Agentic rollout requests and result structures.
   * - :doc:`Algorithm <algorithms>`
     - Advantage, policy-loss, loss-scale, and tool-call parser registries.
   * - :doc:`Channel <channel>`
     - Asynchronous producer-consumer queues with optional collectors and dispatchers.
   * - :doc:`Cluster <cluster>`
     - Ray cluster discovery, node groups, and worker allocation.
   * - :doc:`Embodied Data <embodied_data>`
     - Embodied environment and rollout data structures.
   * - :doc:`Env <env>`
     - Environment worker, factory (``get_env_cls``, ``SupportedEnvType``), and ``RealWorldEnv``.
   * - :doc:`Model <models>`
     - Model registry (``register_model``, ``get_model``) and embodied ``BasePolicy`` contract.
   * - :doc:`Placement <placement>`
     - Placement strategies mapping worker groups onto nodes and accelerators.
   * - :doc:`Replay Buffer <replay_buffer>`
     - ``TrajectoryReplayBuffer``, ``TrajectoryCache``, and preload datasets.
   * - :doc:`Robotics <robotics>`
     - Hardware ``Connection``, ``RobotPart``, ``Robot``, and discovery APIs.
   * - :doc:`Rollout <rollout>`
     - SGLang, vLLM, and Hugging Face rollout workers.
   * - :doc:`Runner <runners>`
     - Top-level training loops for embodied, reasoning, agentic, and SFT workloads.
   * - :doc:`Worker <worker>`
     - ``Worker`` and ``WorkerGroup`` abstractions for distributed execution and communication.

.. toctree::
   :hidden:
   :maxdepth: 1

   Actor <actor>
   Agentic Data <data>
   Algorithm <algorithms>
   Channel <channel>
   Cluster <cluster>
   Embodied Data <embodied_data>
   Env <env>
   Model <models>
   Placement <placement>
   Replay Buffer <replay_buffer>
   Robotics <robotics>
   Rollout <rollout>
   Runner <runners>
   Worker <worker>
