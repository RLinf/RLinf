Guides
======

Follow these guides in the order you scale a run: get logging and checkpoints right, expand to multiple nodes, diagnose bottlenecks and speed up throughput, and customize rollout backends when needed.

Where Are You in Your Run?
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 42 24

   * - Current Goal
     - What You Will Set Up
     - Start Here
   * - Run, monitor, and manage a training job
     - Logging, resume, checkpoint conversion, LoRA, and multi-env training
     - :doc:`Configure & Run <configure/index>`
   * - Scale beyond one GPU or one machine
     - Multi-node Ray, heterogeneous clusters, cloud-edge, placement, 5D parallelism, dynamic scheduling
     - :doc:`Scale to a Cluster <launch-scale/index>`
   * - Training is slow or GPUs sit idle
     - Profiling first, then pipeline mode, decoupled envs, observation compression, async Agent RL, RTC, FUSCO
     - :doc:`Profile & Speed Up <performance/index>`
   * - Switch or serve a dedicated rollout engine
     - SGLang & vLLM versions, SGLang server, HTTP inference client, Session Server, PhyAI, ApxInf
     - :doc:`Rollout Backends <rollout-engines/index>`

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. Configure & Run
      :link: configure/index
      :link-type: doc

      Track metrics, resume runs, convert checkpoints, enable LoRA, and train across multiple environments.

   .. grid-item-card:: 2. Scale to a Cluster
      :link: launch-scale/index
      :link-type: doc

      Launch across nodes, heterogeneous CPU/GPU clusters, cloud-edge links, 5D parallelism, and dynamic scheduling.

   .. grid-item-card:: 3. Profile & Speed Up
      :link: performance/index
      :link-type: doc

      Profile bottlenecks first, then enable pipelining, decoupled envs, observation compression, async Agent RL, RTC, or FUSCO.

   .. grid-item-card:: 4. Rollout Backends
      :link: rollout-engines/index
      :link-type: doc

      Choose SGLang or vLLM versions, run SGLang servers and HTTP clients, or accelerate embodied rollout with PhyAI and ApxInf.

Looking for teleoperation, data collection, or two-node robot launch? See :doc:`Real-World Workflows <../examples/real_world_workflows_index>`.
