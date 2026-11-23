Architecture Overview
=====================

This page gives a five-minute visual mental model of RLinf: how data flows through a training step, how components map to hardware, and which data format connects each stage.

Core RL Loop
------------

Every RLinf run starts from one entry script that creates a Ray-backed ``Cluster``, launches ``Env``, ``Rollout``, ``Reward`` (optional), and ``Actor`` worker groups on hardware, and hands control to a ``Runner`` that repeats rollout, advantage computation, policy update, and weight sync.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/core_dataflow.svg
   :align: center
   :width: 100%
   :alt: RLinf core training loop connecting Env, Rollout, Reward and Buffer, Actor, and Weight Sync

   Core RLinf training loop and weight synchronization.

.. list-table::
   :header-rows: 1
   :widths: 20 26 54

   * - Component
     - Config section
     - Responsibility
   * - Env / Agent
     - ``env.*``, ``agentloop.*``
     - Steps simulators, real-world robots, or tool sandboxes and returns observations and rewards.
   * - Rollout
     - ``rollout.*``
     - Runs policy inference (HuggingFace, SGLang, or vLLM) to produce actions, responses, and logprobs.
   * - Reward & Buffer
     - ``algorithm.*``, ``reward.*``
     - Computes rule-based or learned rewards, estimates GAE / GRPO advantages, or serves off-policy replay batches.
   * - Actor
     - ``actor.*``
     - Updates policy and value weights with FSDP or Megatron, syncs weights back to rollout, and saves checkpoints.

For internal execution details, see :doc:`execution_flow`, :doc:`worker`, and :doc:`channel`.

Deployment Modes
----------------

Collocated, disaggregated, and hybrid execution are three placements of the same workers—change ``cluster.component_placement`` in YAML rather than writing separate code.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/deployment_modes.svg
   :align: center
   :width: 100%
   :alt: Comparison of Collocated, Disaggregated, and Hybrid placement modes

   Collocated, disaggregated, and hybrid component placement.

.. list-table::
   :header-rows: 1
   :widths: 20 24 56

   * - Mode
     - Hardware scale
     - When to choose it
   * - Collocated
     - 1–8 GPUs
     - Default for quickstarts and single-node training; all components share GPUs with offload.
   * - Disaggregated
     - 8+ GPUs
     - Dedicated GPUs per component for zero offload stalls, pipelining, and asynchronous RL.
   * - Hybrid / Heterogeneous
     - Multi-node or robot + GPU
     - Mixes shared GPU components with separate CPU workers or physical robot control hosts.

For placement syntax and multi-node topologies, see :doc:`execution_modes`, :doc:`placement`, and :doc:`../guides/multi_node`.

Data Formats
------------

Embodied workflows connect teleoperation, simulation rollouts, SFT, online RLPD, and reward modeling through three on-disk formats under ``save_dir`` (defaults to ``${runner.logger.log_path}``).

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/data_formats.svg
   :align: center
   :width: 100%
   :alt: RLinf data collection formats flowing to SFT, online RLPD, and reward model training

   Data collection sources, stored formats, and downstream consumers.

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - Format
     - Directory
     - Downstream stage
   * - ``lerobot_dataset``
     - ``{save_dir}/lerobot_dataset/``
     - OpenPI / VLA SFT, HG-DAgger, and offline RL (RECAP, STEAM).
   * - ``replay_buffer``
     - ``{save_dir}/demo_buffer/``
     - Online RLPD / SAC via ``algorithm.demo_buffer.load_path``.
   * - ``pickle_episode``
     - ``{save_dir}/pickle_episode/``
     - ResNet binary and VLM trend reward-model preprocessing.

Next Steps
----------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Training Configuration
      :link: ../reference/configuration
      :link-type: doc

      Read an annotated minimal YAML and Hydra preset map.

   .. grid-item-card:: Data Collection
      :link: ../guides/data_collection
      :link-type: doc

      Configure export formats, streaming shards, and dataset merging.

   .. grid-item-card:: First Training Run
      :link: ../start/vla
      :link-type: doc

      Run a single-GPU Embodied AI or Agent & LLM training example.
