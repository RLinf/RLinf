Extending RLinf
===============

Plug your own model, simulator, physical robot, reward model, or training backend into RLinf without modifying the distributed runtime. Pick the smallest extension layer that matches your goal.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/extending_map.svg
   :align: center
   :width: 100%

   RLinf's four extension layers on top of the shared distributed runtime.

Which Guide Should I Read?
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 38 30

   * - Goal
     - Extension Point
     - Start Here
   * - Train a new VLA or LLM with FSDP
     - ``register_model`` + ``BasePolicy``
     - :doc:`New Model with FSDP <new_model_fsdp>`
   * - Train a large LLM with Megatron + SGLang
     - Megatron actor + rollout adapter
     - :doc:`New Model with Megatron <new_model_megatron>`
   * - Run embodied rollout with SGLang
     - SGLang embodied rollout adapter
     - :doc:`SGLang Embodied Model <sglang_embodied_model>`
   * - Add a supervised fine-tuning model
     - SFT worker model registry
     - :doc:`New SFT Model <new_model_sft>`
   * - Add a new physics simulator
     - ``SupportedEnvType`` + ``get_env_cls()``
     - :doc:`New Environment <new_env>`
   * - Add a task on an already-supported robot
     - Real-world task config overlay (no robotics code)
     - :doc:`New Real-World Tasks <new_task>`
   * - Connect a new sensor, arm, or robot
     - Local part → compose parts → remote placement
     - :doc:`New Robot <new_robot>`
   * - Train and plug in a learned reward model
     - Reward worker + ``reward.use_reward_model``
     - :doc:`Reward Model <reward_model>`
   * - Customize Megatron-Bridge or weight sync
     - Backend bridge / weight syncer
     - :doc:`Advanced Integrations <advanced-integrations/index>`

1. Add a Model
--------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: New Model with FSDP
      :link: new_model_fsdp
      :link-type: doc

      Register a Hugging Face or embodied VLA policy on the FSDP backend.

   .. grid-item-card:: New Model with Megatron
      :link: new_model_megatron
      :link-type: doc

      Add a Hugging Face model on the Megatron + SGLang backend.

   .. grid-item-card:: SGLang Embodied Model
      :link: sglang_embodied_model
      :link-type: doc

      Adapt an embodied policy to the SGLang rollout worker for fast inference and evaluation.

   .. grid-item-card:: New SFT Model
      :link: new_model_sft
      :link-type: doc

      Plug a new model into the RLinf SFT training pipeline.

2. Add an Environment or Robot
------------------------------

Start at the simplest layer that fits: if RLinf already talks to your robot hardware, add a real-world task without writing robotics code; for new hardware, verify one local part before composing or placing it remotely.

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: New Simulator Environment
      :link: new_env
      :link-type: doc

      Wrap a simulator backend, register its environment type, and map action formats.

   .. grid-item-card:: New Real-World Tasks
      :link: new_task
      :link-type: doc

      Define a new task on physical hardware RLinf already supports.

   .. grid-item-card:: New Robot
      :link: new_robot
      :link-type: doc

      Build one local part, compose a robot tree, and move it to remote placement.

3. Add Rewards and Algorithms
-----------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Reward Model
      :link: reward_model
      :link-type: doc

      Collect trajectories, train a ResNet or VLM reward model, and wire it into online RL.

   .. grid-item-card:: Architecture & Extension Overview
      :link: overview
      :link-type: doc

      See how registries (models, envs, advantages, losses, rewards) compose inside RLinf.

4. Backend and Engine Integrations
----------------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Advanced Integrations
      :link: advanced-integrations/index
      :link-type: doc

      Integrate Megatron-Bridge and optimize actor-to-rollout weight synchronization.
