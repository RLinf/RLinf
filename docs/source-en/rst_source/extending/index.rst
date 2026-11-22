Extending RLinf
===============

Plug your own model, simulator, physical robot, algorithm, or reward into RLinf without modifying the distributed runtime. Pick the smallest extension layer that matches your goal.

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
     - :doc:`FSDP Model <new_model_fsdp>`
   * - Train a Hugging Face LLM with Megatron + SGLang
     - Megatron-Bridge actor backend (recommended)
     - :doc:`Megatron-Bridge <mbridge>`
   * - Run embodied rollout with SGLang
     - SGLang embodied rollout adapter
     - :doc:`SGLang Embodied <sglang_embodied_model>`
   * - Add a supervised fine-tuning model
     - SFT worker model registry
     - :doc:`SFT Model <new_model_sft>`
   * - Add a new physics simulator
     - ``SupportedEnvType`` + ``get_env_cls()``
     - :doc:`Simulator Env <new_env>`
   * - Add a task on an already-supported robot
     - Real-world task config overlay (no robotics code)
     - :doc:`Real-World Task <new_task>`
   * - Connect a new sensor, arm, or robot
     - Local part → compose parts → remote placement
     - :doc:`Real-World Robot <new_robot>`
   * - Add an advantage, policy loss, or rule reward
     - ``@register_advantage`` / ``@register_policy_loss`` / ``register_reward``
     - :doc:`Algorithms & Rule Rewards <new_algorithm>`
   * - Train and plug in a learned reward model
     - Reward worker + ``reward.use_reward_model``
     - :doc:`Reward Model <reward_model>`
   * - Write a manual Megatron adapter or custom weight sync
     - Low-level Megatron resharding / weight syncer
     - :doc:`Advanced Integrations <advanced-integrations/index>`

.. _out-of-tree-extensions:

Out-of-Tree Extensions
----------------------

You do not need to fork RLinf to register a custom model, advantage, policy loss, or rule-based reward. Because Ray launches each worker in its own Python process, a registration call made only in your driver script does not reach worker processes.

Put your registration calls inside a ``register()`` function in your own package and export ``RLINF_EXT_MODULE`` before launching training or checkpoint conversion:

.. code-block:: python

   # my_pkg/rlinf_ext.py
   from rlinf.models import register_model

   def build_my_model(cfg, torch_dtype):
       from my_pkg.policy import CustomPolicy
       return CustomPolicy(cfg, torch_dtype)

   def register():
       register_model("my_custom_model", build_my_model, category="embodied")

.. code-block:: bash

   export RLINF_EXT_MODULE=my_pkg.rlinf_ext

RLinf imports the module and calls ``register()`` automatically in every worker process and standalone checkpoint convertor before loading configs or models.

1. Models
---------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: FSDP Model
      :link: new_model_fsdp
      :link-type: doc

      Register a Hugging Face or embodied VLA policy on the FSDP backend.

   .. grid-item-card:: Megatron-Bridge
      :link: mbridge
      :link-type: doc

      Recommended path for adding Hugging Face LLMs to Megatron + SGLang training without hand-written weight conversion.

   .. grid-item-card:: SGLang Embodied
      :link: sglang_embodied_model
      :link-type: doc

      Adapt an embodied policy to the SGLang rollout worker for fast inference and evaluation.

   .. grid-item-card:: SFT Model
      :link: new_model_sft
      :link-type: doc

      Plug a new model into the RLinf SFT training pipeline.

2. Environments & Robots
------------------------

Start at the simplest layer that fits: if RLinf already talks to your robot hardware, add a real-world task without writing robotics code; for new hardware, verify one local part before composing or placing it remotely.

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Simulator Env
      :link: new_env
      :link-type: doc

      Wrap a simulator backend, register its environment type, and map action formats.

   .. grid-item-card:: Real-World Task
      :link: new_task
      :link-type: doc

      Define a new task on physical hardware RLinf already supports.

   .. grid-item-card:: Real-World Robot
      :link: new_robot
      :link-type: doc

      Build one local part, compose a robot tree, and move it to remote placement.

3. Algorithms & Rewards
-----------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Algorithms & Rule Rewards
      :link: new_algorithm
      :link-type: doc

      Register custom advantage estimators, policy losses, and rule-based text rewards in minutes.

   .. grid-item-card:: Reward Model
      :link: reward_model
      :link-type: doc

      Collect trajectories, train a ResNet or VLM reward model, and wire it into online RL.

4. Advanced Integrations
------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Manual Megatron Model
      :link: new_model_megatron
      :link-type: doc

      Low-level manual Megatron + SGLang model integration when Megatron-Bridge does not cover your architecture.

   .. grid-item-card:: Weight Synchronization
      :link: weight_syncer
      :link-type: doc

      Optimize actor-to-rollout weight synchronization (``patch`` and ``bucket`` modes) in embodied training.
