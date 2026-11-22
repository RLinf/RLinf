Extending RLinf
===============

Plug your own model, simulator, physical robot, reward, algorithm, agent tool, or accelerator into RLinf without modifying the distributed runtime. Pick the smallest extension layer that matches your goal.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/extending_map.svg
   :align: center
   :width: 100%

   RLinf's extension layers on top of the shared distributed runtime.

Which Guide Should I Read?
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - Goal
     - Extension Point
     - Start Here
   * - Train a new VLA or embodied RL policy (FSDP)
     - ``register_model`` + ``BasePolicy``
     - :doc:`FSDP Model <new_model_fsdp>`
   * - Add a supervised fine-tuning model
     - ``register_model`` + ``FSDPSftWorker``
     - :doc:`SFT Model <new_model_sft>`
   * - Run embodied evaluation rollout with SGLang
     - SGLang embodied adapter registry
     - :doc:`SGLang Embodied <sglang_embodied_model>`
   * - Add a new physics simulator
     - ``SupportedEnvType`` + ``get_env_cls()``
     - :doc:`Simulator Env <new_env>`
   * - Connect a new sensor, arm, or physical robot
     - Local part → compose parts → remote placement
     - :doc:`Real-World Robot <new_robot>`
   * - Add a task on an already-supported robot
     - Real-world task config overlay (no robotics code)
     - :doc:`Real-World Task <new_task>`
   * - Train or subclass a learned embodied reward model
     - ``BaseRewardModel`` + ``reward.use_reward_model``
     - :doc:`Reward Model <reward_model>`
   * - Add an advantage, policy loss, or rule reward
     - ``@register_advantage`` / ``@register_policy_loss`` / ``register_reward``
     - :doc:`Algorithms & Rule Rewards <new_algorithm>`
   * - Add a multi-turn agent loop, tool, or parser
     - ``AgentLoopWorker`` / ``ToolWorker`` / ``@register_toolcall_parser``
     - :doc:`Agent & Tool Calling <new_agent>`
   * - Train a Hugging Face LLM with Megatron + SGLang
     - Megatron-Bridge actor backend (recommended)
     - :doc:`Megatron-Bridge <mbridge>`
   * - Port RLinf to a new GPU or NPU accelerator
     - ``AcceleratorManager`` subclass + ``install.sh --platform``
     - :doc:`Accelerator Backend <new_accelerator>`
   * - Write a manual Megatron adapter or custom weight sync
     - Low-level Megatron resharding / ``WeightSyncer``
     - :doc:`Advanced Integrations <advanced-integrations/index>`

.. _out-of-tree-extensions:

Out-of-Tree Extensions
----------------------

You do not need to fork RLinf to register a custom model, advantage, policy loss, rule-based reward, or tool-call parser. Because Ray launches each worker in its own Python process, a registration call made only in your driver script does not reach worker processes.

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

1. Embodied Models
------------------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: FSDP Model
      :link: new_model_fsdp
      :link-type: doc

      Register a Hugging Face or embodied VLA policy on the FSDP backend.

   .. grid-item-card:: SFT Model
      :link: new_model_sft
      :link-type: doc

      Plug a new model and dataset into the RLinf SFT training pipeline.

   .. grid-item-card:: SGLang Embodied
      :link: sglang_embodied_model
      :link-type: doc

      Adapt an embodied policy to the SGLang rollout worker for fast evaluation.

2. Embodied Environments, Robots & Rewards
------------------------------------------

Start at the simplest layer that fits: if RLinf already talks to your robot hardware, add a real-world task without writing robotics code; for new hardware, verify one local part before composing or placing it remotely.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Simulator Env
      :link: new_env
      :link-type: doc

      Wrap a simulator backend, register its environment type, and map action formats.

   .. grid-item-card:: Real-World Robot
      :link: new_robot
      :link-type: doc

      Build one local part, compose a robot tree, and move it to remote placement.

   .. grid-item-card:: Real-World Task
      :link: new_task
      :link-type: doc

      Define a new task on physical hardware RLinf already supports.

   .. grid-item-card:: Reward Model
      :link: reward_model
      :link-type: doc

      Extend ``BaseRewardModel`` or train built-in ResNet / VLM reward models for embodied RL.

3. Algorithms, Agents & LLMs
----------------------------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Algorithms & Rule Rewards
      :link: new_algorithm
      :link-type: doc

      Register custom advantage estimators, policy losses, and rule-based text rewards.

   .. grid-item-card:: Agent & Tool Calling
      :link: new_agent
      :link-type: doc

      Implement custom multi-turn agent loops, tool workers, and tool-call parsers.

   .. grid-item-card:: Megatron-Bridge
      :link: mbridge
      :link-type: doc

      Recommended path for training Hugging Face LLMs on Megatron + SGLang without manual weight conversion.

4. Hardware & Advanced Integrations
-----------------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Accelerator Backend
      :link: new_accelerator
      :link-type: doc

      Add a new GPU or NPU accelerator backend to RLinf's hardware scheduler and installer.

   .. grid-item-card:: Advanced Integrations
      :link: advanced-integrations/index
      :link-type: doc

      Manual Megatron model integration and custom actor-to-rollout weight synchronization.
