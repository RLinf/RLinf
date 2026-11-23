Extending RLinf
===============

Plug your own model, simulator, physical robot, reward, algorithm, agent tool, or hardware backend into RLinf without modifying the distributed runtime. Pick the group that matches what you want to add.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/extending_map.svg
   :align: center
   :width: 100%

   RLinf's extension layers on top of the shared distributed runtime.

Which Guide Should I Read?
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 36 28

   * - What You Want to Add
     - Extension Point
     - Guide
   * - New VLA or embodied RL policy (FSDP)
     - ``register_model`` + ``BasePolicy``
     - :doc:`Embodied Policy (FSDP) <new_model_fsdp>`
   * - New supervised fine-tuning model & dataset
     - ``register_model`` + ``FSDPSftWorker``
     - :doc:`SFT Model & Dataset <new_model_sft>`
   * - New embodied rollout model on SGLang
     - SGLang embodied adapter registry
     - :doc:`Embodied Rollout (SGLang) <sglang_embodied_model>`
   * - New Hugging Face LLM on Megatron + SGLang (recommended)
     - Megatron-Bridge actor backend
     - :doc:`LLM via Megatron-Bridge <mbridge>`
   * - Custom Megatron LLM adapter (manual resharding)
     - Manual Megatron actor & weight mapping
     - :doc:`LLM via Manual Megatron <new_model_megatron>`
   * - New physics simulator
     - ``SupportedEnvType`` + ``get_env_cls()``
     - :doc:`Simulator Environment <new_env>`
   * - New sensor, arm, or physical robot
     - Local part → robot tree → remote placement
     - :doc:`Real-World Robot Hardware <new_robot>`
   * - New task on an already-supported robot
     - Real-world task config overlay
     - :doc:`Real-World Task <new_task>`
   * - New advantage, policy loss, or rule reward
     - ``@register_advantage`` / ``@register_policy_loss`` / ``register_reward``
     - :doc:`Advantage, Loss & Rule Reward <new_algorithm>`
   * - New learned embodied reward model
     - ``BaseRewardModel`` + ``reward.use_reward_model``
     - :doc:`Learned Reward Model <reward_model>`
   * - New multi-turn agent loop, tool, or parser
     - ``AgentLoopWorker`` / ``ToolWorker`` / ``@register_toolcall_parser``
     - :doc:`Agent Loop & Tool Calling <new_agent>`
   * - New GPU or NPU accelerator backend
     - ``AcceleratorManager`` + ``install.sh --platform``
     - :doc:`Accelerator Backend <new_accelerator>`
   * - Custom actor-to-rollout weight synchronization
     - ``WeightSyncer`` transport layer
     - :doc:`Weight Synchronization <weight_syncer>`

.. tip::

   **No fork needed:** To register a custom model, advantage, policy loss, rule reward, or tool-call parser from your own repository, expose a ``register()`` hook and export ``RLINF_EXT_MODULE=<your_pkg.module>`` before launching (see :ref:`out-of-tree-extensions` in :doc:`Embodied Policy (FSDP) <new_model_fsdp>` and :doc:`Advantage, Loss & Rule Reward <new_algorithm>`).

Choose an Extension Area
------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. Add a Model
      :link: models/index
      :link-type: doc

      Add an embodied FSDP policy, SFT model, SGLang embodied rollout model, Megatron LLM, or custom weight synchronization.

   .. grid-item-card:: 2. Add an Env or Robot
      :link: envs-robots/index
      :link-type: doc

      Integrate a new physics simulator, bring up new robot hardware, or add a task on a supported robot.

   .. grid-item-card:: 3. Add an Algorithm
      :link: algorithms-agents/index
      :link-type: doc

      Register custom advantages, policy losses, and rule-based or learned reward models.

   .. grid-item-card:: 4. Add an Agent
      :link: new_agent
      :link-type: doc

      Implement custom multi-turn agent loops, tool workers, and tool-call parsers.

   .. grid-item-card:: 5. Add a Hardware Backend
      :link: new_accelerator
      :link-type: doc

      Port RLinf to a new GPU or NPU accelerator backend and installer platform.
