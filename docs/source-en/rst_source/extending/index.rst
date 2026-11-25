Extending
=========

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
     - :doc:`Embodied RL Model (FSDP) <new_model_fsdp>`
   * - New supervised fine-tuning model & dataset
     - ``register_model`` + ``FSDPSftWorker``
     - :doc:`Embodied SFT Model (FSDP) <new_model_sft>`
   * - New embodied rollout model on SGLang
     - SGLang embodied adapter registry
     - :doc:`Embodied Rollout Model (SGLang) <sglang_embodied_model>`
   * - New Hugging Face LLM on Megatron + SGLang (recommended)
     - Megatron-Bridge actor backend
     - :doc:`LLM Model (Megatron-Bridge) <mbridge>`
   * - Custom Megatron LLM adapter (manual resharding)
     - Manual Megatron actor & weight mapping
     - :doc:`LLM Model (Manual Megatron) <new_model_megatron>`
   * - New physics simulator
     - ``SupportedEnvType`` + ``get_env_cls()``
     - :doc:`Simulator Env <new_env>`
   * - New sensor, arm, or physical robot
     - Local part → robot tree → remote placement
     - :doc:`Real-World Hardware <new_robot>`
   * - New task on an already-supported robot
     - Real-world task config overlay
     - :doc:`Real-World Task <new_task>`
   * - New advantage or policy loss
     - ``@register_advantage`` / ``@register_policy_loss``
     - :doc:`Advantage & Policy Loss <new_algorithm>`
   * - New rule-based text reward
     - ``register_reward``
     - :doc:`Rule-Based Reward <rule_reward>`
   * - New model-based embodied reward
     - ``BaseRewardModel`` + ``reward.use_reward_model``
     - :doc:`Model-Based Reward <reward_model>`
   * - Build a white-box agent inside RLinf (token-level, e.g. Search-R1)
     - ``AgentLoopWorker`` / ``ToolWorker`` / ``@register_toolcall_parser``
     - :doc:`White-Box Agent <new_agent>`
   * - Connect a black-box agent over OpenAI API (e.g. OpenAI SDK, AutoGen)
     - Session Server ``/v1/chat/completions`` + ``run_session_trajectory``
     - :doc:`Black-Box Agent <openai_agent>`
   * - New GPU or NPU accelerator backend
     - ``AcceleratorManager`` + ``install.sh --platform``
     - :doc:`Accelerator Backend <new_accelerator>`
   * - Custom actor-to-rollout weight synchronization
     - ``WeightSyncer`` transport layer
     - :doc:`Weight Synchronization <weight_syncer>`
   * - Connect the agent session server to a custom runner
     - ``SessionServerWorker`` / ``create_payload_channels``
     - :doc:`Connect the Session Server to a Runner <session_server_runner>`

.. tip::

   **No fork needed:** To register a custom model, advantage, policy loss, rule reward, or tool-call parser from your own repository, expose a ``register()`` hook and export ``RLINF_EXT_MODULE=<your_pkg.module>`` before launching (see :ref:`out-of-tree-extensions` in :doc:`Embodied RL Model (FSDP) <new_model_fsdp>` and :doc:`Advantage & Policy Loss <new_algorithm>`).

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
      :link: new_algorithm
      :link-type: doc

      Register custom advantage estimators and policy loss functions.

   .. grid-item-card:: 4. Add a Reward
      :link: rewards/index
      :link-type: doc

      Register rule-based text scorers or train and plug in learned embodied reward models.

   .. grid-item-card:: 5. Add an Agent
      :link: agents/index
      :link-type: doc

      Build a token-level white-box agent inside RLinf (for example Search-R1), or plug in an existing black-box OpenAI API agent.

   .. grid-item-card:: 6. Add a Hardware Backend
      :link: new_accelerator
      :link-type: doc

      Port RLinf to a new GPU or NPU accelerator backend and installer platform.
