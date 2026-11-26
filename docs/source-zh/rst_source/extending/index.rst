扩展开发
========

无需修改 RLinf 的分布式调度核心，即可接入自定义模型、仿真器、真机机器人、Reward、算法、Agent 工具或加速卡后端。直接按你要新增的组件类别进入对应分组即可。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/extending_map_zh.svg
   :align: center
   :width: 100%

   RLinf 在共享分布式运行时之上的扩展插槽。

快速选型：我要新增什么？
------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 36 28

   * - 开发目标
     - 对应扩展点
     - 指南入口
   * - 接入新的 VLA 或具身 RL 策略（FSDP）
     - ``register_model`` 与 ``BasePolicy``
     - :doc:`具身 RL 模型（FSDP） <new_model_fsdp>`
   * - 接入新的 SFT 监督微调模型与数据集
     - ``register_model`` 与 ``FSDPSftWorker``
     - :doc:`具身 SFT 模型（FSDP） <new_model_sft>`
   * - 为具身模型接入 SGLang Rollout 加速
     - SGLang 具身适配注册表
     - :doc:`具身 Rollout 模型（SGLang） <sglang_embodied_model>`
   * - 在 Megatron + SGLang 上训练 Hugging Face LLM（推荐）
     - Megatron-Bridge Actor 后端
     - :doc:`LLM 模型（Megatron-Bridge） <mbridge>`
   * - 手动编写 Megatron LLM 适配与权重切分
     - 手动 Megatron Actor 适配
     - :doc:`LLM 模型（手动 Megatron） <new_model_megatron>`
   * - 接入新的物理仿真 Env
     - ``SupportedEnvType`` 与 ``get_env_cls()``
     - :doc:`仿真 Env <new_env>`
   * - 接入新的传感器、机械臂或真机硬件
     - 单部件本地验证 → 组合整机 → 远程 Placement
     - :doc:`真机硬件 <new_robot>`
   * - 在已有真机平台上新增任务
     - 真机任务配置叠加（无需改驱动）
     - :doc:`真机任务 <new_task>`
   * - 新增 Advantage 或 Policy Loss
     - ``@register_advantage`` / ``@register_policy_loss``
     - :doc:`Advantage 与 Policy Loss <new_algorithm>`
   * - 新增基于规则的文本 Reward
     - ``register_reward``
     - :doc:`基于规则的 Reward <rule_reward>`
   * - 训练或自定义基于模型的具身 Reward
     - ``BaseRewardModel`` 与 ``reward.use_reward_model``
     - :doc:`基于模型的 Reward <reward_model>`
   * - 在 RLinf 内编写白盒 Agent（Token 级，如 Search-R1）
     - ``AgentLoopWorker`` / ``ToolWorker`` / ``@register_toolcall_parser``
     - :doc:`白盒 Agent <new_agent>`
   * - 通过 OpenAI API 接入黑盒 Agent（如 OpenAI SDK、AutoGen）
     - Session Server ``/v1/chat/completions`` 与 ``run_session_trajectory``
     - :doc:`黑盒 Agent <openai_agent>`
   * - 适配新的 GPU / NPU 加速卡后端
     - ``AcceleratorManager`` 与 ``install.sh --platform``
     - :doc:`新增加速卡后端 <new_accelerator>`
   * - 定制 Actor 到 Rollout 的权重同步链路
     - ``WeightSyncer`` 传输层
     - :doc:`自定义权重同步 <weight_syncer>`
   * - 将 Agent Session Server 接入自定义 Runner
     - ``SessionServerWorker`` / ``create_payload_channels``
     - :doc:`将 Session Server 接入自定义 Runner <session_server_runner>`

.. tip::

   **无需 fork 主仓库：** 可以在自己的外部仓库里注册自定义模型、Advantage、Policy Loss、基于规则的 Reward 或工具调用解析器——只需暴露 ``register()`` 函数并在启动前导出 ``RLINF_EXT_MODULE=<your_pkg.module>``\ （完整示例见 :doc:`具身 RL 模型（FSDP） <new_model_fsdp>` 中的 :ref:`out-of-tree-extensions` 与 :doc:`Advantage 与 Policy Loss <new_algorithm>`\ ）。

按组件类别进入
--------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. 新增模型
      :link: models/index
      :link-type: doc

      集中覆盖全部模型接入与同步：FSDP 具身模型、SFT 模型、SGLang 具身 Rollout 模型、Megatron-Bridge / 手动 Megatron LLM 与权重同步。

   .. grid-item-card:: 2. 新增 Env 与真机
      :link: envs-robots/index
      :link-type: doc

      接入新物理仿真器、接入新真机硬件，或在已支持真机上定义新任务。

   .. grid-item-card:: 3. 新增算法
      :link: new_algorithm
      :link-type: doc

      注册自定义 Advantage 函数与 Policy Loss 函数。

   .. grid-item-card:: 4. 新增 Reward
      :link: rewards/index
      :link-type: doc

      注册规则文本 Reward，或训练并接入学习型具身 Reward 模型。

   .. grid-item-card:: 5. 新增 Agent
      :link: agents/index
      :link-type: doc

      在 RLinf 内编写 token 级白盒 Agent（如 Search-R1），或经 Session Server 免重写接入外部黑盒 Agent。

   .. grid-item-card:: 6. 新增硬件后端
      :link: new_accelerator
      :link-type: doc

      为 RLinf 的硬件调度层与安装脚本适配新的 GPU / NPU 加速卡。
