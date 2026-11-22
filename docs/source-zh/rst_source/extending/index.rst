扩展开发
========

无需改动 RLinf 的分布式调度核心，即可接入自定义模型、仿真器、真机机器人、奖励模型、算法、智能体工具或加速卡后端。按目标从改动面最小的扩展层切入即可。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/extending_map_zh.svg
   :align: center
   :width: 100%

   RLinf 在共享分布式运行时之上的扩展插槽。

快速选型：我要接入什么？
------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - 开发目标
     - 对应扩展点
     - 推荐指南
   * - 在 FSDP 上接入新的 VLA 或具身 RL 策略
     - ``register_model`` 与 ``BasePolicy``
     - :doc:`FSDP 模型 <new_model_fsdp>`
   * - 接入新的 SFT 监督微调模型
     - ``register_model`` 与 ``FSDPSftWorker``
     - :doc:`SFT 模型 <new_model_sft>`
   * - 用 SGLang 加速具身策略评测 rollout
     - SGLang 具身适配注册表
     - :doc:`SGLang 具身模型 <sglang_embodied_model>`
   * - 接入新的物理仿真器
     - ``SupportedEnvType`` 与 ``get_env_cls()``
     - :doc:`仿真环境 <new_env>`
   * - 接入新传感器、机械臂或整机
     - 单零部件本地验证 → 组合 → 远程部署
     - :doc:`真机机器人 <new_robot>`
   * - 在已有真机上新增任务
     - 真机任务配置叠加（无需写机器人驱动）
     - :doc:`真机任务 <new_task>`
   * - 训练或自定义学习型具身奖励模型
     - ``BaseRewardModel`` 与 ``reward.use_reward_model``
     - :doc:`奖励模型 <reward_model>`
   * - 新增优势估计、策略损失或规则奖励
     - ``@register_advantage`` / ``@register_policy_loss`` / ``register_reward``
     - :doc:`算法与规则奖励 <new_algorithm>`
   * - 接入自定义多轮智能体、工具或解析器
     - ``AgentLoopWorker`` / ``ToolWorker`` / ``@register_toolcall_parser``
     - :doc:`智能体与工具调用 <new_agent>`
   * - 在 Megatron + SGLang 上训练 Hugging Face 大模型
     - Megatron-Bridge actor 后端（推荐）
     - :doc:`Megatron-Bridge <mbridge>`
   * - 适配新的 GPU / NPU 加速卡后端
     - ``AcceleratorManager`` 子类与 ``install.sh --platform``
     - :doc:`加速卡后端 <new_accelerator>`
   * - 手动编写 Megatron 适配或定制权重同步
     - 底层 Megatron 权重切分 / ``WeightSyncer``
     - :doc:`高级集成 <advanced-integrations/index>`

.. tip::

   **无需 fork 主仓库：** 可以在自己的外部仓库里注册自定义模型、优势估计、策略损失、规则奖励或工具调用解析器——只需暴露 ``register()`` 函数并在启动前导出 ``RLINF_EXT_MODULE=<your_pkg.module>``\ （完整示例见 :doc:`FSDP 模型 <new_model_fsdp>` 中的 :ref:`out-of-tree-extensions` 与 :doc:`算法与规则奖励 <new_algorithm>`\ ）。

1. 具身模型
-----------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: FSDP 模型
      :link: new_model_fsdp
      :link-type: doc

      在 FSDP 后端注册 Hugging Face 模型或具身 VLA policy。

   .. grid-item-card:: SFT 模型
      :link: new_model_sft
      :link-type: doc

      将自定义模型与数据集接入 RLinf 的 SFT 训练流水线。

   .. grid-item-card:: SGLang 具身模型
      :link: sglang_embodied_model
      :link-type: doc

      将具身策略适配到 SGLang rollout worker，实现高速评测。

2. 具身环境、真机与奖励模型
---------------------------

按需从最轻量的层次切入：若目标真机已在支持列表中，直接新增真机任务即可，无需修改 robotics 代码；接入新硬件时，先在本机跑通单个零部件，再组合整机与配置远程部署。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 仿真环境
      :link: new_env
      :link-type: doc

      封装仿真器后端、注册环境类型并完成动作格式映射。

   .. grid-item-card:: 真机机器人
      :link: new_robot
      :link-type: doc

      从本地单个零部件起步，完成整机组合与远程部署。

   .. grid-item-card:: 真机任务
      :link: new_task
      :link-type: doc

      在 RLinf 已支持的真机硬件上快速定义新任务。

   .. grid-item-card:: 奖励模型
      :link: reward_model
      :link-type: doc

      扩展 ``BaseRewardModel`` 或训练内置 ResNet / VLM 具身奖励模型。

3. 算法、智能体与大模型
-----------------------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 算法与规则奖励
      :link: new_algorithm
      :link-type: doc

      注册自定义优势估计函数、策略损失函数与基于规则的文本奖励。

   .. grid-item-card:: 智能体与工具调用
      :link: new_agent
      :link-type: doc

      实现自定义多轮智能体循环、工具 worker 与工具调用解析器。

   .. grid-item-card:: Megatron-Bridge
      :link: mbridge
      :link-type: doc

      推荐路径：无需手写权重转换，直接把 Hugging Face 大模型接入 Megatron + SGLang 训练。

4. 硬件与高级集成
-----------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 加速卡后端
      :link: new_accelerator
      :link-type: doc

      为 RLinf 的硬件调度层与安装脚本接入新的 GPU / NPU 加速卡。

   .. grid-item-card:: 高级集成
      :link: advanced-integrations/index
      :link-type: doc

      手动接入 Megatron 模型，以及定制 actor 到 rollout 的权重同步链路。
