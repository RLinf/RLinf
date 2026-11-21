扩展开发
========

无需改动 RLinf 的分布式调度核心，即可接入自定义模型、仿真器、真机机器人、奖励模型或底层训练引擎。从最贴近改动范围的扩展层开始即可。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/extending_map_zh.svg
   :align: center
   :width: 100%

   RLinf 在共享分布式运行时之上的四类扩展插槽。

快速选型：我要接入什么？
------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 38 30

   * - 开发目标
     - 对应扩展点
     - 推荐指南
   * - 在 FSDP 上接入新的 VLA 或 LLM
     - ``register_model`` 与 ``BasePolicy``
     - :doc:`FSDP 新模型 <new_model_fsdp>`
   * - 在 Megatron + SGLang 上训练大模型
     - Megatron actor 与 rollout 适配
     - :doc:`Megatron 新模型 <new_model_megatron>`
   * - 用 SGLang 加速具身策略 rollout
     - SGLang 具身 rollout 适配层
     - :doc:`SGLang 具身模型 <sglang_embodied_model>`
   * - 接入新的 SFT 监督微调模型
     - SFT worker 模型注册表
     - :doc:`新 SFT 模型 <new_model_sft>`
   * - 接入新的物理仿真器
     - ``SupportedEnvType`` 与 ``get_env_cls()``
     - :doc:`新仿真环境 <new_env>`
   * - 在已有真机上新增任务
     - 真机任务配置叠加（无需写机器人驱动）
     - :doc:`新增真机任务 <new_task>`
   * - 接入新传感器、机械臂或整机
     - 单部件本地验证 → 组合 → 远程部署
     - :doc:`新机器人 <new_robot>`
   * - 训练并接入学习型奖励模型
     - Reward worker 与 ``reward.use_reward_model``
     - :doc:`奖励模型 <reward_model>`
   * - 定制 Megatron-Bridge 或权重同步
     - 后端桥接 / 权重同步器
     - :doc:`高级集成 <advanced-integrations/index>`

1. 接入模型
-----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: FSDP 新模型
      :link: new_model_fsdp
      :link-type: doc

      在 FSDP 后端注册 Hugging Face 模型或具身 VLA policy。

   .. grid-item-card:: Megatron 新模型
      :link: new_model_megatron
      :link-type: doc

      在 Megatron + SGLang 后端接入大模型训练与推理。

   .. grid-item-card:: SGLang 具身模型
      :link: sglang_embodied_model
      :link-type: doc

      将具身策略适配到 SGLang rollout worker，实现高速推理与评测。

   .. grid-item-card:: 新 SFT 模型
      :link: new_model_sft
      :link-type: doc

      将自定义模型接入 RLinf 的 SFT 训练流水线。

2. 接入环境与真机
-----------------

按需从最轻量的层次切入：若目标真机已在支持列表中，直接新增真机任务即可，无需修改 robotics 代码；接入新硬件时，先在本机跑通单个部件，再组合整机与配置远程部署。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 新仿真环境
      :link: new_env
      :link-type: doc

      封装仿真器后端、注册环境类型并完成动作格式映射。

   .. grid-item-card:: 新增真机任务
      :link: new_task
      :link-type: doc

      在 RLinf 已支持的真机硬件上快速定义新任务。

   .. grid-item-card:: 新机器人
      :link: new_robot
      :link-type: doc

      从本地单个零部件起步，完成整机组合与远程部署。

3. 接入奖励与算法
-----------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 奖励模型
      :link: reward_model
      :link-type: doc

      采集轨迹、训练 ResNet 或 VLM 奖励模型，并接入在线 RL 训练。

   .. grid-item-card:: 扩展架构概览
      :link: overview
      :link-type: doc

      了解模型、环境、优势估计、损失与奖励等注册表如何在框架内协作。

4. 底层引擎与集成
-----------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 高级集成
      :link: advanced-integrations/index
      :link-type: doc

      接入 Megatron-Bridge 与优化 actor 到 rollout 的权重同步链路。
