扩展开发
========

无需改动 RLinf 的分布式调度核心，即可接入自定义模型、仿真器、真机机器人、算法或奖励函数。按目标从改动面最小的扩展层切入即可。

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
     - :doc:`FSDP 模型 <new_model_fsdp>`
   * - 在 Megatron + SGLang 上训练 Hugging Face 大模型
     - Megatron-Bridge actor 后端（推荐）
     - :doc:`Megatron-Bridge <mbridge>`
   * - 用 SGLang 加速具身策略 rollout
     - SGLang 具身 rollout 适配层
     - :doc:`SGLang 具身模型 <sglang_embodied_model>`
   * - 接入新的 SFT 监督微调模型
     - SFT worker 模型注册表
     - :doc:`SFT 模型 <new_model_sft>`
   * - 接入新的物理仿真器
     - ``SupportedEnvType`` 与 ``get_env_cls()``
     - :doc:`仿真环境 <new_env>`
   * - 在已有真机上新增任务
     - 真机任务配置叠加（无需写机器人驱动）
     - :doc:`真机任务 <new_task>`
   * - 接入新传感器、机械臂或整机
     - 单零部件本地验证 → 组合 → 远程部署
     - :doc:`真机机器人 <new_robot>`
   * - 新增优势估计、策略损失或规则奖励
     - ``@register_advantage`` / ``@register_policy_loss`` / ``register_reward``
     - :doc:`算法与规则奖励 <new_algorithm>`
   * - 训练并接入学习型奖励模型
     - Reward worker 与 ``reward.use_reward_model``
     - :doc:`奖励模型 <reward_model>`
   * - 手动编写 Megatron 适配或定制权重同步
     - 底层 Megatron 权重切分 / 权重同步器
     - :doc:`高级集成 <advanced-integrations/index>`

.. _out-of-tree-extensions:

外部仓库扩展
------------

无需 fork 或修改 RLinf 源码，即可在自己的仓库里注册自定义模型、优势估计、策略损失或规则奖励。由于 Ray 会在独立 Python 进程中拉起各个 worker，仅在入口脚本里调用注册函数不会生效到 worker 进程。

将注册调用放进外部包的 ``register()`` 函数，并在启动训练或 checkpoint 转换前导出 ``RLINF_EXT_MODULE``：

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

RLinf 会在每个 worker 进程及独立转换工具初始化时自动导入该模块并调用 ``register()``，完成与主进程一致的注册。

1. 接入模型
-----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: FSDP 模型
      :link: new_model_fsdp
      :link-type: doc

      在 FSDP 后端注册 Hugging Face 模型或具身 VLA 策略。

   .. grid-item-card:: Megatron-Bridge
      :link: mbridge
      :link-type: doc

      推荐路径：无需手写权重转换，直接把 Hugging Face 大模型接入 Megatron + SGLang 训练。

   .. grid-item-card:: SGLang 具身模型
      :link: sglang_embodied_model
      :link-type: doc

      将具身策略适配到 SGLang rollout worker，实现高速推理与评测。

   .. grid-item-card:: SFT 模型
      :link: new_model_sft
      :link-type: doc

      将自定义模型接入 RLinf 的 SFT 训练流水线。

2. 接入环境与真机
-----------------

按需从最轻量的层次切入：若目标真机已在支持列表中，直接新增真机任务即可，无需修改 robotics 代码；接入新硬件时，先在本机跑通单个零部件，再组合整机与配置远程部署。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 仿真环境
      :link: new_env
      :link-type: doc

      封装仿真器后端、注册环境类型并完成动作格式映射。

   .. grid-item-card:: 真机任务
      :link: new_task
      :link-type: doc

      在 RLinf 已支持的真机硬件上快速定义新任务。

   .. grid-item-card:: 真机机器人
      :link: new_robot
      :link-type: doc

      从本地单个零部件起步，完成整机组合与远程部署。

3. 接入算法与奖励
-----------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 算法与规则奖励
      :link: new_algorithm
      :link-type: doc

      几分钟内注册自定义优势估计函数、策略损失函数或基于规则的文本奖励。

   .. grid-item-card:: 奖励模型
      :link: reward_model
      :link-type: doc

      采集轨迹、训练 ResNet 或 VLM 奖励模型，并接入在线 RL 训练。

4. 高级集成
-----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 手动接入 Megatron 模型
      :link: new_model_megatron
      :link-type: doc

      Megatron-Bridge 暂未覆盖你的模型结构时，按底层步骤手工适配 Megatron + SGLang。

   .. grid-item-card:: 权重同步
      :link: weight_syncer
      :link-type: doc

      优化具身训练中 actor 到 rollout 的权重同步链路（``patch`` 与 ``bucket`` 模式）。
