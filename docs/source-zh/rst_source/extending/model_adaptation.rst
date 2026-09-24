模型适配方案
============

通过现有模型工厂和 worker 接口将模型接入 RLinf。本文先列出已注册模型的实现与源码来源，再说明 LingBot-VLA 2.0 遵循的接入方式。首先确定模型的角色：动作 policy、语言模型、扩散模型和奖励模型分别进入对应的 worker 流程。

现有模型与实现
--------------

2026-09-22 的源码盘点在 ``SupportedModel`` 中找到 39 个不同名称，包含本次新增的 ``lingbotvlav2``。注册名称只表示配置能够识别该名称；实际运行还取决于 builder、算法和环境的支持范围。例如，MolmoAct2 当前仅支持评估 rollout，``cma`` 则是没有内置工厂绑定的历史注册项。

多数 VLA 使用“RLinf 内 adapter + 外部安装的模型包或源码 fork”，部分模型同时保留局部模型实现和 transforms。统一的是工厂入口与 policy 接口，模型主干的代码存放位置可以不同。

.. list-table:: 模型与源码边界
   :header-rows: 1
   :widths: 32 36 32

   * - 注册名称
     - RLinf 实现
     - 模型依赖 / 角色
   * - ``openvla``、``openvla_oft``
     - ``embodiment/openvla``、``embodiment/openvla_oft``、``embodiment/prismatic``
     - 外部 ``prismatic`` 与仓库内的 RL、模型适配代码。
   * - ``openpi``、``openpi_rlinf``
     - ``embodiment/openpi``、``embodiment/openpi_rlinf``
     - 外部 ``openpi``；后者包含更多仓库内模型模块。
   * - ``pi0_fast``
     - ``embodiment/pi0_fast``
     - 外部 LeRobot ``PI0FastPolicy`` 与 RLinf rollout/replay 适配。
   * - ``gr00t``、``gr00t_n1d6``、``gr00t_n1d7``
     - ``embodiment/gr00t``
     - 对应版本的 ``gr00t`` 源码与 adapter。
   * - ``dexbotic_pi``、``dexbotic_dm0``
     - ``embodiment/dexbotic_pi``、``embodiment/dexbotic_dm0``
     - 外部 ``dexbotic`` 与 RL policy 子类。
   * - ``lingbotvla``、``lingbotvlav2``
     - ``embodiment/lingbotvla``、``embodiment/lingbotvlav2``
     - 外部 ``lingbotvla``；V1、V2 使用独立环境。
   * - ``starvla``、``abot_m0``、``evo1``
     - ``embodiment/`` 下的同名目录
     - 分别适配外部 starVLA、ABot、Evo-1 源码。
   * - ``dreamzero``、``fastwam``、``cosmos3``
     - ``embodiment/`` 下的同名目录
     - 外部 ``groot``、``fastwam``、``cosmos_framework``。
   * - ``molmoact2``
     - ``embodiment/molmoact2``
     - RLinf 的 LeRobot fork；仅支持评估 policy 接口。
   * - ``mlp_policy``、``rlt_mlp_policy``、``rlt_td3_mlp_policy``
     - ``embodiment/mlp_policy``
     - RLinf 自有模型，由算法配置选择对应行为。
   * - ``cnn_policy``、``flow_policy``
     - ``embodiment/cnn_policy``、``embodiment/flow_policy``
     - RLinf 自有 policy，使用标准 PyTorch 组件。
   * - ``cfg_model``、``recap_value_model``、``steam_value_model``
     - ``embodiment/openpi_cfg``、``embodiment/value_model``
     - 基于 OpenPI 的 policy/value 模型，训练角色不同。
   * - ``sd3``、``wan22_ti2v_5b``
     - ``diffusion/sd3``、``diffusion/wan``
     - Diffusers pipeline 与 RLinf 扩散模型封装。
   * - ``qwen2.5``、``qwen2.5_vl``、``qwen3``、``qwen3_vl``
     - FSDP model manager 与配置指定的语言模型后端
     - Transformers / 后端模型实现，用于语言、多模态或 SFT。
   * - ``qwen3_moe``、``qwen3_vl_moe``、``deepseek_v3``、``glm4_moe_lite``
     - 配置指定的语言模型后端
     - 依赖对应后端的支持范围，不经过具身 ``BasePolicy`` builder。
   * - ``resnet``
     - ``embodiment/reward``
     - 独立的奖励模型注册表，使用 torchvision。
   * - ``cma``
     - 历史 ``SupportedModel`` 注册项
     - 当前没有内置的 ``rlinf.models.get_model`` 绑定，不能作为可运行的 policy 配方选择。

表中实现路径均相对于 ``rlinf/models/``。以 ``rlinf/config.py``、``rlinf/models/__init__.py`` 及 ``requirements/install.sh`` 中对应的安装函数为准。这里列的是源码接入情况，不表示本次改动已对全部模型与环境组合完成 GPU 实测。

动作 policy 接口
----------------

通过 ``rlinf.models.get_model(cfg)`` 调用注册的 ``get_model(cfg, torch_dtype)`` builder，将可选模型 import 放在 builder 内。返回的 ``BasePolicy`` 提供 ``predict_action_batch`` 用于 rollout，经 ``ForwardType`` 调用 ``default_forward`` 完成 actor 概率重算。不支持的模式应明确报错；只有动作预测接口不代表支持 PPO。

模型包负责主干、配置、模型 transforms 与 kernels；RLinf adapter 负责环境 I/O 映射、RL 采样、概率重算、可训练参数范围和 value head；RLinf workers 负责 FSDP、optimizer、断点续训、placement 和权重同步。

安装与验证
----------

使用 ``requirements/install.sh`` 及公共依赖 helpers，为新增模型固定包版本或源码 revision。所需接口尚未发布时，保留版本化补丁、文件哈希、源码来源和许可证声明，拒绝覆盖冲突的用户改动。V2 使用 ``requirements/embodied/models/lingbotvlav2/source.json`` 和 ``ppo-compat.patch``；Docker 与 CI 复用同一安装器。其他模型历史上较宽松的版本固定方式，应单独验证后再调整。

验证 checkpoint 加载、预处理、rollout/actor 概率一致性、梯度有限、可训练参数范围、optimizer 更新、权重同步和续训。独立评估需要完整的 ``rollout.model``；导出上游部署格式需另做加载测试。中英文说明及 Docker/CI 应保持一致，并明确实际执行的验证项。

worker 接入细节见 :doc:`new_model_fsdp`，V2 安装、PPO 和评估用法见 :doc:`../examples/embodied/lingbotvlav2`。
