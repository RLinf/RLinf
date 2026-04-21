基于EmbodiChain的强化学习训练
=============================

EmbodiChain (`<https://github.com/DexForce/EmbodiChain>`__) 是一个具身智能实验室框架，
通过 Gym 风格接口暴露强化学习任务。RLinf 将其集成为具身环境类型
（``env_type: embodichain``）。

当前 RLinf 对 EmbodiChain 的接入主要面向简单的强化学习任务。也就是说，
RLinf 已经为 EmbodiChain 提供了一个稳定的环境入口，当前已经经过验证的示例是
**CartPole** 搭配 MLP Actor-Critic：配置名为
``embodichain_ppo_cart_pole``，环境配置片段为
``env/embodichain_cart_pole``。

当前支持范围
------------

目前 RLinf 通过CartPole任务来验证 EmbodiChain的接入是否成功：

- RLinf 通过 ``gym_config_path`` 加载 EmbodiChain 的 gym JSON
- RLinf 提取机器人状态字段，并将其拼接为 ``states``
- RLinf 使用 ``mlp_policy`` 这类标准 RL 策略进行训练

当前仓库中随文档提供的官方示例是：

- **CartPole + PPO + MLP**，对应 ``embodichain_ppo_cart_pole``

上游 EmbodiChain 仓库已经包含了更丰富的任务配置，包括操作类任务。但这些任务
**尚未在 RLinf 中打包成官方示例配方**，尤其是那些需要相机观测、语言指令或
VLA 式多模态输入的任务。

环境
----

**环境注册**

- **环境类型**：``embodichain``
- **枚举项**：``SupportedEnvType.EMBODICHAIN``
- **实现类**：``rlinf.envs.embodichain.embodichain_env.EmbodiChainEnv``

**Gym 配置解析**

通过 ``gym_config_path`` 指定 EmbodiChain 任务 JSON 文件。RLinf 按以下顺序解析
该路径：

1. 绝对路径
2. 若设置了 ``EMBODICHAIN_PATH``，则使用 ``${EMBODICHAIN_PATH}/相对路径`` (可选，用于本地仓库覆盖)
3. 安装后的 ``embodichain`` 包附近的路径（``pip install`` 后的默认情况）

正常使用 ``pip install`` 时，无需设置 ``EMBODICHAIN_PATH``，即可从已安装包解析配置。

**观测与动作空间**

- **观测**：RLinf 对外暴露单一张量键 ``states``。
- **状态构造**：``states`` 由 EmbodiChain ``robot`` 字段中
  ``state_keys`` 指定的若干字段拼接而成。
- **默认状态键**：``["qpos", "qvel", "qf"]``
- **动作空间**：连续 Box 动作空间

请确保策略配置与具体任务一致：

- ``actor.model.obs_dim`` 必须等于所选状态字段拼接后的展平维度
- ``actor.model.action_dim`` 必须匹配环境动作维度
- ``actor.model.policy_setup`` 必须匹配任务控制模式

CartPole 示例使用 ``policy_setup: cartpole-delta-qpos``。

**仿真说明**

以下环境字段会传递给 EmbodiChain 的 ``SimulationManagerCfg``：

- ``headless``
- ``enable_rt``
- ``sim_device``

在 RLinf 的 placement 机制下，worker 进程内部始终使用逻辑 GPU id ``0``。
RLinf 会为每个 worker 设置 ``CUDA_VISIBLE_DEVICES``，因此 ``cuda:0`` 实际上
指向的是当前 worker 被分配到的 GPU。

算法
----

内置的 EmbodiChain 示例使用：

- **策略**：``mlp_policy``
- **算法**：PPO
- **优势估计**：GAE
- **损失类型**：actor-critic

这套配置面向低维状态控制任务，并与 :doc:`mlp` 中介绍的通用 MLP 工作流保持一致。

未来扩展到 VLA 微调的路径
--------------------------

EmbodiChain 同样可以作为后续 RLinf 的 VLA 微调基础环境入口，但这并不只是简单
替换 ``gym_config_path``。

对于面向 VLA 的任务，通常还需要在 RLinf 中补齐以下几类能力：

1. **观测接线**

   - 将 EmbodiChain 中的相机图像、mask、语言指令或其他多模态输入，整理为目标
     VLA 模型所需的 observation dictionary。

2. **模型配置**

   - 将 ``mlp_policy`` 替换为对应的 VLA 模型配置，例如 OpenVLA、
     OpenVLA-OFT、OpenPI、Dexbotic 或其他已支持模型，具体取决于任务类型。

3. **动作语义对齐**

   - 根据 EmbodiChain 任务暴露出的机器人控制接口，对齐 ``policy_setup``、
     ``action_dim`` 以及 action chunk 等配置。

4. **任务配方封装**

   - 为每个验证过的 EmbodiChain 任务单独增加 RLinf 配置文件，而不是长期依赖一个
     通用示例页。

换句话说，当前 EmbodiChain 接入已经为 RLinf 提供了稳定的环境入口；后续更丰富的
视觉任务或语言条件任务，可以在同一个 wrapper 基础上逐步扩展。

依赖安装
--------

安装 RLinf 的具身依赖以及 EmbodiChain，无需额外安装 VLA 模型相关依赖：

.. code-block:: bash

   cd <path_to_RLinf_repository>
   bash requirements/install.sh embodied --env embodichain

该命令会从项目配置的额外索引安装 **EmbodiChain 的最新发行版**。任务相关的 gym 配置会从已安装包中加载，一般**无需**设置
``EMBODICHAIN_PATH``。

.. note::

   **当前限制**

   EmbodiChain 依赖的仿真器 ``dexsim`` 需要 ``libpython3.xx.so`` 库。目前 ``dexsim``
   通过系统路径（如 ``/usr/local/lib``）或 Conda 路径解析该库，暂不完全支持 UV 的
   Python 运行时布局。

   如果遇到 ``libpython3.11.so`` 相关的运行时错误，建议使用 Conda 环境替代 UV：

   .. code-block:: bash

      # 使用 Conda 创建环境（推荐）
      conda create -n rlinf python=3.11
      conda activate rlinf
      pip install uv
      # 然后运行安装脚本，但跳过 UV 的 Python 管理
      bash requirements/install.sh embodied --env embodichain --no-root

   我们计划在下一个 ``dexsim`` 版本中改进对 UV 的支持。

若需要使用本地 EmbodiChain 源码树（例如自定义 JSON），可手动导出：

.. code-block:: bash

   export EMBODICHAIN_PATH=/path/to/EmbodiChain

辅助启动脚本 ``examples/embodiment/run_embodiment.sh`` 不再默认写入占位路径；不设置
``EMBODICHAIN_PATH`` 时按已安装包解析配置路径。

快速开始
--------

**1. 环境配置**

参考环境配置文件为
``examples/embodiment/config/env/embodichain_cart_pole.yaml``：

.. code-block:: yaml

   env_type: embodichain
   gym_config_path: configs/agents/rl/basic/cart_pole/gym_config.json
   headless: true
   enable_rt: false
   sim_device: cuda
   state_keys: ["qpos", "qvel", "qf"]

**2. 训练配置**

顶层训练配置文件为
``examples/embodiment/config/embodichain_ppo_cart_pole.yaml``，其中使用了：

- ``model/mlp_policy@actor.model``
- PPO + GAE
- ``obs_dim: 6``
- ``action_dim: 2``
- ``policy_setup: cartpole-delta-qpos``

**3. 启动训练**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh embodichain_ppo_cart_pole

**若出现缺文件或数据集下载相关报错**

并不要求在训练前必须先下载资源；若环境已就绪可直接训练。若在运行时报错，提示找不到
EmbodiChain 任务或仿真资源，可尝试：

1. 设置数据根目录（下载文件存放位置）：

   .. code-block:: bash

      export EMBODICHAIN_DATA_ROOT=/path/to/data

2. 下载 CartPole 与共享仿真资源（在已安装 ``embodichain`` 的同一 Python 环境中执行）：

   .. code-block:: bash

      python -m embodichain.data download --name CartPole
      python -m embodichain.data download --name SimResources

3. 重新执行
   上文第 3 步中的训练命令。

**4. 适配新任务**

如果要切换到其他 EmbodiChain 任务：

1. 将 ``gym_config_path`` 指向新的 EmbodiChain gym JSON
2. 如果任务暴露了不同的状态字段，更新 ``state_keys``
3. 更新 ``actor.model.obs_dim`` 以匹配展平后的状态维度
4. 更新 ``actor.model.action_dim`` 和 ``policy_setup`` 以匹配任务定义

对于低维任务，上述修改通常已经足够。对于视觉任务或语言条件任务，在训练真正跑通前，
你大概率还需要进一步补齐 RLinf 的观测封装与模型接线。

评测与 CI
---------

EmbodiChain CartPole 同时也被用于具身端到端测试，相关测试位于
``tests/e2e_tests/embodied/``。在测试环境中安装 ``embodichain`` 后，配置默认从包内解析；仅在需要非默认
checkout 时设置 ``EMBODICHAIN_PATH``。
