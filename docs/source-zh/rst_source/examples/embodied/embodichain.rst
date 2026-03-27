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
2. ``${EMBODICHAIN_PATH}/<相对路径>``
3. 安装后的 ``embodichain`` 包附近的路径

因此，你既可以使用本地 EmbodiChain 仓库，也可以使用已安装包的目录结构。

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

该命令会克隆 `EmbodiChain <https://github.com/DexForce/EmbodiChain.git>`_
（如果已设置 ``EMBODICHAIN_PATH`` 则直接复用），并在虚拟环境的
``activate`` 脚本中追加 ``export EMBODICHAIN_PATH=...``。

你也可以手动以 editable 模式安装 EmbodiChain，然后显式导出路径：

.. code-block:: bash

   export EMBODICHAIN_PATH=/path/to/EmbodiChain

辅助启动脚本 ``examples/embodiment/run_embodiment.sh`` 在未设置
``EMBODICHAIN_PATH`` 时默认使用 ``$HOME/workspace/research/EmbodiChain``，
如果你的仓库路径不同，请自行调整。

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
``tests/e2e_tests/embodied/``。本地运行这些测试时，请确保
``EMBODICHAIN_PATH`` 指向一个有效的 EmbodiChain 仓库目录。
