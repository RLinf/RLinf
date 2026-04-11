MLP策略强化学习训练
===============================================

本示例展示 RLinf 框架使用 **MLP (Multi-Layer Perceptron)** 策略网络进行强化学习训练的完整流程。
MLP 策略主要面向 **低维状态(state)输入** 的机器人控制任务，支持在仿真环境(ManiSkill3)、仿真环境(FrankaSim)、仿真环境(Libero-Spatial)中进行训练。
当前提供的配置覆盖 **PPO-MLP** 、 **SAC-MLP** 、**GRPO-MLP** 算法设置，可用于快速验证环境、训练管线与网络结构。

主要目标是让模型具备以下能力:

1. **状态理解**:处理来自环境的低维状态(关节角、末端位姿、物体状态等)。
2. **动作生成**:产生连续控制动作(末端位置增量/关节目标/夹爪控制等)。
3. **强化学习**:结合环境反馈，使用 PPO 或 SAC 优化策略。

环境
----
RLinf 目前支持多类具身智能环境，可通过 **defaults** 中的 ``env/<env_name>@env.train`` 与 ``env/<env_name>@env.eval`` 选择不同环境配置，
并在 ``env.train`` / ``env.eval`` 节点对并行环境数、episode 长度、reset 方式、视频保存等进行覆盖。

目前支持(示例中已覆盖)的环境包括:

- ``maniskill_pick_cube`` (ManiSkill3 )
- ``libero_spatial`` (LIBERO Spatial)
- ``frankasim_pickcube_state`` (Mujoco )

也可通过自定义环境配置来训练特定任务，具体可参考以下方式:

1. 在配置文件中通过 defaults 引用环境(训练/评估可分别指定)。

.. code:: yaml

   defaults:
     - env/maniskill_pick_cube@env.train
     - env/maniskill_pick_cube@env.eval

   defaults:
     - env/libero_spatial@env.train
     - env/libero_spatial@env.eval

   defaults:
     - env/frankasim_pickcube_state@env.train
     - env/frankasim_pickcube_state@env.eval

算法
-----------------------------------------

**核心算法组件**

1.  **PPO (Proximal Policy Optimization)**

    -   采用 on-policy 的 Actor-Critic 框架。
    -   使用 GAE(Generalized Advantage Estimation)估计优势函数:``adv_type: gae``。
    -   使用 clip 约束策略更新幅度(ratio clipping)，并可选 KL 约束项。

2.  **SAC (Soft Actor-Critic)**

    -   通过 Bellman 备份与熵正则化学习 Q 值(off-policy)。
    -   使用 MLP 作为 Actor 策略网络，并在配置中启用 Q 相关头/结构(``add_q_head: True``)。
    -   支持自动温度调节（配置 ``entropy_tuning``，如 ``alpha_type: softplus``），平衡探索与利用。

3.  **GRPO(Group Relative Policy Optimization)**

    - 对于每个状态/提示，策略生成 *G* 个独立动作  
    - 以组内平均奖励为基线，计算每个动作的相对优势


依赖安装
---------------

对于在仿真环境运行，请参考 :doc:`../../start/installation` 进行安装。

本系列配置使用 Hydra 的 searchpath 从环境变量引入外部配置目录:

- ``hydra.searchpath: file://${oc.env:EMBODIED_PATH}/config/``

请确保已正确设置 ``EMBODIED_PATH``，并安装 ManiSkill3 / FrankaSim 相关依赖与资源。

运行脚本
--------

**1. 配置文件**

RLinf 提供多份 MLP 默认配置，覆盖不同环境与算法设置:

-   **ManiSkill + PPO + MLP**: ``maniskill_ppo_mlp`` 
-   **ManiSkill + SAC + MLP**: ``maniskill_sac_mlp`` 
-   **FrankaSim + PPO + MLP**: ``franka_sim_ppo_mlp`` 

**2. 关键参数配置**

**2.1 模型参数 (Model)**

MLP 模型由 ``model/mlp_policy@actor.model`` 引入，并在不同配置中做覆盖。常见关键字段如下:

.. code:: yaml

   model_type: "mlp_policy"                # 使用 MLP 策略网络作为 actor（多层感知机；适合低维 state 输入）

   model_path: ""                        

   policy_setup: "panda-qpos"              # 选择动作语义与控制模式；panda-qpos 通常表示关节空间控制（如 qpos/关节目标或增量）

   obs_dim: 42                             # 输入到 MLP 的状态向量维度（需与环境输出的 state 维度严格一致）

   action_dim: 8                           # 策略输出动作向量的维度（需与环境 action space 维度严格一致）

   num_action_chunks: 1                    # 一次 forward 生成的动作 chunk 

   hidden_dim: 256                         # MLP 隐藏层的通道/宽度

   precision: "32"                         # 模型参数与计算精度；
   add_value_head: True                    # 是否在策略网络上额外挂载 value head

   is_lora: False                          # 是否启用 LoRA（

   lora_rank: 32                           # LoRA 的低秩维度 r；仅当 is_lora=True 时生效


**2.2 集群与硬件配置 (Cluster)**
    对于真机训练，使用多节点配置，将 Actor/Policy 部署在 GPU 服务器上，将 Env/Robot 部署在控制机（NUC/工控机）上。具体配置可参考 :doc:`franka` 。

**3. 启动命令**

**ManiSkill(PPO-MLP)**

::

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

**ManiSkill(SAC-MLP)**

::

   bash examples/embodiment/run_embodiment.sh maniskill_sac_mlp

**Libero-Spatial(GRPO-MLP)**

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_0_grpo_mlp

**FrankaSim(PPO-MLP)**

::

   bash examples/embodiment/run_embodiment.sh franka_sim_ppo_mlp

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ../results

**2. 关键监控指标**

- **环境指标**:

  - ``env/episode_len``:回合实际经历的环境步数(单位:step)
  - ``env/return``:回合总回报
  - ``env/reward``:step-level 奖励
  - ``env/success_once``:回合中至少成功一次标志(若环境提供)

- **Training Metrics (SAC)**:

  - ``train/sac/critic_loss``:Q 函数损失
  - ``train/sac/actor_loss``:策略损失
  - ``train/sac/alpha_loss``:温度参数损失
  - ``train/sac/alpha``:温度参数值
  - ``train/replay_buffer/size``:重放缓冲区大小

- **Training Metrics (PPO)**:

  - 策略损失(policy loss)
  - 价值损失(value loss)
  - 估计 KL(approx_kl / kl)
  - clip 比例(clip_frac)
  - 策略熵(entropy)


