基于 Genesis 仿真平台的强化学习训练
======================================

本文档提供了在 RLinf 框架内使用 Genesis 仿真环境启动 **MLP 策略** 训练任务的指南。

Genesis 是一个物理逼真的多物理场仿真平台，支持 GPU 并行计算高精度的接触动力学，适用于复杂的手部操作任务。

环境
-----------------------

**Genesis 环境**

- **Environment**：Genesis Simulation Platform
- **Task**：控制 Franka Panda 机械臂抓取方块
- **Observation**：
  - **Images**：第三人称视角的 RGB 图像 (640×480)
  - **States**：49 维向量（包含机器人关节、末端位姿及物体位姿）
- **Action Space**：9 维连续动作
  - 7 维臂部关节位置控制
  - 2 维夹爪位置控制

依赖安装
---------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

.. code:: bash

   # 使用 install.sh 安装 genesis 相关依赖
   bash requirements/install.sh embodied --env genesis
   source .venv/bin/activate

运行脚本
-------------------

**1. 配置文件**

- 配置文件路径：``examples/embodiment/config/genesis_cubepick_ppo_mlp.yaml``

**2. 启动命令**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_mlp

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs

**2. 关键监控指标**

- **训练指标**：

  - ``train/actor/policy_loss``: PPO策略损失。
  - ``train/actor/clip_fraction``: 触发 PPO 裁剪的样本比例。反映了新旧策略的差异程度。
  - ``train/actor/approx_kl``: 近似 KL 散度。监控策略更新幅度，防止更新过大导致崩溃。
  - ``train/actor/grad_norm``: 梯度范数。用于监控训练稳定性，图中显示随收敛过程梯度范数会有所上升。
  - ``train/critic/value_loss``: 价值函数损失。衡量 Critic 对状态价值估计的准确性。
  - ``train/critic/explained_variance``: 衡量价值函数拟合程度。越接近 1 越好。
  - ``train/actor/total_loss``: 策略损失 + 价值损失 + 熵正则的总和。

- **Rollout 指标**：

  - ``rollout/returns_mean``: 优势函数的均值。
  - ``rollout/advantages_max/mean/min``: 优势函数的最大值/最小值。
  - ``rollout/rewards``: 一个chunk的奖励。

- **环境指标**：

  - ``env/success_once``: 核心指标。表示回合内是否成功抓取并抬起方块。在 400 个 Epoch 内，成功率预期可达到 90% 以上。
  - ``env/episode_len``: 该回合实际经历的环境步数（单位：step）。
  - ``env/return``: 回合总回报。
  - ``env/reward``: 步级奖励均值。


**3. 视频生成**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

**4. 训练日志工具集成**

.. code-block:: yaml

    runner:
        task_type: embodied
        logger:
            log_path: "../results"
            project_name: rlinf
            experiment_name: "genesis_cubepick_ppo_mlp"
            logger_backends: ["tensorboard"]

Genesis 结果
~~~~~~~~~~~~~~~~~~~

在/examples/embodiment/config/genesis_cubepick_ppo_mlp.yaml中默认参数的训练下env/success_once可达到约 92.6% 。