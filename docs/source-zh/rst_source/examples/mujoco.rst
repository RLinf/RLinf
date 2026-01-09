基于 MuJoCo 评测平台的强化学习训练
==================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档给出在 **RLinf** 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务的完整指南，
并介绍如何在 **MuJoCo** 环境中微调 VLA 模型以完成机器人操作任务。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像；
2. **语言理解**：理解自然语言的任务描述；
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）；
4. **强化学习**：结合环境反馈，使用 PPO 优化策略。

环境
----

MuJoCo 环境基于项目 `serl <https://rail-berkeley.github.io/serl/docs/sim_quick_start.html>`_ 构建，
包含两个最小化仿真环境：

- ``PandaPickCube-v0``
- ``PandaPickCubeVision-v0``

任务定义
~~~~~~~~

- **Task**：控制 Franka Panda 机械臂抓取物块并移动至目标位置；
- **Observation**：

  - ``PandaPickCube-v0``：本体感知状态 + 目标位置；
  - ``PandaPickCubeVision-v0``：多视角 RGB 图像（机器人视角 + 腕部相机）+ 本体感知状态；

- **Action Space**：4 维连续动作

  - 三维位置控制（x, y, z）
  - 夹爪控制（开/合）

数据结构
~~~~~~~~

``PandaPickCube-v0``

- **States**：本体感知与目标位置

  - 末端执行器三维位置
  - 末端执行器三维速度
  - 夹爪一维开合
  - 物块三维位置

``PandaPickCubeVision-v0``

- **Images**：第三人称视角与腕部相机视角的 RGB 张量
- **States**：本体感知

  - 末端执行器三维位置
  - 末端执行器三维速度
  - 夹爪一维开合

- **Task Descriptions**：自然语言指令
- **Actions**：归一化连续动作值
- **Rewards**：基于任务完成度的逐步奖励

算法
----

核心算法组件包括：

1. **PPO（近端策略优化）**

   - 使用 GAE（广义优势估计）进行优势估计；
   - 带比例限制（clipping）的策略裁剪；
   - 价值函数裁剪；
   - 熵正则化。

2. **GRPO（组相对策略优化）**

   - 对于每个状态/提示，策略生成 *G* 个独立动作；
   - 通过减去组平均奖励来计算每个动作的优势。

依赖安装
--------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

**选项 1：Docker 镜像**


使用 Docker 镜像运行实验：

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

   # 如果需要国内加速下载镜像，可以使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

对于不同模型上的实验，请通过镜像内置的 ``switch_env`` 工具切换到对应虚拟环境：

.. code-block:: bash

   # 切换到 OpenVLA 环境
   source switch_env openvla

   # 切换到 OpenVLA-OFT 环境
   source switch_env openvla-oft

选项 2：自定义环境
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror 到下面的 install.sh 命令
   # 将 --model 参数改为 openvla-oft 可安装 OpenVLA-OFT 环境
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

资源下载
--------

下载 MuJoCo 资源文件：

.. code-block:: bash

   cd rlinf/envs/mujoco
   git clone https://github.com/zlockewtg/franka-sim.git
   pip install -e .
   pip install -r requirements.txt

运行脚本
--------

1. 关键参数配置
~~~~~~~~~~~~~~~~

示例 1：流水线重叠（推荐）
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env: 0-7
       rollout: 8-15
       actor: 0-15

   rollout:
     pipeline_stage_num: 2

该配置允许 **rollout 与 env** 之间流水线重叠，从而提升 rollout 吞吐。

示例 2：完全共享（env / rollout / actor 共用 GPU）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       env,rollout,actor: all

示例 3：完全分离（互不干扰，无需 offload）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env: 0-3
       rollout: 4-7
       actor: 8-15

该配置实现 env、rollout、actor 各自使用独立 GPU，互不干扰，因此通常不需要 offload 功能。

2. 启动命令
~~~~~~~~~~~

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

暂时只支持在 MuJoCo 环境中使用 PPO 训练 MLP Policy：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh mujoco_ppo_mlp

可视化与结果
------------

1. TensorBoard 日志
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

2. 关键监控指标
~~~~~~~~~~~~~~~

训练指标
^^^^^^^^

- ``train/actor/approx_kl``：近似 KL，用于监控策略更新幅度
- ``train/actor/clip_fraction``：触发 PPO clip 的样本比例
- ``train/actor/clipped_ratio``：裁剪后的概率比均值，用来衡量 clip 的影响程度
- ``train/actor/grad_norm``：梯度范数
- ``train/actor/lr``：学习率
- ``train/actor/policy_loss``：策略损失
- ``train/critic/value_loss``：价值函数损失
- ``train/critic/value_clip_ratio``：值函数裁剪触发比例
- ``train/critic/explained_variance``：价值拟合程度，越接近 1 越好
- ``train/entropy_loss``：策略熵
- ``train/loss``：总损失（actor + critic + entropy regularization）

Rollout 指标
^^^^^^^^^^^^

- ``rollout/advantages_max``：优势最大值
- ``rollout/advantages_mean``：优势均值
- ``rollout/advantages_min``：优势最小值
- ``rollout/rewards``：一个 chunk 的奖励统计

环境指标
^^^^^^^^

- ``env/episode_len``：回合步数（step）
- ``env/return``：回合总回报（在稀疏奖励中参考意义有限）
- ``env/reward``：step-level 奖励
- ``env/success_once``：建议重点监控该指标，反映未归一化成功率，更能体现策略真实性能

3. 视频生成
~~~~~~~~~~~

仅支持 ``PandaPickCubeVision-v0`` 环境下生成视频：

.. code-block:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

4. 训练日志工具集成
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "maniskill_ppo_openvla"
       logger_backends: ["tensorboard"]  # wandb, swanlab
