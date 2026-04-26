XSquare Turtle2 真机强化学习
==============================

本文档给出在 RLinf 框架内，基于 **XSquare Turtle2** 双臂机器人平台启动真机强化学习训练任务的完整指南。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自最多三个机载相机的 RGB 图像。
2. **动作生成**：为一条或两条机械臂产生精确的末端执行器增量动作（位置、旋转及夹爪控制）。
3. **强化学习**：结合真实环境反馈，使用 SAC 优化策略。

环境
----

**真实世界环境**

- **机器人**：XSquare Turtle2 双臂桌面机器人，支持最多 2 条机械臂（左臂 ID ``0``，右臂 ID ``1``）以及最多 3 个 RGB 相机（ID ``0``、``1``、``2``）。
- **任务**：目前支持**按键按压**任务（``ButtonEnv``）：

  - 机械臂末端执行器向下运动，按压位于目标位姿处的按键。
  - 随机复位时加入 ±5 cm 位置噪声和 ±20° 姿态噪声，以提升任务难度。
  - 任务描述字符串：*"Press the button with the end-effector."*

- **观测（Observation）**：

  - 来自一个或多个相机的 RGB 图像（128 × 128），以 ``frames/wrist_<k>`` 形式返回。
  - TCP 位姿：每条激活机械臂的位置（xyz）与四元数（xyzw），拼接为一维向量。

    - 单臂：``[batch_size, 7]``
    - 双臂：``[batch_size, 14]``

- **动作空间（Action Space）**：每条机械臂 7 维连续动作，双臂叠加：

  - 三维增量位置（Δx, Δy, Δz）
  - 三维增量姿态（Δroll, Δpitch, Δyaw）
  - 夹爪宽度指令（开/合）

  单臂：``(7,)``；双臂：``(14,)``；取值归一化至 ``[-1, 1]``。

**数据结构**

- **Images**：RGB 张量 ``[batch_size, 128, 128, 3]``
- **Actions**：归一化连续值，每维度取值范围 ``[-1, 1]``
- **Rewards**：成功时返回 ``1.0``（所有激活机械臂均到达目标位置阈值内），否则返回 ``0.0``；可选稠密指数奖励


算法
----

**核心算法组件**

1. **SAC（Soft Actor-Critic）**

   - 通过 Bellman 公式和熵正则化学习 Q 值。
   - 学习策略网络以最大化熵正则化的 Q 值。
   - 自动调节温度参数（``alpha``）以平衡探索与利用。

2. **Cross-Q** （可选）

   - SAC 的一种变体，去除了目标 Q 网络。
   - 在一个批次中连接当前和下一个观测，结合 BatchNorm 实现稳定的 Q 训练。

3. **RLPD（Reinforcement Learning with Prior Data）** （可选）

   - 在在线 SAC 中融合离线示范数据。
   - 高更新-数据比以充分利用已采集数据。

4. **CNN 策略网络**

   - 基于 ResNet 的视觉编码器，处理 RGB 输入。
   - MLP 层融合图像特征与本体感知状态，输出动作。
   - 独立 Q-head 实现 Critic 功能。


硬件环境搭建
------------

真机实验需要以下硬件：

- **机器人**：XSquare Turtle2 双臂机器人
- **相机**：机器人搭载的最多 3 个 RGB 相机（ID 0–2）
- **训练 / Rollout 节点**：一台带有 GPU 的计算机，运行 CNN 策略
- **机器人控制节点**：一台与机器人处于同一局域网的小型计算机（不需要 GPU）

.. warning::

  请确保训练节点与机器人控制节点处于**同一局域网**中。


依赖安装
--------

控制节点与训练 / Rollout 节点需要安装不同的软件依赖。

机器人控制节点
~~~~~~~~~~~~~~

XSquare Turtle2平台自带SDK和基于ROS的控制器。**请在开始下安装之以前，确保您已进入Xsquare的官方Docker容器**。请联系`XSquare <https://x2robot.com>`_获取准确的Docker镜像和启动说明。

进入 XSquare Docker 容器后，在其中克隆 RLinf 仓库：

.. code:: bash

   # 为了提高国内下载速度，也可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

然后安装真机强化学习所需的 RLinf Python 依赖：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 标志
   bash requirements/install.sh embodied --env xsquare_turtle2
   source .venv/bin/activate

.. note::

   在 Turtle2 控制节点上，推荐采用**双 Docker 分工**来降低 ROS 冲突风险：

   - 控制容器（例如厂商提供的 Turtle2 控制容器，如 ``turtle2_release``）负责 ROS master、UI、底层 bring-up，以及相机/机械臂控制节点。
   - 独立的 RLinf 容器只负责 RLinf 运行时与 Ray worker，不直接接管 ``roscore``、UI 或 ``run.sh``。

   建议 RLinf 容器基于与控制容器**兼容的同源镜像配置**构建，例如复用相同的基础镜像、系统依赖、ROS 运行时和必要挂载，只在其上额外加入 RLinf 所需的 Python 环境与代码。这样可以减少因为容器环境漂移导致的消息定义、设备访问或 ROS 依赖不一致问题。

   一个常见做法是让两个容器都使用 ``host network``，从而共享控制节点的网络栈；这样 RLinf 容器可以直接连接已有的 ROS master，而不需要重新在自己的容器里启动一套控制面。实际训练或评估时，建议仅在控制容器中临时打开 UI 做校准，完成后关闭 UI，再由 RLinf 容器执行任务，以避免多个进程同时占用机械臂控制话题。

训练 / Rollout 节点
~~~~~~~~~~~~~~~~~~~

a. 克隆 RLinf 仓库
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # 为了提高国内下载速度，也可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. 安装依赖
^^^^^^^^^^^

**方式 1：Docker 镜像**

.. code:: bash

   # 训练 / rollout 节点使用 maniskill_libero 镜像
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 为了提高国内下载速度，也可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

**方式 2：自定义环境**

.. code:: bash

   # 在训练 / rollout 节点上安装 openvla + maniskill_libero 环境
   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 标志
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate


模型下载
--------

在开始训练之前，需要下载预训练的 ResNet CNN 骨干网络：

.. code:: bash

   # 方式 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 方式 2：使用 huggingface-hub
   # 为了提高国内下载速度：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

下载完成后，请在对应的配置 YAML 文件中正确填写 ``model_path`` 字段。


运行实验
--------

前置准备
~~~~~~~~

**获取任务目标末端位姿**

对于每个任务，需要记录触发成功信号的目标末端执行器位姿。
通过 XSquare 控制界面手动将机械臂移动至期望目标位置，然后读取当前位姿。

位姿以欧拉角格式存储：``[x, y, z, rz, ry, rx]`` （XSquare 约定）。
如使用双臂模式，需为两条机械臂分别记录目标位姿。

集群配置
~~~~~~~~

在正式开始实验之前，需要先正确搭建 Ray 集群。

.. warning::

  这一步非常关键，请谨慎操作！任何细微的配置错误，都可能导致依赖缺失或无法正确控制机器人。

RLinf 使用 Ray 管理分布式环境。在某个节点上执行 ``ray start`` 时，Ray 会记录当时的
Python 解释器路径和环境变量；之后在该节点上由 Ray 启动的所有进程都会继承同一套配置。

我们提供了脚本 ``ray_utils/realworld/setup_before_ray.sh``，
用于在每个节点启动 Ray 之前统一设置环境。根据自己的环境修改该脚本并在每个节点上 source 它。

脚本主要负责：

1. 激活正确的虚拟 Python 环境（参见依赖安装部分）。
2. 在控制节点上，确保 XSquare SDK 包可被正确找到（使用 XSquare 官方 Docker 镜像时已自动处理）。
3. 在所有节点上设置 RLinf 相关环境变量：

.. code-block:: bash

   export PYTHONPATH=<path_to_your_RLinf_repo>:$PYTHONPATH
   export RLINF_NODE_RANK=<node_rank_of_this_node>
   export RLINF_COMM_NET_DEVICES=<network_device>  # 仅在机器拥有多个网卡时需要设置

``RLINF_NODE_RANK`` 应在集群的 ``N`` 个节点之间设置为 ``0 ~ N-1``。
``RLINF_COMM_NET_DEVICES`` 为可选项，仅在多网卡机器上需要设置；可以通过 ``ifconfig`` 或 ``ip addr`` 查看。

完成上述配置后，在各节点上启动 Ray：

.. code-block:: bash

   # 在 head 节点（节点 rank 0）上
   ray start --head --port=6379 --node-ip-address=<head_node_ip_address>

   # 在 worker 节点（节点 rank 1 ~ N-1）上
   ray start --address='<head_node_ip_address>:6379'

可以通过执行 ``ray status`` 检查集群是否已正确启动。

配置文件
~~~~~~~~

根据实际设置修改配置文件 ``examples/embodiment/config/realworld_button_turtle2_sac_cnn.yaml``。

需要更新的关键字段：

.. code-block:: yaml

  cluster:
    num_nodes: 2  # 1 个训练/rollout 节点 + 1 个控制节点
    component_placement:
      actor:
        node_group: "gpu"
        placement: 0
      rollout:
        node_group: "gpu"
        placement: 0
      env:
        node_group: turtle2
        placement: 0
    node_groups:
      - label: "gpu"
        node_ranks: 0
      - label: turtle2
        node_ranks: 1
        hardware:
          type: Turtle2
          configs:
            - node_rank: 1

  env:
    train:
      override_cfg:
        is_dummy: False
        use_arm_ids: [1]          # 0=左臂，1=右臂；双臂使用 [0,1]
        use_camera_ids: [2]       # 要使用的相机 ID（0、1 或 2）
        target_ee_pose:           # [[左臂目标位姿], [右臂目标位姿]]，欧拉角 [x,y,z,rz,ry,rx]
          - [0, 0, 0, 0, 0, 0]
          - [0.3, 0.0, 0.15, 0.0, 1.0, 0.0]

  actor:
    model:
      model_path: "/path/to/RLinf-ResNet10-pretrained"
      state_dim: 6    # 单臂 6（xyz+euler），双臂 12
      action_dim: 6   # 单臂 6（xyz_delta+rpy_delta），双臂 12

  rollout:
    model:
      model_path: "/path/to/RLinf-ResNet10-pretrained"

对于**按键按压**任务，``target_ee_pose`` 同时定义了成功判断的阈值位置和复位位置
（机械臂复位时会移动到目标位置 Z 轴方向稍高处）。

检查环境（可选）
~~~~~~~~~~~~~~~~

在正式启动实验前，可以使用 dummy 模式验证集群和模型流水线是否正常：

在 ``env.train.override_cfg`` 和 ``env.eval.override_cfg`` 中均将 ``is_dummy`` 设置为 ``True``
以启用 dummy 模式（无需真实机器人）。

在 head 节点上运行：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dummy_turtle2_sac_cnn

运行实验
~~~~~~~~

完成上述检查后，即可在 head 节点上启动真机训练实验：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_button_turtle2_sac_cnn


部署
----

在只做策略 rollout 或真机评估时，RLinf 提供了通用 Turtle2 部署环境，
注册名为 ``Turtle2DeployEnv-v1``。对应的环境片段配置为：

.. code-block:: text

   examples/embodiment/config/env/realworld_turtle2_deploy.yaml

可以将它作为 eval-only 配置中的 ``env.eval`` 使用。例如复制
``examples/embodiment/config/realworld_eval.yaml`` 后，将 env default 替换为：

.. code-block:: yaml

   defaults:
     - env/realworld_turtle2_deploy@env.eval
     - model/pi0@actor.model
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - override hydra/job_logging: stdout

Turtle2 deploy 环境片段会将 ``init_params.id`` 设置为
``Turtle2DeployEnv-v1``，并在 env 层暴露 ``action_mode``：

.. code-block:: yaml

   env:
     eval:
       action_mode: absolute_pose
       override_cfg:
         task_description: "Describe your Turtle2 deployment task here."
         use_arm_ids: [0, 1]
         use_camera_ids: [0, 1, 2]
         reset_ee_pose:
           - [0.20, 0.00, 0.10, 0.00, 0.00, 0.00]
           - [0.20, 0.00, 0.10, 0.00, 0.00, 0.00]
         ee_pose_limit_min:
           - [-0.20, -0.60, -0.05, -3.20, -3.20, -3.20]
           - [-0.20, -0.60, -0.05, -3.20, -3.20, -3.20]
         ee_pose_limit_max:
           - [0.80, 0.60, 0.60, 3.20, 3.20, 3.20]
           - [0.80, 0.60, 0.60, 3.20, 3.20, 3.20]

``apply_dual_pose_action_wrappers`` 会挂载部署动作 wrapper：

- ``absolute_pose`` 使用 ``DualAbsolutePoseActionWrapper``。策略为每条激活机械臂
  输出一个 7 维绝对位姿动作：``[x, y, z, roll, pitch, yaw, gripper]``。
- ``relative_pose`` 使用 ``DualRelativePoseActionWrapper``。默认还会启用
  ``DualRelativeFrame``，将末端坐标系下的增量动作转换回机器人基坐标系。
- 两种模式最后都会经过 ``DualQuat2EulerWrapper``，因此策略看到的是欧拉角格式
  的 TCP 状态。

在 Ray head 节点上运行 eval-only 部署配置：

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh <your_turtle2_eval_config_name>


可视化与结果
------------

**1. TensorBoard 日志**

在 Ray head 节点上运行：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **环境指标**：

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）。
  - ``env/return``：回合总回报。
  - ``env/reward``：环境的 step-level 奖励。
  - ``env/success_once``：回合中至少成功一次标志（0 或 1）。

- **训练指标**：

  - ``train/sac/critic_loss``：Q 函数的损失。
  - ``train/critic/grad_norm``：Q 函数的梯度范数。
  - ``train/sac/actor_loss``：策略损失。
  - ``train/actor/entropy``：策略熵。
  - ``train/actor/grad_norm``：策略的梯度范数。
  - ``train/sac/alpha_loss``：温度参数的损失。
  - ``train/sac/alpha``：温度参数的值。
  - ``train/alpha/grad_norm``：温度参数的梯度范数。
  - ``train/replay_buffer/size``：当前重放缓冲区的大小。
  - ``train/replay_buffer/max_reward``：重放缓冲区中存储的最大奖励。
  - ``train/replay_buffer/min_reward``：重放缓冲区中存储的最小奖励。
  - ``train/replay_buffer/mean_reward``：重放缓冲区中存储的平均奖励。
  - ``train/replay_buffer/utilization``：重放缓冲区的利用率。
