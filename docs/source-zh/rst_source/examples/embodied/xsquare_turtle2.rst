XSquare Turtle2 真机强化学习
========================================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg
   :align: center
   :width: 80%

   用于 SAC/CNN 真机按键训练的 XSquare Turtle2 双臂机器人。

在 XSquare Turtle2 双臂机器人上训练真机策略。你将进入厂商控制容器，安装 RLinf 依赖，设置按键任务目标位姿，并在机器人节点与 GPU 节点上启动 SAC/CNN 训练。

概览
----------------------------------------

在 XSquare Turtle2 上训练用于按键任务的视觉 SAC 策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN policy

   .. grid-item-card:: 算法
      :text-align: center

      SAC · Cross-Q · RLPD

   .. grid-item-card:: 任务
      :text-align: center

      Button pressing

   .. grid-item-card:: 硬件
      :text-align: center

      XSquare Turtle2 · 1–2 arms · cameras

| **你将完成:** 进入厂商容器 → 安装 RLinf env → 设置目标位姿 → 测试 dummy 配置 → 训练.
| **前置条件:** :doc:`安装 </rst_source/start/installation>` · XSquare Docker/controller stack · 局域网.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - 任务
     - 配置 / 入口
     - 说明
   * - Dummy check
     - ``realworld_dummy_turtle2_sac_cnn``
     - 在无硬件运动情况下验证配置和集群连通。
   * - Training
     - ``realworld_button_turtle2_sac_cnn``
     - 使用一条或两条活动机械臂训练按键任务。
   * - Monitoring
     - TensorBoard logs
     - 跟踪 reward、return 与 replay-buffer 统计。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - 最多三路 RGB 相机流，以及每条活动机械臂的 TCP 位姿。
   * - Action
     - 每条机械臂 7 维连续增量位姿与夹爪命令。
   * - Reward
     - 按键成功时为 ``1.0``；可选指数稠密 shaping。
   * - Prompt
     - ``Press the button with the end-effector.``

硬件环境搭建
----------------------------------------

真机实验需要以下硬件：

- **机器人**：XSquare Turtle2 双臂机器人
- **相机**：机器人搭载的最多 3 个 RGB 相机（ID 0–2）
- **训练 / Rollout 节点**：一台带有 GPU 的计算机，运行 CNN 策略
- **机器人控制节点**：一台与机器人处于同一局域网的小型计算机（不需要 GPU）

.. warning::

  请确保训练节点与机器人控制节点处于同一局域网中。


安装
----------------------------------------

控制节点与训练 / Rollout 节点需要安装不同的软件依赖。

机器人控制节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

训练 / Rollout 节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为了提高国内下载速度，也可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

B. 安装依赖
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


下载模型
----------------------------------------

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


运行
----------------------------------------

前置准备
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**获取任务目标末端位姿**

对于每个任务，需要记录触发成功信号的目标末端执行器位姿。
通过 XSquare 控制界面手动将机械臂移动至期望目标位置，然后读取当前位姿。

位姿以欧拉角格式存储：``[x, y, z, rz, ry, rx]`` （XSquare 约定）。
如使用双臂模式，需为两条机械臂分别记录目标位姿。

集群设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

对于按键按压任务，``target_ee_pose`` 同时定义了成功判断的阈值位置和复位位置
（机械臂复位时会移动到目标位置 Z 轴方向稍高处）。

检查环境（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在正式启动实验前，可以使用 dummy 模式验证集群和模型流水线是否正常：

在 ``env.train.override_cfg`` 和 ``env.eval.override_cfg`` 中均将 ``is_dummy`` 设置为 ``True``
以启用 dummy 模式（无需真实机器人）。

在 head 节点上运行：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dummy_turtle2_sac_cnn

运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

完成上述检查后，即可在 head 节点上启动真机训练实验：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_button_turtle2_sac_cnn


可视化与结果
----------------------------------------

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
