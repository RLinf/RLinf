基于 GenieSim 的 Place Workpiece 强化学习训练
==============================================

本文档给出在 **RLinf** 框架内，基于 **GenieSim** 仿真平台运行 **Place Workpiece** （工件放置）
任务的完整指南。涵盖从镜像构建、SpaceMouse 演示数据采集、到 SAC 人机协同训练的完整流程。

主要特点：

1. **Isaac Sim + MuJoCo 双仿真器**：Isaac Sim 提供高保真渲染，MuJoCo 提供高频物理仿真；
2. **SpaceMouse 人机协同 (Human-in-the-Loop)**：训练过程中人类操作员可实时干预策略；
3. **RLPD + BC 正则化**：演示数据同时用于 demo buffer 对称采样和行为克隆正则化；
4. **SAC + CNN Encoder**：使用 ResNet-10 编码器处理相机图像，SAC 优化连续动作策略。

环境
----

**GenieSim Place Workpiece 环境**

- **仿真平台**：GenieSim（Isaac Sim 渲染 + MuJoCo 物理）
- **任务**：控制 G2 机器人右臂将工件放入工位目标槽位
- **观测空间**：

  - 右腕相机 RGB 图像（128×128，从 480×480 缩放）
  - 右臂末端执行器状态（12 维：EE 位置、姿态、线速度、角速度）

- **动作空间**：7 维连续 delta 动作

  - 三维位置增量（dx, dy, dz）
  - 三维旋转增量（droll, dpitch, dyaw）
  - 1 维夹爪控制指令

- **奖励**：

  - ``r_alive``：基于工件与目标的三维距离和姿态误差的指数衰减存活奖励
  - ``r_below``：工件低于目标高度时的惩罚，防止过度下压
  - ``r_success``：工件成功放入目标位置并保持静止后的稀疏奖励

算法
----

核心算法组件包括：

1. **SAC (Soft Actor-Critic)**

   - 双 Critic 网络减少 Q 值过估计；
   - 自动熵温度调节；
   - RLPD 对称采样：replay buffer 与 demo buffer 各 50%。

2. **Human-in-the-Loop**

   - env_0 接受 SpaceMouse 实时干预；
   - 干预动作存入 replay buffer 并提取到 demo buffer；
   - BC 正则化损失引导策略向专家行为靠拢。

3. **CNN Encoder**

   - ResNet-10 编码右腕相机图像（128×128）；
   - 图像特征与右臂 EE 状态（12 维）拼接后送入 Actor/Critic。

前置要求
--------

硬件
~~~~

- NVIDIA GPU（推荐 RTX 3090 及以上，显存 ≥ 24GB）
- 3Dconnexion SpaceMouse（采集数据时需要）

软件
~~~~

- Ubuntu 22.04 / 24.04
- Docker（带 NVIDIA Container Toolkit）
- NVIDIA 驱动 ≥ 535

仓库准备
~~~~~~~~

.. code:: bash

   mkdir workspace && cd workspace
   git clone https://github.com/AgibotTech/genie_sim.git
   git clone -b dev/geniesim https://github.com/RLinf/RLinf.git

.. code:: text

   workspace/
   ├── genie_sim/     # GenieSim 仓库
   └── RLinf/         # RLinf 仓库

确保两个仓库在同一父目录下。GenieSim 的安装与资产下载请参考
`GenieSim 官方文档 <https://agibot-world.com/sim-evaluation/docs/#/v3>`_ 。

依赖安装
--------

1. 构建基础镜像
~~~~~~~~~~~~~~~~

a. 请先参考 `GenieSim 官方文档 <https://agibot-world.com/sim-evaluation/docs/#/v3>`_
   的 2.3.1 Docker Container 部分构建 GenieSim 基础镜像。

b. 在 GenieSim 基础镜像之上构建 RLinf 集成镜像：

   .. code:: bash

      cd workspace/
      bash genie_sim/scripts/build_geniesim_rlinf_image.sh

   构建产物：``geniesim-rlinf:latest``。构建脚本使用 ``genie_sim/scripts/`` 作为 build context，不会传输大型资产目录。

2. 构建训练镜像
~~~~~~~~~~~~~~~~

.. code:: bash

   cd workspace/RLinf
   docker build \
     --build-arg BUILD_TARGET=embodied-geniesim \
     -t geniesim-rlinf-train:latest \
     -f docker/Dockerfile \
     .

3. 验证镜像
~~~~~~~~~~~~

.. code:: bash

   docker run --rm --gpus all geniesim-rlinf-train:latest nvidia-smi

4. SpaceMouse 依赖
~~~~~~~~~~~~~~~~~~~

SpaceMouse 通过 HID 协议通信。请在 **宿主机**（非容器内）上安装依赖库并配置 udev 规则：

.. code:: bash

   sudo apt-get install -y libhidapi-dev

创建 udev 规则，使设备无需 root 权限即可访问：

.. code:: bash

   sudo tee /etc/udev/rules.d/99-spacemouse.rules > /dev/null <<'EOF'
   SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666"
   SUBSYSTEM=="input",  ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666"
   EOF

重新加载规则并验证：

.. code:: bash

   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ls -l /dev/hidraw*   # 应显示 0666 权限

资源下载
--------

下载 CNN encoder 预训练权重：

.. code:: bash

   # 下载模型（两种方式二选一）
   # 方式 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 方式 2：使用 huggingface-hub
   # 为了提高国内下载速度，可以添加以下环境变量：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

下载完成后，请在对应的配置 YAML 文件中正确填写模型路径。

运行脚本
--------

**1. 采集演示数据**

将 SpaceMouse 通过 USB 连接到宿主机后：

.. code:: bash

   cd workspace/
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh collect --num-demos 50

可通过 ``--save-dir`` 指定容器内的保存路径（默认：``/geniesim/RLinf/sac_demo``）：

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh collect --num-demos 50 --save-dir /geniesim/RLinf/my_demos

SpaceMouse 操作方式：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 操作
     - 效果
   * - 平移设备
     - 移动右臂末端执行器 (x/y/z)
   * - 旋转设备
     - 旋转右臂末端执行器 (roll/pitch/yaw)
   * - 按下左键
     - 结束轨迹 → 保存演示 → 环境重置
   * - 按下右键
     - 结束轨迹 → 丢弃演示 → 环境重置

采集完成后，演示数据默认保存在 ``workspace/RLinf/sac_demo/`` 下（对应容器内 ``/geniesim/RLinf/sac_demo``）。

**2. 转换演示数据**

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh convert

默认从 ``/geniesim/RLinf/sac_demo`` 读取，输出到 ``/geniesim/RLinf/sac_demo_buffer``。可自定义路径：

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh convert --demo-dir /geniesim/RLinf/my_demos --output-dir /geniesim/RLinf/my_demos_buffer

**3. 启动训练**

启动训练前，请先编辑训练配置文件
``examples/embodiment/config/geniesim_sac_spacemouse.yaml``，将以下路径修改为容器内路径
（RLinf 在容器内挂载于 ``/geniesim/RLinf``）：

- ``rollout.model.model_path`` 和 ``actor.model.model_path``：设为预训练模型目录，
  例如 ``/geniesim/RLinf/examples/embodiment/config/RLinf-ResNet10-pretrained``
- ``algorithm.demo_buffer.load_path``：如果已转换演示数据，设为 buffer 目录，
  例如 ``/geniesim/RLinf/sac_demo_buffer``

配置完成后启动训练：

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train

训练过程中可使用 SpaceMouse 在 env_0 上实时干预策略，其余环境由策略自主运行。

**4. 关键配置文件**

- **环境配置**：``examples/embodiment/config/env/geniesim_place_workpiece.yaml``
- **训练配置**：``examples/embodiment/config/geniesim_sac_spacemouse.yaml``
- **模型配置**：``examples/embodiment/config/model/cnn_policy.yaml``

**5. Hydra 参数覆盖**

.. code:: bash

   # 调整折扣因子
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.gamma=0.97

   # 调整 BC 正则化系数
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.bc_coef=5.0

   # 修改日志路径
   bash RLinf/rlinf/envs/geniesim/scripts/run.sh train runner.logger.log_path=../my_results

**6. 调试与日志**

启动一个空白容器进行交互式调试：

.. code:: bash

   bash RLinf/rlinf/envs/geniesim/scripts/run.sh shell

查看正在运行的容器（如训练或数据采集中）的仿真日志，可通过 ``docker exec`` 接入容器。日志位于 ``/tmp/geniesim_logs/``：

.. code:: bash

   docker exec -it <container_name> bash
   ls /tmp/geniesim_logs/

可视化与结果
------------

**1. TensorBoard 日志**

日志目录取决于训练配置中 ``runner.logger.log_path`` 的设置（默认：``results``）：

.. code:: bash

   tensorboard --logdir <log_path>

**2. 关键监控指标**

- **训练指标**：

  - ``critic_loss``：Critic 损失，应逐步下降
  - ``actor_loss``：Actor 策略损失
  - ``q_values``：Q 值估计，应逐步上升但不应爆炸
  - ``entropy``：策略熵，反映探索程度
  - ``bc_loss``：行为克隆损失（仅在有 demo buffer 时）

- **环境指标**：

  - ``eval/success_rate``：评估环境成功率
  - ``env/episode_len``：回合步数
  - ``env/reward``：每步奖励
