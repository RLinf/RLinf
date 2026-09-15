Franka 真机强化学习
====================

本示例介绍如何使用 RLinf 在 Franka 机械臂上收集演示并通过 RLPD 在线训练 CNN policy。默认配置在一台配备 GPU 的计算机上运行 Franky 控制、rollout 和训练，使用 Ubuntu 22.04 与 CUDA，无需 ROS。先完成下方的插孔示例，再根据需要配置独立控制节点或其他硬件与 policy。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: 用于真机强化学习的 Franka 机械臂

   用于真机强化学习的 Franka 机械臂。

概览
----

Policy 从相机图像和机器人状态中学习，成功演示用于提供初始回放数据。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN policy

   .. grid-item-card:: 算法
      :text-align: center

      SAC / RLPD

   .. grid-item-card:: 任务
      :text-align: center

      插孔

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · RealSense · NVIDIA GPU

任务
~~~~

本示例将插销插入测得的目标位姿。``realworld_peginsertion_rlpd_cnn_async`` 配置使用演示数据和实时机器人交互数据进行异步训练。SpaceMouse 用于收集演示，也可在训练期间提供人工干预。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 说明
   * - 观测
     - 第一个配置的相机（``wrist_1``）的 RGB 图像和机器人状态。
   * - 动作
     - 六维笛卡尔位置与旋转增量；夹爪保持闭合。
   * - 奖励
     - 末端位姿达到配置的目标容差时判定成功。

硬件准备
--------

通过有线网络将 Franka 连接到计算机，并通过 USB 连接 RealSense 相机和 SpaceMouse。下方命令以 x86-64 Ubuntu 22.04 主机为例。按照 `Ubuntu 驱动安装指南 <https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/>`_ 安装 NVIDIA 驱动；如果选择实时内核，则先完成下方的内核与 CUDA 安装步骤。使用 Docker 时，还需按照 NVIDIA 官方指南完成 `Container Toolkit 安装与 Docker 配置 <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_。

.. warning::

   每次运行真机时都应由操作员全程监控，并确保急停装置触手可及。固定插销与夹具，清空工作区域，检查复位路径是否安全。此任务会复位到目标上方约 10 cm 的位置，并随机改变水平位置和偏航角。不要直接使用其他机器人的目标位姿。

在机器人的地址打开 Franka Desk，记录 Control 固件版本，并根据 `Franka 兼容性表 <https://frankarobotics.github.io/docs/compatibility.html>`_ 选择 libfranka 版本。镜像内置 libfranka 0.19.0；如果固件要求其他版本，请使用下方的自定义安装方式。不要仅为匹配示例而修改机器人固件。

实时内核安装（可选）
~~~~~~~~~~~~~~~~~~~~

建议使用 PREEMPT_RT 内核，以满足 Franky 控制循环对时序的要求。在 RLinf 工作流中，实时内核是可选的：驱动会尝试设置实时调度和锁定内存，无法启用时仍可继续运行。没有实时内核时，较重的 CPU 或 GPU 训练负载可能使控制响应变慢；错过控制周期也可能导致运动停止。

在宿主机上安装内核，不要在 Docker 容器内安装，因为容器与宿主机共享内核。Ubuntu 22.04 优先使用下方的 Ubuntu Pro 安装方式。没有 Ubuntu Pro 订阅时，可按照 Franka 的 `手动内核安装指南 <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_ 编译安装。

.. warning::

   NVIDIA 驱动未正式支持 PREEMPT_RT 内核。``IGNORE_PREEMPT_RT_PRESENCE=1`` 仅跳过驱动构建时的检查，不保证兼容性。请保留原内核作为 GRUB 回退选项，并在训练前检查 GPU 和机器人。

安装 Ubuntu Pro 内核
^^^^^^^^^^^^^^^^^^^^

在宿主机上更新 Pro 客户端并关联符合条件的 Ubuntu Pro 订阅。已关联的机器可跳过 ``pro attach``。以下命令参考 `Ubuntu Pro 安装指南 <https://ubuntu.com/pro-client/docs/en/docs/howtoguides/enable_realtime_kernel/>`_：

.. code:: bash

   sudo apt update
   sudo apt install ubuntu-pro-client
   sudo pro attach
   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 pro enable realtime-kernel

阅读并确认交互提示；如提示与 Livepatch 冲突，按提示停用 Livepatch。环境变量允许已有的 NVIDIA DKMS 驱动尝试为新内核重新构建。安装成功后，保存工作并重启：

.. code:: bash

   sudo reboot

如果 GRUB 未自动选择实时内核，请手动选择。重新登录后检查当前内核：

.. code:: bash

   uname -r
   cat /sys/kernel/realtime

第二条命令必须输出 ``1``，否则请先从 GRUB 的高级选项中选择已安装的实时内核。

在实时内核上使用 CUDA
^^^^^^^^^^^^^^^^^^^^^^

GPU 主机需要为当前实时内核构建的 NVIDIA 驱动。如果重启后 ``nvidia-smi`` 已能列出 GPU，可保留现有驱动；否则，使用下方 APT 命令安装适用于 Turing 及更新架构 GPU 的 NVIDIA 开放驱动。旧架构 GPU 或已有 runfile 安装请参考 NVIDIA 的 `驱动安装指南 <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_，选择兼容驱动，避免混用安装方式。

.. code:: bash

   sudo apt install build-essential dkms wget "linux-headers-$(uname -r)"
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt install nvidia-open
   sudo reboot

如果 Secure Boot 要求注册 Machine Owner Key（MOK），请在重启时完成注册，以便加载新驱动。在实时内核下再次检查 ``nvidia-smi``；如果 GPU 不可用，不要开始训练，可先使用原内核恢复系统。后续内核更新需要重新构建驱动时，同样需要设置实时检查的覆盖变量。

使用 Docker 时，Franka 镜像已包含 CUDA；宿主机只需按前文硬件准备中的说明安装驱动和 NVIDIA Container Toolkit。使用自定义环境时，还需从上方配置的 NVIDIA APT 源安装 CUDA 12.8。如果跳过了驱动安装，请先执行其中的 ``wget``、``dpkg`` 和 ``apt update`` 命令注册软件源。使用 `仅包含 toolkit 的安装包 <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/#meta-packages>`_，避免此步骤替换驱动：

.. code:: bash

   sudo apt install cuda-toolkit-12-8
   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH="$CUDA_HOME/bin:$PATH"
   nvcc --version

在每个训练终端保留这两条环境变量设置，或将其加入 shell 启动文件。``nvcc`` 应显示 CUDA 12.8；下方的环境检查还会验证 PyTorch 能否使用 GPU。

授予实时调度权限
^^^^^^^^^^^^^^^^

下方 Docker 命令已授予实时优先级和内存锁定权限。使用自定义环境时，在宿主机上将登录账户加入专用用户组：

.. code:: bash

   getent group realtime || sudo groupadd realtime
   sudo usermod -aG realtime "$(id -un)"
   sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

将以下内容写入该文件，然后退出并重新登录：

.. code:: text

   @realtime - rtprio 99
   @realtime - memlock unlimited

在新登录的终端中，``ulimit -r`` 应输出 ``99``，``ulimit -l`` 应输出 ``unlimited``。开始训练前，应在操作员监控下检查预期训练负载对控制的影响。

安装
----

克隆 RLinf，后续命令均在仓库根目录执行：

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

选择 Docker 或自定义环境。两种方式均安装 Franky 以及 CNN 训练示例所需的依赖。

Docker（推荐）
~~~~~~~~~~~~~~

拉取 Franka 镜像并启动容器：

.. code:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.4-franka

   docker run -it --name rlinf-franka --gpus all \
     --network host --privileged --shm-size 20g \
     --ulimit rtprio=99 --ulimit memlock=-1 \
     -v "$PWD:/workspace/RLinf" -w /workspace/RLinf \
     rlinf/rlinf:agentic-rlinf0.4-franka bash

镜像使用 CUDA 和 Ubuntu 22.04。在容器内激活已安装的 Franky 环境，并在后续步骤中保持一致：

.. code:: bash

   source switch_env franky-0.19.0

如需另开终端，在宿主机执行 ``docker exec -it rlinf-franka bash``，然后再次激活相同环境。

自定义环境
~~~~~~~~~~

不使用 Docker 时，执行以下命令：

.. code:: bash

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

安装脚本默认使用 libfranka 0.19.0；仅当固件需要时，才在命令前设置 ``LIBFRANKA_VERSION=0.15.0``。安装脚本使用包含 libfranka 的版本化 Franky wheel，无需另行安装 libfranka 或 ROS。在 Docker 外运行时，当前用户必须具有相机和 SpaceMouse USB 设备的读取权限；参见 :doc:`/rst_source/start/installation` 和 `SpaceMouse 安装说明 <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_。

检查环境
~~~~~~~~

在已激活的环境中，先检查 Franky 和 CUDA，再连接机械臂：

.. code:: bash

   python -c "import franky, torch; assert torch.cuda.is_available(); print(torch.__version__)"

下载模型
--------

将预训练 ResNet encoder 下载到仓库内：

.. code:: bash

   hf download RLinf/RLinf-ResNet10-pretrained \
     --local-dir ./models/RLinf-ResNet10-pretrained

后面的训练命令会将该目录同时传给 actor 和 rollout。

运行
----

检查相机与目标位姿
~~~~~~~~~~~~~~~~~~

先检查相机数据流，记录输出的序列号：

.. code:: bash

   python toolkits/realworld_check/test_franka_camera.py

在以下两个现有配置中填写机器人信息：

- ``examples/embodiment/config/realworld_collect_data.yaml``
- ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``

在每个文件的 ``cluster.node_groups`` 中，仅将 ``label: franka`` 对应的条目替换为下方内容。将 ``ROBOT_IP`` 替换为机械臂地址，将 ``CAMERA_SERIAL`` 替换为输出的相机序列号：

.. code:: yaml

   - label: franka
     node_ranks: 0
     hardware:
       type: Franka
       configs:
         - robot_ip: ROBOT_IP
           node_rank: 0
           camera_serials: ["CAMERA_SERIAL"]

将训练配置中的 ``cluster.num_nodes`` 也设为 1；采集配置已经使用一个节点。训练配置中的 ``4090`` 节点组和组件放置保持不变：这个组名表示 GPU 节点 0，不要求 GPU 型号为 4090。本示例使用一个相机，自动命名为 ``wrist_1``。

通过机器人的引导模式将插销放到成功插入时的目标位姿，然后解锁机械臂并在 Franka Desk 中启用 FCI。使用 Franky 读取位姿：

.. code:: bash

   export FRANKA_ROBOT_IP=192.168.1.10  # 替换为机器人的地址。
   python -m toolkits.realworld_check.test_franka_controller

在提示符后输入 ``getpos_euler``，再输入 ``q`` 释放机器人。输出顺序为 ``[x, y, z, roll, pitch, yaw]``，单位为米和弧度。将测得的六个数值填入下方列表，使用逗号分隔，并保存在当前终端中：

.. code:: bash

   export FRANKA_TARGET_POSE='[x, y, z, roll, pitch, yaw]'  # 替换全部六个数值。

继续之前，确认目标位姿及上文所述的复位区域均处于安全工作范围内。关闭其他控制机械臂的程序；同一时刻只能有一个进程持有控制连接。

收集演示
~~~~~~~~

启动 Ray 前先设置节点编号。如果硬件检查脚本已经启动了本地 Ray 实例，先停止该实例，让 Ray 重新读取已激活的环境和正确的节点编号：

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=0
   ray start --head

移动或旋转 SpaceMouse 控制末端。达到目标容差时，任务会自动标记成功并复位，随后开始下一条演示。收集 20 条成功演示：

.. code:: bash

   RLINF_LOG_DIR="$PWD/logs/franka-demo" \
     bash examples/embodiment/collect_data.sh realworld_collect_data \
     "env.eval.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE"

成功轨迹保存在 ``logs/franka-demo/demos`` 下。RLPD 使用这个 replay buffer 目录，而不是 ``collected_data`` 下的可选 episode 导出文件。等待采集程序退出并释放机械臂后，再启动训练。再次采集时请更换日志目录，避免混合不同批次的演示。

训练 Policy
~~~~~~~~~~~

保持相同的环境、Ray 实例和目标位姿，启动训练：

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh \
     realworld_peginsertion_rlpd_cnn_async \
     "env.train.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE" \
     "algorithm.demo_buffer.load_path=$PWD/logs/franka-demo/demos" \
     "actor.model.model_path=$PWD/models/RLinf-ResNet10-pretrained" \
     "rollout.model.model_path=$PWD/models/RLinf-ResNet10-pretrained"

按上述设置，actor、rollout 和 reward 运行在 GPU 0 上，机器人控制运行在节点 0 上。运行期间持续监控机械臂，必要时通过 SpaceMouse 干预。结束实验时，中断启动脚本并等待机器人停止；程序退出后，使用 ``ray stop`` 停止本机 Ray 进程。

如果 Franky 阻抗控制器意外停止，RLinf 会报告运动错误，不会静默重启控制。解决原因后再重新启动训练。直接使用 Python API 时，应先调用 ``disconnect()``，再调用 ``connect()``，然后恢复发送指令；``clear_errors()`` 不会重启已经失败的跟踪控制。

可视化与结果
------------

在另一个已激活环境的终端中启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

打开 ``http://localhost:6006``，关注 ``env/success_once``、``env/return`` 以及 SAC actor 和 critic 的 loss。日志配置参见 :doc:`/rst_source/guides/logger`。下方曲线和视频展示了插孔与充电器任务的实验结果，不代表所有环境都能在相同时间内完成训练。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld-curve.png
   :align: center
   :width: 100%

   真机训练曲线。

.. raw:: html

   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
   </video>
   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/charger-compressed.mp4" type="video/mp4">
   </video>

多节点配置
----------

如需将机器人控制与训练负载分开，可以使用独立的控制计算机。控制节点不需要 GPU、CUDA 或 ROS。将机械臂、相机和 SpaceMouse 接到控制节点，由 GPU 计算机运行 actor 和 rollout。

准备控制节点
~~~~~~~~~~~~

在 Ubuntu 22.04 控制节点上克隆相同版本的 RLinf，在仓库根目录安装 CPU 环境：

.. code:: bash

   UV_TORCH_BACKEND=cpu bash requirements/install.sh embodied --env franka
   source .venv/bin/activate
   python -c "import franky, torch; assert torch.version.cuda is None"

libfranka 版本选择与前文一致，自定义环境的设备权限说明和实时内核建议同样适用于控制节点。加入多节点集群前，先按前文采集步骤在控制节点上收集演示，此时使用节点编号 0。将完整的 ``logs/franka-demo/demos`` 目录复制到 GPU 计算机。

配置并启动集群
~~~~~~~~~~~~~~

在 GPU 计算机上使用默认 CUDA 镜像。两台计算机应使用相同的 RLinf 代码版本、Python 版本和 Ray 版本。在 GPU 计算机上编辑 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``：将 ``cluster.num_nodes`` 改为 2，将 Franka 组的 ``node_ranks`` 和硬件配置中的 ``node_rank`` 都改为 1。保留机械臂 IP 和相机序列号。Actor 和 rollout 仍放在节点 0 的 GPU 0 上。

选择两台计算机在互通网络上的 IP，不要使用机械臂的 IP。在 GPU 计算机已激活环境的终端中执行：

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=0
   export HEAD_IP=192.168.10.10  # 替换为 GPU 计算机的地址。
   ray start --head --port=6379 --node-ip-address="$HEAD_IP"

在控制节点的 CPU 环境中执行：

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=1
   export HEAD_IP=192.168.10.10        # 与 GPU 计算机地址一致。
   export CONTROLLER_IP=192.168.10.11 # 替换为控制计算机的地址。
   ray start --address="$HEAD_IP:6379" --node-ip-address="$CONTROLLER_IP"

在 GPU 计算机上运行 ``ray status``，确认两个节点均已启动。然后仅在 GPU 计算机上执行训练命令，使用本机模型与演示数据路径以及测得的目标位姿。训练时，控制节点不需要模型权重或演示文件。结束后在两个节点分别停止 Ray。多网卡或更多机器人的配置参见 :doc:`/rst_source/guides/hetero`。

兼容已有 ROS 环境
~~~~~~~~~~~~~~~~~

已有的 ROS Noetic 部署可使用 ``rlinf/rlinf:agentic-rlinf0.4-franka-ros`` Docker 镜像，或在 Ubuntu 20.04 上执行 ``bash requirements/install.sh embodied --env franka-ros``。在每个 Franka 硬件配置中设置 ``backend: franka_ros``，并在启动 Ray 前激活匹配的 ``franka-<libfranka-version>`` 环境。仍需遵守 ROS 控制器的固件与实时性要求；Franky 的可选实时调度行为不会改变 ROS 的要求。

其他 Franka 工作流
------------------

使用 VLA policy 时，同一 Docker 镜像提供包含 Franka 依赖的 ``openvla``、``openvla-oft``、``openpi`` 和 ``gr00t`` 环境。激活与模型对应的环境，例如 ``source switch_env openpi``。自定义安装时，将模型与 Franka 一起安装：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env franka --venv openpi
   source openpi/bin/activate

按需将两条命令中的 ``openpi`` 替换为 ``openvla``、``openvla-oft`` 或 ``gr00t``。这些命令仅安装依赖；模型权重、任务配置和训练步骤参见对应工作流。

完成基础示例后，可按需要参考以下指南：

- :doc:`franka_gello` 与 :doc:`franka_vr`：GELLO 或 PICO 遥操作。
- :doc:`franka_pi0_sft_deploy` 与 :doc:`hg-dagger`：OpenPI policy。
- :doc:`franka_reward_model`：学习奖励模型。
- :doc:`franka_zed_robotiq` 与 :doc:`franka_dexhand`：其他相机与末端执行器。
- :doc:`dual_franka`：双臂控制；:doc:`/rst_source/guides/rtc`：重叠执行动作与推理。
