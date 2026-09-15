Franka 真机强化学习
====================

本页介绍如何使用 RLinf 在 Franka 机械臂上训练 CNN policy，从收集演示到 RLPD 在线训练。默认配置只用一台装有 NVIDIA GPU 的 Ubuntu 20.04 计算机，由它同时运行 ROS Noetic、机器人连接、rollout 和训练。你将先准备这台主机（检查固件、安装 NVIDIA 驱动和实时内核），再安装 RLinf，然后运行插孔示例。后面几节分别介绍独立控制节点、不依赖 ROS 的可选 Franky 后端，以及其他 Franka 工作流。

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

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 任务
     - 配置
     - 说明
   * - 插孔
     - ``realworld_collect_data``、``realworld_peginsertion_rlpd_cnn_async``
     - 先用 SpaceMouse 收集演示，再结合演示数据和实时机器人交互数据异步训练。训练期间也可通过 SpaceMouse 人工干预。

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

需要一台 Franka 机械臂、一个 RealSense 相机、一个 SpaceMouse，以及一台装有 NVIDIA GPU 的 x86-64 计算机。通过有线网口将机械臂连接到计算机，再通过 USB 连接相机和 SpaceMouse。

同一套软件支持两种部署方式。本页按单机方式展开；多节点方式复用这些步骤，具体见 `多节点配置`_。

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - 部署方式
     - 计算机
     - 适用场景
   * - 单机（默认）
     - 一台 Ubuntu 20.04 GPU 主机运行 ROS、机器人连接、rollout 和训练。
     - 大多数场景，只需安装和维护一台计算机。
   * - 多节点
     - 一台无 GPU 的控制计算机运行 ROS 和机器人连接；GPU 服务器运行 actor 和 rollout。
     - 希望将机器人控制与训练负载隔离，或 GPU 服务器无法使用 Ubuntu 20.04。

.. warning::

   每次运行真机时都应由操作员全程监控，并确保急停装置触手可及。固定插销与夹具，清空工作区域，检查复位路径是否安全。此任务会复位到目标上方约 10 cm 的位置，并随机改变水平位置和偏航角。不要直接使用其他机器人的目标位姿。

准备机器人主机
--------------

安装 RLinf 之前，需要先在机器人主机上确定三件事：固件决定编译哪个 libfranka 版本；NVIDIA 驱动决定安装脚本能否选用 CUDA 版 PyTorch；内核决定 libfranka 的 1 kHz 控制循环能否实时运行。请在 Ubuntu 20.04 主机上依次完成以下步骤。

检查固件并选择 libfranka 版本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

打开 ``http://<robot_ip>/desk`` 进入 Franka Desk，点击 ``SETTINGS``，记录仪表盘中 ``Control`` 后面的版本号：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_firmware.png
   :align: center
   :width: 60%
   :alt: Franka Desk 仪表盘中的 Control 固件版本

   Franka Desk 中的 Control 固件版本。

在 `Franka 兼容性表 <https://frankarobotics.github.io/docs/compatibility.html>`_ 中查找该版本，确定 libfranka 版本。安装脚本默认编译 libfranka 0.15.0，其他版本在安装时通过 ``LIBFRANKA_VERSION`` 指定。RLinf 已测试过两种组合：固件 5.7.2 至 5.9.0 搭配 libfranka 0.15.0，运行在实时内核上；固件 5.9.2 搭配 libfranka 0.19.0，关闭实时检查后运行在 Ubuntu 20.04 标准内核上。不要仅为匹配示例而修改机器人固件。

在 Ubuntu 20.04 上安装 NVIDIA 驱动
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

驱动必须在 RLinf 之前安装。``requirements/install.sh`` 会读取 ``nvidia-smi`` 报告的 CUDA 版本，并安装该驱动支持的最新 CUDA 版 PyTorch：CUDA 12.6 wheel 需要 560 及以上的驱动，CUDA 12.8 wheel 需要 570 及以上。检测不到驱动时，安装脚本会退回到仅 CPU 的 PyTorch，训练将无法使用 GPU。

如果 ``nvidia-smi`` 已显示 570 或更新的驱动，保留现有驱动并跳过本步骤。否则，添加 NVIDIA 为 Ubuntu 20.04 提供的 CUDA 软件源，并从中安装驱动：

.. code:: bash

   sudo apt-get install -y build-essential dkms wget "linux-headers-$(uname -r)"
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install -y cuda-drivers-575
   sudo reboot

这些命令依次完成：

1. 安装编译驱动内核模块所需的编译器、DKMS 和内核头文件。
2. 通过 ``cuda-keyring`` 注册 NVIDIA 的 APT 软件源和签名密钥。
3. 安装支持 CUDA 12.9 的 575 驱动，并为当前内核编译内核模块。

如果 Secure Boot 要求注册 Machine Owner Key（MOK），请在重启时完成注册，否则模块无法加载。重新登录后，``nvidia-smi`` 应能列出 GPU。不要将该软件源与 ``.run`` 文件或其他软件源安装的驱动混用；如有旧驱动，先按 NVIDIA 的 `驱动安装指南 <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_ 卸载。

``cuda-drivers-575`` 编译的是 NVIDIA 专有内核模块，Ubuntu 20.04 软件源不提供开源内核模块的软件包。GeForce RTX 50 系列等必须使用开源内核模块的 GPU，需要改从 NVIDIA `驱动下载 <https://www.nvidia.com/en-us/drivers/>`_ 页面获取驱动。

本示例不需要 CUDA toolkit，PyTorch wheel 已自带 CUDA 运行时。只有需要编译 CUDA 扩展时才安装 toolkit，并使用仅包含 toolkit 的安装包，避免替换驱动：

.. code:: bash

   sudo apt-get install -y cuda-toolkit-12-8

安装实时内核（推荐）
~~~~~~~~~~~~~~~~~~~~

libfranka 每毫秒向机械臂发送一次指令。PREEMPT_RT 内核能在 rollout 和训练占用 CPU、GPU 时保证这个循环按时执行。``franka_control`` 默认强制要求实时内核，在非 PREEMPT_RT 内核上会拒绝启动。

内核需要安装在宿主机上，不能在 Docker 容器内安装，因为容器与宿主机共享内核。Ubuntu 20.04 请按照 Franka 的 `实时内核指南 <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_ 编译打过补丁的内核，并安装生成的 ``linux-image`` 和 ``linux-headers`` 软件包。保留原内核作为 GRUB 回退选项。重启进入新内核后检查：

.. code:: bash

   uname -r
   cat /sys/kernel/realtime

第二条命令必须输出 ``1``，否则请先从 GRUB 的高级选项中选择实时内核。

在实时内核上使用 NVIDIA 驱动
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA 内核模块也必须针对实时内核重新编译，而它的编译过程会拒绝 PREEMPT_RT 内核，除非设置 ``IGNORE_PREEMPT_RT_PRESENCE=1``。如果进入实时内核后 ``nvidia-smi`` 报错，带上该变量重新编译模块并重启：

.. code:: bash

   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 dkms autoinstall -k "$(uname -r)"
   sudo reboot

如果在实时内核运行后才安装驱动，则将同一变量传给 APT：``sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt-get install -y cuda-drivers-575``。以后驱动或内核更新触发模块重新编译时，同样需要设置该变量。

.. warning::

   NVIDIA 驱动未正式支持 PREEMPT_RT 内核。``IGNORE_PREEMPT_RT_PRESENCE=1`` 仅跳过编译时的检查，不保证兼容性。请在实时内核上确认 ``nvidia-smi`` 正常；GPU 不可用时不要开始训练，可切回原内核恢复系统。

授予实时调度权限
^^^^^^^^^^^^^^^^

实时内核只有在当前账户能够提升线程优先级、锁定内存时才能发挥作用。将登录账户加入专用用户组：

.. code:: bash

   getent group realtime || sudo groupadd realtime
   sudo usermod -aG realtime "$(id -un)"
   sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

将以下内容写入该文件，然后退出并重新登录：

.. code:: text

   @realtime - rtprio 99
   @realtime - memlock unlimited

在新登录的终端中，``ulimit -r`` 应输出 ``99``，``ulimit -l`` 应输出 ``unlimited``。

不使用实时内核运行
^^^^^^^^^^^^^^^^^^

如果无法使用实时内核，安装 RLinf 时设置 ``FRANKA_REALTIME_CONFIG=ignore``\ （见 `安装`_）。安装脚本会将该值写入 ``franka_control_node.yaml`` 的 ``realtime_config``，``franka_control`` 每次启动时都会读取这个文件，libfranka 由此可以在标准内核上运行。若要恢复强制实时，用 ``FRANKA_REALTIME_CONFIG=enforce`` 重新运行安装脚本即可，无需重新编译。

.. warning::

   在标准内核上，较重的训练负载可能导致 libfranka 错过控制周期，机器人会因 ``communication_constraints_violation`` reflex 而停止。仍然推荐使用实时内核。开始训练前，应在操作员监控下，以预期的训练负载检查控制是否稳定。

安装
----

驱动和内核就绪后，在 GPU 主机上完成一次安装，即可同时获得 ROS 控制栈和训练依赖。克隆 RLinf，后续命令均在仓库根目录执行：

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

安装 Franka 环境
~~~~~~~~~~~~~~~~

使用前面选定的 libfranka 版本运行安装脚本。只有在不使用实时内核时，才额外设置 ``FRANKA_REALTIME_CONFIG=ignore``：

.. code:: bash

   LIBFRANKA_VERSION=0.15.0 bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

安装脚本依次完成：

1. 创建 ``.venv`` 虚拟环境，安装 RLinf、Franka 依赖和具身训练依赖，其中包括与驱动匹配的 CUDA 版 PyTorch。
2. 通过 APT 安装系统软件包和 ROS Noetic。这一步要求 Ubuntu 20.04 和 sudo 权限。
3. 在 catkin 工作区 ``.venv/franka_catkin_ws`` 中编译 libfranka、RLinf 维护的 ``franka_ros`` 分支和 ``serl_franka_controllers``，并设置其中的 ``realtime_config``。
4. 将 ``source /opt/ros/noetic/setup.bash`` 和工作区的 ``devel/setup.bash`` 追加到 ``.venv/bin/activate``，激活环境时会一并加载 ROS。

安装脚本读取以下环境变量：

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - 变量
     - 默认值
     - 作用
   * - ``LIBFRANKA_VERSION``
     - ``0.15.0``
     - 要编译的 libfranka 版本，必须与机器人固件兼容。
   * - ``FRANKA_ROS_VERSION``
     - ``0.10.0``
     - 要编译的 ``franka_ros`` 分支。
   * - ``FRANKA_REALTIME_CONFIG``
     - ``enforce``
     - 设为 ``ignore`` 时，libfranka 可在非 PREEMPT_RT 内核上运行。
   * - ``SKIP_ROS``
     - ``0``
     - 设为 ``1`` 时跳过 ROS Noetic 安装和 catkin 编译。

使用 ``--venv <name>`` 可安装到其他目录；中国大陆用户可添加 ``--use-mirror`` 加快下载。

.. warning::

   设置 ``SKIP_ROS=1`` 后，ROS Noetic、libfranka、``franka_ros`` 和 ``serl_franka_controllers`` 需要自行安装。每次执行 ``ray start`` 之前，都要在该终端中 source ``/opt/ros/noetic/setup.bash`` 和 catkin 工作区的 ``devel/setup.bash``，并确保 libfranka 位于 ``LD_LIBRARY_PATH`` 中，因为 Ray worker 会继承启动 Ray 的终端环境。手动安装请参考 `ROS Noetic <https://wiki.ros.org/noetic/Installation/Ubuntu>`_、`libfranka <https://frankarobotics.github.io/docs/libfranka/docs/installation.html>`_ 和 `serl_franka_controllers <https://github.com/rail-berkeley/serl_franka_controllers>`_ 的安装说明。

当前用户必须具有相机和 SpaceMouse USB 设备的读取权限；SpaceMouse 的 udev 规则参见 `SpaceMouse 安装说明 <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_。

检查环境
~~~~~~~~

在已激活的环境中，确认 PyTorch 能使用 GPU，且 ROS 能找到控制器：

.. code:: bash

   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   rospack find serl_franka_controllers

第一条命令必须输出 ``True``；如果输出 ``False``，请检查 ``nvidia-smi``，修复驱动后重新运行安装脚本。第二条命令会输出 ``.venv/franka_catkin_ws`` 中控制器软件包的路径。

使用 Docker 镜像
~~~~~~~~~~~~~~~~~~~~

除本地安装外，也可以直接运行 ``rlinf/rlinf:agentic-rlinf0.4-franka`` 镜像。该镜像基于 CUDA 12.8、Ubuntu 20.04 和 ROS Noetic 构建，环境中的 PyTorch 为 CUDA 版本，因此在单机方式的 GPU 主机上，一个容器即可同时运行 actor、rollout 和机器人控制。宿主机仍需按 `在 Ubuntu 20.04 上安装 NVIDIA 驱动`_ 安装 570 及以上版本的驱动，并安装 `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_。该镜像也可以用作多节点方式中的控制计算机。镜像包含以下环境，通过 ``source switch_env <name>`` 切换：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 环境
     - 内容
   * - ``franka-0.10.0``、``franka-0.13.3``、``franka-0.14.1``、``franka-0.15.0``、``franka-0.18.0``、``franka-0.19.0``
     - 使用对应 libfranka 版本的 ROS 后端，默认激活 ``franka-0.15.0``。
   * - ``franky``
     - 可选的 Franky 后端，内置 libfranka 0.19.0，见 `Franky 后端（可选）`_。
   * - ``franka-dexhand``
     - ROS 后端及灵巧手依赖。

启动容器时授予访问机械臂、相机和 SpaceMouse 的权限，然后选择与固件匹配的环境：

.. code:: bash

   docker run -it --name rlinf-franka \
     --gpus all --network host --privileged \
     --ulimit rtprio=99 --ulimit memlock=-1 \
     -v "$PWD:/workspace/RLinf" -w /workspace/RLinf \
     rlinf/rlinf:agentic-rlinf0.4-franka bash
   source switch_env franka-0.15.0

镜像中的 ROS 环境保持 ``realtime_config: enforce``，因此宿主机需要使用实时内核。如需另开终端，执行 ``docker exec -it rlinf-franka bash``，然后再次选择相同环境。

下载模型
--------

将预训练 ResNet encoder 下载到仓库内：

.. code:: bash

   hf download RLinf/RLinf-ResNet10-pretrained \
     --local-dir ./models/RLinf-ResNet10-pretrained

后面的训练命令会将该目录同时传给 actor 和 rollout。

运行
----

在机器人主机上分三步运行：配置相机并测量目标位姿，收集演示，然后训练。每个终端都需要保持环境激活。

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

配置中没有设置 ``backend``，机械臂使用默认的 ``franka_ros`` 后端。将训练配置中的 ``cluster.num_nodes`` 也设为 1；采集配置已经使用一个节点。训练配置中的 ``4090`` 节点组和组件放置保持不变：这个组名表示 GPU 节点 0，不要求 GPU 型号为 4090。本示例使用一个相机，自动命名为 ``wrist_1``。

通过机器人的引导模式将插销放到成功插入时的目标位姿，然后解锁机械臂并在 Franka Desk 中启用 FCI。使用控制器检查工具读取位姿，该工具会为机械臂启动 ROS 控制器：

.. code:: bash

   export FRANKA_ROBOT_IP=192.168.1.10  # 替换为机器人的地址。
   python -m toolkits.realworld_check.test_franka_controller

在提示符后输入 ``getpos_euler``，再输入 ``q`` 释放机器人。输出顺序为 ``[x, y, z, roll, pitch, yaw]``，单位为米和弧度。工具还支持 ``getpos``、``getjoint``、``getstate``、``gethand``、``clear``、``home``、``open`` 和 ``close``。将测得的六个数值以逗号分隔，保存在当前终端中：

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

成功轨迹保存在 ``logs/franka-demo/demos`` 下。RLPD 使用这个 replay buffer 目录，而不是 ``collected_data`` 下的可选 episode 导出文件。等待采集程序退出并释放机械臂后，再启动训练。再次采集时请更换日志目录，避免混合不同批次的演示。使用 GELLO 代替 SpaceMouse 采集时，参见 :doc:`franka_gello`。

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

通过键盘标注奖励（可选）
~~~~~~~~~~~~~~~~~~~~~~~~

插孔任务根据目标位姿自动计算奖励。对于没有自动成功信号的任务，操作员可以用实体键盘标注奖励。在训练配置中启用键盘 wrapper：

.. code:: yaml

   env:
     train:
       keyboard_reward_wrapper: single_stage  # 或 multi_stage

``single_stage`` 模式下，``a``、``b``、``c`` 分别输出失败、中性和成功奖励。``multi_stage`` 模式下，``a``、``b``、``c`` 用于切换奖励阶段，``q`` 输出负奖励。

监听器直接读取 Linux 输入设备，因此机器人主机需要在启动 Ray 之前知道设备路径。先找到键盘对应的 event 设备：

.. code:: bash

   ls -l /dev/input/by-id/*-event-kbd

例如 ``usb-Logitech_USB_Keyboard-event-kbd -> ../event20`` 表示设备为 ``/dev/input/event20``。授予该设备的访问权限，并在执行 ``ray start`` 的终端中导出路径：

.. code:: bash

   sudo chmod 666 /dev/input/event20
   export RLINF_KEYBOARD_DEVICE=/dev/input/event20

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

如需将机器人控制与训练负载隔离，或 GPU 服务器无法使用 Ubuntu 20.04，可以使用独立的控制计算机。机械臂、相机和 SpaceMouse 接到控制计算机上，由它在没有 GPU 的情况下运行 ROS 和机器人控制；GPU 服务器运行 actor 和 rollout，不需要 ROS。

准备两台计算机
~~~~~~~~~~~~~~

控制计算机按前文的机器人主机准备，但不需要 NVIDIA 驱动：检查固件，配置实时内核或采用不使用实时内核的方式。随后可以按 `使用 Docker 镜像`_ 启动容器（没有 GPU 的计算机去掉 ``--gpus all``），也可以执行 ``bash requirements/install.sh embodied --env franka`` 进行本地安装。没有 NVIDIA 驱动时，本地安装会自动选择仅 CPU 的 PyTorch。

在 GPU 服务器上安装好 NVIDIA 驱动后，克隆相同版本的 RLinf，安装不含 ROS 的同一环境：

.. code:: bash

   SKIP_ROS=1 bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

两台计算机必须使用相同的 RLinf 代码版本、Python 版本和 Ray 版本。加入多节点集群前，先按前文采集步骤在控制计算机上收集演示，此时使用节点编号 0。然后将完整的 ``logs/franka-demo/demos`` 目录复制到 GPU 服务器。

配置并启动集群
~~~~~~~~~~~~~~

在 GPU 服务器上编辑 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``：将 ``cluster.num_nodes`` 改为 2，将 Franka 组的 ``node_ranks`` 和硬件配置中的 ``node_rank`` 都改为 1。保留机械臂 IP 和相机序列号。Actor 和 rollout 仍放在节点 0 的 GPU 0 上。

.. warning::

   Ray 会记录执行 ``ray start`` 的终端中的 Python 解释器和环境变量，该节点上的所有 worker 都会继承它们。每台计算机都要先导出 ``RLINF_NODE_RANK`` 并激活环境，再执行 ``ray start``；启动时缺失的节点编号或 ROS 环境，只能通过重启 Ray 修正。可以参考 ``ray_utils/realworld/setup_before_ray.sh`` 模板编写启动前的环境设置。

选择两台计算机在互通网络上的 IP，不要使用机械臂的 IP。如果计算机有多个网卡，还需通过 ``RLINF_COMM_NET_DEVICES`` 指定承载该 IP 的网卡。在 GPU 服务器已激活环境的终端中执行：

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=0
   export HEAD_IP=192.168.10.10  # 替换为 GPU 服务器的地址。
   ray start --head --port=6379 --node-ip-address="$HEAD_IP"

在控制计算机已激活环境的终端中执行：

.. code:: bash

   ray stop
   export RLINF_NODE_RANK=1
   export HEAD_IP=192.168.10.10        # 与 GPU 服务器地址一致。
   export CONTROLLER_IP=192.168.10.11 # 替换为控制计算机的地址。
   ray start --address="$HEAD_IP:6379" --node-ip-address="$CONTROLLER_IP"

在 GPU 服务器上运行 ``ray status``，确认两个节点均已启动。然后仅在 GPU 服务器上执行训练命令，使用本机模型与演示数据路径以及测得的目标位姿。训练时，控制计算机不需要模型权重或演示文件。结束后在两个节点分别停止 Ray。多台机器人的配置参见 :doc:`/rst_source/guides/realworld_robot` 和 :doc:`/rst_source/guides/hetero`。

Franky 后端（可选）
-------------------

`Franky <https://github.com/TimSchneider42/franky>`_ 通过 libfranka 的 Python 绑定控制机械臂，不依赖 ROS。如果机器人主机无法使用 Ubuntu 20.04 和 ROS Noetic，或者不想编译 catkin 工作区，可以考虑使用它。选定该后端后，本页其余步骤保持不变。

Franky 环境安装预编译的 ``franky-control`` wheel，其中已包含 libfranka。目前只提供 x86-64 平台上 libfranka 0.15.0 和 0.19.0（默认）的 wheel，因此固件必须与其中之一兼容。请安装到单独的环境中，避免覆盖 ROS 环境：

.. code:: bash

   LIBFRANKA_VERSION=0.19.0 bash requirements/install.sh embodied --env franka-franky --venv franky
   source franky/bin/activate

如果主机无法从 GitHub 下载，可将 ``FRANKY_WHEEL`` 设为 wheel 的 URL 或本地路径。使用 Docker 镜像时，直接执行 ``source switch_env franky``；该环境内置 libfranka 0.19.0，固件需要 0.15.0 时请在主机上直接安装。

在每个 Franka 硬件配置中选择该后端。Franky 同样从这份配置读取 libfranka 的实时模式：默认的 ``enforce`` 会拒绝非 PREEMPT_RT 内核；``ignore`` 可在标准内核上运行，但存在 `不使用实时内核运行`_ 中所述的控制风险：

.. code:: yaml

   configs:
     - robot_ip: ROBOT_IP
       node_rank: 0
       camera_serials: ["CAMERA_SERIAL"]
       backend: franky
       realtime_config: ignore  # 使用实时内核时删除此行。

``realtime_config`` 只能与 ``backend: franky`` 一起使用。默认的 ``franka_ros`` 后端会拒绝该字段，其实时模式在安装时由 ``FRANKA_REALTIME_CONFIG`` 决定。控制器检查工具通过参数接受相同的选项：

.. code:: bash

   python -m toolkits.realworld_check.test_franka_controller \
     --backend franky --realtime-config ignore

Franky 也会尝试锁定内存并提升线程优先级，因此 `授予实时调度权限`_ 中的设置同样适用。如果 Franky 阻抗控制器意外停止，RLinf 会报告运动错误，不会静默重启控制。解决原因后再重新启动训练。直接使用 Python API 时，应先调用 ``disconnect()``，再调用 ``connect()``，然后恢复发送指令；``clear_errors()`` 不会重启已经失败的跟踪控制。

其他 Franka 工作流
------------------

完成基础示例后，可按需要参考以下指南：

- :doc:`franka_gello` 与 :doc:`franka_vr`：GELLO 或 PICO 遥操作。
- :doc:`franka_pi0_sft_deploy` 与 :doc:`hg-dagger`：OpenPI policy。
- :doc:`franka_reward_model`：学习奖励模型。
- :doc:`franka_zed_robotiq` 与 :doc:`franka_dexhand`：其他相机与末端执行器。
- :doc:`dual_franka`：双臂控制；:doc:`/rst_source/guides/rtc`：重叠执行动作与推理。
