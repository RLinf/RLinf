Franka 真机强化学习
====================

本页介绍如何使用 RLinf 在 Franka 机械臂上训练 CNN policy，从收集演示到 RLPD 在线训练。默认配置只用一台装有 NVIDIA GPU、运行 Ubuntu 20.04 或 22.04 的 x86-64 计算机。Franky 通过 libfranka 的 Python 绑定控制机械臂，同一台计算机还负责 rollout 和训练。你将先准备这台主机（检查固件、安装实时内核并为其配置 GPU 驱动），再通过本地安装或 Docker 部署 RLinf，然后运行插孔示例。后面几节分别介绍独立控制节点、面向已有部署的旧版 ROS 后端，以及其他 Franka 工作流。

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
     - 一台 Ubuntu 20.04 或 22.04 GPU 主机运行 Franky 机器人控制、rollout 和训练。
     - 大多数场景，只需安装和维护一台计算机。
   * - 多节点（可选）
     - 一台控制计算机运行 Franky 机器人控制，不需要 GPU；GPU 服务器运行 actor 和 rollout。
     - 希望将机器人控制与训练负载隔离，或 GPU 服务器不在机器人旁边。

.. warning::

   每次运行真机时都应由操作员全程监控，并确保急停装置触手可及。固定插销与夹具，清空工作区域，检查复位路径是否安全。此任务会复位到目标上方约 10 cm 的位置，并随机改变水平位置和偏航角。不要直接使用其他机器人的目标位姿。

准备机器人主机
--------------

安装 RLinf 之前，先检查固件并选择兼容的 libfranka 版本，再为机械臂的 1 kHz 控制循环准备内核。如果主机配有 GPU，请在运行 ``requirements/install.sh`` 前确认 ``nvidia-smi`` 在该内核下正常工作，否则安装脚本会选择仅 CPU 的 PyTorch。

检查固件并选择 libfranka 版本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

打开 ``http://<robot_ip>/desk`` 进入 Franka Desk，点击 ``SETTINGS``，记录仪表盘中 ``Control`` 后面的版本号：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_firmware.png
   :align: center
   :width: 60%
   :alt: Franka Desk 仪表盘中的 Control 固件版本

   Franka Desk 中的 Control 固件版本。

在 `Franka 兼容性表 <https://frankarobotics.github.io/docs/compatibility.html>`_ 中查找该版本，确定 libfranka 版本。Franky 以预编译 wheel 的形式发布，wheel 中已包含 libfranka，目前提供 libfranka 0.19.0（安装脚本默认值）和 0.15.0 两个版本。如果固件需要其他 libfranka 版本，请使用 `旧版 ROS 后端（可选）`_，它会从源码编译 libfranka。

RLinf 已测试过两种组合：固件 5.9.2 搭配 libfranka 0.19.0，关闭实时检查后运行在标准内核上；固件 5.7.2 至 5.9.0 搭配 libfranka 0.15.0，运行在实时内核上。不要仅为匹配示例而修改机器人固件。

安装实时内核（推荐）
~~~~~~~~~~~~~~~~~~~~

libfranka 每毫秒向机械臂发送一次指令。PREEMPT_RT 内核能在 rollout 和训练占用 CPU、GPU 时保证这个循环按时执行。Franky 在标准内核上也能启动（见 `不使用实时内核运行`_），但仍然推荐使用实时内核。

内核需要安装在宿主机上，不能在 Docker 容器内安装，因为容器与宿主机共享内核。请按照 Franka 的 `实时内核指南 <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_ 中与你的 Ubuntu 版本对应的步骤，编译打过补丁的内核，并安装生成的 ``linux-image`` 和 ``linux-headers`` 软件包。保留原内核作为 GRUB 回退选项。重启进入新内核后检查：

.. code:: bash

   uname -r
   cat /sys/kernel/realtime

第二条命令必须输出 ``1``，否则请先从 GRUB 的高级选项中选择实时内核。

在实时内核上安装 NVIDIA 驱动
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

进入实时内核后，需要使用上一步安装的对应版本 ``linux-headers`` 编译 NVIDIA 内核模块。安装时传入 ``IGNORE_PREEMPT_RT_PRESENCE=1``，即可跳过驱动对 PREEMPT_RT 的编译检查。使用 APT 安装时，先按 NVIDIA 的 `驱动安装指南 <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_ 配置软件源并选择适合 GPU 和 Ubuntu 版本的软件包，再将下面的 ``DRIVER_PACKAGE`` 替换为实际包名：

.. code:: bash

   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt-get install -y DRIVER_PACKAGE
   sudo reboot

如果已经通过 DKMS 安装过驱动，只需为当前实时内核编译模块：

.. code:: bash

   sudo env IGNORE_PREEMPT_RT_PRESENCE=1 dkms autoinstall -k "$(uname -r)"
   sudo reboot

重启进入实时内核后，运行 ``nvidia-smi``，确认能列出 GPU，再安装 RLinf。以后驱动或内核更新触发模块重新编译时，同样需要传入该环境变量。

.. warning::

   ``IGNORE_PREEMPT_RT_PRESENCE=1`` 仅跳过编译检查；NVIDIA 未正式支持 PREEMPT_RT 内核。如果 GPU 不可用，可切回原内核恢复系统。

授予实时调度权限
^^^^^^^^^^^^^^^^

RLinf 启动 Franky 时，会尝试锁定控制进程的内存、以 ``SCHED_FIFO`` 优先级 80 运行该进程，并将其绑定到指定 CPU 核。每一步都需要相应权限；权限不足时，RLinf 会记录警告并继续运行。将登录账户加入专用用户组即可授予这些权限：

.. code:: bash

   getent group realtime || sudo groupadd realtime
   sudo usermod -aG realtime "$(id -un)"
   sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

将以下内容写入该文件，然后退出并重新登录：

.. code:: text

   @realtime - rtprio 99
   @realtime - memlock unlimited

在新登录的终端中，``ulimit -r`` 应输出 ``99``，``ulimit -l`` 应输出 ``unlimited``。即使使用标准内核，这些限制也有帮助。

不使用实时内核运行
^^^^^^^^^^^^^^^^^^

无法使用实时内核时，不需要额外操作。Franky 从机器人硬件配置的 ``realtime_config`` 读取 libfranka 的实时模式，默认值 ``ignore`` 允许 libfranka 在非 PREEMPT_RT 内核上运行。当前内核不是实时内核时，RLinf 会记录一条警告。

如果要确保机械臂只在实时内核上运行，在 `运行`_ 一节介绍的硬件配置中设置 ``enforce``。此时 Franky 会拒绝在非 PREEMPT_RT 内核上启动：

.. code:: yaml

   configs:
     - robot_ip: ROBOT_IP
       node_rank: 0
       camera_serials: ["CAMERA_SERIAL"]
       realtime_config: enforce

.. warning::

   在标准内核上，较重的训练负载可能导致 libfranka 错过控制周期，机器人会因 ``communication_constraints_violation`` reflex 而停止。开始训练前，应在操作员监控下，以预期的训练负载检查控制是否稳定。

安装
----

驱动和内核就绪后，在 GPU 主机上完成一次安装，即可同时获得 Franky 机器人控制和训练依赖。可以选择本地安装，也可以使用本节末尾介绍的 Docker 镜像。克隆 RLinf，后续命令均在仓库根目录执行：

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

安装 Franka 环境
~~~~~~~~~~~~~~~~

运行安装脚本。只有固件需要 libfranka 0.15.0 时，才在命令前加上 ``LIBFRANKA_VERSION=0.15.0``：

.. code:: bash

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

安装脚本依次完成：

1. 创建 ``.venv`` 虚拟环境，安装 RLinf 和具身训练依赖，其中包括与驱动匹配的 PyTorch。
2. 通过 APT 安装系统软件包，这一步需要 sudo 权限。只有这些软件包已经装好时，才传入 ``--no-root``。
3. 安装 Franka 依赖、LeRobot，以及包含 libfranka 的预编译 ``franky-control`` wheel。

该 wheel 面向 manylinux 2.28 构建，因此同一个环境可在 Ubuntu 20.04 和 22.04 上使用。安装脚本读取以下环境变量：

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 变量
     - 默认值
     - 作用
   * - ``LIBFRANKA_VERSION``
     - ``0.19.0``
     - Franky wheel 对应的 libfranka 版本。预编译 wheel 只提供 x86-64 平台上的 ``0.15.0`` 和 ``0.19.0``；其他版本或架构会让安装脚本提前报错退出，除非设置了 ``FRANKY_WHEEL``。
   * - ``FRANKY_WHEEL``
     - 未设置
     - 用于替代预编译 wheel 的 ``franky-control`` wheel URL 或本地路径，例如主机无法访问 GitHub，或你为其他 libfranka 版本自行构建了 wheel。

使用 ``--venv <name>`` 可安装到其他目录；中国大陆用户可添加 ``--use-mirror`` 加快下载。

当前用户必须具有相机和 SpaceMouse USB 设备的读取权限；SpaceMouse 的 udev 规则参见 `SpaceMouse 安装说明 <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_。

检查环境
~~~~~~~~

在已激活的环境中，确认 PyTorch 能使用 GPU，且 Franky 能正常加载：

.. code:: bash

   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   python -c "import franky"

第一条命令必须输出 ``True``；如果输出 ``False``，请检查 ``nvidia-smi``，修复驱动后重新运行安装脚本。Franky 及其内置的 libfranka 加载成功时，第二条命令没有任何输出。

使用 Docker 镜像
~~~~~~~~~~~~~~~~~~~~

除本地安装外，也可以直接运行 ``rlinf/rlinf:agentic-rlinf0.4-franka`` 镜像。该镜像基于 CUDA 12.8 和 Ubuntu 22.04 构建，环境中的 PyTorch 为 CUDA 版本，因此在单机方式的 GPU 主机上，一个容器即可同时运行 actor、rollout 和机器人控制。宿主机仍需安装 570 及以上版本的 NVIDIA 驱动，以及 `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_。镜像包含以下环境，通过 ``source switch_env <name>`` 切换：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 环境
     - 内容
   * - ``franky``
     - Franky 后端，内置 libfranka 0.19.0 和灵巧手依赖，默认激活。
   * - ``openvla``、``openvla-oft``、``openpi``、``gr00t``
     - ``franky`` 的全部内容及对应的 VLA policy。

启动容器时授予访问 GPU、机械臂、相机和 SpaceMouse 的权限：

.. code:: bash

   docker run -it --name rlinf-franka \
     --gpus all --network host --privileged \
     --ulimit rtprio=99 --ulimit memlock=-1 \
     -v "$PWD:/workspace/RLinf" -w /workspace/RLinf \
     rlinf/rlinf:agentic-rlinf0.4-franka bash

国内下载时，可以将镜像名称中的 ``rlinf/rlinf`` 替换为 ``infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf``，保留原有 tag。

容器启动后已处于 ``franky`` 环境。``--ulimit`` 参数在容器内授予 `授予实时调度权限`_ 中的调度权限，但内核仍是宿主机的内核。镜像中的 Franky 环境内置 libfranka 0.19.0，固件需要 0.15.0 时，请在主机上用 ``LIBFRANKA_VERSION=0.15.0`` 本地安装。如需另开终端，执行 ``docker exec -it rlinf-franka bash``；如果切换过环境，需要再次选择相同环境。

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

配置中不需要 ``backend`` 字段，机械臂默认使用 Franky 后端；除非按 `不使用实时内核运行`_ 添加 ``enforce``，否则 ``realtime_config`` 为 ``ignore``。将训练配置中的 ``cluster.num_nodes`` 也设为 1；采集配置已经使用一个节点。训练配置中的 ``4090`` 节点组和组件放置保持不变：这个组名表示 GPU 节点 0，不要求 GPU 型号为 4090。本示例使用一个相机，自动命名为 ``wrist_1``。

通过机器人的引导模式将插销放到成功插入时的目标位姿，然后解锁机械臂并在 Franka Desk 中启用 FCI。使用控制器检查工具读取位姿，该工具通过 Franky 连接机械臂：

.. code:: bash

   export FRANKA_ROBOT_IP=192.168.1.10  # 替换为机器人的地址。
   python -m toolkits.realworld_check.test_franka_controller

在提示符后输入 ``getpos_euler``，再输入 ``q`` 释放机器人。输出顺序为 ``[x, y, z, roll, pitch, yaw]``，单位为米和弧度。工具还支持 ``getpos``、``getjoint``、``getstate``、``gethand``、``clear``、``home``、``open`` 和 ``close`` 命令；也可以用 ``--robot-ip`` 代替环境变量，用 ``--realtime-config enforce`` 要求实时内核。将测得的六个数值以逗号分隔，保存在当前终端中：

.. code:: bash

   export FRANKA_TARGET_POSE='[x, y, z, roll, pitch, yaw]'  # 替换全部六个数值。

继续之前，确认目标位姿及上文所述的复位区域均处于安全工作范围内。关闭其他控制机械臂的程序；同一时刻只能有一个进程持有控制连接。

.. _franka-motion-settings:

配置机械臂运动
~~~~~~~~~~~~~~

数采、训练和评估使用同一个 Franky Cartesian 控制器。硬件条目未填写 ``compliance`` 时，控制器采用下表中的默认值，因此上述配置无需额外补充控制参数。PICO、SpaceMouse 遥操作，以及下发 Cartesian 位姿的 GELLO 遥操作，也使用这些默认值。

.. list-table:: Franky 控制器默认值
   :header-rows: 1
   :widths: 30 15 55

   * - 参数
     - 默认值
     - 含义
   * - ``translational_stiffness``
     - 1000 N/m
     - 位置误差对应的初始刚度。
   * - ``rotational_stiffness``
     - 50 Nm/rad
     - 姿态误差对应的初始刚度。
   * - ``translational_clip``
     - 0.008 m
     - 计算恢复力矩时使用的初始位置误差上限。
   * - ``rotational_clip``
     - 0.04 rad
     - 计算恢复力矩时使用的初始姿态误差上限。
   * - ``max_step``
     - 0.03 m
     - 每次调用时，目标位置相对上一次指令目标的最大变化量。
   * - ``max_step_rad``
     - 0.10 rad
     - 每次调用时，目标姿态相对上一次指令目标的最大变化量。

第一次下发目标时，以实测位姿作为限幅基准。目标变化上限按调用次数生效，不表示运动速度；误差限幅约束控制器用于计算力矩的误差，不表示工作空间边界。环境仍会先应用自身的动作缩放和工作空间边界，再下发目标。

任务可以在 reset 时通过 ``compliance_param`` 请求不同的刚度和误差限幅，Franky 再按硬件配置中的限制处理这些请求：默认刚度上限为 1200 N/m 和 80 Nm/rad，误差限幅下限为 5 mm 和 0.02 rad。插孔任务 reset 后的刚度达到上述上限，位置误差限幅为 x/y 方向 5 mm、z 方向 10 mm，姿态误差限幅为 0.02 rad。``max_step`` 和 ``max_step_rad`` 在 reset 后继续有效。其他任务仍使用各自的 reset 参数请求。

需要调整硬件参数时，在需要使用该参数的各个配置中，将差异项写在与 ``robot_ip`` 同级的 ``compliance`` 下。例如，下方配置将每次位置目标的变化限制为 2 cm，其余参数沿用 Franky 默认值：

.. code:: yaml

   compliance:
     max_step: 0.02

参数名拼写错误会在声明 Franky 机械臂时、建立连接之前报错。Python 中的 ``FrankyArm(..., compliance={"max_step": 0.02})`` 也遵循相同规则；如果传入 ``CartesianCompliance`` 对象，则表示提供一套完整参数，对象中所有字段均按原值使用，包括该对象自身的默认值。

旧版 ``franka_ros`` backend 保留自身的控制器配置，并忽略此硬件 mapping。关节指令使用关节控制，不受这些 Cartesian 参数影响，双臂 GELLO 关节遥操作也属于这种情况。

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

如果 Franky 阻抗控制器意外停止，RLinf 会报告运动错误，不会静默重启控制。先解决问题，再重新启动训练。直接使用 Python API 时，应先调用 ``disconnect()``，再调用 ``connect()``，然后恢复发送指令；``clear_errors()`` 不会重启已经失败的跟踪控制。

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

如需将机器人控制与训练负载隔离，或 GPU 服务器不在机器人旁边，可以使用独立的控制计算机。机械臂、相机和 SpaceMouse 接到控制计算机上，由它运行 Franky 机器人控制，不需要 GPU；GPU 服务器运行 actor 和 rollout。

准备两台计算机
~~~~~~~~~~~~~~

控制计算机按前文的机器人主机准备，但不需要 NVIDIA 驱动：检查固件，配置实时内核或采用不使用实时内核的方式。随后可以用 `安装`_ 中的命令本地安装，没有 NVIDIA 驱动时安装脚本会自动选择仅 CPU 的 PyTorch；也可以按 `使用 Docker 镜像`_ 启动容器，并去掉 ``--gpus all``。

在 GPU 服务器上安装好 NVIDIA 驱动后，克隆相同版本的 RLinf，并执行相同的安装命令：

.. code:: bash

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

两台计算机必须使用相同的 RLinf 代码版本、Python 版本和 Ray 版本。加入多节点集群前，先按前文采集步骤在控制计算机上收集演示，此时使用节点编号 0。然后将完整的 ``logs/franka-demo/demos`` 目录复制到 GPU 服务器。

配置并启动集群
~~~~~~~~~~~~~~

在 GPU 服务器上编辑 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``：将 ``cluster.num_nodes`` 改为 2，将 Franka 组的 ``node_ranks`` 和硬件配置中的 ``node_rank`` 都改为 1。保留机械臂 IP 和相机序列号。Actor 和 rollout 仍放在节点 0 的 GPU 0 上。

.. warning::

   Ray 会记录执行 ``ray start`` 的终端中的 Python 解释器和环境变量，该节点上的所有 worker 都会继承它们。每台计算机都要先导出 ``RLINF_NODE_RANK`` 并激活环境，再执行 ``ray start``；启动时缺失的节点编号或环境，只能通过重启 Ray 修正。可以参考 ``ray_utils/realworld/setup_before_ray.sh`` 模板编写启动前的环境设置。

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

旧版 ROS 后端（可选）
---------------------

除 Franky 外，RLinf 也可以通过 ROS Noetic、``franka_ros`` 和 ``serl_franka_controllers`` 控制机械臂。已有 ROS 部署，或者固件需要 0.15.0、0.19.0 以外的 libfranka 版本时，可以使用这一后端。该后端运行在 ROS Noetic 支持的 Ubuntu 20.04 上，需要在主机上本地安装。安装并选定后端后，本页其余步骤保持不变。

安装 ROS 环境
~~~~~~~~~~~~~

请安装到单独的环境中，避免覆盖 Franky 环境，并传入前面选定的 libfranka 版本：

.. code:: bash

   LIBFRANKA_VERSION=0.15.0 bash requirements/install.sh embodied --env franka-ros --venv franka-ros
   source franka-ros/bin/activate

安装脚本依次完成：

1. 安装与 Franky 环境相同的 RLinf 和训练依赖，并通过 APT 安装 ROS Noetic。
2. 在 catkin 工作区 ``franka-ros/franka_catkin_ws`` 中编译 libfranka、RLinf 维护的 ``franka_ros`` 分支和 ``serl_franka_controllers``，并将 ``realtime_config`` 写入其中的 ``franka_control_node.yaml``。
3. 将 ROS 和 catkin 的 ``setup.bash`` 追加到环境的 activate 脚本，激活环境时会一并加载 ROS。

ROS 环境读取以下环境变量：

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

用 ``rospack find serl_franka_controllers`` 检查编译结果，它会输出 catkin 工作区中控制器软件包的路径。

.. warning::

   设置 ``SKIP_ROS=1`` 后，ROS Noetic、libfranka、``franka_ros`` 和 ``serl_franka_controllers`` 需要自行安装。每次执行 ``ray start`` 之前，都要在该终端中 source ``/opt/ros/noetic/setup.bash`` 和 catkin 工作区的 ``devel/setup.bash``，并确保 libfranka 位于 ``LD_LIBRARY_PATH`` 中，因为 Ray worker 会继承启动 Ray 的终端环境。手动安装请参考 `ROS Noetic <https://wiki.ros.org/noetic/Installation/Ubuntu>`_、`libfranka <https://frankarobotics.github.io/docs/libfranka/docs/installation.html>`_ 和 `serl_franka_controllers <https://github.com/rail-berkeley/serl_franka_controllers>`_ 的安装说明。

选择后端与实时模式
~~~~~~~~~~~~~~~~~~

在每个 Franka 硬件配置中设置 ``backend: franka_ros``：

.. code:: yaml

   configs:
     - robot_ip: ROBOT_IP
       node_rank: 0
       camera_serials: ["CAMERA_SERIAL"]
       backend: franka_ros

ROS 后端的实时模式在安装时由 ``FRANKA_REALTIME_CONFIG`` 决定，不从硬件配置读取。``franka_control`` 每次启动时读取安装写入的值；默认的 ``enforce`` 会拒绝非 PREEMPT_RT 内核。若要修改，用 ``FRANKA_REALTIME_CONFIG=ignore`` 或 ``enforce`` 重新运行安装脚本即可，无需重新编译。在 ``franka_ros`` 硬件配置中写入 ``realtime_config`` 会直接报错，错误信息会提示改用这个变量。

控制器检查工具通过参数选择后端：

.. code:: bash

   python -m toolkits.realworld_check.test_franka_controller --backend franka_ros

使用 ROS 后端运行
~~~~~~~~~~~~~~~~~~~~

使用 ROS 后端时，运行步骤与 `运行`_ 相同，只有环境和硬件配置不同。每个运行 RLinf 的终端都要先激活 ROS 环境，尤其是在执行 ``ray start`` 之前：激活时会加载 ROS 和 catkin 工作区，而 Ray worker 会继承该环境。

.. code:: bash

   source franka-ros/bin/activate
   ray stop
   export RLINF_NODE_RANK=0
   ray start --head

在两个配置文件中设置 ``backend: franka_ros`` 后，`收集演示`_ 和 `训练 Policy`_ 中的命令无需修改即可运行。

机械臂连接时，RLinf 会自行启动 ROS，依次：

1. 复用已在运行的 ``roscore``，没有则启动一个，并为当前进程创建一个 ROS 节点。
2. 针对机械臂的 ``robot_ip`` 执行 ``roslaunch serl_franka_controllers impedance.launch``，启动 ``franka_control``、Franka Hand 驱动和笛卡尔阻抗控制器。
3. 进行关节复位时切换到 ``joint.launch``，机械臂到达复位位姿后再重新启动阻抗控制器。

机械臂的 FCI 连接同一时间只能由一个程序占用，请先关闭其他控制程序。首次运行前，用控制器检查工具确认启动流程正常：

.. code:: bash

   python -m toolkits.realworld_check.test_franka_controller --backend franka_ros

与 Franky 的差异
^^^^^^^^^^^^^^^^^^

- 机械臂只接受笛卡尔空间的 ``tcp_pose`` 目标。单臂 Franka 任务本身就发送这类目标；关节位置控制（例如双臂关节任务）需要使用 Franky。
- ``gripper_type: franka`` 通过机械臂启动文件拉起的 ``/franka_gripper`` 话题控制 Franka Hand，因此只有机械臂保持连接时手爪才会响应。Robotiq 等其他末端执行器与使用 Franky 时一样独立连接。
- 阻抗控制器自行管理增益。任务的 ``compliance_param`` 通过 ``dynamic_reconfigure`` 生效，硬件配置中的 ``compliance`` 设置不起作用。

在多节点方式中，只有控制计算机需要 ROS 环境。GPU 服务器运行 actor 和 rollout，按 `准备两台计算机`_ 安装 Franky 环境即可。

常见问题
^^^^^^^^^^

- 出现 ``Running kernel does not have realtime capabilities``：启动实时内核，或用 ``FRANKA_REALTIME_CONFIG=ignore`` 重新安装。
- 出现 libfranka 版本不兼容的错误：按兼容性表中与固件对应的 ``LIBFRANKA_VERSION`` 重新安装。
- 机械臂一直没有就绪：确认 Franka Desk 中已激活 FCI，然后在控制器运行期间，于已激活的环境中执行 ``rostopic echo -n 1 /franka_state_controller/franka_states``，正常时会输出一条状态消息。
- 运行崩溃后可能残留仍占用机械臂的 ``roslaunch`` 进程。重新启动前先停止 Ray，并执行 ``pkill -f roslaunch``。

其他 Franka 工作流
------------------

安装脚本可以将 Franka 依赖与其他模型组合。所有 Franky 环境都包含灵巧手依赖。使用 VLA policy 时，将模型和 Franka 一起安装：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env franka --venv openpi
   source openpi/bin/activate

按需将 ``openpi`` 替换为 ``openvla``、``openvla-oft`` 或 ``gr00t``。这些命令只安装依赖，模型权重、任务配置和训练步骤见以下指南：

- :doc:`franka_gello` 与 :doc:`franka_vr`：GELLO 或 PICO 遥操作。
- :doc:`franka_pi0_sft_deploy` 与 :doc:`hg-dagger`：OpenPI policy。
- :doc:`franka_reward_model`：学习奖励模型。
- :doc:`franka_zed_robotiq` 与 :doc:`franka_dexhand`：其他相机与末端执行器。
- :doc:`dual_franka`：双臂控制；:doc:`/rst_source/guides/rtc`：重叠执行动作与推理。
