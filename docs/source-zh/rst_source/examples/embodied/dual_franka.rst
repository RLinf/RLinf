双 Franka 真机：GELLO 数据采集、π₀.₅ SFT 与部署
====================================================

本指南是 RLinf 中 **双臂 Franka** 真机的端到端流程：双节点环境搭建、
1 kHz GELLO 关节空间双臂数据采集、π₀.₅ 在 20 维 rot6d 动作空间上的
SFT 微调，以及通过脚踏开关将训练好的策略部署回真机。

阅读本页前请先阅读：

* :doc:`franka` — 单臂 Franka 基础、Ray cluster 搭建、RealSense +
  SpaceMouse 数据采集路径。如果尚不熟悉 ``FrankaController`` /
  ``FCI`` / ``RLINF_NODE_RANK``，请先完整阅读该页。
* :doc:`franka_gello` — GELLO 硬件安装、Dynamixel SDK、
  ``gello-teleop`` 包、USB-FTDI 权限。

本页只覆盖双臂 rig 的差异点：

* **franky** 底层后端（``franky-control`` 包封装的 libfranka），
  替代 :doc:`franka` 使用的 ROS / serl 路径，左右两台机械臂共用；
* 三个新双臂环境 —— ``DualFrankaEnv``\ （旧版 14 维 Cartesian
  delta）、``DualFrankaJointEnv``\ （16 维 joint，采集用）、
  ``DualFrankaTcpEnv``\ （20 维 TCP-rot6d，SFT 与部署用）；
* 用 **rot6d / SE(3) body-frame delta** 替换 openpi 自带的
  component-wise ``DeltaActions``；
* 由 3 键脚踏驱动、可断点续采的数据采集流；
* 双物理节点 Ray cluster：每个节点拥有一条 Franka 的 controller，
  env worker 与 GPU 均运行在 node 0。


该 rig 的适用范围与非适用范围
--------------------------------

RLinf 的双 Franka rig 面向 **双臂操作的 SFT** —— 采集高质量的遥操作
数据，并在 π₀ / π₀.₅ 等 VLA 上进行微调。与 :doc:`franka` 的单臂
SAC / PPO 训练相比，该 rig：

* **面向模仿学习，不执行在线 RL。** 采集使用 GELLO 关节空间遥操作；
  部署使用 SFT 模型自主推理 + 脚踏控制 episode 边界。不包含
  reward 标注，也没有 RL 更新。
* **左右臂统一使用 ``franky-control`` 这一 libfranka 后端。**
  所有底层控制均在 ``franky`` 内部的 C++ 1 kHz 循环中运行，Python
  仅更新参考点。该设计规避了"纯 Python 控制循环 + ROS"路径下的
  GIL 抖动问题。
* **方向用 6D 表示，不再用 Euler。** Euler 状态/动作
  会向 π₀.₅ 引入 ±π wrap 不连续点（上一帧 roll = +3.14 rad，
  下一帧 roll = −3.14 rad ⇒ 一个 "−2π" 的伪 delta，被策略当作
  规律学习）。改用 rot6d + SE(3) body-frame delta 后可消除此类
  问题。
* **左右臂分到两台机器上。** 每个节点用一根专线直连一根 Franka
  的 FCI 端口（一根线、一张网卡、一台机器对一台机械臂）。两个节点
  之间走另一张共享 LAN，仅用于 Ray 控制流和张量同步。

如需进行单臂 Franka 在线 RL（SAC / PPO），请参考 :doc:`franka`，
而非本页。


硬件拓扑
--------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - 节点
     - 角色
     - 节点上的硬件
   * - **node 0**\ （head）
     - Ray head；env worker；左 ``FrankyController``；
       部署阶段的 actor / rollout；所有相机和 GELLO 采集
     - 1× GPU（如 RTX 4090，仅 SFT 与部署阶段使用）；
       左 Franka FR3 直连一张网卡，对接 FCI 端口；
       左 Robotiq 2F-85（USB-RS485 Modbus）；
       **左右两台 GELLO** Dynamixel 链（USB-FTDI）；
       **三台相机全部在此**\ —— base RealSense D435i（第三人称）+
       左腕 Lumos USB-3 + 右腕 Lumos USB-3；
       PCsensor 3 键脚踏（放在 node 0）
   * - **node 1**\ （worker）
     - Ray worker；只跑右 ``FrankyController``
     - 可选 GPU（推理不需要）；
       右 Franka FR3 直连自己的网卡，对接 FCI 端口；
       右 Robotiq 2F-85

.. note::

   两臂的 FCI IP 与网卡名按你机器实际网络情况填写到下文
   Hardware YAML 中即可。

相机分工：wrapper 栈把 ``main_image_key`` 设为 ``left_wrist_0_rgb``，
所以**左腕相机**充当 π₀.₅ 的主图像（``observation/image`` 槽位），
``base_0_rgb`` 与 ``right_wrist_0_rgb`` 作为辅助视角进入
``observation/extra_view_image-{0,1}``。三台相机的 USB 全部接到
**node 0**\ —— env worker 在 node 0，由它统一打开
``/dev/v4l/by-id/...`` 与 ``rs.pipeline()``。所以右腕 Lumos 的 USB
线虽然要拉回 node 0，机械臂本体仍然挂在 node 1：

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - 相机槽位
     - 后端
     - 用途
   * - ``base_0_rgb``
     - RealSense D435i
     - 第三人称视角，左右臂共用
   * - ``left_wrist_0_rgb``
     - Lumos USB 3（XVisio vSLAM）
     - 左臂腕相机，作为 π₀.₅ 主 ``image``
   * - ``right_wrist_0_rgb``
     - Lumos USB 3（XVisio vSLAM）
     - 右臂腕相机

脚踏：3 键 PCsensor FootSwitch，3 个踏板烧成键码 ``a`` / ``b`` /
``c`` （使用厂家提供的 Windows 工具刷写一次，键码写入固件，重启后保留）。
**脚踏必须接在 node 0**（env worker 默认 placement 钉死在 node 0）。
node 0 在 ``ray start`` **之前** 导出
``RLINF_KEYBOARD_DEVICE=/dev/input/eventXX`` ，让 Ray 把这个变量
打包进 worker 环境。``KeyboardListener`` 直接走 ``evdev``，不依赖
``DISPLAY`` / ``xev``，也不依赖终端焦点。


软件栈
------

**采集** 阶段的数据通路::

  GELLO 机械臂（Dynamixel）                  env worker (node 0)
        │                                          │
        ▼                                          ▼
  GelloJointExpert（1 kHz 读）              DualFrankaJointEnv.step
        │ ±2π 反 wrap                              │ 10 Hz
        ▼                                          │
  DualGelloJointIntervention                       │
   （direct_stream 守护线程，1 kHz）               │ （env.step 只读
        │                                          │  state + grippers，
        └─move_joints─► FrankyController(left)  ◄──┘  不发 motion）
        └─move_joints─► FrankyController(right)
                              │ C++ 1 kHz JointImpedanceTracker
                              ▼
                        Franka FR3

**部署** 阶段的数据通路::

  observation（state[20] + 3 个相机）
        │
        ▼
  DualFrankaRot6dInputs ─► RigidBodyDeltaActions ─► π₀ / π₀.₅
                                                       │
                                                       ▼
                            RigidBodyAbsoluteActions ◄┘  （T_abs = T_state @ T_delta）
                                       │
                                       ▼
                            DualFrankaRot6dOutputs（切回 20 维）
                                       │
                                       ▼
                  DualFrankaTcpEnv.step（每台机械臂 move_tcp_pose）
                                       │ C++ 1 kHz CartesianImpedanceTracker
                                       ▼
                                 Franka FR3

``FrankyController`` 内部的两个 tracker
（``JointImpedanceTracker`` 和 ``CartesianImpedanceTracker``）
是**互斥**的 —— 从采集（关节阻抗）切到部署（笛卡尔阻抗）时会
自动停掉前一个，因此跨场景切换不需要重启 franky 进程。


安装（每个节点都执行）
----------------------

以下步骤需在 ``node 0`` 和 ``node 1`` 上**分别执行一次**。两个节点是
独立 checkout、独立 venv，只共享 LAN 网络。

1. PREEMPT_RT 内核与 rtprio 限额
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

franky 后端假设主机已经在 PREEMPT_RT 内核上运行。请按 Franka 官方文档
`Setting up the real-time kernel
<https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html>`_
编译并启动；本项目验证过的版本为 ``5.15.133-rt69``\ 。验证：

.. code-block:: bash

   uname -a | grep -o PREEMPT_RT   # 必须输出 PREEMPT_RT

直连的千兆网卡指向 Franka 的 FCI 口（通常 ``172.16.0.2``\ ），中间
不要有交换机；同时检查 ``/proc/cmdline`` 没有奇怪的 ``iommu`` /
``apic`` 选项干扰 RT 线程。

放置 ``/etc/security/limits.d/99-realtime.conf``\ ，让 PAM 给当前用户
开放 ``rtprio 99`` 和 ``memlock unlimited``：

.. code-block:: text

   *  -  rtprio    99
   *  -  memlock   unlimited

退出登录再重新登录让 PAM 重新读取限额；然后 ``ulimit -r`` 应当返回
``99`` 或 ``unlimited``\ ，``ulimit -l`` 应当返回 ``unlimited``\ 。
否则 ``FrankyController.__init__`` 会打印 ``SCHED_FIFO denied`` /
``mlockall failed`` 并 fallback 到默认调度——控制器仍能运行，但
RT 抖动会回来。

.. note::

   这些限额由
   ``rlinf/envs/realworld/franka/franky_controller.py`` 中的
   ``_apply_rt_hardening()`` 在启动时检查；如果 ``SCHED_FIFO``
   被拒绝或 ``mlockall`` 失败，控制器会以 best-effort 模式继续
   运行并打 warning，而不会直接退出，warning 文本里附带具体的
   修复指引。

2. 每次开机的 RT 调优
~~~~~~~~~~~~~~~~~~~~~~

下面这些参数每次重启都会被重置。每次启动会话跑一次，或者写到
systemd one-shot / ``rc.local`` 里持久化：

.. code-block:: bash

   # 1. CPU governor → performance（防止 P-state 切换引入 µs 级抖动）
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
       echo performance > "$g"
   done'
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # 期望: performance

   # 2. 放开 SCHED_FIFO 95% throttle（默认 950000/1000000）
   sudo sysctl -w kernel.sched_rt_runtime_us=-1
   cat /proc/sys/kernel/sched_rt_runtime_us                    # 期望: -1

   # 3. 关掉 Franka 链路的网卡 interrupt coalescing
   sudo ethtool -C eno1 rx-usecs 0 tx-usecs 0                  # 把 eno1 换成你的网卡

用 ``ip -br a`` 确认实际网卡名。如果想让 ``rt_runtime`` 持久化：

.. code-block:: bash

   echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

.. note::

   ``requirements/install.sh embodied --env franka-franky`` 在安装结束时
   （内部调用 ``requirements/embodied/franky_install.sh``）会把以上三条
   命令打印出来。本节是这几条命令的权威版本。

3. RLinf + franky
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # 一次装齐：系统依赖（rt-tests, ethtool, eigen, pinocchio，
   # 由 install.sh 内部调 franky_install.sh 处理）+
   # RLinf Python 依赖 + PyPI 的 franky-control wheel。
   # 非 root 用户需要 sudo（系统依赖安装那一步会提示输密码）。
   bash requirements/install.sh embodied --env franka-franky --use-mirror

   source .venv/bin/activate

``--env franka-franky`` 固定使用 franky 路径
（PyPI 的 ``franky-control >= 0.15.0``），**跳过**
:doc:`franka` 使用的 ``serl_franka_controllers`` ROS / catkin 编译流。
``--use-mirror`` 面向国内用户（自动切换 PyPI / GitHub /
HuggingFace 镜像）。

.. note::

   ``requirements/install.sh embodied --env franka-franky`` **一条命令搞定**
   ：uv venv → 内部调 ``franky_install.sh`` 装系统级依赖
   （``rt-tests``、``ethtool``、``cmake``、``libeigen3-dev``、
   ``libpoco-dev``、``libfmt-dev``、pinocchio 等）→ ``franky-control``
   wheel。**不需要单独跑** ``franky_install.sh``。

**libfranka 不由以上脚本安装。** ``franky-control`` 在 PyPI 上以 manylinux
wheel 形式分发，**libfranka 已经内嵌在 wheel 里**——标准 Ubuntu + 主流
Python 版本（默认匹配组合：Python 3.11 + libfranka 0.15.x）拿来直接用，
无需任何额外操作。

只有当当前 Python / 系统 ABI 与 wheel 不匹配时，``pip`` 会回退到源码编译
**franky-control**（这一步会顺带把它依赖的 libfranka 一起处理）。
``franky_install.sh`` 已经预装好 cmake / eigen / poco / fmt / pinocchio
这些构建依赖；这种情况下直接从 GitHub 装源码版 franky：

.. code-block:: bash

   pip install git+https://github.com/TimSchneider42/franky.git

完整源码编译选项、libfranka 子模块版本对照、CMake 配置等参见 franky 官方
仓库 README：https://github.com/TimSchneider42/franky 。

.. warning::

   **请避开 libfranka 0.18.0**。Franka 官方 0.18.0 release notes
   标注了阻抗 / 笛卡尔控制路径的回归 bug；在本文使用的 joint /
   Cartesian impedance tracker 下，这个 bug 表现为机械臂
   **严重出力不足** —— 无力、无法承载自身重力，甚至无法跟踪轻量级
   GELLO 动作。版本请按 Franka firmware 在官方
   `compatibility matrix
   <https://frankarobotics.github.io/docs/compatibility.html>`_
   中查找匹配版本，**不要选择 0.18.0**\ （已在 0.19.0 上验证通过）。
   安装后可检查 ``franky.__libfranka_version__``。

4. GELLO（env worker 所在节点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

两台 GELLO 的 USB-FTDI 都接到 env worker 所在节点（仓库默认 placement
下是 **node 0** ），整个数据采集过程都保持在那里。
``DualGelloJointIntervention`` 在 env worker 进程里直接打开两个
串口，以 ~1 kHz 读取 —— 跨 LAN 访问 node 1 上的 GELLO 会超出实时性
预算、丢失采样，并导致 impedance tracker 参考点抖动。

具体安装命令（``gello`` + ``gello-teleop`` + USB-FTDI 权限，以及
"为什么只 init ``DynamixelSDK`` 这个 submodule"的背景）见
:doc:`franka_gello`。在 **node 0 上单独执行** 这些命令，并安装到与
RLinf 同一个 venv —— ``DualGelloJointIntervention`` 在 env wrapper
栈构建时会 in-process 直接 import 这两个包。

5. 脚踏
~~~~~~~

PCsensor FootSwitch 通过厂家提供的 Windows 工具把 3 个踏板烧成
键码 ``a`` / ``b`` / ``c``\ （写入一次后进入固件，重启后保留）。验证 +
授权：

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   #  期望: usb-PCsensor_FootSwitch-event-kbd → ../eventXX

   sudo chmod 666 /dev/input/eventXX

   # 一定要在 `ray start` 之前 export，让 Ray 把变量打包进 worker
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

``KeyboardListener`` 直接使用 ``evdev``，支持 ``ENODEV`` 自动重连
（USB 短时断连后也可恢复），并用边沿触发的 press 队列保证短于轮询周期
的按压也不会丢失。


硬件验证
--------

启动 Ray 之前，每个节点需先对各硬件执行单项测试。

相机
~~~~

.. code-block:: bash

   # RealSense：枚举总线，确认协商到 USB-3。
   rs-enumerate-devices | grep -E "Name|Serial|USB Type"

   # Lumos（XVisio vSLAM）：确认两个 /dev/v4l/by-id 节点都在。
   ls /dev/v4l/by-id/

   # USB 拓扑：Lumos 和 RealSense 应当协商到 5000M。
   # 任何掉到 "480M" 都是 USB-2 fallback（线缆或 hub 不行）。
   lsusb -t

GELLO
~~~~~

**1. 找 FTDI 串口路径并分辨左右**

GELLO 走 FTDI USB→Dynamixel 适配器，每条 GELLO 链对应 ``/dev/serial/by-id/``
下一个 ``usb-FTDI_..._<unique_id>-if00-port0`` 节点（``<unique_id>`` 来自
FTDI 芯片硬件序列号，每条线唯一、跨重启不变）：

.. code-block:: bash

   # 列所有 FTDI 转换器
   ls -l /dev/serial/by-id/ | grep -i ftdi

如果两条 GELLO 都接上，会看到两个候选。**分辨左右**用拔插对照法：

.. code-block:: bash

   # 先只插左 GELLO，记一遍
   ls /dev/serial/by-id/ | grep -i ftdi    # → LEFT_PATH
   # 再插上右 GELLO 一起列
   ls /dev/serial/by-id/ | grep -i ftdi    # → 新出现的那条就是 RIGHT_PATH

记住 ``<unique_id>`` 写下来。**这两条 by-id 路径写进 yaml**
（``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
的 ``env.eval.left_gello_port`` / ``right_gello_port``）：

.. code-block:: yaml

   env:
     eval:
       left_gello_port:  /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0
       right_gello_port: /dev/serial/by-id/usb-FTDI_..._<RIGHT_ID>-if00-port0

**2. 单独验每条 GELLO 关节读数实时性**

.. code-block:: bash

   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0

操作 GELLO 时应当看到关节读数实时变化。如果数值阻塞或者突然跳
±2π，请执行下一节的标定流程。右臂换 ``<RIGHT_ID>`` 重复。

每台机械臂单独验
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   FRANKA_ROBOT_IP=172.16.0.2 \
   FRANKA_GRIPPER_TYPE=robotiq \
   FRANKA_GRIPPER_PORT=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id>-if00-port0 \
       python toolkits/realworld_check/test_franky_controller.py

REPL 命令：

* ``getjoint`` —— 打印当前关节角
* ``home`` —— 同步复位到 ``HOME_JOINTS``
* ``hold 30`` —— 静置 30 s，听有没有嗡鸣
* ``stream 4 0.001 500`` —— 1 kHz 推 500 条 J4 += 0.001 rad
  （streaming preemption 压测）
* ``impedance 300 300 300 300 150 80 30`` —— 降低关节阻抗后再压测一次
* ``open`` / ``close`` —— gripper sanity

每节点对自己那台机械臂单独执行：静置无可听嗡鸣、``stream 4 0.001 1000``
能跑 ≥ 800 Hz、``home`` 从任意合法位姿都能干净复位即可。**两台机械臂
都通过验证之前不要启动 Ray。**


GELLO 标定
----------

1. **标定**\ （每台 GELLO 一次，更换电机后再标）：

   .. code-block:: bash

      python toolkits/realworld_check/test_gello.py calibrate

   脚本会将机器人安全地依次移动到两个已知姿态（``POSE_A`` =
   Franka 原点，``POSE_B`` = π/4 倍数），让操作员将 GELLO 各
   摆成相同姿态，然后从两次差值解出 ``joint_signs`` 和
   ``joint_offsets``，最后打印一段可直接粘贴到
   ``gello_software/gello/agents/gello_agent.py`` 的
   ``DynamixelRobotConfig`` 块::

       "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0":
           DynamixelRobotConfig(
               joint_ids=(1, 2, 3, 4, 5, 6, 7),
               joint_offsets=(...),
               joint_signs=(...),
               gripper_config=(8, ..., ...),
               baudrate=1_000_000,
           ),

   ``gello`` 是 editable install，粘贴完成后不需要重新安装，只需重启 ``gello`` 的进程即可。

2. **对齐**\ （观察到 GELLO leader 和机械臂位姿不一致时执行 ——
   例如手动移动过机械臂、长时间未使用、或采集会话开始前需要确认状态
   时）：

   .. code-block:: bash

      python toolkits/realworld_check/test_gello.py align-sequential

   脚本会将机器人移动到一个固定的对齐 HOME 位姿（J4 = −π/2、
   J6 = +π/2 等），然后逐关节 J1 → J7 引导操作员对齐，每个关节带
   live progress bar。一旦某关节连续 8 帧都在 ±0.10 rad 内就
   自动跳到下一个。

两个脚本都会用 glob ``/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*-if00-port0``
自动找到本机 Robotiq 串口，因此无需关心当前位于左节点还是右节点。


硬件 YAML
---------

双 Franka 的硬件配置写在
``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
（采集）和
``examples/embodiment/config/env/realworld_dual_franka_rot6d.yaml``
（rot6d 部署）。参考这两份示例改即可，需要按本机替换的占位符：

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP`` —— 左右臂 FCI IP（如
  ``172.16.0.2``）。
* ``BASE_CAMERA_SERIAL`` —— base 相机 serial（RealSense 用
  ``rs.context().devices`` 报告的；按 ``base_camera_type`` 后端
  改成对应 SDK 的 serial）。
* ``LEFT_CAMERA_SERIAL`` / ``RIGHT_CAMERA_SERIAL`` —— 两腕相机
  serial（Lumos 用 ``/dev/v4l/by-id/usb-XVisio_..._video-index0``
  路径；按 ``*_camera_type`` 后端改）。
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION``
  —— Robotiq 2F-85 的 RS-485 串口，固定用
  ``/dev/serial/by-id/usb-FTDI_..._<id>-if00-port0``，**不要**
  用 ``/dev/ttyUSB*``\ （重启 / 热插拔后会换号）。
* ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT`` —— GELLO 主手的
  ``/dev/serial/by-id`` 路径（两个都插在 env worker 所在节点，
  即 ``node_rank: 0``）。
* override 段内的 ``ee_pose_limit_min`` / ``ee_pose_limit_max``
  —— 按本机工作空间安全箱调；行 0 是左臂、行 1 是右臂，每行
  ``[x, y, z, roll, pitch, yaw]``。

``left_controller_node_rank`` / ``right_controller_node_rank``
（默认 ``0`` / ``1``\ ，每节点各管一台）和 ``node_rank``\ （env
worker + 相机所在节点）通常不用改。


Ray cluster 启动
-----------------

Ray 在 ``ray start`` 时会捕获当前的 Python 解释器和**已 export 的
环境变量**，worker actor 都继承这个快照。``ray start`` 之后再
``pip install`` 进 venv 的包，下次 import 时仍可见（Ray 不会
冻结 ``site-packages`` ）；但环境变量不会同步更新 —— 未 export 的变量，
worker 永远拿不到。顺序：

1. **每个节点上**：激活 venv，export ``RLINF_NODE_RANK``，export
   ``RLINF_COMM_NET_DEVICES``\ （可选，如果有多个网口请指定两个
   node 间通信的网口），如果脚踏在该节点就 export
   ``RLINF_KEYBOARD_DEVICE``。验证 ``franky``、``gello``、
   ``gello_teleop`` 都能 import。
2. **然后** ``ray start`` —— node 0 head，node 1 worker。

在每个节点上激活 venv，export rank 相关环境变量，再 ``ray start``。
``HEAD_IP`` / ``WORKER_IP`` 是两台机器相互通信用的局域网 IP（不是
``127.0.0.1``，也不是公网 IP）。

.. code-block:: bash

   # node 0（Ray head）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX  # 若脚踏在这台

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1（Ray worker）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1

   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

在 node 0 验证：

.. code-block:: bash

   ray status
   # 期望：2 个节点都 ALIVE，cluster 里 GPU/CPU 资源对得上

.. warning::

   两个节点是**独立 checkout**。node 0 改完代码要 rsync 到 node 1
   （``rsync -av --delete RLinf/ <node1>:/path/to/RLinf/``）
   **并** 在该节点重启 Ray，让新代码进 Ray 的环境快照。忘了同步
   会出现"node 0 上可运行、node 1 上 ImportError"或者更隐蔽的
   "feature 在某些 worker 上行为不一致"问题。


数据采集（GELLO 关节空间）
--------------------------

采集路径是 ``DualFrankaJointEnv-v1`` + ``teleop_direct_stream:
true``。``DualGelloJointIntervention`` 内部的守护线程以 ~1 kHz
读 GELLO Dynamixel 电机位置，直接推到两个 ``FrankyController``
actor（再转发给 franky 的 ``JointImpedanceTracker``）。
``env.step`` 以 10 Hz 运行，只读取 state、在状态翻转时触发 gripper
开/合、采集相机帧 —— 它**不调用** ``move_joints``。

为什么使用 direct-stream 而不是 env-step gating？10 Hz 采样会把
高频腕部动作抹掉。1 kHz 守护线程按 GELLO 原生频率采样操作员实际
的手部运动，env.step 再以 10 Hz 读取**已经发生**的关节状态 ——
因此数据集记录的是操作员实际执行的动作，而不是被 100 ms 网格
截断后的轨迹。10 Hz 这个状态读取频率也正是 π₀.₅ 推理时接收的输入
频率。

配置
~~~~

用仓库里的
``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``。
开采集前会改的字段：

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - YAML 字段
     - 含义
   * - ``runner.num_data_episodes``
     - 目标 episode 数。结合 ``data_collection.resume`` 之后这是
       *跨多次会话累积* 的总目标，不是单次会话目标。
   * - ``env.eval.left_gello_port`` / ``right_gello_port``
     - 这次会话临时换 GELLO 单元时在这里覆盖。否则继承自 env yaml。
   * - ``env.eval.override_cfg.task_description``
     - 写到每帧 ``task`` 字段的 prompt。
   * - ``env.eval.override_cfg.joint_action_mode``
     - ``absolute``\ （采集用，1:1 映射 GELLO 关节）；``delta``
       用于同一 env 类的离线 RL。
   * - ``env.eval.override_cfg.teleop_direct_stream``
     - ``true`` 开 1 kHz 守护线程。设成 ``false`` 会回到 env.step
       gating 路径，**不是推荐的采集路径**。
   * - ``data_collection.save_dir``
     - 数据集根目录。默认每次会话写到
       ``${runner.logger.log_path}/collected_data``；命令行 override
       同一根目录，让多次会话累积。
   * - ``data_collection.resume``
     - ``true`` 时会从 ``save_dir/rank_0`` 下已有的 ``id_*``
       shard 累计 episode 数继续。

启动
~~~~

Ray 启动后打开 2 个终端。

**终端 1** —— launcher（在 node 0）：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka 2>&1 \
        | tee logs/collect.log

**终端 2** —— 实时进度条（在 node 0）：

.. code-block:: bash

   python toolkits/realworld_check/collect_monitor.py logs/collect.log

单独使用一个 monitor，是因为采集器作为 Ray worker 运行，stdout
被 Ray 的 log monitor 批量缓冲（~500 ms），会破坏 ``tqdm`` 的
``\r`` 原位刷新。Monitor 在自己的 TTY 里 tail 日志文件，渲染一
条干净的 tqdm 进度条，显示成功计数、最近脚踏事件、最后一次
reward。默认启动会 replay 已有日志，所以监视器开晚也能对齐进度
（``--no-replay`` 切到只 tail EOF）。``--source=worker``\ （默认
``auto``）会直接 tail Ray worker 的 stdout 文件
（``/tmp/ray/session_latest/logs/worker-*-<pid>.out``），完全
绕开 log monitor batching（快 1-2 分钟），找不到时回退到 tee 日志。

每个 episode 的工作流
~~~~~~~~~~~~~~~~~~~~~~

确认 ``test_gello.py align-sequential`` 报告 ``ALL JOINTS ALIGNED`` 之后：

1. **(pre)** 每次 reset 时，机械臂会跟着 GELLO 当前位姿对齐
   （``KeyboardStartEndWrapper`` + ``DualGelloJointIntervention``
   会通过 ``options["skip_reset_to_home"]=True`` 跳过 home 复位）。
   机械臂保持在操作员当前手部位置。
2. **踩下 ``a``** —— 从当前位姿开始记录第 0 帧。
3. **演示任务**。每一步数据进 buffer。机械臂以 1 kHz 跟踪 GELLO；
   相机以 10 Hz 抓帧。
4. **踩下 ``b``** —— 子任务边界：``segment_id`` +1
   （1 s 防抖；窗口内的二次按下被忽略）。用来标 "approach" /
   "grasp" / "transfer" / "place" 等阶段，方便下游策略按
   segment_id 条件化。
5. **踩下 ``c``** —— 标记成功：reward = 1.0、``terminated=True``、
   ``CollectEpisode`` 把 buffer 写到 LeRobot shard。
6. **录制中再次踩下 ``a``** —— 中止：丢弃 buffer，回到 pre 阶段。
   机械臂不复位 home，停在当前位置（便于操作员立即重试，不打断
   GELLO 跟踪）。

输出格式
~~~~~~~~

LeRobot v2.1，每次会话一个 shard，路径是
``<save_dir>/rank_0/id_{N}/``：

* ``meta/info.json`` —— feature schema。``state`` 固定 ``[68]``；
  ``actions`` 在 joint 模式下 ``[16]``，rot6d 模式下 ``[20]``。
* ``meta/episodes_stats.jsonl`` —— 每条 episode 的 ``state`` /
  ``actions`` min / max / mean / std。
* ``data/episode_NNNNNN.parquet`` —— 每一步一行。

每帧字段：

* ``state`` —— ``DualFrankaJointEnv.STATE_LAYOUT`` 拼接
  ``[gripper_position(2), joint_position(14), joint_velocity(14),
  tcp_force(6), tcp_pose(14), tcp_torque(6), tcp_vel(12)]`` = 68。
  前 2 个 slot 特意设计成 ``[L_grip, R_grip]``，匹配 rot6d 策略
  ``_rearrange_state`` 的切片假设。
* ``actions`` —— GELLO 守护线程那一步派发的动作（joint 模式
  16 维：``[L_jpos(7), L_grip, R_jpos(7), R_grip]``）。
* ``image`` —— ``left_wrist_0_rgb``\ （``main_image_key``）。
* ``wrist_image-0`` / ``wrist_image-1`` —— 通过
  ``CollectEpisode._expand_multi_view_images`` 展开的左右腕视图。
* ``extra_view_image-0`` / ``extra_view_image-1`` —— base + 右腕
  视图，**顺序锁死** ``("base_0_rgb", "right_wrist_0_rgb")``。
  这个顺序在 ``DualFrankaRot6dInputs._extract_extra_views`` 里被
  断言，rig 重命名时会显式报错，避免静默调换相机含义。
* ``task`` —— 该 episode 所属任务的 prompt。
* ``is_success`` —— sticky flag，整条 episode 都为 ``True``
  当且仅当 episode 由踩下 ``c`` 结束。
* ``done`` —— 只有 episode 最后一帧为 ``True``。
* ``intervene_flag`` —— 采集阶段始终 ``True``\ （GELLO 守护
  线程的命令就是 action）。
* ``segment_id`` —— uint8，踩下 ``b`` 时 +1。

断点续采
~~~~~~~~

``data_collection.resume: true`` 加上原 ``save_dir`` 重新执行：
``CollectEpisode._count_existing_lerobot_episodes`` 会扫
``id_*`` shard 累计 ``total_episodes``\ （恶意损坏的 shard 自动
跳过，避免上次中断留下的脏 shard 阻塞 resume），新会话写到
新建的 ``id_{N}`` shard，已 finalize 的数据保持不变。

进度条初始位置会以已有计数初始化 —— ``num_data_episodes: 200``
加上之前已存的 50 条 success，新会话还需要采集 150 条。


回填 rot6d 与 norm_stats
-------------------------

采集出来的是 16 维 joint 动作 + 68 维 state；π₀.₅ SFT 要求 20 维
rot6d（state 和 action 都是 ``[xyz(3) + rot6d(6) + grip(1)] × 2``）。
SFT 前用以下脚本离线转换：

.. code-block:: bash

   export PYTHONPATH=$(pwd)
   python toolkits/dual_franka/backfill_rot6d.py \
       --src $HF_LEROBOT_HOME/<repo_id>/joint_v1 \
       --dst $HF_LEROBOT_HOME/<repo_id>/rot6d_v1

脚本做三件事：把 state 前 20 维改写成 rot6d 布局（直接从原 state
里的 tcp_pose 切出来转 rot6d，不跑 FK）；把 actions 从 16 维扩到
20 维，xyz/rot6d 用**下一帧** tcp_pose 近似操作员当前帧的目标，
gripper 槽位沿用原信号；更新 parquet schema 的 ``actions.length``
并重算每条 episode 的 stats。已回填过的数据集会直接报错，不会重复
写入。

回填完之后算 norm stats：

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_rot6d \
       --repo-id <repo_id>/rot6d_v1

脚本会按 SFT 数据 pipeline 执行一遍数据集
（``RepackTransform`` → ``DualFrankaRot6dInputs`` →
``RigidBodyDeltaActions`` ），把 ``norm_stats.json`` 写到
``<openpi_assets_dirs>/<data_config.repo_id>/`` 下。同一个
``<repo_id>`` 也是部署阶段 rollout worker 查找 norm_stats 的
key —— 详见下文"ckpt / norm_stats 锁步"完整路径解析规则。

norm stats 必须**在回填之后**计算，不能在之前计算 —— 它需要看到策略
真正会预测的 body-frame delta，而不是磁盘上的绝对目标。


SFT（π₀.₅，rot6d_v1）
---------------------

配置
~~~~

``examples/sft/config/realworld_sft_openpi_dual_franka_rot6d.yaml``。启动前
需要修改的字段：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 字段
     - 设成
   * - ``data.train_data_paths``
     - 已完成回填的 rot6d_v1 数据集所在的 LeRobot 根目录。这个值
       会被 ``train_vla_sft.py`` 在 validate 之前 export 成
       ``HF_LEROBOT_HOME``，openpi 数据加载器会自动用。
   * - ``actor.model.model_path``
     - π₀ / π₀.₅ base ckpt（torch 转换后的 weights，例如
       ``checkpoints/torch/pi05_base/``）。
   * - ``actor.model.action_dim``
     - ``20``\ （必须匹配 rot6d 数据 layout）。
   * - ``actor.model.num_action_chunks``
     - ``20``\ （匹配 ``pi05_dualfranka_rot6d`` TrainConfig 里的
       ``action_horizon``）。
   * - ``actor.model.openpi.config_name``
     - ``pi05_dualfranka_rot6d``。
   * - ``actor.optim.lr``
     - π₀.₅ 在此类数据集上 ``7.91e-6`` 是一个合理默认值。
   * - ``actor.fsdp_config.sharding_strategy``
     - ``full_shard``\ （若 GPU 数量超过 8 张、希望使用跨副本
       all-reduce 而非 all-gather，则改为 ``hybrid_shard``）。
   * - ``runner.save_interval``
     - ``500`` 步保存一次 ckpt 到
       ``${runner.logger.log_path}/checkpoints/global_step_<N>/``。

启动
~~~~

.. code-block:: bash

   # 单节点、4 张 GPU —— cluster.num_nodes: 1，
   # component_placement.actor,env,rollout 用 GPU 0..3。
   bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_rot6d

Runner 每 ``runner.save_interval`` 步保存一次 ckpt，目录布局：

.. code-block:: text

   <log_path>/checkpoints/global_step_<N>/
   ├── actor/
   │   └── model_state_dict/
   │       └── full_weights.pt
   └── <asset_id>/                        # 例如 "<your-hf-user>/<your-dataset>"
       └── norm_stats.json                # 推理使用的 pinned norm stats

真机部署时，rollout worker 会从
``<model_path>/actor/model_state_dict/full_weights.pt`` 读策略
权重，从 ``<model_path>/<asset_id>/norm_stats.json`` 读 norm
stats。

真机部署
--------

跟采集用同一套 Ray cluster，换入口脚本 + 配置。

配置
~~~~

``examples/embodiment/config/realworld_eval_dual_franka.yaml``。
占位符标注为 ``# Replace:``。最常修改的字段：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 字段
     - 设成
   * - ``rollout.model.model_path``
     - ``<sft_log>/checkpoints/global_step_<N>/`` —— 必须包含
       ``actor/model_state_dict/full_weights.pt`` 和
       ``<data_config.repo_id>/norm_stats.json`` （
       ``data_config.repo_id`` 怎么算见下文"ckpt / norm_stats
       锁步"）。
   * - ``actor.model.openpi_data.repo_id``
     - 作为 ``data_kwargs`` 传给 ``get_openpi_config`` ，会覆盖
       ``data_config.repo_id`` ；这个 ``repo_id`` 就是部署时
       ``norm_stats.json`` 的查找 key。和
       ``calculate_norm_stats.py --repo-id`` 时给的值保持一致。
   * - ``env.eval.override_cfg.task_description``
     - 跟训练 prompt 一致。
   * - ``env.eval.override_cfg.joint_reset_qpos``
     - 从 SFT 数据集首帧 joint 均值反算回来；用过期值会让
       reset 后第一帧 obs 在训练分布外。
   * - ``env.eval.override_cfg.target_ee_pose`` / ``reset_ee_pose``
     - 跟采集时的 workspace 对齐。
   * - ``cluster.node_groups[*].env_configs[0].python_interpreter_path``
     - node 0 上 openpi venv 的 python 路径（env worker / rollout
       actor 用这个起 worker 进程）。

硬件 ``configs`` 与采集 yaml 完全一致 —— 同 IP、同相机 serial、
同 gripper 串口。Wrapper 是按 ``env.eval.use_*`` flag 装的，所以
采集 vs 部署的 yaml 差别只有 3 个：

* ``use_gello_joint: false``\ （采集是 ``true``）
* ``keyboard_reward_wrapper: eval_control``\ （采集是 ``start_end``）
* ``use_relative_frame: false`` —— rot6d 部署必须，否则
  ``DualRelativeFrame`` 会破坏 rot6d state。

启动
~~~~

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

   # Hydra override 示例：
   #   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
   #        rollout.model.model_path=/sft/global_step_5000 \
   #        env.eval.override_cfg.task_description="pour water"

每个 episode 的部署工作流
~~~~~~~~~~~~~~~~~~~~~~~~~

``KeyboardEvalControlWrapper`` 把脚踏 wrapper 切成自主推理模式：

1. ``env.reset()`` 之后两台机械臂保持在 reset 位姿。``env.step()``
   被截到 **idle** 模式 —— 不向内层 env 转发（impedance 控制器
   保持上一次 reset 时的目标，机械臂原地静止），但 wrapper 仍会把
   最近一次 obs 返回，让策略的 chunked rollout 循环空转，不下发
   任何关节指令。
2. 踩下 ``a`` —— wrapper 切到 **running**。下一步 ``env.step``
   开始向内层 env 转发策略输出。
3. 踩下 ``c`` —— 成功：``terminated=True``、``reward=1.0``、
   ``info["eval_result"]="success"``。Wrapper 内部立刻调
   ``env.reset()`` 让机械臂回 home，然后回到 idle 等下一次 ``a``
   —— 这是脚踏可连续操作的关键，即使 eval ``env_worker``
   是 ``auto_reset=False``。
4. 踩下 ``b`` —— 失败：行为同 ``c``，但 ``reward=0.0``、
   ``info["eval_result"]="failure"``。
5. running 阶段，wrapper 强制把 ``terminated`` / ``truncated``
   置 False，除非脚踏触发 —— env 自己的 ``max_episode_steps``
   不会切断策略。把 ``max_episode_steps`` 设大一点（仓库 yaml
   是 ``10000``），让脚踏始终是边界 owner。

ckpt / norm_stats 锁步
~~~~~~~~~~~~~~~~~~~~~~~

部署时最常见的崩盘原因是 ``norm_stats`` 不匹配。Rollout worker
的 norm_stats 路径解析在
``rlinf/models/embodiment/openpi/__init__.py``::

   pinned_path = <model_path>/<data_config.asset_id>/norm_stats.json
   if pinned_path 存在:
       用它
   else:
       退回 data_config.norm_stats，并输出明确告警

``data_config.asset_id`` 是 SFT 阶段 ``DualFrankaRot6dDataConfig.create()``
解析出来的（继承自 ``AssetsConfig.asset_id``，没显式设的话会回退到
``data_config.repo_id`` ）。同一个 key 也被
``calculate_norm_stats.py`` 用来写出，路径是
``<openpi_assets_dirs>/<data_config.repo_id>/`` 下。所以
``<model_path>/...`` 下的路径必须与 SFT 实际使用的路径一致。

实际操作：

* 如果保留 ``actor.model.openpi_data.repo_id`` 的默认值
  （ ``<your-hf-user>/<your-dataset>`` ），norm_stats 在
  ``<model_path>/<your-hf-user>/<your-dataset>/norm_stats.json``。
* 如果将 ``actor.model.openpi_data.repo_id`` （以
  ``data_kwargs`` 形式被透传）覆盖成本地回填数据集，
  ``data_config.repo_id`` 会被替换，查找 key 也跟着变成新值。
  **务必用同一个 ``--repo-id`` 执行**
  ``calculate_norm_stats.py`` ，再把结果拷到
  ``<model_path>/<那个 repo_id>/norm_stats.json``。

启动前自检：

.. code-block:: bash

   # 直接 grep SFT 日志中 rollout worker 实际查找的路径：
   grep "norm_stats" <sft_log>/run_embodiment.log | tail
   # 或者直接确认 model_path 下能找到 norm_stats.json：
   find <model_path> -maxdepth 3 -name norm_stats.json
   ls <model_path>/actor/model_state_dict/full_weights.pt

不匹配的 stats 会静默产生分布外的 state；策略会塌缩成一个固定动作
（向角落移动、gripper 锁死在打开状态等），且不会显式报错。fallback 路径**会**
输出 ``"norm_stats fallback: ... verify they match training or
inference will be wrong"`` warning —— 在判定 rollout 健康之前请先
grep log。


故障排查
--------

**GELLO 守护线程未启动**
   GELLO 重新上电、FTDI 重插，然后用
   ``python -m gello_teleop.gello_expert --port /dev/...`` 验证
   两侧都能持续输出 Dynamixel 读数。

**Ray worker 静默死在 import**
   在跑 ``ray start`` 的同一 shell 里执行
   ``which python && python -c "import franky, gello, gello_teleop"``
   确认 venv 和已装包一致；具体报错看
   ``/tmp/ray/session_latest/logs/worker-*.err``。

**有一台机械臂 reset 时挂住**
   在 controller 节点 ``ping -c 100 <robot_ip>``，若丢包就重启
   该机械臂再跑。

**开机后 ``move_joints`` 一直报错**
   释放白色急停按钮 → Desk 网页（\ ``http://<robot_ip>/desk/``\ ）
   点 *Activate FCI* → 等关节 LED 由白转蓝 → 再启动。

**GELLO 守护线程和 env reset 互相 race**
   reset 期间把 GELLO leader 放稳在支架上，等
   ``KeyboardStartEndWrapper`` 报告 reset 结束再继续操作。

**脚踏报 "Permission denied"**
   ``sudo chmod 666 /dev/input/eventXX``；要持久化就写 udev rule
   （``KERNEL=="event*", SUBSYSTEM=="input",
   ATTRS{name}=="PCsensor FootSwitch", MODE="0666"``）。

**RealSense 退到 USB 2.x**
   换 USB 线缆，插到主板的蓝色 USB-3 端口，``lsusb -t`` 确认显示
   ``5000M`` 而不是 ``480M``。

**Lumos 冷启动第一次失败**
   重新插拔 USB 线。

**部署时 idle 一直不响应**
   确认 ``RLINF_KEYBOARD_DEVICE`` 指向正确的
   ``/dev/input/eventXX`` 且 ``chmod 666`` 仍生效，然后踩 ``a``
   触发。

**部署阶段跟踪抖动**
   降 ``RLINF_CART_K_R``、提高 ``RLINF_CART_GAINS_TC``、把
   ``RLINF_CART_MAX_STEP_RAD`` 收紧；仍不行就缩短策略 chunk
   长度。

**部署时找不到 ``norm_stats.json``**
   把 ``calculate_norm_stats.py`` 写出的
   ``<openpi_assets_dirs>/<repo_id>/`` 复制到
   ``<model_path>/<repo_id>/``；先 grep
   ``"norm_stats fallback"`` warning 判断是否走到 fallback 路径。

**collect_monitor 无进展**
   launcher 加 ``2>&1 | tee logs/collect.log``；env worker 在另一
   节点时给 monitor 加 ``--source=worker``。

**controller 启动时输出 ``sched_setaffinity failed`` warning**
   换 6+ 核机器，或对 venv 解释器执行
   ``sudo setcap cap_sys_nice=eip $(which python)``。

**reset 时两台机械臂都动了，但之后只有一根跟踪 GELLO**
   每台 GELLO 单独跑
   ``python toolkits/realworld_check/test_gello.py align-check``
   确认都在持续输出读数，再重启。
