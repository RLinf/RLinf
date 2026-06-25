Franka 真机使用 VR 遥操作设备
====================================

本指南介绍如何在 RLinf 的 Franka 真实世界环境中配置和使用 **VR / PICO**
遥操作设备。本文是基础 :doc:`franka` 文档的扩展，仅涵盖 VR 遥操作链路所需的
**额外** 步骤。

.. note::

   如果你还没有阅读过基础的 Franka 指南，请先参考 :doc:`franka`。
   本页默认 Franka 控制、ROS、Ray 集群和相机已经按基础指南完成配置。


硬件架构概览
-----------------

当前 VR 遥操作链路以 PICO 头显和手柄为例。PICO 数据由外部服务采集并通过
ZeroMQ 发布，RLinf 中的 PICO intervention wrapper 订阅该数据流，并将手柄相对位姿
转换为 Franka 环境使用的归一化末端 delta action。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 节点
     - 角色
     - 硬件 / 软件
   * - **GPU 服务器** (node 0)
     - Actor、rollout、可选相机采集
     - NVIDIA GPU、RLinf
   * - **Franka 控制节点** (node 1 或单节点)
     - FrankaController、env worker、VR 数据订阅
     - Franka、ROS Noetic、serl_franka_controllers、pyzmq
   * - **VR / PICO PC**
     - 运行 XRoboToolkit 和 VR 数据发布进程
     - PICO 头显、手柄、VR publisher

如果 VR publisher 与 RLinf env worker 在同一台机器，可以使用 IPC 地址；
如果在不同机器上运行，推荐使用 TCP 地址。


VR 软件准备
------------------------------

VR 数据发布进程不直接由 RLinf 启动。它需要先从 PICO / XRoboToolkit 读取头显和手柄数据，
再将数据发布到 ZeroMQ。

1. 准备 PICO / XRoboToolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 PICO 头显和运行 publisher 的机器上完成以下准备：

- 在 PICO 头显端，从
  `XRoboToolkit-Unity-Client releases <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases>`_
  选择与头显和 XRoboToolkit 版本适配的 APK 并安装。
- 在负责收发 PICO 数据的机器上，从
  `XRoboToolkit-PC-Service releases <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases>`_
  选择适配版本并安装 PC Service。当前 release 页面提供的 PC Service 安装包仅覆盖
  Ubuntu 22.04 和 Ubuntu 24.04，请使用这两个系统版本之一进行安装；其他系统版本需要自行适配。
- 确认 publisher 机器与 Franka 控制节点在同一网络中，或者与 RLinf env worker 在同一台机器上。

在 PICO 头显连接前，先在收发数据的机器上启动 XRoboToolkit PC Service：

.. code-block:: bash

   cd /opt/apps/roboticsservice
   bash runService.sh

启动 PC Service 后，再在 PICO 头显端连接该服务，并确认 PICO 头显、左右手柄均已连接，
手柄位姿和按键数据会持续更新。

选择一个安装目录，然后克隆仓库：

.. code-block:: bash

   cd /path/to/install/pico
   git clone git@github.com:tiny-xie/pico_software.git
   cd pico_software/XRoboToolkit-Teleop-Sample-Python

.. note::

   ``pico_software`` 中使用的 ``XRoboToolkit-Teleop-Sample-Python`` 模块来自官方开源仓库
   `XR-Robotics/XRoboToolkit-Teleop-Sample-Python <https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python>`_。

配置环境：

.. code-block:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的setup_uv.sh命令
   # bash setup_uv.sh --use-mirror

   bash setup_uv.sh

启动 VR 数据 publisher 前，可使用刚刚创建的虚拟环境，在 ``pico_software`` 路径下验证
PICO 与收发数据机器的 XRT 链路是否已经连通：

.. code-block:: bash

   cd /path/to/pico_software/XRoboToolkit-Teleop-Sample-Python
   source .venv/bin/activate
   cd /path/to/pico_software
   python test_pico_xrt_pipeline.py

如果连接成功，移动 PICO 头显和手柄时，终端中应能看到持续变化的数据流。

启动 VR 数据 publisher 前，先在 ``pico_software/configs/vr_bridge.yaml`` 中配置
publisher 侧的 ZeroMQ 地址。

同机运行时可以使用 IPC：

.. code-block:: yaml

   zmq:
     ipc_addr: "ipc:///tmp/vr_data.ipc"

跨机器运行时，publisher 侧建议绑定 TCP：

.. code-block:: yaml

   zmq:
     ipc_addr: "tcp://0.0.0.0:<port>"

然后在同一台收发数据的机器上，使用该配置文件启动 VR 数据 publisher：

.. code-block:: bash

   cd /path/to/pico_software/XRoboToolkit-Teleop-Sample-Python
   source .venv/bin/activate
   cd /path/to/pico_software
   python -m vr_data_publisher --config configs/vr_bridge.yaml

2. 安装 RLinf 侧依赖
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行 PICO intervention 的 RLinf 环境请直接使用 ``franka`` env 安装。
该 env 包含基础 Franka 真机依赖，以及 VR / PICO 链路所需的 ``pyzmq``：

.. code-block:: bash

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

如果使用 Ray，请在 ``ray start`` **之前** 完成安装并 source 对应环境。

.. warning::

   Ray 会在 ``ray start`` 时捕获 Python 解释器和环境变量。若 ``pyzmq``、
   ROS 环境或 ``PYTHONPATH`` 在 ``ray start`` 之后才配置，worker 进程可能无法导入
   ``PicoIntervention`` 或无法连接 ZeroMQ。


3. 验证 PICO 数据流
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

启动 VR 数据 publisher 后，可以在运行 ``PicoIntervention`` 的节点上先执行
RLinf 内置检查脚本，确认已经能收到 PICO / ZeroMQ 数据：

.. code-block:: bash

   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr ipc:///tmp/vr_data.ipc

如果 publisher 和 RLinf env worker 跨机器运行，请把 ``--zmq-addr`` 改成
YAML 中 ``pico.zmq_addr`` 使用的 TCP 地址：

.. code-block:: bash

   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr tcp://<vr_publisher_ip>:<port>

你应该看到持续更新的输出，类似于：

.. code-block:: text

   [000012] recv_rate=79.8Hz | headset: pos=[0.120 1.430 0.210] quat=[0.000 0.707 0.000 0.707] | right: pos=[0.320 1.120 0.850] quat=[0.010 0.690 0.020 0.724] grip=0.000 trigger=0.000 | buttons_active=none

如果输出持续刷新，就说明 RLinf 侧已经能收到 PICO 数据流；移动手柄、按下
``grip`` / ``trigger`` / ``A`` / ``B`` 时，对应数值或按键状态也会随数据变化。


YAML 配置说明
-------------------

要使用 PICO 进行数据采集，请使用配置文件
``examples/embodiment/config/realworld_collect_data_pico.yaml``。
RLinf consumer 侧的 ZeroMQ 连接地址配置在 ``env.eval.pico.zmq_addr``。
该地址必须与 ``configs/vr_bridge.yaml`` 中的 publisher bind 地址匹配：
同机运行时可使用 ``ipc:///tmp/vr_data.ipc``；跨机器运行时应使用
``tcp://<vr_publisher_ip>:<port>``，不要使用 ``tcp://0.0.0.0:<port>`` 作为
consumer 连接地址。

关键配置如下：

.. code-block:: yaml

   env:
     eval:
       use_spacemouse: False
       use_pico: True

       pico:
         zmq_addr: "tcp://<vr_publisher_ip>:<port>"
         position_scale: 1.0
         rotation_scale: 1.0
         calibration:
           button: "trigger"


夹爪配置
-------------------

如果 ``no_gripper: true``，PICO 只控制机械臂 6D 末端运动，不会实际控制夹爪。

若需要 VR 控制夹爪，请确保环境动作维度包含夹爪，并使用支持的末端执行器配置，例如：

.. code-block:: yaml

   env:
     eval:
       no_gripper: False
       pico:
         gripper_close_button: "A"
         gripper_open_button: "B"
         gripper_close_threshold: 0.5

当前按键语义：

.. code-block:: text

   A -> close gripper
   B -> open gripper

如果 A/B 都不按，夹爪动作输出为 ``0.0``，不会因为松开按钮而自动打开。

集群配置注意事项
---------------------

集群配置步骤与 :doc:`franka` 中描述的相同，主要额外要求如下：

- 运行 ``PicoIntervention`` 的节点必须能访问 VR publisher 的 ZeroMQ 地址。
- 如果 VR publisher 使用 TCP，确认防火墙和网卡路由允许访问对应端口。
- 如果 VR publisher 使用 IPC，publisher 和 RLinf env worker 必须在同一台机器上。
- ``RLINF_NODE_RANK``、YAML 中的 ``cluster.node_groups[*].node_ranks`` 和
  Franka hardware config 中的 ``node_rank`` 必须一致。


安全建议
----------------

- 首次上真机将 ``position_scale`` 调低，例如 ``0.3`` 到 ``0.5``。
- 工作空间安全盒、单步 action scale 和 Franka Desk 状态都确认后再放任务物体。
- 每次修改 Ray 环境、Python 依赖、ROS 环境变量或 ZeroMQ 地址后，先 ``ray stop`` 再重启。
- 若控制方向明显不对，先松开 ``grip``，重新站位并扣下 ``trigger`` 标定，不要在接管中强行纠正。

启动顺序
---------------------

1. 在 Franka 控制节点上完成 ROS、catkin workspace、RLinf venv 和 ``PYTHONPATH`` 配置。
2. 在启动 Ray 前确认已经安装并 source ``franka`` 环境。
3. 启动 Ray 集群。单节点或多节点步骤与 :doc:`franka` 相同。
4. 启动 PICO / XRoboToolkit PC Service，并确认头显和手柄已连接。
5. 启动 VR 数据 publisher。
6. 首次运行或修改 ZeroMQ 地址后，运行 ``test_pico_data.py`` 确认 PICO 数据可达。
7. 在 Ray head 节点启动采集脚本，并确认采集脚本已经接入 ``PicoIntervention``。

.. code-block:: bash

   cd /path/to/RLinf
   bash examples/embodiment/collect_data.sh realworld_collect_data_pico

.. warning::

   同一时间只允许一个程序持有 Franka 控制权。运行 RLinf 采集时，不要同时运行
   ``robo_avatar.slave.main``、其他 ``franka_control_node`` 或另一个 RLinf 真机任务。


VR 操作步骤
----------------

首次上真机建议不放任务物体，只做小幅运动验证。

1. 站到期望的操作者初始姿态，头显朝向机器人工作台正前方。
2. 扣下 ``trigger`` 做 PICO base 标定。
3. 按住手柄 ``grip`` 开始接管。按下瞬间会记录手柄 reference pose 和当前 TCP reference pose。
4. 小幅移动手柄，确认方向：

.. code-block:: text

   手柄前推 -> 末端 +X
   手柄后拉 -> 末端 -X
   手柄左/右 -> 末端 +/-Y
   手柄上/下 -> 末端 +/-Z
   手柄旋转 -> 末端 roll / pitch / yaw

5. 如果启用了夹爪，确认：

.. code-block:: text

   A -> 关夹爪
   B -> 开夹爪

6. 松开 ``grip`` 会停止接管；重新按住 ``grip`` 时会重新记录 reference pose。



故障排查
----------------

**启动后没有 VR 数据**

- 确认 XRoboToolkit PC Service / runService 正在运行。
- 确认 VR publisher 进程没有退出。
- 先运行 ``test_pico_data.py``，确认 RLinf 侧能直接收到 PICO 数据。
- 确认 ``pico.zmq_addr`` 与 publisher 地址匹配。
- TCP 场景下确认可以从 RLinf env worker 节点访问 publisher IP 和端口。

**按住 grip 机械臂不动**

- 确认 ``use_spacemouse: False`` 且 ``use_pico: True``。
- 确认当前代码已经在 ``apply.py`` 中接入 ``PicoIntervention``。
- 确认 ``grip`` 数值超过 ``control_threshold``。
- 确认已经完成标定；若 ``calibration.required=True`` 且未标定，PICO 不会接管。
- 检查是否被 Franka 安全盒或 ``action_scale`` 限制。

**方向不对或启动瞬间跳变**

- 松开 ``grip``，回到舒适位置后重新扣下 ``trigger`` 标定。
- 重新按住 ``grip``，让系统记录新的 reference pose。
- 如果整体方向偏转，调整 ``operator_to_robot_yaw``，例如 ``1.5708``、
  ``3.1416`` 或 ``-1.5708``。

**夹爪没有反应**

- 确认 ``no_gripper: False``。
- 确认 ``override_cfg.end_effector_type`` 不是 ``none``。
- 确认环境 action space 包含第 7 维夹爪动作。
- 确认没有同时启用会覆盖夹爪动作的其他 wrapper。

**FrankaHW 初始化失败 / UDP timeout**

如果日志中出现：

.. code-block:: text

   FrankaHW: Failed to initialize libfranka robot. libfranka: UDP receive: Timeout

这通常不是 VR 数据问题，而是 Franka 控制节点没有和机器人建立 FCI UDP 通信。
请检查 ``robot_ip``、机器人是否进入可编程模式、控制节点网卡路由，以及是否有其他程序占用 Franka。
