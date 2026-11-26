Franka PICO VR 遥操作
=====================

概览
----

使用 PICO VR 头显与手柄遥操作 Franka 真机，用于示教采集和训练过程中的人工接管。外部 VR publisher 通过 XRoboToolkit 读取头显与手柄位姿，并经 ZeroMQ 推送给 RLinf env worker，将手柄相对运动映射为归一化末端增量动作。开始前请先按 :doc:`franka` 完成基础 Franka 控制、相机与 Ray 集群配置。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · PICO 头显与手柄

   .. grid-item-card:: 连接方式
      :text-align: center

      ZeroMQ\ （同机 IPC / 跨机 TCP）

   .. grid-item-card:: 用途
      :text-align: center

      VR 遥操作采集与人工接管

   .. grid-item-card:: 前置依赖
      :text-align: center

      :doc:`franka` · XRoboToolkit · Ubuntu 22.04/24.04

硬件与安全准备
--------------

VR 链路最多涉及三类节点角色。publisher 与 RLinf env worker 在同一台机器时可用 IPC 地址，跨机器部署时在同一局域网内使用 TCP 地址。

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 节点
     - 角色
     - 硬件 / 软件
   * - **GPU 服务器**\ （node 0）
     - Actor、rollout、可选相机采集
     - NVIDIA GPU、RLinf
   * - **Franka 控制节点**\ （node 1 或单节点）
     - Franka 机械臂、env worker、VR 数据订阅
     - Franky、``pyzmq``
   * - **VR / PICO PC**
     - 运行 XRoboToolkit PC Service 与 VR 数据发布进程
     - PICO 头显、手柄、``pico_software``

.. warning::

   将 Franka 急停按钮放在伸手可及处，首次上真机先清空工作台物体。首次测试请把 ``position_scale`` 调低到 ``0.3``–``0.5``，确认工作空间安全盒、单步 ``action_scale`` 和 Franka Desk 状态后再放置任务物体。如果控制方向明显不对，先松开 ``grip``，重新站位并扣下 ``trigger`` 标定，切勿在按住 ``grip`` 接管时强行纠正。每次修改 Ray 环境、Python 依赖、机器人配置或 ZeroMQ 地址后，先执行 ``ray stop`` 再重启。

.. _pico-software-setup:

在 PICO 头显和 publisher 机器上部署 XRoboToolkit，启动 PC Service 并确认 XRT 数据流连通后再运行 ZeroMQ 桥接：

.. dropdown:: 安装 PICO APK、XRoboToolkit PC Service 并验证 XRT 链路

   1. 在 PICO 头显端，从 `XRoboToolkit-Unity-Client releases <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases>`_ 安装适配版本的 APK。
   2. 在 publisher 机器上（官方 release 覆盖 Ubuntu 22.04 和 Ubuntu 24.04，其他系统需自行适配），从 `XRoboToolkit-PC-Service releases <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases>`_ 安装对应 PC Service。确认该机器与控制节点在同一网络，或与 RLinf env worker 共用主机。
   3. 连接头显前，先启动 XRoboToolkit PC Service：

      .. code-block:: bash

         (cd /opt/apps/roboticsservice && bash runService.sh)

   4. 在 PICO 头显端连接该服务，确认头显和手柄均已连接，且位姿与按键数据持续刷新。
   5. 克隆 ``pico_software``\ （内含官方 `XR-Robotics/XRoboToolkit-Teleop-Sample-Python <https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python>`_ 模块），安装环境并验证 XRT 链路：

      .. code-block:: bash

         git clone https://github.com/tiny-xie/pico_software.git && cd pico_software
         (cd XRoboToolkit-Teleop-Sample-Python && bash setup_uv.sh)
         source XRoboToolkit-Teleop-Sample-Python/.venv/bin/activate
         python test_pico_xrt_pipeline.py

      移动头显和手柄后，终端应持续输出变化的数据流。

.. _pico-publisher-setup:

在 ``pico_software/configs/vr_bridge.yaml`` 中配置 publisher 侧的 ZeroMQ 监听地址，然后启动 VR 数据发布进程：

.. dropdown:: 配置并启动 ZeroMQ VR 数据 publisher

   同机部署时绑定 IPC：

   .. code-block:: yaml

      zmq:
        ipc_addr: "ipc:///tmp/vr_data.ipc"

   跨机器部署时绑定 TCP：

   .. code-block:: yaml

      zmq:
        ipc_addr: "tcp://0.0.0.0:<port>"

   在 VR / PICO PC 的 ``pico_software`` 目录下激活前面创建的虚拟环境并启动 publisher：

   .. code-block:: bash

      source XRoboToolkit-Teleop-Sample-Python/.venv/bin/activate
      python -m vr_data_publisher --config configs/vr_bridge.yaml

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与 VR 桥接

      在 Franka 控制节点安装 ``franka`` 环境（已包含基础 Franka 真机依赖与 ``pyzmq``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      .. warning::

         必须在执行 ``ray start`` 之前完成安装并激活环境。Ray 会在启动时捕获 Python 解释器和环境变量，之后再配置 ``pyzmq`` 或 Franka 环境会导致 worker 无法读取控制器或连接 ZeroMQ。

      启动 VR 数据 publisher 后，在读取控制器的节点运行内置脚本验证 ZeroMQ 数据流：

      .. code-block:: bash

         # 同机 IPC
         python -m toolkits.realworld_check.test_pico_data \
             --zmq-addr ipc:///tmp/vr_data.ipc

         # 跨机 TCP（替换为实际 publisher IP:端口，与 env.eval.pico.zmq_addr 保持一致）
         python -m toolkits.realworld_check.test_pico_data \
             --zmq-addr tcp://192.168.1.100:5555

      移动手柄并按下 ``grip``、``trigger``、``A`` 或 ``B``，确认终端持续刷新如下输出：

      .. code-block:: text

         [000012] recv_rate=79.8Hz | headset: pos=[0.120 1.430 0.210] quat=[0.000 0.707 0.000 0.707] | right: pos=[0.320 1.120 0.850] quat=[0.010 0.690 0.020 0.724] grip=0.000 trigger=0.000 | buttons_active=none

   .. tab-item:: 2. 配置与遥操作运行

      编辑 ``examples/embodiment/config/realworld_collect_data_pico.yaml``，将 ``env.eval.pico.zmq_addr`` 与 publisher 地址对齐（同机填 ``ipc:///tmp/vr_data.ipc``，跨机填 ``tcp://<vr_publisher_ip>:<port>``，不要填 ``tcp://0.0.0.0:<port>``）：

      .. code-block:: yaml

         env:
           eval:
             teleop: pico
             pico:
               zmq_addr: "tcp://<vr_publisher_ip>:<port>"
               position_scale: 1.0
               rotation_scale: 1.0
               calibration:
                 button: "trigger"

      **启动顺序**

      1. 在 Franka 控制节点激活 ``franka`` 环境（``source .venv/bin/activate``）。
      2. 多机运行时按 :doc:`franka` 启动 Ray 集群（单机运行会自动启动本地 Ray，无需手动执行 ``ray start``）。
      3. 启动 XRoboToolkit PC Service，确认头显与手柄已连接。
      4. 启动 ``vr_data_publisher``。
      5. 首次运行或更改 ZeroMQ 地址后，先用 ``test_pico_data.py`` 验证数据可达。
      6. 在 Ray head 节点的 RLinf 仓库根目录启动采集脚本：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data_pico

      .. warning::

         同一时间只允许一个程序持有 Franka 控制权。运行 RLinf 采集时，不要同时运行 ``robo_avatar.slave.main``、其他 ``franka_control_node`` 或另一个 RLinf 真机任务。

      **VR 手柄操作映射**

      面向工作台正前方站好，首次测试先不放任务物体，只做小幅运动验证：

      .. list-table::
         :header-rows: 1
         :widths: 34 66

         * - 手柄操作
           - 机械臂响应
         * - 扣下 ``trigger``
           - 按当前站姿标定 PICO base 坐标系。
         * - 按住 / 松开 ``grip``
           - 按住瞬间记录手柄与当前 TCP reference pose 并开始接管；松开停止接管，再次按住重新记录 reference pose。
         * - 手柄前推 / 后拉
           - 末端 ``+X`` / ``-X``。
         * - 手柄左移 / 右移
           - 末端 ``+/-Y``。
         * - 手柄上抬 / 下压
           - 末端 ``+/-Z``。
         * - 旋转手柄
           - 末端 roll / pitch / yaw。
         * - 按下 ``A`` / ``B``
           - 关夹爪 / 开夹爪（启用夹爪控制时）。

.. dropdown:: 机械臂运动参数

   采集示例沿用 Franky 默认运动参数，单次调用的目标变化上限为 3 cm 和 0.10 rad，无需为 PICO 另加 ``compliance``。数据采集和策略执行共用这些默认值，完整参数、reset 行为及硬件覆盖方法见 :ref:`配置机械臂运动 <franka-motion-settings>`。

   ``pico.position_scale`` 和 ``pico.rotation_scale`` 控制手柄运动映射为请求运动的比例，只作用于 PICO 输入（包含训练期间的人工接管）。环境先应用动作缩放和工作空间边界，再下发给 Franky。

.. dropdown:: 夹爪配置

   ``no_gripper: true`` 时，PICO 只控制 6D 末端运动。若需要通过手柄控制夹爪，先确认环境动作空间包含夹爪维度，再配置：

   .. code-block:: yaml

      env:
        eval:
          no_gripper: False
          pico:
            gripper_close_button: "A"
            gripper_open_button: "B"
            gripper_close_threshold: 0.5

   ``A`` 键闭合夹爪，``B`` 键打开夹爪。两个按键均未按下时，夹爪动作输出 ``0.0``，松开按键不会自动张开夹爪。

.. dropdown:: 集群配置注意事项与完整 YAML 参考

   - 读取控制器的节点必须能访问 ``env.eval.pico.zmq_addr``。
   - 使用 TCP 时，确认防火墙与网卡路由放行对应端口；使用 IPC 时，publisher 与 RLinf env worker 必须在同一台主机。
   - 保持 ``RLINF_NODE_RANK``、YAML 中的 ``cluster.node_groups[*].node_ranks`` 以及 Franka 硬件配置中的 ``node_rank`` 一致。
   - 以 ``examples/embodiment/config/realworld_collect_data_pico.yaml`` 为起点，按现场硬件替换 ``robot_ip``、``camera_serials``、``target_ee_pose`` 与 ``env.eval.pico.zmq_addr``。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 解决方法
   * - 启动后没有 VR 数据
     - 确认 XRoboToolkit PC Service\ （``runService.sh``）和 ``vr_data_publisher`` 都在运行，先用 ``test_pico_data.py`` 验证接收，核对 ``pico.zmq_addr`` 与 publisher 地址一致，并检查跨机 TCP 连通性。
   * - 按住 ``grip`` 机械臂不动
     - 确认环境配置已设 ``teleop: pico``、``grip`` 读数超过 ``control_threshold``、已完成 ``trigger`` 标定（``calibration.required=True`` 时必需），并检查是否触达 Franka 安全盒或 ``action_scale`` 限制。
   * - 方向不对或启动瞬间跳变
     - 松开 ``grip``，回到舒适站姿后重新扣下 ``trigger`` 标定，再按住 ``grip`` 记录新 reference pose。整体坐标系偏转时，调整 ``operator_to_robot_yaw``\ （例如 ``1.5708``、``3.1416`` 或 ``-1.5708``）。
   * - 夹爪没有反应
     - 确认 ``no_gripper: False``、环境动作空间包含第 7 维夹爪动作，且没有其他 wrapper 覆盖夹爪输出。
   * - ``FrankaHW: Failed to initialize libfranka robot. libfranka: UDP receive: Timeout``
     - 控制节点未能与机械臂建立 FCI UDP 通信。检查 ``robot_ip``、Franka Desk 是否处于可编程模式、控制节点网卡路由，以及是否有其他程序占用 Franka。
