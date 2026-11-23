GimArm 在线强化学习
===================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png
   :align: center
   :width: 80%

   用于 SocketCAN 控制、电机校准和插孔训练的 GimArm 6 自由度机械臂。

通过 Linux SocketCAN 将 GimArm 6 自由度机械臂接入 RLinf，完成电机零位校准、目标位姿采集、VR 遥操作数据采集和真机插孔训练。

概览
----

本页介绍双节点部署方案：控制主机通过 CAN FD 直连机械臂，GPU 服务器运行 SAC / RLPD 训练与 rollout。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      GimArm 6 自由度 · 平行夹爪

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      RealSense（可选）· PICO 4 Ultra VR

   .. grid-item-card:: 工作流
      :text-align: center

      CAN 初始化 · 零位校准 · VR 遥操作 · SAC / RLPD

   .. grid-item-card:: 硬件
      :text-align: center

      SocketCAN FD · 控制主机 · GPU 节点

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - 任务
     - 配置 / 入口
     - 说明
   * - 硬件检查
     - ``test_gim_arm_env.py``
     - 验证控制器启动、状态读取、关节运动、复位与夹爪开合。
   * - 数据采集
     - ``gim_arm_teleop``
     - 使用 PICO 4 Ultra 头显采集 VR 遥操作示教。
   * - 真机训练
     - GimArm 真机配置
     - 以目标位姿奖励训练 CNN 插孔 policy。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 可选 RealSense 相机画面，以及 GimArm 关节与 TCP 状态。
   * - 动作
     - 按配置输出 6 自由度机械臂控制与夹爪指令。
   * - 奖励
     - 到 ``target_ee_pose`` 的稀疏或稠密正运动学距离。
   * - 指令
     - 环境配置中的插孔任务描述文本。

硬件与安全准备
--------------

将 GimArm 通过 Linux SocketCAN FD 适配器直连到控制主机（标准波特率 1 Mbps，数据波特率 5 Mbps），并保证控制主机与 GPU 训练节点处于同一局域网。

.. warning::

   上电和运动测试前请确保急停开关触手可及并清空工作空间。电机零位校准必须在机械臂处于机械零位时执行，错误的零位会导致机械臂异常运动。系统重启后须重新初始化 CAN 接口。

.. dropdown:: 手动配置 CAN FD 与单电机校零

   如果不使用 ``gim_arm_control`` 自带的 ``sh/init_can.sh``，可以手动启用 ``can0``：

   .. code-block:: bash

      sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
      sudo ip link set can0 txqueuelen 1000
      sudo ip link set can0 up

   电机校零依赖 ``can-utils``\ （执行 ``sudo apt install can-utils`` 安装）。按十六进制 CAN ID 单独校准某台达妙电机（而非全部 ``001``–``008``）：

   .. code-block:: bash

      bash ~/gim_arm_control/sh/set_zero.sh can0 001

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 机器人控制节点

      **1. 安装 RLinf 与 gim_arm_control SDK**

      安装 GimArm 环境后，编译安装底层 C++ CAN 驱动与 Python 绑定（需 ``scikit-build-core>=0.5``、``nanobind>=2.0`` 和 C++17 编译器，自动安装 ``numpy`` 与 ``pinocchio``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env gim_arm && source .venv/bin/activate

         git clone https://github.com/RLinf/gim_arm_control.git ~/gim_arm_control
         uv pip install -e ~/gim_arm_control/python

      若环境需要兼容 NumPy 1.x，改用 ``uv pip install -e "~/gim_arm_control/python[pin270]"``。

      **2. 初始化 CAN FD 并校准电机零位**

      运行 ``gim_arm_control`` 中的脚本启用 ``can0``，并确认接口已启动：

      .. code-block:: bash

         bash ~/gim_arm_control/sh/init_can.sh can0
         ip link show can0

      首次使用或更换电机后，将机械臂摆回机械零位，再对全部达妙电机（``001``–``008``）校准零点：

      .. code-block:: bash

         bash ~/gim_arm_control/sh/set_zero.sh can0 --all

      **3. 运行硬件检查并获取目标位姿**

      运行测试脚本，验证控制器启动、在线状态、状态向量形状、关节运动、复位与夹爪开合：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_gim_arm_env --can can0 --variant gim_arm_xl

      脚本会对当前关节角做正运动学计算，并打印 TCP 位置与四元数。手动将机械臂移到插孔目标位姿，记录输出并将四元数转为 Euler XYZ 弧度，填入 ``target_ee_pose``。

      .. dropdown:: VR 遥操作数据采集（可选）

         推荐使用 Ubuntu 22.04（x86）控制主机与同一局域网下的 PICO 4 Ultra 头显，配合 `gim_arm_teleop <https://github.com/RLinf/gim_arm_teleop>`__ 采集示教（如需头显 VST 画面，先开启相机权限）：

         1. 安装并启动 `XRoboToolkit-PC-Service <https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.1.1/XRoboToolkit-PC-Service-1.1.1.deb>`__：

            .. code-block:: bash

               sudo dpkg -i XRoboToolkit-PC-Service-1.1.1.deb
               bash /opt/apps/roboticsservice/runService.sh

         2. 在头显安装 `XRoboToolkit-PICO-1.1.1.apk <https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk>`__，连接控制主机 IP，并按需勾选 ``head``、``hand`` 与 ``controller``。

         3. 按 `gim_arm_teleop <https://github.com/RLinf/gim_arm_teleop>`__ 说明配置环境后，启动硬件遥操作脚本同步记录 VR 指令与机械臂状态：

            .. code-block:: bash

               cd gim_arm_teleop
               python scripts/hardware/teleop_gim_arm_hardware.py

   .. tab-item:: 训练与 Rollout 节点

      **1. 安装 RLinf**

      在 GPU 节点安装 GimArm 环境（如需特定 VLA 模型依赖，追加 ``--model <model>``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env gim_arm && source .venv/bin/activate

      **2. 修改硬件与目标位姿配置**

      在 GimArm 训练 YAML 中填入硬件参数和任务位姿：

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: gim_arm
               node_ranks: 1
               hardware:
                 type: GimArm
                 configs:
                   - can_interface: can0
                     arm_variant: gim_arm_xl
                     camera_serials: ["YOUR_CAMERA_SERIAL"]  # 纯状态测试可设为 []
                     node_rank: 1

         env:
           train:
             override_cfg:
               target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]  # [x, y, z, rx, ry, rz]（米 / 弧度）
               reset_joint_qpos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
               safe_retract_qpos: [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
               is_dummy: false

      - ``target_ee_pose`` 为插孔目标末端位姿（米与 Euler XYZ 弧度）。
      - ``reset_joint_qpos`` 和 ``safe_retract_qpos`` 分别设置回合起始关节角与复位安全回缩关节角。
      - ``is_dummy: true`` 可在无硬件环境下测试流程；``camera_serials: []`` 跳过相机，观测中的 ``frames`` 为空字典。

      .. dropdown:: 完整配置参考（可选）

         .. code-block:: yaml

            cluster:
              num_nodes: 2
              component_placement:
                actor:
                  node_group: "4090"
                  placement: 0
                env:
                  node_group: gim_arm
                  placement: 0
                rollout:
                  node_group: "4090"
                  placement: 0
              node_groups:
                - label: "4090"
                  node_ranks: 0
                - label: gim_arm
                  node_ranks: 1
                  hardware:
                    type: GimArm
                    configs:
                      - can_interface: can0
                        arm_variant: gim_arm_xl
                        camera_serials: ["YOUR_CAMERA_SERIAL"]  # 无可用相机时设为 []
                        camera_type: realsense
                        enable_gripper: true
                        gripper_type: parallel
                        node_rank: 1

            env:
              train:
                override_cfg:
                  target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
                  reset_joint_qpos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                  safe_retract_qpos: [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
                  is_dummy: false
              eval:
                override_cfg:
                  target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]

      **3. 启动 Ray 集群并开始训练**

      ``examples/embodiment/config/`` 尚未内置独立的 GimArm 训练配置，可基于现有真机 SAC/RLPD 配置（如 ``realworld_peginsertion_rlpd_cnn_async.yaml``）替换上方的硬件与环境配置块。按 :doc:`../../guides/multi_node` 启动双节点 Ray 集群（GPU 节点为 head rank ``0``，控制节点为 worker rank ``1``），再在 head 节点启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh <your_config>

结果与常见问题
--------------

在 Ray head 节点运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once``、``env/return`` 与 SAC 损失，指标说明见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 现象
     - 解决方法
   * - 重启后 CAN 接口丢失或未启用
     - 重新运行 ``bash ~/gim_arm_control/sh/init_can.sh can0``，并用 ``ip link show can0`` 确认状态为 ``UP``。
   * - 更换电机后机械臂运动异常
     - 将机械臂摆回机械零位，再运行 ``bash ~/gim_arm_control/sh/set_zero.sh can0 --all`` 重新校零。
   * - ``pinocchio`` 或 NumPy 1.x 版本冲突
     - 使用 ``uv pip install -e "~/gim_arm_control/python[pin270]"`` 重装 SDK 绑定。
   * - 尚未接入 RealSense 相机
     - 在硬件配置中将 ``camera_serials`` 设为 ``[]``，以纯状态模式运行。

如需在 GimArm 上开发自定义真机任务，参见 :doc:`新增真机任务 <../../extending/new_task>`。
