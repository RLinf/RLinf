AgileX Piper 配置与控制
=======================

.. figure:: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
   :align: center
   :width: 70%

   Piper 机械臂与夹爪模型。图片来源：AgileX Robotics，piper_ros。

通过 Linux SocketCAN 配置 AgileX Piper 6 自由度机械臂，并使用 RLinf 硬件测试脚本读取反馈、验证关节与夹爪运动。

概览
----

本页覆盖与机械臂相连的 Linux 控制主机上的硬件配置与运动自检，无需 GPU 或模型 checkpoint；RLinf 尚未内置 Piper 训练任务。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      AgileX Piper 6 自由度 · AgxGripper

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      关节与 TCP 反馈 · 可选相机

   .. grid-item-card:: 工作流
      :text-align: center

      CAN 配置 · 只读探测 · 关节与夹爪检查

   .. grid-item-card:: 硬件
      :text-align: center

      Linux 主机 · USB-to-CAN（SocketCAN）· 无需 GPU

硬件与安全准备
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 组件
     - 要求
   * - 机械臂与夹爪
     - AgileX Piper 及其电源；如安装夹爪，使用 AgxGripper。
   * - 控制主机
     - Linux 计算机、处于 SocketCAN 模式的 USB-to-CAN 适配器及连接机械臂的 CAN 线缆。

将机械臂固定在稳固台面上，为小幅关节运动预留空间；上电前按厂商说明接好电源和 CAN 线缆。

.. warning::

   即使传入 ``--read-only``，建立连接时也会使能电机。连接前请清空运动区域，执行关节与夹爪指令时保持手部远离机器人。退出测试脚本不等同于急停，不会切断电机电源。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: AgileX Piper

      **1. 安装 RLinf**

      在与机械臂相连的控制主机上安装 Piper 运行环境（包含固定版本 ``pyAgxArm`` 驱动）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env piper && source .venv/bin/activate

      **2. 启动 SocketCAN 接口**

      运行 ``ip -brief link`` 查找接口名（通常为 ``can0``，如不同请替换），以 1 Mbit/s 启动，并在机械臂上电后检查 5 帧反馈报文：

      .. code-block:: bash

         sudo ip link set can0 down
         sudo ip link set can0 type can bitrate 1000000
         sudo ip link set can0 up
         ip -details link show can0
         candump -n 5 can0

      ``ip`` 输出应显示接口已启用且包含 ``bitrate 1000000``。SDK 不会配置 CAN 接口，适配器设置见 `pyAgxArm 文档 <https://github.com/agilexrobotics/pyAgxArm>`__。

      **3. 读取反馈并检查关节与夹爪**

      先以只读模式读取反馈（打印 6 个关节角、米制位置与 xyzw 四元数工具位姿、``[0, 1]`` 夹爪开度及检测到的固件 profile，不发送运动指令或复位）：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_piper_controller \
           --channel can0 --read-only

      若 AgxGripper 行程为 0.1 m，追加 ``--gripper-max-width 0.1``；未安装夹爪时追加 ``--no-gripper``。

      确认反馈正常后，去掉 ``--read-only`` 以 10% 速度测试小幅运动：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_piper_controller \
           --channel can0 --speed-percent 10

      逐条输入以下指令，每次运动完成后再继续：

      .. code-block:: text

         where
         joint 1 2
         where
         joint 1 -2
         grip 0.5
         where
         quit

      ``joint 1 2`` 将关节 1 增加 2 度（支持关节 1–6，单次增量不超过 5 度并受行程限位裁剪）；``grip 0.5`` 移动到半开行程；``where`` 读取最新反馈；``quit``、EOF 或 Ctrl+C 关闭 CAN 会话，电机继续保持位置。

结果与常见问题
--------------

``candump -n 5 can0`` 正常打印 5 帧报文、``--read-only`` 返回实时关节与夹爪读数且 ``where`` 反映 2 度步进与夹爪运动，即说明硬件链路就绪。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 现象
     - 解决方法
   * - ``candump -n 5 can0`` 阻塞且无报文
     - 按 Ctrl+C 退出，检查机械臂电源、CAN 接线、``ip -brief link`` 接口名和 ``bitrate 1000000``。
   * - 夹爪开度比例不符或未装夹爪
     - 0.1 m 行程的 AgxGripper 传入 ``--gripper-max-width 0.1``，裸臂传入 ``--no-gripper``。
   * - 执行 ``quit`` 后电机仍保持力矩
     - 属正常行为，如需断电请使用机械臂硬件电源或急停开关。

如需在自有代码中控制 Piper 或接入相机，参见 :doc:`机器人接口 <../../concepts/robotics>`；开始训练前构建真机任务，参见 :doc:`新增真机任务 <../../extending/new_task>`。
