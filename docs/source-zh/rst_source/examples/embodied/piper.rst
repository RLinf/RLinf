AgileX Piper 配置与控制
=======================

本文介绍如何配置 AgileX Piper 机械臂，并运行 RLinf 硬件测试脚本，读取反馈、检查关节运动与夹爪开合。

.. figure:: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
   :align: center
   :width: 70%

   Piper 机械臂与夹爪模型。图片来源：AgileX Robotics，piper_ros。

概览
----

测试在连接机械臂的 Linux 主机上运行，无需 GPU 或模型 checkpoint。RLinf 目前尚未提供适用于 AgileX Piper 的真机任务或训练流程；本指南仅介绍硬件检查。

硬件配置
--------

安装软件前，先准备机械臂及其控制主机。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 要求
   * - 机械臂与夹爪
     - AgileX Piper 及其电源；若安装夹爪，则使用 AgxGripper。
   * - 控制主机
     - Linux 计算机、处于 SocketCAN 模式的 USB-to-CAN 适配器，以及连接机械臂的 CAN 线缆。

将机械臂固定在稳定的台面上，并为小幅关节运动留出空间。上电前，按厂商说明连接电源与 CAN 线缆。

安装
----

在控制主机上按下方自定义环境流程安装机器人依赖。后续检查均在同一仓库目录和已激活的环境中运行。

机器人控制节点
~~~~~~~~~~~~~~

在连接机器人的主机上克隆 RLinf，然后安装对应环境。

.. include:: _setup_common.rst
   :end-before: 然后，使用

在仓库根目录安装机器人依赖：

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env piper
   source .venv/bin/activate
   export REPO_PATH="$PWD"

后续命令均在此仓库目录中运行，并保持环境已激活。安装脚本会安装 ``PiperArm`` 使用的固定版本 ``pyAgxArm`` 驱动；平台要求见 :doc:`安装 </rst_source/start/installation>`。

准备 CAN 连接
^^^^^^^^^^^^^

运行 ``ip -brief link`` 查找 SocketCAN 接口。处于 SocketCAN 模式的 USB-to-CAN 适配器通常显示为 ``can0``；如果名称不同，请替换下列命令中的接口名。连接机器人前，将接口配置为 1 Mbit/s：

.. code-block:: bash

   sudo ip link set can0 down
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set can0 up
   ip -details link show can0

输出中应显示接口已启用，且包含 ``bitrate 1000000``。机械臂上电后，运行 ``candump -n 5 can0`` 查看 5 帧反馈报文。如果一直没有报文，按 Ctrl+C 退出，检查电源、CAN 接线、适配器和波特率，再继续操作。SDK 不会配置 CAN 接口；适配器配置与硬件支持说明见 `pyAgxArm 文档 <https://github.com/agilexrobotics/pyAgxArm>`_。

运行
----

CAN 配置完成且机械臂上电后，先读取反馈：

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --read-only

命令打印 6 个关节角（单位为度）、工具位姿（位置单位为米，旋转为 xyzw 四元数）和范围为 ``[0, 1]`` 的夹爪开度，然后退出。此命令使能电机并读取反馈，不发送位置目标，也不复位机械臂。日志中会显示检测到的固件 profile。若 AgxGripper 行程为 0.1 m，添加 ``--gripper-max-width 0.1``；未安装夹爪时，添加 ``--no-gripper``。

去掉 ``--read-only`` 即可测试小幅运动。逐条输入下列指令，每次运动结束后再继续：

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --speed-percent 10

.. code-block:: text

   where
   joint 1 2
   where
   joint 1 -2
   grip 0.5
   where
   quit

``joint 1 2`` 以关节 1 的实测位置为起点，增加 2 度。脚本接受关节编号 1 至 6，单次增量不超过 5 度；机械臂驱动还会按关节行程裁剪目标。``grip 0.5`` 将夹爪目标设为完整行程的一半。``where`` 读取反馈，夹爪停止后可再次执行。``quit``、EOF 或 Ctrl+C 关闭 CAN 会话，电机继续保持位置。

.. warning::

   即使指定 ``--read-only``，连接时也会使能电机。连接前清空运动区域，执行关节与夹爪指令时保持手部远离机器人。退出测试不等同于急停，也不会切断电机电源。

在自己的程序中控制机械臂或接入相机，请参考 :doc:`机器人接口 </rst_source/concepts/robotics>`。

新增真机任务
------------

硬件检查通过后，参考 :doc:`新增真机任务 </rst_source/extending/new_task>` 定义任务的复位流程、观测、动作、奖励和成功条件。开始训练前，需要先实现可运行的真机任务。
