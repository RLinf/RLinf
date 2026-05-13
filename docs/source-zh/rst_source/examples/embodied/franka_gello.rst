Franka 真机使用 GELLO 遥操作设备
====================================

本指南介绍如何在 RLinf 的 Franka 真实世界环境中配置和使用 **GELLO** 遥操作设备。
本文是基础 :doc:`franka` 文档的扩展，仅涵盖 GELLO 硬件所需的 **额外** 步骤。

.. note::

   如果你还没有阅读过基础的 Franka 指南，请先参考 :doc:`franka`。
   本页仅涉及 GELLO 硬件相关的额外配置。


硬件架构概览
-----------------

`GELLO <https://github.com/wuphilipp/gello_software>`_ 是一种关节级遥操作设备，
其运动学结构与 Franka 机械臂一致，操控比空间鼠标更直观、精确，并原生支持夹爪控制。

典型的 GELLO 部署方式是将设备连接到 **控制节点**
（通常为 NUC 或与机械臂物理连接的机器），通过 USB 串口适配器（FTDI）通信。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 节点
     - 角色
     - 硬件
   * - **GPU 服务器** (node 0)
     - Actor、rollout、env worker；相机采集
     - NVIDIA GPU（如 RTX 4090），RealSense 相机
   * - **NUC** (node 1)
     - FrankaController、GELLO 遥操作
     - Franka 机械臂、GELLO 设备（USB-FTDI）


GELLO 软件安装
------------------------------

GELLO 依赖两个软件包，必须 **按顺序** 安装：

1. ``gello`` — 来自 `gello_software <https://github.com/wuphilipp/gello_software>`_ 的底层驱动。
2. ``gello-teleop`` — RLinf 使用的正运动学和遥操作代理接口。

两个软件包都应安装在运行 GELLO 设备的节点上（通常为 NUC / 控制节点）。

1. 安装 ``gello`` （gello_software）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

选择一个安装目录，然后克隆仓库并**仅**初始化 **Dynamixel SDK** 子模块：

.. code-block:: bash

   cd /path/to/install/gello
   git clone https://github.com/wuphilipp/gello_software.git
   cd gello_software
   git submodule update --init third_party/DynamixelSDK

.. note::

   ``gello_software`` 还注册了 ``third_party/mujoco_menagerie``
   （一个体量较大的机器人 MJCF 资产仓库，仅被上游的 mujoco 演示脚本
   使用）。RLinf 的 GELLO 遥操作走 ``gello-teleop``，它自带 Franka
   的 MJCF，并不需要 menagerie 子模块。
   ``git submodule update --init <path>`` 只会注册并克隆指定的子模块；
   如果执行裸的 ``git submodule init``，则会把 menagerie 一并排入队列。

安装 ``gello`` 包和 **Dynamixel SDK** （作为第三方子模块）：

.. code-block:: bash

   pip install -e .
   pip install -e third_party/DynamixelSDK/python

Dynamixel SDK 用于与 GELLO 设备内部的 Dynamixel 舵机通信。
若缺少该依赖，``GelloAgent`` 将无法读取关节位置。

.. note::

   如果遇到串口访问权限问题，可将当前用户添加到 ``dialout`` 组：

   .. code-block:: bash

      sudo usermod -aG dialout $USER

   然后注销并重新登录以使更改生效。

有关更多硬件配置信息（如设置唯一的电机 ID、DynamixelRobotConfig 和端口映射），
请参考
`gello_software README <https://github.com/wuphilipp/gello_software#readme>`_。

2. 安装 ``gello-teleop``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``gello-teleop`` 封装了 ``gello`` 驱动，并集成了 Franka 正运动学
（使用 dm_control/MuJoCo）和遥操作代理接口。直接从 GitHub 仓库安装：

.. code-block:: bash

   pip install git+https://github.com/RLinf/gello-teleop.git

或者，如果你希望使用可编辑安装：

.. code-block:: bash

   git clone https://github.com/RLinf/gello-teleop.git
   cd gello-teleop
   pip install -e .

如果需要同时自动安装 ``gello`` 依赖（如果尚未单独安装）：

.. code-block:: bash

   pip install "gello-teleop[gello] @ git+https://github.com/RLinf/gello-teleop.git"


3. 配置串口设备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

将 GELLO 设备通过 USB-FTDI 适配器插入控制节点。识别串口设备：

.. code-block:: bash

   ls /dev/serial/by-id/
   # 查找类似如下路径：
   # usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

赋予权限：

.. code-block:: bash

   sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
   # 或者将用户添加到 dialout 组以获得持久访问权限：
   sudo usermod -aG dialout $USER

.. tip::

   建议使用 ``/dev/serial/by-id/`` 路径而非 ``/dev/ttyUSB*``，
   因为前者在重启和插拔其他 USB 设备后保持不变。

4. 验证 GELLO 设备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行内置测试脚本，确认 GELLO 设备通信正常并能产生有效的关节数据：

.. code-block:: bash

   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

你应该看到持续更新的输出，类似于：

.. code-block:: text

   pos=[0.500 0.000 0.300]  quat=[1.000 0.000 0.000 0.000]  gripper=[0.040]

如果在移动 GELLO 设备时输出数据持续更新，则说明安装成功。


YAML 配置说明
-------------------

要使用 GELLO 进行数据采集，请使用配置文件
``examples/embodiment/config/realworld_collect_data_gello.yaml``。
与标准空间鼠标配置的关键区别如下：

.. code-block:: yaml

   env:
     eval:
       use_spacemouse: False
       use_gello: True
       gello_port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"

.. list-table:: GELLO 相关配置字段
   :header-rows: 1
   :widths: 25 15 60

   * - 字段
     - 默认值
     - 说明
   * - ``use_gello``
     - ``False``
     - 启用 GELLO 遥操作。设置为 ``True`` 以使用 GELLO 替代空间鼠标。
   * - ``gello_port``
     - ``null``
     - GELLO 设备的串口路径。当 ``use_gello`` 为 ``True`` 时必须指定。
   * - ``use_spacemouse``
     - ``True``
     - 使用 GELLO 时必须设置为 ``False``。

有关完整的数据采集流程，请参考 :doc:`franka` 中的
**使用 GELLO 进行数据采集** 章节。


采集进度监视
-------------

采集脚本以 Ray Worker 运行，stdout 会被 Ray 的 log monitor 批量缓冲，
``tqdm`` 的 ``\r`` 原位刷新因此失效。要拿到实时进度条，请在
**另一个终端** 运行 ``toolkits/realworld_check/collect_monitor.py``：
它通过 tail 采集日志渲染一个 ``tqdm`` 进度条，显示成功计数、最近的
脚踏事件以及被丢弃的 episode。

.. code-block:: bash

   # 终端 1 —— 启动采集（stdout tee 到日志文件）
   bash examples/embodiment/collect_data.sh \
       realworld_collect_data_gello_joint_dual_franka 2>&1 \
       | tee logs/collect.log

   # 终端 2 —— 实时进度条（日志文件未出现时会等待）
   python toolkits/realworld_check/collect_monitor.py logs/collect.log

启动时 monitor 会回放已有日志，使进度条初始位置对齐到此前已保存的
episode 数；若希望直接从 EOF 开始 tail，请加 ``--no-replay``。


集群配置注意事项
---------------------

集群配置步骤与 :doc:`franka` 中描述的相同，主要额外要求如下：

- 在 **控制节点** (NUC) 上：确保在运行 ``ray start`` **之前** 已在虚拟环境中
  安装好 ``gello`` 和 ``gello-teleop``。

.. warning::

   请记住 Ray 会在 ``ray start`` 时捕获 Python 解释器和环境变量。任何在
   ``ray start`` **之后** 安装的软件包对 Ray worker 不可见。请务必先安装
   ``gello`` 和 ``gello-teleop``，然后再启动 Ray。


故障排查
----------------

**GELLO 设备未检测到**

- 确认 USB-FTDI 适配器已连接：``ls /dev/serial/by-id/``。
- 运行 ``lsusb`` 查看是否有 ``FTDI`` 设备。
- 确保 Dynamixel 舵机已上电（GELLO 设备的舵机需要外部供电）。

**串口权限被拒绝**

- 执行 ``sudo chmod 666 /dev/serial/by-id/<your-device>``。
- 或者将用户添加到 ``dialout`` 组：
  ``sudo usermod -aG dialout $USER`` （需要重新登录）。

**导入错误：``No module named 'gello'``**

- 确保 ``gello`` 包（来自 ``gello_software``）已安装在同一虚拟环境中。
  运行 ``pip show gello`` 进行验证。

**导入错误：``No module named 'gello_teleop'``**

- 确保 ``gello-teleop`` 已安装：运行 ``pip show gello-teleop`` 进行验证。
- 如果使用可编辑安装，请确保仓库路径正确。

**GELLO 读数不更新**

- 检查 Dynamixel 舵机 ID 是否与 ``gello_software`` 中的配置匹配。
- 如果通信不稳定，尝试降低 GELLO 配置中的波特率。
- 确保没有其他进程正在使用同一串口。
