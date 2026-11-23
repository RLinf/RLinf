Franka GELLO 遥操作
===================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gello.jpeg
   :align: center
   :width: 80%

   用于采集 Franka 示教数据的 GELLO 关节级遥操作设备。

概览
----

使用 GELLO 主臂替代 SpaceMouse，以关节级遥操作采集 Franka 示教数据。在控制节点上按顺序安装 ``gello`` 与 ``gello-teleop``，验证 USB-FTDI 串口数据流，更新采集配置，并通过日志监控脚本实时查看采集进度。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · GELLO 主臂 · 夹爪

   .. grid-item-card:: 连接方式
      :text-align: center

      USB-FTDI 串口\ （``/dev/serial/by-id/``）

   .. grid-item-card:: 用途
      :text-align: center

      关节级遥操作数据采集

   .. grid-item-card:: 前置依赖
      :text-align: center

      :doc:`franka` · 舵机外部供电 · ``dialout`` 权限

硬件与安全准备
--------------

先为 GELLO 内部的 Dynamixel 舵机接通外部供电，再通过 USB-FTDI 适配器将设备连接到 Franka 控制节点（通常为 NUC）。推荐使用 ``/dev/serial/by-id/`` 下的固定路径，避免重启或插拔其他 USB 设备后 ``/dev/ttyUSB*`` 编号漂移：

.. code-block:: bash

   ls /dev/serial/by-id/
   # 示例输出：
   # usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

为当前会话赋予串口读写权限，或将当前用户加入 ``dialout`` 组以获得持久权限（加入后需注销并重新登录）：

.. code-block:: bash

   sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
   # 或持久授权：
   sudo usermod -aG dialout $USER

电机 ID、舵机参数和端口映射等硬件配置详见 `gello_software README <https://github.com/wuphilipp/gello_software#readme>`_。

.. warning::

   启动遥操作前，请将 Franka 急停按钮放在伸手可及处并清空工作空间障碍物。先将 GELLO 主臂摆到接近 Franka 安全初始姿态的位置，防止接管瞬间机械臂跳变；同时确认 Franka Desk 处于可编程模式，且没有其他进程占用机械臂控制权。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Franka + GELLO

      **1. 按顺序安装 RLinf Franka 与 GELLO 依赖**

      在控制节点安装基础 ``franka`` 环境，随后在 RLinf 仓库根目录依次克隆并安装 ``gello_software``\ （仅初始化 Dynamixel SDK 子模块）和 ``gello-teleop``：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

         git clone https://github.com/wuphilipp/gello_software.git ../gello_software
         git -C ../gello_software submodule update --init third_party/DynamixelSDK
         uv pip install -e ../gello_software -e ../gello_software/third_party/DynamixelSDK/python

         git clone https://github.com/RLinf/gello-teleop.git ../gello-teleop
         uv pip install -e ../gello-teleop

      .. note::

         ``git -C ../gello_software submodule update --init third_party/DynamixelSDK`` 只拉取 Dynamixel SDK，跳过体积较大的 ``third_party/mujoco_menagerie``；``gello-teleop`` 已自带 Franka MJCF。

      **2. 验证串口与实时读数**

      在 RLinf 仓库根目录运行内置检测脚本，传入你的 ``/dev/serial/by-id/`` 路径：

      .. code-block:: bash

         python -m rlinf.robotics.parts.teleop.gello \
             --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

      移动 GELLO 主臂，确认终端持续刷新末端位姿与夹爪读数：

      .. code-block:: text

         pos=[0.500 0.000 0.300]  quat=[1.000 0.000 0.000 0.000]  gripper=[0.040]

      **3. 配置并启动采集**

      在 ``examples/embodiment/config/realworld_collect_data_gello.yaml`` 中填写 ``robot_ip`` 和 GELLO 串口路径：

      .. code-block:: yaml

         env:
           eval:
             teleop: gello
             gello_port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"

      由于采集器以 Ray worker 运行，stdout 会被 Ray 批量缓冲导致 ``tqdm`` 原位刷新失效。在终端 1 启动采集并将日志写入文件，在终端 2 运行监控脚本查看实时进度条：

      .. code-block:: bash

         # 终端 1 —— 启动单臂采集（双臂关节采集使用 realworld_collect_data_gello_joint_dual_franka）
         bash examples/embodiment/collect_data.sh \
             realworld_collect_data_gello 2>&1 \
             | tee logs/collect.log

         # 终端 2 —— 实时进度条（传 --no-replay 可直接从日志末尾开始监听）
         python -m toolkits.realworld_check.collect_monitor logs/collect.log

.. dropdown:: GELLO 配置项与双臂说明

   .. list-table::
      :header-rows: 1
      :widths: 25 15 60

      * - 字段
        - 默认值
        - 说明
      * - ``env.eval.teleop``
        - ``spacemouse``
        - 设为 ``gello`` 后，由 GELLO 主臂驱动从臂。
      * - ``env.eval.gello_port``
        - ``null``
        - GELLO 设备的串口路径，``teleop`` 为 ``gello`` 时必填。

   双臂关节采集请向 ``examples/embodiment/collect_data.sh`` 传入 ``realworld_collect_data_gello_joint_dual_franka``。完整的数据采集流程和导出格式见 :doc:`franka`。

.. dropdown:: 集群配置注意事项

   多机部署按 :doc:`franka` 配置 Ray 集群，但须注意顺序：在控制节点（NUC）上执行 ``ray start`` 前，先在虚拟环境中安装 ``gello`` 和 ``gello-teleop``。

   .. warning::

      Ray 会在 ``ray start`` 时捕获 Python 解释器和环境变量，之后安装的包对 Ray worker 不可见。务必先安装 ``gello`` 和 ``gello-teleop``，再启动 Ray。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 解决方法
   * - 未检测到 GELLO 设备
     - 运行 ``ls /dev/serial/by-id/`` 与 ``lsusb`` 确认存在 ``FTDI`` 设备，并检查 Dynamixel 舵机的外部电源是否已打开。
   * - 串口权限被拒绝
     - 执行 ``sudo chmod 666 /dev/serial/by-id/<your-device>``，或运行 ``sudo usermod -aG dialout $USER`` 后注销重登。
   * - 报错 ``No module named 'gello'``
     - 运行 ``pip show gello`` 确认 ``gello_software`` 已装入当前虚拟环境（多机部署须重启 Ray）。
   * - 报错 ``No module named 'gello_teleop'``
     - 运行 ``pip show gello-teleop`` 检查安装状态，确认可编辑安装指向的仓库路径有效。
   * - GELLO 读数不更新
     - 核对 Dynamixel 舵机 ID 是否与 ``gello_software`` 配置一致，通信不稳定时降低波特率，并关闭占用同一串口的其他进程。
