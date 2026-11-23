.. Copyright 2026 The RLinf Authors.
   Licensed under the Apache License, Version 2.0.

Franka Orbbec 相机
==================

概览
----

通过 RLinf 相机接口读取 Orbbec USB 相机的彩色图像和对齐深度，并将同一 backend 接入 Franka 配置。该 backend 基于 Orbbec SDK v2，真机验证目标为 Femto Bolt；其他 Orbbec 型号需使用兼容 SDK v2 的固件与 stream profile。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 硬件
      :text-align: center

      Orbbec Femto Bolt\ （SDK v2 USB）

   .. grid-item-card:: 连接方式
      :text-align: center

      USB 3.0\ （5 Gbit/s）+ 12 V / 2 A 电源

   .. grid-item-card:: 用途
      :text-align: center

      Franka RGB 与对齐深度采集

   .. grid-item-card:: 前置依赖
      :text-align: center

      :doc:`franka` · ``pyorbbecsdk2==2.1.2``

硬件与安全准备
--------------

测试 Femto Bolt 深度时，连接 USB 3 数据线以及 12 V / 2 A DC 电源适配器（或提供 5 V / 3 A 的 Type-C 电源）。在 Linux 下运行 ``lsusb -t`` 查看协商速率：``5000M`` 表示 5 Gbit/s USB 3，``480M`` 表示 USB 2.0。始终通过 Orbbec SDK 查询到的设备序列号选择相机，因为重新插接后 USB 枚举顺序与 ``/dev/videoN`` 名称可能变化。

.. note::

   **仅通过 USB 供电。** 只需要彩色图像时保持 ``enable_depth=False``，驱动只启动彩色流，不修改激光或供电设置。按 `Femto Bolt 数据手册 <https://www.orbbec.com/wp-content/uploads/2023/08/ORBBEC_Datasheet_Femto-Bolt-0816-v01-1.pdf>`_，仅靠 USB 供电时深度/红外最高支持 640 × 576\ （Y16），彩色最高支持 1920 × 1080\ （YUY2/MJPG），且普通 USB 3 数据口不一定满足 5 V / 3 A 供电。启用深度后，若无法获取深度流，RLinf 会直接报错而不会静默退回纯彩色模式。其他 Orbbec 型号的供电要求请参考各自手册。

.. warning::

   每台物理相机只应由一个进程管理。打开 Orbbec 设备会初始化传感器，连接前请关闭占用该相机的其他应用。RLinf 只打开配置中指定的序列号，不会自动替换成另一台相机。在 Franka 机械臂附近安装相机时，请理顺线缆避开工作空间，并将急停按钮放在伸手可及处。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Orbbec SDK 安装与验证

      **1. 安装 RLinf Franka 环境与 Orbbec SDK v2**

      在连接 USB 相机的主机上安装 ``franka`` 环境（跨机部署时安装在承载该相机的 worker 节点），再用 ``--no-deps`` 安装 ``pyorbbecsdk2==2.1.2``，避免引入示例依赖（Open3D、pygame、PyAV、pynput）及冲突的 NumPy 版本约束（RLinf 机器人环境已包含所需的 NumPy 和 OpenCV）。包名为 ``pyorbbecsdk2``，Python 导入名为 ``pyorbbecsdk``；设备 udev 权限与平台要求见 `官方 Linux 安装指南 <https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html>`_：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate
         uv pip install --no-deps pyorbbecsdk2==2.1.2

      **2. 查询相机序列号**

      在物理连接相机的主机上运行以下脚本，打印每台已连接 USB 相机的 SDK 序列号（不启动采集；多台相机可逐台接入记录）：

      .. code-block:: bash

         python - <<'PY'
         import pyorbbecsdk as ob

         context = ob.Context()
         context.enable_net_device_enumeration(False)
         devices = context.query_devices()
         for index in range(devices.get_count()):
             print(devices.get_device_serial_number_by_index(index))
         PY

      将下方示例中的 ``YOUR_CAMERA_SERIAL`` 替换为查询结果。

      **3. 验证彩色图像与对齐深度**

      先设置 ``enable_depth=False`` 验证彩色采集（声明相机描述符时不会访问硬件，连接后才开始采集）：

      .. code-block:: python

         from rlinf.robotics import Camera, CameraInfo

         camera = Camera.of(
             CameraInfo(
                 name="scene",
                 serial_number="YOUR_CAMERA_SERIAL",
                 camera_type="orbbec",
                 resolution=(1280, 720),
                 fps=15,
                 enable_depth=False,
             )
         )
         try:
             camera.connect()
             observation = camera.get_observation()
             image = observation["frame"]  # (720, 1280, 3), BGR uint8
         finally:
             camera.disconnect()

      彩色采集正常后，将 ``enable_depth`` 改为 ``True`` 即可同时获取 ``observation["depth"]``：shape 为 ``(720, 1280)``、单位为米的 ``float32`` 数组，由 SDK 标定对齐到彩色图像（0 表示无效深度）。断开连接后会释放设备，可再次连接。

      **4. 接入 Franka 配置**

      修改 ``cluster.node_groups[].hardware.configs[]`` 中的相机字段：

      .. code-block:: yaml

         camera_type: orbbec
         camera_serials: ["YOUR_CAMERA_SERIAL"]

      需要深度时，在 Franka 环境的 ``init_params`` 中设置 ``enable_camera_depth: true``\ （默认 ``false``），环境会通过观测中的 ``depths`` mapping 暴露对齐深度供 policy 读取。若需与其他机器人零部件组合或通过 ``node_rank`` 部署到远程 worker，见 :doc:`../../concepts/robotics`；完整启动流程见 :doc:`../../guides/realworld_robot`。

.. dropdown:: 流分辨率、对齐与重采样说明

   - ``resolution`` 按 ``(width, height)`` 指定输出尺寸。驱动在所需 ``fps`` 下选择最接近的原生彩色分辨率，深度采用该 ``fps`` 下的设备默认分辨率（Femto Bolt 为 640 × 576）。
   - 深度先对齐到彩色图像，再与彩色一起缩放到 ``resolution``：彩色使用面积重采样，深度使用最近邻重采样以保留实测数值。
   - 连接时日志会记录所选原生 profile。输出宽高比与原生彩色流不一致时图像会被拉伸并输出警告；不支持的帧率会报错并列出可用 profile。
   - 读取超时与重连沿用公共相机接口行为，详见 :doc:`../../concepts/robotics_architecture`。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 解决方法
   * - 查询脚本无序列号输出
     - 检查 USB 3 数据线并用 ``lsusb -t`` 确认设备在线，按 `官方 Linux 安装指南 <https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html>`_ 配置 udev 设备权限。
   * - 深度启动失败或断流
     - 接通 12 V / 2 A DC 电源（或 5 V / 3 A Type-C 供电），检查 ``lsusb -t`` 为 ``5000M`` 且 USB 带宽充足，关闭占用相机的进程后重连。
   * - 设备忙或传感器初始化失败
     - 关闭所有正在访问该 Orbbec 相机的其他程序，确保每台物理相机仅由单个进程独占。
   * - 报错帧率不支持或图像被拉伸
     - 按报错列出的原生 profile 调整 ``fps``，并将 ``resolution`` 宽高比与原生彩色流对齐。
