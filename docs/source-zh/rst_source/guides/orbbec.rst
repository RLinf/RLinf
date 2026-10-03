.. Copyright 2026 The RLinf Authors.
   Licensed under the Apache License, Version 2.0.

Orbbec 相机
===========

通过 RLinf 的相机接口读取 Orbbec USB 相机的彩色图像和对齐深度。先单独验证相机，再将同一 backend 接入机器人配置。该 backend 使用 Orbbec SDK v2，真机验证以 Femto Bolt 为目标；其他型号需要兼容 SDK v2 的固件及 stream profile。

安装与连接
----------

在运行相机的 Python 环境中安装 SDK。如果相机连接在另一个节点上，应安装到该节点的 worker 环境：

.. code-block:: bash

   python -m pip install pyorbbecsdk2==2.1.2

安装包名为 ``pyorbbecsdk2``，Python import 名为 ``pyorbbecsdk``。设备权限和平台要求见 `官方 Linux 安装指南 <https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html>`_。发现或连接相机时才会导入 SDK。

验证 Femto Bolt 深度时，连接 USB 3 数据线及 12 V 电源适配器。Linux 下可用 ``lsusb -t`` 查看实际连接速率：``5000M`` 表示 5 Gbit/s，``480M`` 表示 USB 2.0。使用相机机身上的序列号选择设备；重新插接后，USB 枚举顺序及 ``/dev/videoN`` 名称可能变化。

.. warning::

   同一台物理相机应由一个进程管理。打开 Orbbec 设备可能会初始化传感器，因此连接前应关闭占用该相机的其他应用。RLinf 只打开指定序列号的设备，不会改用另一台相机。

先读取彩色图像，再启用深度
--------------------------

先设置 ``enable_depth=False`` 验证彩色采集。通过 ``Camera.of(CameraInfo(...))`` 声明相机，再调用 ``connect()`` 打开设备；构造对象时不会访问硬件：

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

``get_observation()`` 返回新的相机观测，``disconnect()`` 停止采集并释放设备，之后可以再次连接。彩色采集正常后，将 descriptor 中的 ``enable_depth`` 改为 ``True``，观测中便会增加 ``observation["depth"]``：shape 为 ``(720, 1280)``、单位为米的 ``float32`` 数组。SDK 根据标定数据将深度像素对齐到彩色图像；深度值为零表示该像素没有有效测量。

``resolution`` 按 ``(width, height)`` 指定输出尺寸。驱动会在所需 ``fps`` 下选择最接近的原生分辨率，先将深度对齐到彩色图像，再将两者缩放到输出尺寸。彩色采用面积重采样，深度采用最近邻重采样，以保留测量值。如果输出与原生彩色图像的宽高比不同，图像会被拉伸。不支持的帧率会报错并列出可用 profile，不会静默改用其他帧率。

仅通过 USB 供电
---------------

只需要彩色图像时，保持 ``enable_depth=False``。驱动只启动彩色流，不启动深度流，也不修改激光或供电设置。

`Femto Bolt 数据手册 <https://www.orbbec.com/wp-content/uploads/2023/08/ORBBEC_Datasheet_Femto-Bolt-0816-v01-1.pdf>`_ 规定使用 12 V / 2 A DC 适配器，或提供 5 V / 3 A 的 Type-C 电源。仅通过 USB 供电时，深度与红外最高支持 640 × 576 的 Y16 模式，彩色最高支持 1920 × 1080 的 YUY2/MJPG 模式。USB 3 数据连接本身不能保证这一供电能力，其他 Orbbec 型号的供电要求也可能不同。

请求深度后，如果无法获得深度流，RLinf 会报错，不会静默切换成只有彩色图像的观测。启动失败或断流时，检查电源适配器、数据线、USB 带宽、设备权限，以及是否有其他进程占用相机。修复连接后重新连接相机。读取超时与重连沿用公共相机接口的行为，详见 :doc:`../concepts/robotics_architecture`。

接入机器人
----------

单独读取成功后，可按 :doc:`../concepts/robotics` 将相机与其他机器人零部件组合。若相机位于 worker 节点，在 ``Camera.of`` 中传入 ``node_rank``；该 worker 管理 SDK 和 USB 连接，返回的观测 key 和单位保持一致。

对于已有的 Franka 任务，修改所选 ``cluster.node_groups[].hardware.configs[]`` 条目中的相机字段：

.. code-block:: yaml

   camera_type: orbbec
   camera_serials: ["YOUR_CAMERA_SERIAL"]

这两个字段指定相机 backend 及机器人将打开的设备。需要深度时，在 Franka 环境的 ``init_params`` 中设置 ``enable_camera_depth: true``，默认值为 ``false``。Franka 环境通过观测中的 ``depths`` mapping 返回深度；policy 需要读取该 mapping 才能将深度用于学习。机械臂与任务设置继续沿用已有配置，完整启动流程见 :doc:`realworld_robot`。
