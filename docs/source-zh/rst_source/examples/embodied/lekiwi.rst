LeKiwi 移动底盘配置与控制
==========================

本文从拆箱开始，带你配置 LeKiwi 全向移动底盘与前置摄像头，直到跑起 ``LeKiwiDriveEnv-v1`` 环境。文中先安装环境，再为轮电机编号，用一条命令检查每个部件，最后给出把底盘交给策略或控制器的配置。

请按顺序完成这三项检查。每项只验证一个部件，因此后续步骤出错时，你已经知道车轮、摄像头和键盘都是好的。组装步骤见 `LeKiwi 组装指南 <https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md>`_；LeRobot 的 `LeKiwi 指南 <https://huggingface.co/docs/lerobot/en/lekiwi>`_ 介绍同一款机器人在其自身框架中的用法。

准备工作
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 要求
   * - 底盘
     - LeKiwi 移动底盘，包含三个 Feetech STS3215 轮电机、舵机控制板和套件电源。
   * - 摄像头
     - 安装在底盘前方的 UVC 摄像头，与舵机控制板连接到同一台计算机。
   * - 控制主机
     - 带 USB 接口的 Linux 计算机。常用选择是底盘上的树莓派，不需要 GPU。

各部分如何协作
--------------

底盘是三个互成 120 度的轮电机，挂在同一条串口总线上，硬件只认识每个轮子的速度。RLinf 把机体速度 ``[x, y, theta]``（单位分别为 m/s、m/s、deg/s）换算成各轮速度；当某个轮子超出上限时，所有轮子按同一比例缩小，因此达不到满速的方向会变慢，但行进方向不变。机器人层用底盘和摄像头组装 ``LeKiwiRobot``，环境层再把它包装成 ``LeKiwiDriveEnv-v1``。

安装环境
--------

以下步骤请在连接舵机控制板的计算机（例如树莓派 5）上执行，而不是你的笔记本。

.. include:: _setup_common.rst
   :end-before: 然后，使用

在仓库根目录安装机器人依赖：

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env lekiwi --install-rlinf --no-root
   source .venv/bin/activate

安装脚本包含 LeRobot 及 Feetech 支持和 OpenCV 的无界面（headless）构建，它只采集画面，不需要打开窗口所需的系统 GL 库。``--install-rlinf`` 会安装 RLinf 本身，下文使用的三个 ``rlinf-lekiwi-*`` 命令即由它提供；不加该参数时只安装依赖。``--no-root`` 会跳过安装脚本原本要用 sudo 安装的系统库：下文检查只用到串口和这个无界面构建，底盘计算机并不需要它们。若还要在这一台机器上训练或跑仿真，则不要加该参数。平台要求见 :doc:`安装 </rst_source/start/installation>`。

找到舵机串口
------------

每条命令都需要舵机控制板的串口。插上控制板、且该 USB 路径上没有其他设备时，运行：

.. code-block:: bash

   lerobot-find-port

按提示拔下控制板，再重新插上，它报告的设备就是下文要用的串口。本文示例假设为 ``/dev/ttyACM0``，macOS 上是 ``/dev/tty.usbmodem...`` 之类的路径。``/dev/serial/by-id/`` 下的名字在换 USB 口后仍然有效，底盘固定使用时不妨优先选它。后续检查一直使用这个串口。

设置轮电机 ID
-------------

全新的 STS3215 都回复同一个默认 ID，因此需要逐个编号。LeRobot 的配置命令会逐个提示，只接上它指定的那个轮电机：

.. code-block:: bash

   lerobot-setup-motors --robot.type=lekiwi --robot.port=/dev/ttyACM0

它写入左轮 7、后轮 8、右轮 9，与运行时驱动一致。车轮已编号的底盘可跳过此步。

检查车轮
--------

把底盘垫起使车轮悬空，接好三个电机后运行：

.. code-block:: bash

   rlinf-lekiwi-test-wheels --port /dev/ttyACM0

每个轮子转动半秒，命令报告编码器读回的速度：

.. code-block:: text

   INFO: Spinning the left wheel at raw 100
   INFO: left  commanded    100  measured     98  ok
   INFO: back  commanded    100  measured     97  ok
   INFO: right commanded    100  measured     99  ok

``--raw`` 让转动更快，``--seconds`` 让转动更久。

检查摄像头
----------

.. code-block:: bash

   rlinf-lekiwi-check-camera --save check.png

命令列出发现的摄像头、打开其中一个，并报告它实际送出的画面——因为摄像头可能打开成功却一帧都不给：

.. code-block:: text

   INFO: Webcams found: usb-USB_Camera_USB_Camera_2505280051-video-index0
   INFO: Using usb-USB_Camera_USB_Camera_2505280051-video-index0 at 640x480
   INFO: Captured 10 frames in 0.34 s (29.4 fps)
   INFO: Frame shape (480, 640, 3), dtype uint8, mean brightness 96.2

把打印出的标识符填入下文的 ``camera_names``。打开 ``check.png`` 查看机器人看到的画面，并在采集数据前调整安装角度。接有多个摄像头时，用 ``--device`` 指定。

用键盘遥控底盘
--------------

首次运行请让车轮悬空，并保证舵机电源触手可及。先在不驱动电机的情况下确认按键：

.. code-block:: bash

   rlinf-lekiwi-teleoperate --dry-run

然后正式遥控：

.. code-block:: bash

   rlinf-lekiwi-teleoperate --port /dev/ttyACM0

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 按键
     - 动作
   * - W / S
     - 前进 / 后退
   * - A / D
     - 左移 / 右移
   * - Z / X
     - 左转 / 右转
   * - R / F
     - 每次按下升高 / 降低一档速度
   * - Q / Esc / Ctrl+C
     - 退出，停车并松开力矩
   * - 松开移动键
     - 下发零速度

三档速度从最慢档开始：

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 档位
     - 平移
     - 转向
   * - 慢（初始）
     - 0.1 m/s
     - 30 deg/s
   * - 中
     - 0.2 m/s
     - 60 deg/s
   * - 快
     - 0.3 m/s
     - 90 deg/s

若轮半径或轮心到车体中心的距离不是默认的 0.05 与 0.125 米，用 ``--wheel-radius`` 和 ``--base-radius`` 指定；``--max-raw`` 可降低舵机上限，``--fps`` 改变控制频率。

退出、键盘输入丢失或出现异常时，命令会停车并松开每个轮子的力矩。USB 线被扯断或进程被杀死时，车轮仍可能继续转动：这时请使用舵机电源开关，并保证同一时间只有一个程序控制该串口。

键盘监听读取本机桌面输入，需要本地 X11 会话；它读不到 SSH 无头会话或 Wayland。改用另一台机器遥控：``--listen`` 在机器人上根据 TCP 收到的按键驱动车轮，``--remote HOST`` 在有键盘的机器上把按键发送过去。

.. code-block:: bash

   # 在机器人上，连接电机：
   rlinf-lekiwi-teleoperate --listen --port /dev/ttyACM0

   # 在接键盘的机器上：
   rlinf-lekiwi-teleoperate --remote <机器人 IP>

按键与速度档位完全相同。两端默认使用 TCP 端口 5555，可用 ``--remote-port`` 修改。``--listen`` 还接受 ``--fps``、``--wheel-radius``、``--base-radius``、``--max-raw`` 和 ``--dry-run``，``--connect-timeout`` 决定 ``--remote`` 持续重试多久。接收端监视发送端的心跳：发送端被杀、网络断开或笔记本休眠时，它会把车轮归零并等待新的发送端。用 Ctrl+C 停止发送端；接收端关闭时它也会退出。

驱动环境
--------

``LeKiwiDriveEnv-v1`` 是已注册的 Gymnasium 任务：按硬件配置组装整机，上报实测的机体速度与摄像头画面，并下发有界的速度指令。一个 step 保持该速度一个控制周期，随后停止车轮，因此只有调用方持续 step 时底盘才会移动。

与其他真机任务一样，在 env 配置文件中用 ``init_params.id`` 选择任务；``examples/embodiment/config/env/so101_reach.yaml`` 是一个这样配置的机械臂示例：

.. code-block:: yaml

   # examples/embodiment/config/env/lekiwi_drive.yaml
   env_type: real
   total_num_envs: null
   main_image_key: front

   init_params:
     id: "LeKiwiDriveEnv-v1"
     num_envs: null

串口、摄像头和轮子几何参数来自运行环境所在节点组的 hardware 配置：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: lekiwi
         node_ranks: 1
         hardware:
           type: LeKiwi
           configs:
             - port: /dev/ttyACM0
               camera_serials: ["<摄像头序列号>"]
               node_rank: 1

单次运行的参数写在训练配置的 ``override_cfg`` 下：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         camera_names:
           "<摄像头序列号>": front
         frame_size: [128, 128]
         max_linear_velocity: 0.5
         max_angular_velocity: 90.0
         step_frequency: 10.0

``camera_names`` 的键就是 ``rlinf-lekiwi-check-camera`` 打印的标识符。把摄像头命名为 ``front``，其画面就会出现在 ``main_image_key`` 指向的键下；未命名的摄像头以 ``camera_1`` 等上报。

``frame_size`` 决定观测中每帧的宽和高，默认 128×128 像素。策略通常需要小画面，而实时预览窗口需要更大的画面。摄像头驱动会向设备申请该尺寸，设备无法提供时在软件中缩放，因此请选择摄像头原生支持的分辨率。

奖励恒为零，回合也不会提前终止：任务成功与否由任务本身判定，参见 :doc:`新真机任务 </rst_source/extending/new_task>`。

在自己的程序中使用底盘
----------------------

要在实验之外使用底盘，直接用机器人层组装 ``LeKiwiRobot`` 并调用各部件；速度换算与限幅完全一致：

.. code-block:: python

   from rlinf.robotics.parts.cameras import CameraInfo
   from rlinf.robotics.robots import LeKiwiRobot

   robot = LeKiwiRobot.build(
       port="/dev/ttyACM0",
       node_rank=None,  # 在本进程中直接打开硬件。
       cameras={
           "front": CameraInfo(
               name="front",
               serial_number="<摄像头序列号>",
               camera_type="webcam",
               resolution=(320, 240),
           )
       },
   )
   robot.connect()
   try:
       robot.send_action({"base": {"velocity": [0.1, 0.0, 0.0]}})
   finally:
       robot.disconnect()

各部件的观测与动作契约见 :doc:`机器人接口 </rst_source/concepts/robotics>`。
