LeKiwi 移动底盘配置与控制
==========================

本文从拆箱开始，带你配置 LeKiwi 全向移动底盘与前置摄像头，再到运行 ``LeKiwiDriveEnv-v1`` 环境。文中先安装环境，再为轮电机编号，用一条命令检查每个零部件，最后给出把底盘交给 policy 或控制器的配置。

请按顺序完成这三项检查。每项只验证一个零部件，因此后续步骤出错时，你已经知道车轮、摄像头和键盘均工作正常。

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
   * - 键盘主机
     - 可选。同一网络内的任意计算机（macOS 也可），只需一个装好 ``torch``、``omegaconf``、``numpy`` 和 ``pynput`` 的 Python 3.11+ 环境，无需安装 RLinf。

本文不介绍组装：步骤见 `LeKiwi 组装指南 <https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md>`_，同一款机器人在其自身框架中的用法见 LeRobot 的 `LeKiwi 指南 <https://huggingface.co/docs/lerobot/en/lekiwi>`_。

各部分如何协作
--------------

LeKiwi 底盘由三个工作在速度模式下的 Feetech STS3215 舵机驱动全向轮。policy 输出平面运动指令 ``[x, y, theta]``\ （前进速度 ``m/s``、横移速度 ``m/s``、偏航角速度 ``deg/s``），而舵机控制器接收的是每个车轮的原始速度指令。

``LeKiwiBase`` 零部件负责衔接这两层：通过平面运动学将 ``[x, y, theta]`` 映射为三个车轮的速度；如果任一车轮的计算速度超出 ``max_raw``，它会按相同比例缩放三个车轮的速度，使底盘整体降速但保持原有的移动与转向比例不变。``LeKiwiRobot`` 将该底盘与 ``WebcamCamera`` 摄像头零部件组合为整机，``LeKiwiDriveEnv-v1`` 再将其封装为 Gymnasium 环境，使 policy 能以 ``step_frequency`` 的频率接收图像与底盘速度观测并下发 ``[x, y, theta]`` 动作。各零部件的观测与动作契约见 :doc:`机器人接口 </rst_source/concepts/robotics>`。

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

安装脚本包含 LeRobot 及 Feetech 支持和 OpenCV 的无界面（headless）构建，它只采集画面，不需要打开窗口所需的系统 GL 库。``--no-root`` 会跳过安装脚本原本要用 sudo 安装的系统库：下文检查只用到串口和这个无界面构建，底盘计算机并不需要它们。若还要在这一台机器上训练或跑仿真，则不要加该参数。``--install-rlinf`` 会安装 RLinf 本身。平台要求见 :doc:`安装 </rst_source/start/installation>`。请激活该虚拟环境，并在仓库根目录下使用 ``toolkits/realworld_check/test_lekiwi_env.py`` 运行下文的所有检查命令。

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

   python -m toolkits.realworld_check.test_lekiwi_env wheels --port /dev/ttyACM0

每个车轮以原始速度指令 100 转动 0.6 秒后停下，命令报告编码器读回的速度：

.. code-block:: text

   INFO: Spinning the left wheel at raw 100
   INFO: left  commanded    100  measured     98  ok
   INFO: back  commanded    100  measured     97  ok
   INFO: right commanded    100  measured     99  ok

``--raw`` 让转动更快，``--seconds`` 让转动更久。

检查摄像头
----------

.. code-block:: bash

   python -m toolkits.realworld_check.test_lekiwi_env camera --save check.png

命令列出发现的摄像头、打开其中一个，并报告它实际送出的画面，避免摄像头虽然成功打开设备但无法返回图像帧：

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

   python -m toolkits.realworld_check.test_lekiwi_env teleop --dry-run

然后正式遥控：

.. code-block:: bash

   python -m toolkits.realworld_check.test_lekiwi_env teleop --port /dev/ttyACM0

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
   python -m toolkits.realworld_check.test_lekiwi_env teleop --listen --port /dev/ttyACM0

   # 在接键盘的机器上：
   python -m toolkits.realworld_check.test_lekiwi_env teleop --remote <机器人 IP>

按键与速度档位完全相同。两端默认使用 TCP 端口 5555，可用 ``--remote-port`` 修改。``--listen`` 还接受 ``--fps``、``--wheel-radius``、``--base-radius``、``--max-raw`` 和 ``--dry-run``，``--connect-timeout`` 决定 ``--remote`` 持续重试多久。接收端监视发送端的心跳：发送端被杀、网络断开或笔记本休眠时，它会把车轮归零并等待新的发送端。用 Ctrl+C 停止发送端；接收端关闭时它也会退出。

按上文安装好的机器已经带有 ``pynput``，该包会随 LeRobot 一起安装。在单独的笔记本或 macOS（``install.sh`` 不支持 macOS）上，只需准备一个 Python 3.11+ 虚拟环境，安装运行时依赖与 ``pynput`` 后直接在仓库里运行发送端即可，无需运行安装脚本或安装 RLinf：

.. code-block:: bash

   cd /path/to/RLinf
   pip install torch omegaconf numpy pynput
   PYTHONPATH=$PWD python -m toolkits.realworld_check.test_lekiwi_env teleop --remote <机器人 IP>

macOS 还需在「系统设置 > 隐私与安全性」中为终端打开「辅助功能」和「输入监控」，然后重启终端；否则发送端能连上并持续发送心跳，但按键不会到达机器人。

驱动环境
--------

``LeKiwiDriveEnv-v1`` 是已注册的 Gymnasium 任务：按硬件配置组装整机，上报实测的机体速度与摄像头画面，并下发有界的速度指令。一个 step 保持该速度一个控制周期，随后停止车轮，因此只有调用方持续 step 时底盘才会移动。

``examples/embodiment`` 下没有 LeKiwi 的配置文件：这个环境由 RPent 承载，而不是由一次训练运行承载。由自己承载该任务的配置与其他真机任务一样，用 ``init_params.id`` 选择它；``examples/embodiment/config/env/so101_reach.yaml`` 是同样写法的机械臂示例：

.. code-block:: yaml

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

``camera_names`` 的键就是 ``python -m toolkits.realworld_check.test_lekiwi_env camera`` 打印的标识符。把摄像头命名为 ``front``，其画面就会出现在 ``main_image_key`` 指向的键下；未命名的摄像头以 ``camera_1`` 等上报。

``frame_size`` 决定观测中每帧的宽和高，默认 128×128 像素。policy 通常需要小画面，而实时预览窗口需要更大的画面。驱动默认向设备申请 ``640x480`` 采集分辨率，并保留设备实际返回的画面尺寸；环境层先按目标宽高比居中裁剪，再缩放到观测尺寸，避免拉伸物体。

奖励恒为零，回合也不会提前终止：任务成功与否由任务本身判定，参见 :doc:`新增真机任务 </rst_source/extending/new_task>`。

RPent 部署
----------

智能体控制与部署步骤见 `RPent LeKiwi 指南 <https://github.com/RLinf/RPent/blob/main/docs/source-zh/rst_source/real_world_robots/lekiwi.rst>`_，其中包含环境服务、机器人配置与规划器的使用方法。
