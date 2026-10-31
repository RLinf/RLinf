LeKiwi 移动底盘配置与控制
==========================

本文从拆箱开始，带你配置 LeKiwi 全向移动底盘与前置摄像头，直到跑起 ``LeKiwiDriveEnv-v1`` 环境。文中先安装环境，再为轮电机编号，用一条命令检查每个部件，最后给出把底盘交给策略或控制器的配置。

请按顺序完成这三项检查。每项只验证一个部件，因此后续步骤出错时，你已经知道车轮、摄像头和键盘都是好的。

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
     - 可选。同一网络内的任意计算机（macOS 也可），只需一个装好 RLinf 运行时依赖和 pynput 的 Python 环境，无需安装 RLinf。

本文不介绍组装：步骤见 `LeKiwi 组装指南 <https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md>`_，同一款机器人在其自身框架中的用法见 LeRobot 的 `LeKiwi 指南 <https://huggingface.co/docs/lerobot/en/lekiwi>`_。

各部分如何协作
--------------

底盘是三个互成 120 度的轮电机，挂在同一条串口总线上，硬件只认识每个轮子的速度。RLinf 把机体速度 ``[x, y, theta]``（单位分别为 m/s、m/s、deg/s）换算成各轮速度；当某个轮子超出上限时，所有轮子按同一比例缩小，因此达不到满速的方向会变慢，但行进方向不变。机器人层用底盘和摄像头组装 ``LeKiwiRobot``，环境层再把它包装成 ``LeKiwiDriveEnv-v1``。各部件的观测与动作契约见 :doc:`机器人接口 </rst_source/concepts/robotics>`。

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

按上文安装好的机器已经带有 pynput，它由 LeRobot 引入。若一台机器只有一个含 RLinf 运行时依赖（``torch`` 和 ``omegaconf``，例如现成的 RLinf 虚拟环境）的 Python 环境，则装好 pynput 后直接在仓库里运行发送端即可，既不需要安装脚本，也不需要安装 RLinf。macOS 走这条路，因为 ``install.sh`` 不支持 macOS。

.. code-block:: bash

   cd <RLinf 仓库路径>
   pip install pynput
   PYTHONPATH=$PWD python -m toolkits.lekiwi.teleop --remote <机器人 IP>

最后一条命令与 ``rlinf-lekiwi-teleoperate`` 脚本执行的是同一个。macOS 还需在“系统设置 > 隐私与安全性”中为终端打开“辅助功能”和“输入监控”，然后重启终端；否则发送端能连上并持续发送心跳，但按键不会到达机器人。

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

RPent 部署
----------

基于 RLinf 构建的智能体框架 RPent 通过其 LeKiwi 扩展控制这台底盘：机器人计算机用 HTTP 提供环境服务，智能体计算机运行规划器，由规划器调用 ``drive(velocity, duration_s)``、``stop()`` 和只读的 ``view_env_state(step)``。机器人计算机上不运行 Ray 集群。

在两台机器上都安装带 ``lekiwi`` 可选依赖的 RPent。机器人计算机直接装进上面建好的环境，不必再建一个：``lekiwi`` 是唯一不引入 RLinf 的可选依赖，因此从源码安装的 RLinf 保持生效，无需重装。

.. code-block:: bash

   # 机器人计算机，在上面建好的环境中：
   source .venv/bin/activate
   uv pip install -e '/path/to/RPent[lekiwi]'

   # 智能体计算机，在 RPent 仓库中；它不需要 RLinf：
   uv pip install -e '/path/to/RPent[lekiwi]'

``uv pip check`` 会报告安装过程中改动的共享依赖。之后再运行 ``install.sh`` 会以 ``uv sync`` 结束，它会移除 RLinf 未声明的包（包括 RPent），因此在重装 RLinf 之后需要再装一次 RPent。可选依赖要加引号：未加引号的 ``[lekiwi]`` 会被 zsh 当作通配符展开。

用检测脚本按硬件生成站点 YAML，其中的舵机串口和摄像头标识符就是上面检查得到的值；该脚本不会让底盘运动：

.. code-block:: bash

   python -m robots.lekiwi.detect --output /path/to/lekiwi-site.yaml

在机器人计算机上启动环境服务，然后在智能体计算机上运行任务或 Dashboard：

.. code-block:: bash

   # 机器人计算机：
   python -m robots.lekiwi.env_server \
     --robot-config /path/to/lekiwi-site.yaml \
     --transport http --host 0.0.0.0 --port 8120

   # 智能体计算机，运行任务 0（drive_smoke_test），使用 API 规划器：
   rpent --robot lekiwi --task-id 0 \
     --env-endpoint http://<机器人 IP>:8120 \
     --robot-config /path/to/lekiwi-site.yaml \
     --planner api --model <模型 ID>

   # 智能体计算机，在浏览器中手动驾驶：
   rpent --robot lekiwi --dashboard --planner manual \
     --env-endpoint http://<机器人 IP>:8120 \
     --robot-config /path/to/lekiwi-site.yaml

任务 1 ``room_survey`` 在房间里分段行驶并报告摄像头看到的画面。站点 YAML 各字段、Dashboard 用法和故障排查见 `RPent LeKiwi 说明 <https://github.com/RLinf/RPent/blob/main/docs/source-zh/rst_source/real_world_robots/lekiwi.rst>`_。
