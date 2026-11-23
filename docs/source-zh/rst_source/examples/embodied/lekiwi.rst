LeKiwi 移动底盘
===============

从拆箱开始配置 LeKiwi 三全向轮移动底盘与前置 UVC 摄像头，按车轮、摄像头、键盘遥控的顺序逐项自检，直至运行 ``LeKiwiDriveEnv-v1`` 环境。

概览
----

按顺序完成车轮、摄像头和键盘三项检查，后续出错时可直接定位新引入的环节。底盘驱动将平面速度指令换算为三个 Feetech STS3215 轮电机速度，若任一车轮超出 ``max_raw`` 则按同比例整体降速，保持移动方向和转向比例不变。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      LeKiwi · 3 路 Feetech STS3215 全向轮

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      前置 UVC 摄像头 · 本地 / TCP 键盘

   .. grid-item-card:: 工作流
      :text-align: center

      电机编号 · 部件自检 · 遥控 · 驱动环境

   .. grid-item-card:: 硬件
      :text-align: center

      树莓派 / Linux · USB 舵机 · 无需 GPU

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 实测平面机体速度与 RGB 相机画面（默认按 ``640x480`` 采集，按目标宽高比居中裁剪后缩放到 ``frame_size``，默认 ``128x128``）。
   * - 动作
     - 平面底盘速度 ``[x, y, theta]``\ （前进 ``m/s``、横移 ``m/s``、偏航 ``deg/s``），按 ``step_frequency`` 保持一个控制周期后自动停车。
   * - 奖励
     - 恒为 ``0`` 且不提前终止，成功条件由调用方或自定义任务决定。

硬件与安全准备
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 组件
     - 要求
   * - 底盘与摄像头
     - LeKiwi 移动底盘（三个 Feetech STS3215 轮电机、舵机控制板与套件电源）及接在同一主机上的前置 UVC 摄像头。
   * - 控制主机
     - 带 USB 接口的 Linux 主机（常用树莓派 5），无需 GPU。
   * - 键盘主机（可选）
     - 同一局域网内的任意计算机（含 macOS），装有 Python 3.11+ 与 ``pynput``，无需安装 RLinf。

机械组装见 `LeKiwi 组装指南 <https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md>`__，上游框架用法见 LeRobot `LeKiwi 指南 <https://huggingface.co/docs/lerobot/en/lekiwi>`__。

.. warning::

   首次车轮自检与键盘遥控时请垫起底盘使三轮悬空，并确保舵机电源开关触手可及。同一时间只允许一个程序占用舵机串口。正常退出、输入超时或异常时会停车并松开力矩，但 USB 断线或进程被强杀仍可能导致车轮继续转动，遇此情况请立即切断舵机电源。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与硬件自检

      **1. 在底盘控制主机上安装 RLinf**

      在连接舵机控制板的主机（例如树莓派 5）上执行。``--no-root`` 跳过需 sudo 安装的系统 GL 库，因为无界面 OpenCV 仅用于采帧（若同机还需训练或运行模拟器则去掉 ``--no-root``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env lekiwi --install-rlinf --no-root && source .venv/bin/activate

      **2. 查找串口并为轮电机编号**

      查找舵机控制板串口（示例假设为 ``/dev/ttyACM0``，固定底盘优先使用 ``/dev/serial/by-id/...``），随后逐个为全新 STS3215 电机编号（写入左轮 7、后轮 8、右轮 9；已编号可跳过）：

      .. code-block:: bash

         lerobot-find-port
         lerobot-setup-motors --robot.type=lekiwi --robot.port=/dev/ttyACM0

      **3. 检查车轮**

      垫起底盘悬空车轮并接好三个电机，以原始指令 100 逐轮转动 0.6 秒（可用 ``--raw`` 和 ``--seconds`` 调整）：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_lekiwi_env wheels --port /dev/ttyACM0

      预期输出：

      .. code-block:: text

         Spinning the left wheel at raw 100
         left  commanded    100  measured     98  ok
         back  commanded    100  measured     97  ok
         right commanded    100  measured     99  ok

      **4. 检查前置摄像头**

      验证摄像头能正常送帧并查看 ``check.png``\ （多摄像头用 ``--device`` 指定）：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_lekiwi_env camera --save check.png

      预期输出（将打印出的标识符填入下方 ``camera_names``）：

      .. code-block:: text

         Webcams found: /dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0
         Using /dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0 at 640x480
         Captured 10 frames in 0.34 s (29.4 fps)
         Frame shape (480, 640, 3), dtype uint8, mean brightness 96.2

   .. tab-item:: 2. 键盘遥控与环境运行

      **1. 测试按键并遥控底盘**

      首次测试保持车轮悬空，先运行 ``--dry-run`` 确认按键捕获，再驱动电机（几何尺寸或上限不同时可用 ``--wheel-radius`` 0.05、``--base-radius`` 0.125、``--max-raw`` 与 ``--fps`` 覆盖）：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_lekiwi_env teleop --dry-run
         python -m toolkits.realworld_check.test_lekiwi_env teleop --port /dev/ttyACM0

      .. list-table::
         :header-rows: 1
         :widths: 35 65

         * - 按键
           - 动作
         * - ``W / S`` · ``A / D`` · ``Z / X``
           - 前进 / 后退 · 左移 / 右移 · 左转 / 右转
         * - ``R / F``
           - 在慢（0.1 m/s、30 deg/s）、中（0.2 m/s、60 deg/s）、快（0.3 m/s、90 deg/s）三档间升降速
         * - ``Q / Esc / Ctrl+C`` · 松开移动键
           - 退出、停车并松开力矩 · 下发零速度

      .. dropdown:: 跨主机键盘遥控（无头 SSH / Wayland / macOS）

         本机键盘监听依赖桌面 X11 会话。在无头 SSH、Wayland 或另一台主机遥控时采用 TCP 模式（先启接收端；默认端口 5555 可用 ``--remote-port`` 修改，``--fps`` 默认 30 Hz，支持 ``--connect-timeout`` 与默认 2 秒的 ``--input-timeout``）：

         .. code-block:: bash

            # 机器人端（接收端）：
            python -m toolkits.realworld_check.test_lekiwi_env teleop --listen --port /dev/ttyACM0

            # 键盘主机端（替换为机器人 IP）：
            python -m toolkits.realworld_check.test_lekiwi_env teleop --remote 192.168.1.100

         在单独的笔记本或 macOS（``install.sh`` 不支持 macOS）上，只需在 Python 3.11+ 虚拟环境中安装 ``pynput`` 即可运行发送端（macOS 须在“系统设置 > 隐私与安全性”中为终端开启“辅助功能”和“输入监控”并重启终端，启用电机前先用接收端 ``--dry-run`` 验证）：

         .. code-block:: bash

            git clone https://github.com/RLinf/RLinf.git && cd RLinf
            pip install pynput
            python -m toolkits.realworld_check.test_lekiwi_env teleop --remote 192.168.1.100

      **2. 配置 LeKiwiDriveEnv-v1 环境**

      使用 ``env_type: real`` 选择 ``LeKiwiDriveEnv-v1``\ （结构同 ``examples/embodiment/config/env/so101_reach.yaml``），声明 ``hardware.type: LeKiwi``，并将摄像头标识符映射为 ``front`` 以匹配 ``main_image_key``：

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: lekiwi
               node_ranks: 1
               hardware:
                 type: LeKiwi
                 configs:
                   - serial_port: /dev/ttyACM0
                     camera_serials: ["/dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0"]
                     node_rank: 1

         env:
           eval:
             env_type: real
             main_image_key: front
             init_params:
               id: "LeKiwiDriveEnv-v1"
             override_cfg:
               camera_names:
                 "/dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0": front
               frame_size: [128, 128]
               max_linear_velocity: 0.5
               max_angular_velocity: 90.0
               step_frequency: 10.0

      .. dropdown:: RPent 部署（可选）

         RPent 直接运行 ``LeKiwiDriveEnv-v1``，因此 ``examples/embodiment`` 下不提供训练配置。环境服务、机器人配置与规划器用法见 `RPent LeKiwi 指南 <https://github.com/RLinf/RPent/blob/main/docs/source-zh/rst_source/real_world_robots/lekiwi.rst>`__。

结果与常见问题
--------------

三轮自检均输出 ``ok``、``camera --save check.png`` 成功采帧且亮度非零、``teleop --dry-run`` 正常响应按键，即表示底盘软硬件就绪。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 现象
     - 解决方法
   * - SSH 或 Wayland 下本机 ``teleop`` 收不到按键
     - 机器人端运行 ``teleop --listen``，在桌面主机运行 ``teleop --remote <机器人 IP>``。
   * - macOS 远程发送端无法捕获按键
     - 在“系统设置 > 隐私与安全性”中为终端开启“辅助功能”与“输入监控”，重启终端并先用 ``--dry-run`` 验证。
   * - 摄像头画面上报在 ``camera_1`` 而非 ``front``
     - 将 ``test_lekiwi_env camera`` 打印的完整标识符在 ``override_cfg.camera_names`` 中映射为 ``front``。
   * - 接收端中途自动停车退出
     - 断线或超过 ``--input-timeout``\ （默认 2 秒）无输入会自动停车退出，重新启动两端命令即可开始新会话。

各零部件的观测与动作契约见 :doc:`机器人接口 <../../concepts/robotics>`；如需定义任务奖励与终止条件，参见 :doc:`新增真机任务 <../../extending/new_task>`。
