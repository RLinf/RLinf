UR5e 配置与控制
================

本文介绍如何在同一控制主机上配置 UR5e、通过 USB-RS485 连接的 Robotiq 2F-85、一台固定在环境中的 Orbbec Femto Bolt 相机和 SpaceMouse。使用相机的 RGB 图像完成驱动与遥操作检查后，将演示保存为 LeRobot 数据集，微调 π₀.₅，再使用相同的观测和笛卡尔增量动作部署 policy。

概览
----

``UR5e`` 机器人包含使用 ``ur_rtde`` 后端的机械臂、可选的独立 Robotiq 夹爪，以及显式配置的相机。机械臂动作是在机器人基座坐标系中表示的绝对 TCP 位姿，由本地循环通过 SDK 的 ``servoL`` 位置控制器发送。采集环境将较低频率的 SpaceMouse 增量转换为这些绝对目标。

驱动使用控制器当前配置的 TCP 和负载。安装工具后，请先在 PolyScope 中完成对应设置。本指南中的控制器仿真使用 PolyScope 5.26.3 的 URSim。

硬件配置
--------

打开 RLinf 连接前，先准备可访问的控制器。初次检查使用 URSim；连接实机时，按厂商流程配置机械臂。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 要求
   * - 控制器
     - UR5e，或模型设置为 UR5 的 URSim e-Series。控制会话需要 Remote Control 和 NORMAL / RUNNING 状态；仅接收状态时，需要开启且可访问的 RTDE。
   * - 网络服务
     - 在 PolyScope 中开启 Dashboard（29999）、Primary / Secondary（30001 / 30002）、Real-time（30003）和 RTDE（30004），并确保驱动能够访问这些端口。SDK 通过这些控制器接口建立连接并运行控制脚本。
   * - 控制主机
     - Linux 计算机，具有 RLinf Python 环境，能够通过网络访问机器人控制器。
   * - 采集设备
     - 连接到同一 Linux 计算机的 SpaceMouse、实体键盘或 a/b/c 踏板、相机，以及通过 USB-RS485 连接的 Robotiq 2F-85。

按 `ur_rtde 控制器配置说明 <https://sdurobotics.gitlab.io/ur_rtde/pages/getting_started/environment_setup.html>`_ 开启服务和 Remote Control。连接前，按现有控制器流程处理上电、刹车与安全状态恢复。

安装
----

按 :doc:`安装 </rst_source/start/installation>` 准备平台依赖并克隆仓库。在仓库根目录安装并激活 UR5e 与 OpenPI 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env ur5e --venv .venv-ur5e
   source .venv-ur5e/bin/activate
   export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

该流程安装 OpenPI，以及 ``ur_rtde``、Orbbec 采集、SpaceMouse、串口 Modbus、键盘输入和 LeRobot 采集依赖。后续命令均使用这个已激活的环境。

Orbbec SDK 的安装包为 ``pyorbbecsdk2==2.1.2``，通过 ``pyorbbecsdk`` 导入。设备权限和图像流检查见下文。

运行
----

组合采集设备前，先检查驱动。使用仿真时，将 URSim e-Series 设置为 UR5；本集成使用 ``universalrobots/ursim_e-series:5.26.3`` 验证。按 `官方 URSim Docker 说明 <https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/master/doc/setup/ursim_docker.rst>`_ 配置网络，使驱动所在的计算机或容器能够访问控制器端口。将下列地址替换为你的控制器地址。

可以先用 fake SDK 检查机器人注册和观测 shape：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5e --mock \
     --arg robot_ip=192.168.56.101 --arg node_rank=0

成功时输出以 ``OK: every part connected, observed what it declares, and let go.`` 结束。随后使用纯接收模式读取控制器，不启动控制脚本：

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --receive-only --frequency 10 --samples 10

命令记录十个反馈样本后断开连接。输出的 UR 原生 TCP 位姿为 ``[x, y, z, rx, ry, rz]``，姿态使用旋转向量，单位为弧度。要获取复位和工作空间配置采用的 xyz Euler 表示，运行：

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --read-only

记录 ``TCP pose (m, xyz Euler rad)`` 行用于配置。此模式启动 SDK 控制脚本，打印实测状态后断开连接，不发送位置目标；需要控制器处于 Remote Control 和 NORMAL / RUNNING 状态。

.. warning::

   驱动应独占控制器。打开控制会话会启动 SDK 的 URScript 程序，断开连接会停止该程序。连接前，在 PolyScope 中处理上电、刹车与安全状态恢复。

在 URSim 中交互检查运动时，去掉 ``--read-only``，逐条输入：

.. code-block:: text

   where
   offset 1 0 0
   offset -1 0 0
   quit

每条 offset 的单位为毫米，采用基座坐标系并保持姿态，在一秒内插值发送，长度最多为 5 mm。等待运动稳定后再输入下一条命令。``quit``、EOF 和 Ctrl-C 关闭会话。连接实机检查时，按台架测试流程操作，并保持常规停止控制可用；关闭会话不等同于机械臂急停。

伺服循环运行在拥有机械臂连接的计算机上。按 `ur_rtde 实时配置指南 <https://sdurobotics.gitlab.io/ur_rtde/pages/core_concepts/guides/realtime_setup.html>`_，在预期负载下验证控制周期。

观测与动作
----------

定义任务时，使用机械臂声明的字段。观测均为 ``float32`` 数组，笛卡尔量在机器人基座坐标系中表示。

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - 字段
     - Shape
     - 含义
   * - ``tcp_pose``
     - ``(7,)``
     - 位置，单位为米；随后为四元数 ``[qx, qy, qz, qw]``。驱动将 UR 的旋转向量转换为该表示。
   * - ``tcp_vel``
     - ``(6,)``
     - 线速度，单位为 m/s；随后为角速度，单位为 rad/s。
   * - ``arm_joint_position``
     - ``(6,)``
     - 关节角，单位为弧度。
   * - ``arm_joint_velocity``
     - ``(6,)``
     - 关节速度，单位为 rad/s。

驱动接收基座坐标系下的绝对 TCP 目标 ``[x, y, z, qx, qy, qz, qw]``，并将四元数转换为 ``servoL`` 使用的旋转向量。本地伺服在环境 step 之间保持最新目标，由首条运动指令启动。

在硬件配置中，通过 ``frequency``\ （默认 500 Hz）、``lookahead_time``\ （0.1 s）和 ``gain``\ （300）设置位置伺服。

若伺服循环失败，后续状态读取和动作调用会报告错误。断开连接并处理控制器故障后，再重新连接。采集环境的 TCP 复位见 `采集演示`_。

检查 Orbbec RGB 相机
--------------------

按 :doc:`Orbbec 相机配置 <franka_orbbec>` 查询 Femto Bolt 序列号并检查独立 RGB 采集。UR 安装流程已包含 SDK。采集模板请求 1280 × 720、15 fps 的彩色流，关闭深度；后端返回 BGR，UR 采集环境记录 RGB。

采集与 policy 执行时，保持相机安装位置、视角、裁剪和图像名称一致。将选定序列号填入下述采集配置。

采集演示
--------

驱动与相机检查通过后，配置 ``examples/embodiment/config/realworld_ur5e_collect_data.yaml``。该流程将机械臂、夹爪、环境相机和环境都放在节点 0。替换硬件条目中的 ``ROBOT_IP``、Robotiq 串口路径和 ``CAMERA_SERIAL``，并替换 ``env.eval.override_cfg.camera_names`` 中同一个序列号，将所选视角命名为 ``main``。``main_image_key`` 选择该名称。

启动前，在 ``env/realworld_ur5e_collect.yaml`` 的 ``override_cfg`` 中填写 ``reset_ee_pose``、``ee_pose_limit_min`` 和 ``ee_pose_limit_max``。这些字段沿用 Franka 的位姿配置方式，均为 UR 基座坐标系下的 ``[x, y, z, rx, ry, rz]``，位置单位为米，姿态为 xyz Euler 角，单位为弧度。将模板中的 ``null`` 替换为实测复位目标和任务边界。复位目标必须在边界内。目标先按 XYZ 边界和以复位姿态为中心的连续 Euler 窗口裁剪，随后限制每条指令相对实测位姿的步长。姿态限步沿最短旋转路径执行，中间指令可能处于配置边界外。实测位姿越界本身不会中断运行。按采集任务设置 ``task_description``。

在硬件条目中，``camera_resolution`` 按宽、高设置相机输出尺寸，``camera_fps`` 设置帧率。在 ``env.eval.override_cfg`` 中，``image_size`` 设置裁剪缩放后的记录尺寸，按高、宽排列，默认为 224 × 224；``camera_names`` 与 ``camera_crop_regions`` 配置视角。``step_frequency`` 与数据集 ``fps`` 默认均为 10，调整采集频率时应保持一致。将 ``env.eval.override_cfg.enable_camera_player`` 设为 ``true`` 可预览图像。

启动采集和保存成功 episode 后，环境会将 TCP 返回 ``reset_ee_pose``。``open_gripper_on_reset`` 默认为 ``false``，保留夹爪状态；设为 ``true`` 后，会在 TCP 回位前打开已关闭的夹爪，并沿用 Franka 抓放任务的 0.6 秒等待。回位至少持续 ``reset_duration`` 秒，每步相对实测位姿的变化受 ``action_scale`` 限制。插值结束后，机械臂继续向复位目标推进，最多等待 ``reset_timeout`` 秒。位置与姿态到位容差分别由 ``reset_position_tolerance``\ （单位为米）和 ``reset_orientation_tolerance``\ （单位为弧度）设置；超时会报告失败，并尝试保持实测位姿。采集前先确认复位路径。

推拉或旋转 SpaceMouse 摇杆按与 Franka 相同的映射，在基座坐标系中发送笛卡尔增量。左键关闭夹爪，右键打开夹爪。松开摇杆后停止添加增量，运动后的第一条零输入会保持实测位姿。

.. warning::

   连接 Robotiq 夹爪时会执行清除与激活流程，手指可能移动。每次连接前都应清空手指周围区域。启用 ``open_gripper_on_reset`` 后，每次 TCP 复位前还会松开夹持。

采集使用现有 Linux evdev episode 监听器。运行用户需要能够访问 SpaceMouse HID 设备、相机、串口适配器和实体键盘或踏板。设备发现有歧义时，通过 ``env.eval.keyboard_device`` 或 ``RLINF_KEYBOARD_DEVICE`` 选择后者。在 SSH 终端输入的按键不属于该设备的 evdev 输入。

激活环境并完成设备配置后，启动采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_ur5e_collect_data

runner 创建带时间戳的日志目录，并将 LeRobot 数据集写入其中的 ``lerobot_dataset/``。在实体输入设备上按 ``a`` 开始记录，按 ``b`` 标记分段，按 ``c`` 结束并保存成功 episode。再次按 ``a`` 会丢弃当前记录；为避免误触双击，wrapper 会忽略开始后 1.5 s 内的第二次 ``a``。模板只保留成功 episode，目标数量由 ``runner.num_data_episodes`` 设置，默认为 20。保存成功 episode 后，采集流程将 TCP 复位到配置的起点；开始下一条记录前，先还原任务物体。

.. warning::

   按 ``c`` 保存成功 episode 后会执行配置的 TCP 回位，保存前应确认该路径。开始记录前和放弃 episode 后，SpaceMouse 输入仍然有效。

检查记录格式
~~~~~~~~~~~~

大量采集前，先检查一小段 episode。安装 Robotiq 夹爪并使用默认的 ``use_relative_frame: True`` 时，记录与 policy 动作均为七个归一化通道 ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``，笛卡尔增量采用当前末端坐标系。SpaceMouse 保留 Franka 的基座系映射，其记录动作会转换到与 policy 相同的坐标系。``action_scale`` 默认为 ``[0.02, 0.1, 1]``，前两项在工作空间限幅前缩放基座系平移和 Euler 增量。夹爪信号为负时关闭，为正时打开。

记录状态为 ``[gripper_position, x_rel, y_rel, z_rel, roll_rel, pitch_rel, yaw_rel]``，位姿相对于复位返回的实测位姿，六个相对位姿值在刚复位时均为零。开度与位置的单位为米，xyz Euler 角的单位为弧度。``main`` 视角的 RGB 图像记录为 ``image``。

训练和部署使用 ``pi05_ur5e`` 读取该格式，并保持相对坐标系设置、状态顺序、相机视角、任务描述、采集频率和动作比例一致。

微调 π₀.₅
----------

将记录的 LeRobot 目录复制到 GPU 训练主机，并在该主机准备同一版本的 RLinf 和 OpenPI 环境。先使用训练和推理共同采用的 ``pi05_ur5e`` 配置计算归一化统计：

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_ur5e \
     --repo-id /path/to/ur5e_dataset

传入包含 ``meta/info.json`` 的数据集目录。命令会报告 ``norm_stats.json`` 的输出位置；训练和部署均使用该文件。统计采用与训练相同的数据转换：状态保留七维，记录的七维增量动作补齐到模型维度。

按 :doc:`OpenPI SFT <sft_openpi>` 中的下载与转换步骤准备 RLinf OpenPI 格式的基础权重。本流程转换时使用 ``--action-dim 32``、``--action-horizon 20`` 和 ``--max-token-len 200``。模型内部补齐后的 32 维动作与 UR 环境接收的七维动作分别配置。

编辑 ``examples/sft/config/realworld_ur5e_sft_openpi_pi05.yaml``，填写训练主机上的数据集、基础权重和统计文件路径：

.. code-block:: yaml

   data:
     train_data_paths: /path/to/ur5e_dataset
   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: /path/to/ur5e_dataset/norm_stats.json

模板使用一张 RGB 图像、记录的状态和 20 步预测长度。按训练主机配置 batch size 与 GPU placement，然后在仓库根目录启动 SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_ur5e_sft_openpi_pi05

SFT 将 checkpoint 保存在本次运行的 ``ur5e_openpi_sft/checkpoints/global_step_<N>/`` 目录。将数据集统计文件与实验一起保存，部署配置会分别加载 checkpoint 和统计文件。训练指标、checkpoint 格式与恢复训练见 :doc:`OpenPI SFT <sft_openpi>`。

部署训练后的 Policy
--------------------

将训练后的 checkpoint 和同一份统计文件放到机器人主机。编辑 ``evaluations/realworld/realworld_ur5e_openpi_pi05_eval.yaml``，填写采集时使用的硬件地址、串口路径、复位位姿、位姿边界、相机设置和任务描述。硬件与环境字段沿用采集配置；保持环境相机的安装位置、``action_scale`` 和 ``step_frequency`` 一致。

评估模板设置 ``open_gripper_on_reset: true``，启动和每次 episode 复位时，会先打开已关闭的夹爪，再返回 TCP 起点。任务需要在回位时保持夹持状态时，将该值设为 ``false``。

将模型路径指向 SFT checkpoint 目录，并填写训练时的统计文件路径：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/ur5e_sft/checkpoints/global_step_30000
       openpi_data:
         norm_stats_path: /path/to/ur5e_dataset/norm_stats.json

模板以评估模式使用 ``pi05_ur5e``，每次预测 20 步动作，执行其中五步后再请求下一组动作。环境按配置的比例缩放每个解码后的增量。通过现有真机评估入口启动部署：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_ur5e_openpi_pi05_eval

使用实体键盘或踏板控制 policy 执行：按 ``a`` 开始 rollout，按 ``c`` 标记成功，按 ``b`` 标记失败。模板启用 ``auto_reset: true``，两种结束操作都会结束 episode，将 TCP 复位到 ``reset_ee_pose``，然后重新等待 ``a``。开始下一轮前，先还原任务物体。``max_steps_per_rollout_epoch`` 设置本次运行的步数预算，等待操作员开始时也可用 Ctrl-C 关闭会话。模板将评估视频保存在本次运行的日志目录中。通用评估流程见 :doc:`真机评估 <../../evaluations/guides/realworld>`。
