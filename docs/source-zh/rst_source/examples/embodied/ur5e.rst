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

UR5e 安装流程通过 ``--no-deps`` 安装 ``pyorbbecsdk2==2.1.2``，保留 OpenPI 使用的 NumPy 和 OpenCV 版本。该包通过 ``pyorbbecsdk`` 导入。USB 设备权限和 SDK 图像流要求见 :doc:`Orbbec 相机配置 <franka_orbbec>`。

检查机器人组合
--------------

连接控制器前，先用 fake SDK 运行实际的机器人类：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5e --mock \
     --arg robot_ip=192.168.56.101 --arg node_rank=0

命令构建 ``UR5e`` 机器人、连接 ``arm``、检查观测名称与 shape，再断开连接。成功时，输出以 ``OK: every part connected, observed what it declares, and let go.`` 结束。这里的地址仅供 SDK fake 使用。``node_rank=0`` 通过 scheduler 路径将机械臂连接放置到节点 0。

连接 URSim
----------

机器人组合检查通过后，启动模型设置为 UR5 的 e-Series URSim。使用固定镜像 ``universalrobots/ursim_e-series:5.26.3``，与本指南覆盖的控制器版本保持一致；容器与网络配置见 `官方 URSim Docker 说明 <https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/master/doc/setup/ursim_docker.rst>`_。

驱动可以运行在 Linux Docker 宿主机上，也可以在与 URSim 连接到同一 Docker bridge 的容器中运行。``192.168.56.101`` 这类 bridge 地址属于对应的 Docker 宿主机；另一台计算机需要单独配置网络路径，或将控制进程部署到 URSim 宿主机。仅发布 VNC 或网页端口，并不会暴露 RTDE 和脚本接口。

将以下地址替换为 URSim 容器地址，先仅接收反馈，不打开控制连接：

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --receive-only --frequency 10 --samples 10

命令仅创建 ``RTDEReceiveInterface``，按请求的 10 Hz 读取并记录十个样本，然后断开连接。输出包含关节位置与速度、TCP 位姿与速度，以及机器人和安全模式。这里的 TCP 位姿使用 UR 原生六维表示 ``[x, y, z, rx, ry, rz]``，旋转部分为以弧度表示的旋转向量。该模式不上传、启动或停止控制脚本，也不发送机器人指令。

检查完整机器人组合时，在 PolyScope 中开启 Remote Control，使仿真机器人进入 NORMAL / RUNNING 状态，再使用实际 SDK 检查连接：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5e \
     --arg robot_ip=192.168.56.101 --arg node_rank=0

命令上传 SDK 控制脚本，读取四项机械臂观测，然后关闭两个 SDK 连接。检查过程不发送位置目标。确认输出中的 shape 正确，再运行任务或运动检查。

.. warning::

   打开控制连接时会上传并启动 SDK 的 URScript 程序，关闭连接时会停止该程序。即使只检查观测，也应确保控制器由该驱动独占。驱动不会自动上电、释放刹车或恢复安全停止。

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

通过 ``Robot.send_action`` 发送 ``{"arm": {"tcp_pose": pose}}``。位姿为绝对值 ``[x, y, z, qx, qy, qz, qw]``；驱动归一化四元数，再转换为 ``servoL`` 接受的旋转向量。首次动作启动伺服循环；在两次任务动作之间，循环持续保持最新目标。连接和读取观测时，该循环保持空闲。

``URRTDEArm`` 接受 ``frequency``\ （默认 500 Hz）、``lookahead_time``\ （默认 0.1 s）和 ``gain``\ （默认 300）。这些参数配置本地位置伺服，``gain`` 不代表笛卡尔刚度。驱动拒绝笛卡尔柔顺配置，不将其转换为上述参数。观测声明仅包含表中的四项，任务应按这些实际能力读取状态。

若伺服循环失败，后续状态读取和动作调用会报告错误。断开连接并处理控制器故障后，再重新连接。采集环境的 TCP 复位见 `采集演示`_。

运行
----

连接与观测检查通过后，使用交互式控制脚本检查 TCP 动作接口。先通过本地驱动读取同一个控制器：

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101 --read-only

命令打印关节角（单位为弧度）与 TCP 位姿（位置单位为米，旋转为 xyzw 四元数），然后断开连接。额外输出的 ``TCP pose (m, xyz Euler rad)`` 行采用与 ``reset_ee_pose`` 和位姿边界相同的六维表示，可据此记录复位目标；``--receive-only`` 输出的旋转向量采用不同的姿态表示。与机器人组合检查一样，该命令上传 SDK 控制脚本，但不发送位置目标。

在 URSim 中显式测试运动时，去掉 ``--read-only``，逐条输入以下命令：

.. code-block:: bash

   python -m toolkits.realworld_check.test_ur_controller \
     --robot-ip 192.168.56.101

.. code-block:: text

   where
   offset 1 0 0
   where
   offset -1 0 0
   where
   quit

``offset 1 0 0`` 以实测 TCP 位置为起点，沿基座坐标系的 x 轴平移 1 mm，并保持姿态。每条 offset 的长度最多为 5 mm，在一秒内插值发送。``where`` 读取实测状态。等待运动稳定后，再发送下一条 offset。``quit``、EOF 或 Ctrl-C 停止伺服会话并断开连接。

.. warning::

   先在 URSim 中验证运动流程。连接实机测试时，按机械臂的台架测试流程操作，并保持常规停止控制可用。退出脚本会关闭控制会话，不等同于机械臂急停。

伺服循环在机械臂连接所在的计算机上运行，与环境提交目标的频率分开。按预期负载验证该计算机上的控制周期，参考 `ur_rtde 实时配置指南 <https://sdurobotics.gitlab.io/ur_rtde/pages/core_concepts/guides/realtime_setup.html>`_。组合与节点放置见 :doc:`机器人接口 </rst_source/concepts/robotics>`。

检查 Orbbec RGB 相机
--------------------

机械臂检查完成后，先验证相机图像流，再组合设备进行遥操作。采集模板使用 ``camera_type: orbbec`` 和已安装的 ``pyorbbecsdk`` SDK。将 Femto Bolt 连接到控制主机，查询序列号：

.. code-block:: bash

   python - <<'PY'
   import pyorbbecsdk as ob

   context = ob.Context()
   context.enable_net_device_enumeration(False)
   devices = context.query_devices()
   for index in range(devices.get_count()):
       print(devices.get_device_serial_number_by_index(index))
   PY

记录要使用的相机序列号。在以下读取示例中，将 ``CAMERA_SERIAL`` 替换为该值，并将 ``resolution`` 按宽、高顺序填写，``fps`` 设为设备支持的彩色流帧率。示例值与可编辑的采集模板一致；``enable_depth=False`` 仅启用彩色流：

.. code-block:: bash

   python - <<'PY'
   from rlinf.robotics import Camera, CameraInfo

   camera = Camera.of(CameraInfo(
       name="main",
       serial_number="CAMERA_SERIAL",
       camera_type="orbbec",
       resolution=(1280, 720),
       fps=15,
       enable_depth=False,
   ))
   try:
       camera.connect()
       frame = camera.get_observation()["frame"]
       print(frame.shape, frame.dtype)
   finally:
       camera.disconnect()
   PY

后端返回 BGR 彩色图像，输出应为 ``(720, 1280, 3)``，dtype 为 ``uint8``。UR 采集环境将图像转换为 RGB 后记录。``finally`` 在读取结束后关闭图像流。将相同的序列号、分辨率和帧率填入采集硬件条目的 ``camera_serials``、``camera_resolution`` 和 ``camera_fps``，并在 ``camera_names`` 中将该序列号映射为 ``main``，与这里的视角名称一致。

演示采集与 policy 执行时，应保持相机安装位置、视角、裁剪、图像名称和颜色处理一致。在 PolyScope 中，按安装的夹爪配置工具 TCP 与负载。

采集演示
--------

驱动与相机检查通过后，配置 ``examples/embodiment/config/realworld_ur5e_collect_data.yaml``。该流程将机械臂、夹爪、环境相机和环境都放在节点 0。替换硬件条目中的 ``ROBOT_IP``、Robotiq 串口路径和 ``CAMERA_SERIAL``，并替换 ``env.eval.override_cfg.camera_names`` 中同一个序列号，将所选视角命名为 ``main``。``main_image_key`` 选择该名称。

启动前，在 ``env/realworld_ur5e_collect.yaml`` 的 ``override_cfg`` 中填写 ``reset_ee_pose``、``ee_pose_limit_min`` 和 ``ee_pose_limit_max``。这些字段沿用 Franka 的位姿配置方式，均为 UR 基座坐标系下的 ``[x, y, z, rx, ry, rz]``，位置单位为米，姿态为 xyz Euler 角，单位为弧度。将模板中的 ``null`` 替换为实测复位目标和任务边界。复位目标与启动时的实测 TCP 均应在边界内，环境会拒绝从范围外开始运动。位置与姿态目标会裁剪到该范围，姿态使用以复位姿态为中心的连续窗口。按采集任务设置 ``task_description``。

在硬件条目中，``camera_resolution`` 按宽、高设置相机输出尺寸，``camera_fps`` 设置帧率。Orbbec 后端按指定帧率选择最接近输出尺寸的原生彩色流，再缩放到输出尺寸；模板使用 1280 × 720、15 fps，仅采集彩色图像。在 ``env.eval.override_cfg`` 中，``image_size`` 设置裁剪缩放后的记录尺寸，按高、宽排列，默认为 224 × 224；``camera_names`` 与 ``camera_crop_regions`` 配置视角。环境通过组合机器人读取机械臂、夹爪和相机观测，再构造 policy 输入。``step_frequency`` 与数据集 ``fps`` 默认均为 10，调整采集频率时应保持一致。将 ``env.eval.override_cfg.enable_camera_player`` 设为 ``true`` 可预览图像。

启动采集和保存成功 episode 后，环境会将 TCP 返回 ``reset_ee_pose``，并保持夹爪状态。插值至少持续 ``reset_duration`` 秒，每一步受 ``action_scale`` 限制，因此距离较远时回位时间会增加。插值结束后，``reset_timeout`` 设置到位检查超时参数，位置与姿态容差分别由 ``reset_position_tolerance``\ （单位为米）和 ``reset_orientation_tolerance``\ （单位为弧度）设置。启动采集前，先确认该复位路径。

推拉或旋转 SpaceMouse 摇杆按与 Franka 相同的映射，在基座坐标系中发送笛卡尔增量。左键关闭夹爪，右键打开夹爪。松开摇杆后停止添加增量，运动后的第一条零输入会保持实测位姿。

.. warning::

   连接 Robotiq 夹爪时会执行清除与激活流程，手指可能移动。每次连接前都应清空手指周围区域。TCP 复位保持夹爪状态。

采集使用现有 Linux evdev episode 监听器。运行用户需要能够访问 SpaceMouse HID 设备、相机、串口适配器和实体键盘或踏板。设备发现有歧义时，通过 ``env.eval.keyboard_device`` 或 ``RLINF_KEYBOARD_DEVICE`` 选择后者。在 SSH 终端输入的按键不属于该设备的 evdev 输入。

激活环境并完成设备配置后，启动采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_ur5e_collect_data

runner 创建带时间戳的日志目录，并将 LeRobot 数据集写入其中的 ``lerobot_dataset/``。在实体输入设备上按 ``a`` 开始记录，按 ``b`` 标记分段，按 ``c`` 结束并保存成功 episode。再次按 ``a`` 会丢弃当前记录；为避免误触双击，wrapper 会忽略开始后 1.5 s 内的第二次 ``a``。模板只保留成功 episode，目标数量由 ``runner.num_data_episodes`` 设置，默认为 20。保存成功 episode 后，采集流程将 TCP 复位到配置的起点；开始下一条记录前，先还原任务物体。

.. warning::

   按 ``c`` 保存成功 episode 后会执行配置的 TCP 回位，保存前应确认该路径。开始记录前和放弃 episode 后，SpaceMouse 输入仍然有效。

检查记录格式
~~~~~~~~~~~~

大量采集前，先检查一小段 episode。采集动作与机械臂后端的绝对 TCP 位姿不同：安装 Robotiq 夹爪时，动作包含七个归一化通道 ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``。平移和旋转是在基座坐标系中表示的增量，按 ``action_scale`` 缩放。默认 ``[0.02, 0.1, 1]`` 表示满输入 step 的每个平移轴最多为 0.02 m，每个旋转轴最多为 0.1 rad；最后一个信号为负时关闭夹爪，为正时打开夹爪。

原始状态包含 ``tcp_pose``\ （七个值，位置单位为米，姿态为 xyzw 四元数）和 ``gripper_position``\ （一个值，为以米表示的开度）。``RealWorldEnv`` 在展开状态前按 key 排序，因此记录的八维状态为 ``[gripper_position, x, y, z, qx, qy, qz, qw]``。环境相机记录一张 RGB 图像，数据集中的 key 为 ``image``；环境使用名称 ``main`` 选择该主图像。

``pi05_ur5e`` 数据配置将该图像、八维状态、七维动作和任务描述映射到 OpenPI，并保留记录的增量动作。采集与部署时，应保持状态顺序、相机视角、任务描述、采集频率和 ``action_scale`` 一致。

微调 π₀.₅
----------

将记录的 LeRobot 目录复制到 GPU 训练主机，并在该主机准备同一版本的 RLinf 和 OpenPI 环境。先使用训练和推理共同采用的 ``pi05_ur5e`` 配置计算归一化统计：

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_ur5e \
     --repo-id /path/to/ur5e_dataset

传入包含 ``meta/info.json`` 的数据集目录。命令会报告 ``norm_stats.json`` 的输出位置；训练和部署均使用该文件。统计采用与训练相同的数据转换：状态保留八维，记录的七维增量动作补齐到模型维度。

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
