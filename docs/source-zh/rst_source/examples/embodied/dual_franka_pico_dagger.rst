.. _dual-franka-pico-dagger-zh:

双臂 Franka PICO VR HG-DAgger
=============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-vr.jpg
   :align: center
   :width: 80%
   :alt: 双 Franka VR 遥操作

   使用 VR / PICO 采集双 Franka 遥操作示教数据。

用两只 PICO 手柄采集双臂 Franka ``tcp_rot6d`` 示教数据，准备 OpenPI π₀.₅ 基础 checkpoint，并在真机上运行支持按臂接管的在线 Human-Gated DAgger（HG-DAgger）。

概览
----

使用 PICO 左右手柄在 ``tcp_rot6d`` 环境中遥操作两台 Franka 机械臂，参照 :doc:`dual_franka` 准备 OpenPI π₀.₅ student checkpoint，再启动在线 HG-DAgger 归档完整成功轨迹并只用全接管 action chunk 训练。双臂硬件、实时内核与相机自检见 :doc:`dual_franka`，PICO / XRoboToolkit 发布链路见 :doc:`franka_vr`，单臂 HG-DAgger 见 :doc:`hg-dagger`。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      2× Franka（Franky）· 2× Robotiq

   .. grid-item-card:: 遥操作与相机
      :text-align: center

      PICO 双手柄 · USB 脚踏 · 3 路相机

   .. grid-item-card:: 工作流
      :text-align: center

      PICO 数据采集 → π₀.₅ SFT → 在线 HG-DAgger

   .. grid-item-card:: 硬件
      :text-align: center

      单机（数据采集）· 3 节点（GPU + 2 台机器人节点跑 DAgger）

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 观测
     - 左腕、右腕与全局相机图像，加上双臂 TCP 与夹爪状态。
   * - 动作
     - 20 维双臂 ``tcp_rot6d`` 目标：``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``。
   * - 奖励
     - 来自 USB 键盘或脚踏（``keyboard_device`` / ``RLINF_KEYBOARD_DEVICE``）的人工标记。
   * - 提示词
     - 写入数据集并作为 OpenPI 语言条件的 ``task_description``。

回合标记来自 USB 键盘或脚踏，PICO 手柄上的 ``X/Y`` 与 ``A/B`` 仅用于开合左右夹爪：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 模式
     - USB 键盘 / 脚踏按键
   * - 数据采集（``start_end``）
     - ``a`` 开始或中止录制；``b`` 递增 ``segment_id``；``c`` 标记成功并落盘。
   * - HG-DAgger（``eval_control``）
     - ``a`` 启动 policy；``b`` 标记失败；``c`` 标记成功。

硬件与安全准备
--------------

数据采集默认使用单机：rank 0 通过双网卡或交换机直连两台机械臂、两个夹爪、三路相机、USB 键盘或脚踏与 PICO consumer。采集脚本会自动拉起本地 Ray，无需手动执行 ``ray start``。

在线 HG-DAgger 使用三个 Ray 节点：

.. list-table::
   :header-rows: 1
   :widths: 14 34 52

   * - Rank
     - 角色（HG-DAgger）
     - 接线与环境要求
   * - ``0``
     - OpenPI actor / rollout / inference
     - 运行 student 模型与在线训练的 GPU 节点。
   * - ``1``
     - 左臂控制、env worker、三路相机、PICO consumer
     - 接入 USB 键盘或脚踏并在 ``ray start`` 前导出 ``RLINF_KEYBOARD_DEVICE``；确保 PICO ZeroMQ 地址可达。
   * - ``2``
     - 右臂控制
     - 仅需右臂 Franka 与 Robotiq 控制链路。

.. warning::

   启用运动前，确保急停开关触手可及并清空双臂工作空间。

   在 DAgger 配置中，将全零占位的 ``joint_reset_qpos`` 替换为数据集首帧关节均值或安全 home 姿态（每臂 7 个关节角）。确认每臂 ``target_ee_pose`` 落在共享环境 ``realworld_dual_franka_tcp_rot6d`` 的限位盒内：x 在 [0.3, 0.9] m，y 在 [-0.4, 0.4] m，z 在 [0.02, 0.7] m。

   Ray 在 ``ray start`` 时捕获 Python 虚拟环境与环境变量，多机运行前务必先导出 ``RLINF_NODE_RANK`` 与 ``RLINF_KEYBOARD_DEVICE``。

.. dropdown:: 多机 Ray 集群启动（HG-DAgger 或双机数据采集必填）

   单机数据采集会自动启动 Ray，跳过此项。三节点 HG-DAgger，或两台机器人分机数据采集（设 ``num_nodes: 2``、``node_rank: 0``、``left_controller_node_rank: 0``、``right_controller_node_rank: 1``）时，在每个节点激活 ``.venv`` 并导出环境变量，再启动 Ray：

   .. code-block:: bash

      # 每个节点均需执行（按节点设置 RLINF_NODE_RANK 为 0/1/2；单网卡可省略 NIC）
      source .venv/bin/activate
      export RLINF_NODE_RANK=0
      export RLINF_COMM_NET_DEVICES=eth0

      # 在 DAgger 的 env 节点（rank 1）上导出 USB 键盘或脚踏设备
      export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/usb-KEYBOARD-event-kbd

      # head 节点（rank 0）
      export HEAD_NODE_IP=192.168.1.10
      ray start --head --port=6379 --node-ip-address="$HEAD_NODE_IP"

      # worker 节点（rank 1、2）
      export HEAD_NODE_IP=192.168.1.10
      ray start --address="${HEAD_NODE_IP}:6379"

   在 head 节点执行 ``ray status`` 确认全部节点在线。

   配置中的 rank 均为集群全局 rank。DAgger 默认配置为 ``node_rank: 1``\ （属于 ``franka`` 节点组）、``left_controller_node_rank: 1`` 与 ``right_controller_node_rank: 2``。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与启动 PICO

      **机器人 / 环境节点**\ （每台直连 Franka 的机器）：预编译 x86_64 Franky wheel 内置 libfranka 0.15.0 或 0.19.0（默认）。按 Franka 官方 `兼容性表 <https://frankarobotics.github.io/docs/compatibility.html>`_ 匹配 ``LIBFRANKA_VERSION``\ （其他固件版本可通过 ``FRANKY_WHEEL`` 传入自编译 wheel）；``franka`` 环境已包含相机、输入设备与 ZeroMQ 依赖，PICO / XRoboToolkit 安装见 :doc:`franka_vr`。

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **训练与 Rollout 节点**\ （在线 DAgger GPU 节点）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **启动 PICO 数据流并在 env 节点验流**\ （确认数据正常后再启用机械臂）：

      .. code-block:: bash

         # publisher 机器终端 1：启动 XRoboToolkit PC Service
         (cd /opt/apps/roboticsservice && bash runService.sh)

         # publisher 机器终端 2（在 pico_software 目录下）：启动 VR 数据发布进程
         source XRoboToolkit-Teleop-Sample-Python/.venv/bin/activate
         python -m vr_data_publisher --config configs/vr_bridge.yaml

         # env 节点（在 RLinf 根目录且 .venv 已激活；将 IP:端口 替换为 publisher 地址）
         python -m toolkits.realworld_check.test_pico_data \
             --zmq-addr tcp://192.168.1.20:5555

      确认终端输出持续刷新，且按下 ``grip``、``trigger``、``A/B`` 与 ``X/Y`` 时数值变化后再继续。

   .. tab-item:: 2. 数据采集与准备权重

      **启动 PICO 双臂采集**\ （在单机机器人主机上激活 ``.venv`` 后直接运行，单节点会自动启动 Ray）：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_dual_franka_collect_data_pico

      **采集操作步骤：**

      1. 戴好头显面向工作台正前方，扣下 ``trigger`` 完成操作者基准标定。

      2. 在 USB 键盘或脚踏上按 ``a`` 开始录制（1.5 秒内误触忽略，之后再按中止并丢弃缓存），按 ``b`` 递增 ``segment_id``\ （1 秒防抖），按 ``c`` 标记成功并写入 LeRobot shard。

      3. 按住左 / 右手 ``grip`` 接管对应机械臂（松开后保持当前 TCP），用 ``X/Y`` 控制左夹爪、``A/B`` 控制右夹爪。

      数据默认写入 ``logs/<timestamp>/`` 下的 ``demo_buffer/`` 与 ``lerobot_dataset/rank_0/id_0/``。

      无图形界面保持 ``enable_camera_player: False``。需跨批次续写时，设 ``data_collection.resume: True`` 并将 ``data_collection.save_dir`` 改为固定路径。

      **准备 OpenPI π₀.₅ checkpoint**\ ：按 :doc:`dual_franka` 中的 SFT 与部署权重准备步骤执行。PICO 采集已直接记录 ``tcp_rot6d`` 动作，跳过 ``backfill_tcp_rot6d.py``。随后在 ``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml`` 中配置权重路径（rollout 自动沿用 ``actor.model.model_path``）：

      .. code-block:: yaml

         actor:
           model:
             model_path: /path/to/deploy/global_step_N
             openpi_data:
               repo_id: your_repo/tcp_rot6d_v1

   .. tab-item:: 3. 启动在线 HG-DAgger

      确认三节点 Ray 集群已就绪（见“硬件与安全准备”中的多机集群折叠项），在 head 节点启动在线训练：

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_dual_franka_dagger_openpi

      **运行期按键与接管：**

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - 控制方式
           - 行为
         * - ``a``
           - 从 idle 状态启动一次自主 rollout。
         * - 左 / 右手 ``grip``
           - 接管对应机械臂，另一侧继续执行 policy；任一侧按下即合成为 20 维 ``[PICO 10D, rollout 10D]`` 记录并标记为接管帧。
         * - ``b``
           - 标记失败，丢弃当前 rollout 并复位。
         * - ``c``
           - 标记成功，将完整回合发送给 actor 归档为在线 LeRobot shard 并复位。

      每回合复位后环境等待再次按 ``a``。actor 保留完整成功轨迹，但训练仅采样所有非 padding 帧均带接管标记的 action chunk。

.. dropdown:: 硬件占位符与 PICO 配置

   主要配置文件：

   .. list-table::
      :header-rows: 1
      :widths: 52 48

      * - 配置文件
        - 用途
      * - ``examples/embodiment/config/realworld_dual_franka_collect_data_pico.yaml``
        - PICO 数据采集（默认三路 ``realsense`` 相机）。
      * - ``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml``
        - 三节点在线 HG-DAgger（默认底座 ``realsense`` + 双腕 ``lumos``）。
      * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
        - 双臂 ``tcp_rot6d`` 共享默认配置。

   按工作站实际硬件替换以下占位符：

   .. list-table::
      :header-rows: 1
      :widths: 44 56

      * - 占位符或配置项
        - 填写内容
      * - ``LEFT_ROBOT_IP``、``RIGHT_ROBOT_IP``
        - 左、右两台 Franka 机械臂的 IP 地址。
      * - ``BASE_CAMERA_SERIAL``、``LEFT_CAMERA_SERIAL``、``RIGHT_CAMERA_SERIAL``
        - RealSense 填 librealsense ASIC serial，Lumos 填 ``/dev/v4l/by-id`` 路径。
      * - ``left_gripper_type``、``right_gripper_type``、``LEFT_GRIPPER_CONNECTION``、``RIGHT_GRIPPER_CONNECTION``
        - 左右夹爪型号与连接地址。
      * - ``keyboard_device``
        - USB 键盘或脚踏事件路径（仅数据采集配置）。
      * - ``TASK_DESCRIPTION``
        - 写入数据集并作为 OpenPI 输入的语言提示词。
      * - ``joint_reset_qpos``
        - 数据集首帧关节均值或每臂安全 7 关节 home 姿态（DAgger）。
      * - ``target_ee_pose``
        - 落在 TCP 限位盒内的目标位姿（例如 ``[0.5, ±0.2, 0.5, -3.14, 0, 0]``）。

   ``teleop`` 按臂绑定遥操作设备，``pico.hand: dual`` 将左右手柄分别对应左右臂。

   采集在 ``env.eval`` 中设 ``hold_current_when_inactive: True``\ （松开手柄保持当前 TCP），DAgger 在 ``env.train`` 中设为 ``False``\ （松开后继续执行 policy rollout）。

   跨机通信时，publisher 绑定 ``tcp://0.0.0.0:<port>``，consumer 的 ``zmq_addr`` 填 ``tcp://<vr_publisher_ip>:<port>``\ （切勿填 ``0.0.0.0``）：

   .. code-block:: yaml

      env:
        eval:
          teleop:
            - {pico: {drives: left}}
            - {pico: {drives: right}}
          keyboard_device: null  # 或 /dev/input/by-id/usb-...-event-kbd
          keyboard_reward_wrapper: start_end
          pico:
            zmq_addr: "ipc:///tmp/vr_data.ipc"
            hand: "dual"
            hold_current_when_inactive: True
            control_trigger: "grip"
            calibration:
              button: "trigger"

.. dropdown:: 机械臂柔顺性参数（可选）

   采集配置在硬件条目中设置了双臂共用的 ``compliance``\ （默认值与复位影响见 :ref:`配置机械臂运动 <franka-motion-settings>`）：

   .. list-table::
      :header-rows: 1
      :widths: 45 55

      * - 参数
        - 采集默认值
      * - 平移 / 旋转刚度
        - 1000 N/m / 56 Nm/rad
      * - 平移 / 旋转误差限幅
        - 0.024 m / 0.15 rad
      * - 每步平移 / 旋转上限
        - 0.06 m / 0.15 rad
      * - 默认 ``max_delta_tau``
        - 0.3 Nm / 控制周期

   带 Robotiq 负载时，10 Hz 下 ``max_step: 0.03`` 可能触发 ``joint_velocity_violation``，出现时调小 ``max_step``。DAgger 配置省略 ``compliance`` 并使用 Franky 默认值，如需与采集手感一致可将同一段复制到 DAgger 硬件条目。

   填写单侧 ``left_compliance`` 或 ``right_compliance`` 会替换该臂的共享参数，未写字段回退为 Franky 默认值（``{}`` 表示全部使用后端默认值）：

   .. code-block:: yaml

      compliance:
        translational_stiffness: 900.0
      left_compliance:
        max_step: 0.02

   这些笛卡尔参数同时作用于 DAgger 中的 policy 目标，不影响 GELLO 关节遥操作。

.. dropdown:: 在线 HG-DAgger 进阶配置（可选）

   ``realworld_dual_franka_dagger_openpi.yaml`` 已默认启用在线 LeRobot 归档、``smooth_intervene``、双臂 PICO 接管与纯 policy 评测：

   .. code-block:: yaml

      algorithm:
        dagger:
          only_save_expert: True
          online_lerobot:
            enabled: True
            only_success: True
            robot_type: "dual_FR3"
            fps: 10
            finalize_interval: 1
            data_path: ${runner.logger.log_path}/online_lerobot
            rolling_lerobot_window_size: 50000
            min_frames: 1
            lerobot_num_workers: 0

      env:
        train:
          smooth_intervene: True
          teleop:
            - {pico: {drives: left}}
            - {pico: {drives: right}}
          keyboard_reward_wrapper: eval_control
          pico:
            zmq_addr: "ipc:///tmp/vr_data.ipc"
            hand: "dual"
            hold_current_when_inactive: False
        eval:
          teleop: none

   .. list-table::
      :header-rows: 1
      :widths: 32 68

      * - 配置项
        - 说明
      * - ``online_lerobot.enabled: True``
        - 将过滤后的回合直接送入 actor 滚动 LeRobot 数据集，归档于 ``${runner.logger.log_path}/online_lerobot/rank_0/id_<N>/``。
      * - ``smooth_intervene: True``
        - 消除接管期间 action chunk 边界停顿：若 chunk 末帧仍由人工接管，跳过下一次推理并继续输出兼容动作。要求 ``env.train.teleop`` 全部为 ``pico`` 且每个 pipeline stage 仅运行一个环境。
      * - ``only_success: True``
        - 丢弃失败回合。
      * - ``only_save_expert: True``
        - 只采样所有非 padding 帧均带 ``intervene_flag=True`` 的 action chunk。
      * - ``env.eval.teleop: none``
        - 评测阶段仅运行 policy。人工接管来自 PICO，无 ``rollout.expert_model``，配置省略 beta 相关字段。

结果与常见问题
--------------

运行 ``tensorboard --logdir ./logs`` 监控在线 HG-DAgger 训练，并在 ``./logs`` 下查看运行日志与归档 shard：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 监控项或产物
     - 指标名或路径
   * - Actor loss 与优化器
     - ``train/dagger/actor_loss``、``train/actor/lr``、``train/actor/grad_norm``
   * - 在线数据集计数
     - ``train/lerobot_dataset/total_episodes``、``train/lerobot_dataset/physical_frames``、``train/lerobot_dataset/logical_samples``、``train/lerobot_dataset/num_sub_datasets``
   * - 回合日志
     - ``logs/<timestamp>/run_embodiment.log``
   * - 归档 shard
     - ``logs/<timestamp>-realworld_dual_franka_dagger_openpi/online_lerobot/rank_0/``

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理方法
   * - DAgger 启动后长时间等待
     - ``keyboard_reward_wrapper: eval_control`` 的预期行为；在 USB 键盘或脚踏上按 ``a`` 启动 rollout。
   * - 回合按键无响应
     - 确认按键来自 ``keyboard_device`` / ``RLINF_KEYBOARD_DEVICE`` 对应的 USB 键盘或脚踏（SSH 终端与 PICO 正面按键无效），且 DAgger 需在 ``ray start`` 前导出该环境变量。
   * - ``test_pico_data`` 收不到数据
     - 检查 XRoboToolkit PC Service 与 ``vr_data_publisher`` 是否运行，并将 ``zmq_addr`` 设为可达的 publisher IP（勿填 ``0.0.0.0``）。
   * - 快速运动触发 ``joint_velocity_violation``
     - 针对当前 Robotiq 负载与 10 Hz 控制频率调小 ``compliance.max_step``\ （或单侧 ``left_compliance`` / ``right_compliance``）。
   * - 单机数据采集误连到旧 Ray 集群
     - 启动 ``collect_data.sh`` 前执行 ``ray stop --force``，并确认已激活 Python 3.11 虚拟环境。
