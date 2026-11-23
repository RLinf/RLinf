Franka 在线强化学习
===================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: 用于真机强化学习的 Franka 机械臂

   用于真机强化学习的 Franka 机械臂。

概览
----

使用 RLinf 在 Franka 机械臂上训练 CNN policy，涵盖从 SpaceMouse 数据采集到异步在线 RLPD 训练的全流程。默认方案在一台运行 Ubuntu 20.04 或 22.04 的 x86-64 NVIDIA GPU 主机上同时运行 Franky（libfranka 的 Python 绑定）、rollout 与训练，也支持控制机与 GPU 服务器分离的多机部署。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      Franka Emika Panda（Franky）

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      Intel RealSense · SpaceMouse

   .. grid-item-card:: 流程
      :text-align: center

      数据采集 → 在线 RLPD（SAC）

   .. grid-item-card:: 硬件
      :text-align: center

      1× NVIDIA GPU（或控制机 + GPU）

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 任务
     - 配置
     - 说明
   * - 插孔
     - ``realworld_collect_data``、``realworld_peginsertion_rlpd_cnn_async``
     - 先用 SpaceMouse 收集演示，再结合演示数据和实时真机交互数据异步训练，并支持 SpaceMouse 人工干预。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 说明
   * - 观测
     - 第一个配置相机（``wrist_1``）的 RGB 图像与机器人状态。
   * - 动作
     - 6 维笛卡尔位置与旋转增量，夹爪保持闭合。
   * - 奖励
     - 末端位姿进入目标容差范围时判定成功。

硬件与安全准备
--------------

用有线网口将 Franka 机械臂连到 Ubuntu 20.04 或 22.04 x86-64 主机，再用 USB 连接 Intel RealSense 相机与 SpaceMouse。当前 Linux 账户需具备相机和 SpaceMouse 的 USB 读取权限（SpaceMouse 的 udev 规则见 `SpaceMouse 安装说明 <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_）。

连接机械臂前，在 Franka Desk（``http://<robot_ip>/desk``）中解锁关节并激活 FCI，同时关闭其他占用 FCI 连接的程序。在 GPU 主机上安装 RLinf 前，先确认 ``nvidia-smi`` 正常工作，否则安装脚本会选择仅 CPU 的 PyTorch。

.. list-table::
   :header-rows: 1
   :widths: 22 43 35

   * - 部署方式
     - 机器配置
     - 适用场景
   * - 单机部署（默认）
     - 一台 Ubuntu 20.04 或 22.04 GPU 主机运行 Franky 控制、rollout 和训练。
     - 大多数场景，只需安装和维护一台机器。
   * - 多机部署（可选）
     - 独立控制机（无需 GPU）运行 Franky 控制；GPU 服务器运行 actor 与 rollout。
     - 隔离机械臂控制与训练负载，或 GPU 服务器不在机器人旁边。

.. warning::

   每次真机运行均须由操作员全程监控，并确保急停按钮触手可及。固定插销与夹具，清空工作区域，确认复位路径安全。插孔任务会复位到目标上方约 10 cm 处并随机扰动水平位置和偏航角，切勿直接复用其他机器人的目标位姿。

.. dropdown:: 固件检查与 libfranka 兼容性

   打开 ``http://<robot_ip>/desk`` 进入 Franka Desk，点击 ``SETTINGS``，记录仪表盘中 ``Control`` 后的固件版本号：

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_firmware.png
      :align: center
      :width: 60%
      :alt: Franka Desk 仪表盘中的 Control 固件版本

      Franka Desk 中的 Control 固件版本。

   对照 `Franka 兼容性表 <https://frankarobotics.github.io/docs/compatibility.html>`_ 选定兼容的 libfranka 版本。预编译 Franky wheel 在 x86-64 上内置 libfranka 0.19.0（默认）和 0.15.0；若固件需要其他版本，请使用下方的\ **旧版 ROS 后端（Ubuntu 20.04）**\ 标签。

   RLinf 已验证两种组合：固件 5.9.2 + libfranka 0.19.0（标准内核，关闭实时检查）和固件 5.7.2 至 5.9.0 + libfranka 0.15.0（实时内核）。不要仅为匹配示例而修改机器人固件。

.. dropdown:: 实时内核、NVIDIA 驱动与主机配置（首次准备）

   **1. 安装 PREEMPT_RT 实时内核（推荐）**

   libfranka 每毫秒发送一次控制指令，PREEMPT_RT 内核可在训练负载下保证 1 kHz 控制周期准时执行。请在宿主机上安装实时内核（容器共享宿主机内核，勿在 Docker 内安装），步骤见 Franka `实时内核指南 <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_，并保留原内核作为 GRUB 回退项。重启后确认第二条命令输出 ``1``：

   .. code-block:: bash

      uname -r
      cat /sys/kernel/realtime

   **2. 在实时内核上构建 NVIDIA 驱动**

   使用匹配当前内核的 ``linux-headers`` 编译 NVIDIA 驱动，并传入 ``IGNORE_PREEMPT_RT_PRESENCE=1`` 跳过 PREEMPT_RT 检查。APT 安装先按 NVIDIA `驱动安装指南 <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_ 配置软件源并替换 ``DRIVER_PACKAGE``；已通过 DKMS 安装驱动时执行第二段命令：

   .. code-block:: bash

      # APT 安装
      sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt-get install -y DRIVER_PACKAGE
      sudo reboot

      # 或为已有 DKMS 驱动重新编译内核模块
      sudo env IGNORE_PREEMPT_RT_PRESENCE=1 dkms autoinstall -k "$(uname -r)"
      sudo reboot

   重启后确认 ``nvidia-smi`` 能列出 GPU 再安装 RLinf。后续驱动或内核升级重建模块时同样需要传入该环境变量。

   .. warning::

      ``IGNORE_PREEMPT_RT_PRESENCE=1`` 仅跳过编译检查，NVIDIA 未正式支持 PREEMPT_RT 内核。若 GPU 不可用，可从 GRUB 切回原内核恢复。

   **3. 授予实时调度权限**

   RLinf 启动 Franky 时会锁定控制进程内存、设置 ``SCHED_FIFO`` 优先级 80 并绑核。将当前账户加入 ``realtime`` 用户组并配置 ``/etc/security/limits.d/99-rlinf-realtime.conf``：

   .. code-block:: bash

      getent group realtime || sudo groupadd realtime
      sudo usermod -aG realtime "$(id -un)"
      sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

   写入以下内容后重新登录，确认 ``ulimit -r`` 输出 ``99`` 且 ``ulimit -l`` 输出 ``unlimited``\ （标准内核同样受益）：

   .. code-block:: text

      @realtime - rtprio 99
      @realtime - memlock unlimited

   **4. 不使用实时内核运行（可选）**

   Franky 默认使用 ``realtime_config: ignore``，在标准内核上仅记录警告即可启动。若希望强制要求 PREEMPT_RT 内核，在硬件配置中与 ``robot_ip`` 同级添加 ``realtime_config: enforce``。

   .. warning::

      在标准内核上，较重的训练负载可能导致 libfranka 错过控制周期并触发 ``communication_constraints_violation`` reflex。开始训练前，应在操作员监控下以目标训练负载验证控制稳定性。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 单机部署（默认）

      **1. 安装依赖**

      克隆仓库并安装 Franka 环境（内置 libfranka 0.19.0，采用 manylinux 2.28 wheel，兼容 Ubuntu 20.04/22.04）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      若固件需要 0.15.0，在命令前加 ``LIBFRANKA_VERSION=0.15.0``；需要自定义 wheel 时用 ``FRANKY_WHEEL`` 指定 URL 或本地路径；系统 APT 依赖已装好时可追加 ``--no-root``。

      **2. 下载权重**

      下载 actor 与 rollout 共用的预训练 ResNet-10 编码器：

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./models/RLinf-ResNet10-pretrained

      **3. 启动运行**

      先检查相机数据流并记录序列号：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_franka_camera

      在 ``examples/embodiment/config/realworld_collect_data.yaml`` 和 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml`` 中，更新 ``cluster.node_groups`` 下 ``label: franka`` 对应的机械臂 IP 与相机序列号。

      在训练配置中同时将 ``cluster.num_nodes`` 设为 1，保持 ``4090`` 节点组不变（该名称只指代 GPU 节点 0，不限 GPU 型号）：

      .. code-block:: yaml

         - label: franka
           node_ranks: 0
           hardware:
             type: Franka
             configs:
               - robot_ip: ROBOT_IP
                 node_rank: 0
                 camera_serials: ["CAMERA_SERIAL"]

      用引导模式将插销移至目标插入位姿，在 Franka Desk 中解锁机械臂并启用 FCI，再读取目标位姿（输入 ``getpos_euler``，再输入 ``q`` 释放机械臂）：

      .. code-block:: bash

         export FRANKA_ROBOT_IP=192.168.1.10  # 替换为机械臂 IP
         python -m toolkits.realworld_check.test_franka_controller
         export FRANKA_TARGET_POSE='[x, y, z, roll, pitch, yaw]'  # 替换全部六个数值

      .. dropdown:: 控制器检查工具更多命令（可选）

         .. list-table::
            :header-rows: 1
            :widths: 28 72

            * - 类别
              - 可用项
            * - 交互命令
              - ``getpos``、``getpos_euler``、``getjoint``、``getstate``、``gethand``、``clear``、``home``、``open``、``close``、``q``
            * - 命令行参数
              - ``--robot-ip``、``--realtime-config enforce``

      用 SpaceMouse 采集 20 条演示（保存至 ``logs/franka-demo/demo_buffer``，同级可选导出 ``pickle_episode/`` 与 ``lerobot_dataset/``；使用 GELLO 采集见 :doc:`franka_gello`），待采集退出并释放机械臂后启动在线 RLPD 训练：

      .. code-block:: bash

         RLINF_LOG_DIR="$PWD/logs/franka-demo" \
           bash examples/embodiment/collect_data.sh realworld_collect_data \
           "env.eval.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE"

         bash examples/embodiment/run_realworld_async.sh \
           realworld_peginsertion_rlpd_cnn_async \
           "env.train.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE" \
           "algorithm.demo_buffer.load_path=$PWD/logs/franka-demo/demo_buffer" \
           "actor.model.model_path=$PWD/models/RLinf-ResNet10-pretrained"

      每次重新采集请更换 ``RLINF_LOG_DIR``，避免混入不同批次数据。运行中全程监控机械臂，按需用 SpaceMouse 干预；结束时中断启动脚本并等待机械臂停稳。

   .. tab-item:: 多机部署（控制机 + GPU）

      **1. 在两台机器上安装**

      在控制机（无需 GPU，安装脚本自动选择 CPU 版 PyTorch）与 GPU 服务器上安装相同版本的 RLinf、Python 与 Ray：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. 采集演示并下载权重**

      先在控制机上按“单机部署”步骤采集演示，再将 ``logs/franka-demo/demo_buffer`` 复制到 GPU 服务器。在 GPU 服务器上下载预训练 ResNet-10 编码器：

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./models/RLinf-ResNet10-pretrained

      **3. 启动 Ray 集群并训练**

      在 GPU 服务器上编辑 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``：将 ``cluster.num_nodes`` 设为 2，Franka 节点组的 ``node_ranks`` 与硬件条目的 ``node_rank`` 均设为 1（actor 与 rollout 留在节点 0 的 GPU 0 上）。

      .. warning::

         Ray 会在 ``ray start`` 时记录当前终端的 Python 环境与环境变量。两台机器均须先激活 ``.venv`` 并导出 ``RLINF_NODE_RANK``，再执行 ``ray start``。填写两台机器在互通网络上的 IP（不要填机械臂 IP）；存在多张网卡时导出 ``RLINF_COMM_NET_DEVICES`` 指定通信网卡。

      在两台机器上启动 Ray：

      .. code-block:: bash

         # 在 GPU 服务器（head 节点）
         export RLINF_NODE_RANK=0
         export HEAD_IP=192.168.10.10
         ray start --head --port=6379 --node-ip-address="$HEAD_IP"

         # 在控制机（worker 节点）
         export RLINF_NODE_RANK=1
         export HEAD_IP=192.168.10.10
         export CONTROLLER_IP=192.168.10.11
         ray start --address="$HEAD_IP:6379" --node-ip-address="$CONTROLLER_IP"

      在 GPU 服务器上用 ``ray status`` 确认两个节点均在线后，仅在 GPU 服务器上运行“单机部署”中的训练命令（控制机无需模型权重与 ``demo_buffer``）。多机器人部署见 :doc:`/rst_source/guides/realworld_robot` 与 :doc:`/rst_source/guides/hetero`。

   .. tab-item:: 旧版 ROS 后端（Ubuntu 20.04）

      已有 ROS Noetic 部署，或固件需要 0.15.0、0.19.0 以外的 libfranka 版本时，在 Ubuntu 20.04 宿主机上本地安装 ``franka_ros`` 与 ``serl_franka_controllers`` 后端到独立虚拟环境：

      **1. 安装依赖**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         LIBFRANKA_VERSION=0.15.0 bash requirements/install.sh embodied --env franka-ros --venv franka-ros && source franka-ros/bin/activate

      安装脚本在 ``franka-ros/franka_catkin_ws`` 编译 libfranka、RLinf 的 ``franka_ros`` 分支及 ``serl_franka_controllers``，并将 ROS 初始化脚本追加到 ``franka-ros/bin/activate``。

      执行 ``rospack find serl_franka_controllers`` 确认编译成功。可用以下环境变量覆盖安装默认值：

      .. list-table::
         :header-rows: 1
         :widths: 32 16 52

         * - 环境变量
           - 默认值
           - 说明
         * - ``LIBFRANKA_VERSION``
           - ``0.15.0``
           - 与固件匹配的 libfranka 版本。
         * - ``FRANKA_ROS_VERSION``
           - ``0.10.0``
           - 编译的 ``franka_ros`` 版本。
         * - ``FRANKA_REALTIME_CONFIG``
           - ``enforce``
           - 非实时内核设为 ``ignore`` 后重跑安装脚本即可切换，无需重编。
         * - ``SKIP_ROS``
           - ``0``
           - 设为 ``1`` 时复用已有 ROS 与 catkin 安装。

      .. warning::

         设置 ``SKIP_ROS=1`` 时需自行准备 ROS Noetic、libfranka、``franka_ros`` 与 ``serl_franka_controllers``\ （手动安装见 `ROS Noetic <https://wiki.ros.org/noetic/Installation/Ubuntu>`_、`libfranka <https://frankarobotics.github.io/docs/libfranka/docs/installation.html>`_ 与 `serl_franka_controllers <https://github.com/rail-berkeley/serl_franka_controllers>`_）。

         每次启动前（多机部署须在 ``ray start`` 前）source ``/opt/ros/noetic/setup.bash`` 与 catkin ``devel/setup.bash``，同时确保 libfranka 位于 ``LD_LIBRARY_PATH``。

      **2. 配置并运行**

      在 Franka 硬件配置中添加 ``backend: franka_ros``\ （不要在 YAML 中填写 ``realtime_config``，ROS 后端从安装时的 ``FRANKA_REALTIME_CONFIG`` 读取实时模式）：

      .. code-block:: yaml

         configs:
           - robot_ip: ROBOT_IP
             node_rank: 0
             camera_serials: ["CAMERA_SERIAL"]
             backend: franka_ros

      在已激活的 ``franka-ros`` 环境中检查控制器，随后运行“单机部署”中的采集与训练命令（多机部署按“多机部署”步骤配置 Ray 集群）：

      .. code-block:: bash

         python -m toolkits.realworld_check.test_franka_controller --backend franka_ros

      RLinf 会自动管理 ``roscore``、``roslaunch serl_franka_controllers impedance.launch`` 以及关节复位时的 ``joint.launch``。多机部署时仅控制机需要 ``franka-ros`` 环境，GPU 服务器使用标准 ``franka`` 环境。

      **与 Franky 的差异及 ROS 常见问题：**

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - 项目
           - 旧版 ROS 后端说明
         * - 控制模式
           - 仅接受笛卡尔位姿目标，关节位置控制需使用 Franky。
         * - 夹爪
           - ``gripper_type: franka`` 通过 ``/franka_gripper`` 话题通信，仅在机械臂连接时响应。
         * - 阻抗增益
           - 由 ``dynamic_reconfigure`` 控制，YAML 硬件配置中的 ``compliance`` 字段无效。
         * - 非实时内核报错
           - 出现 ``Running kernel does not have realtime capabilities`` 时，用 ``FRANKA_REALTIME_CONFIG=ignore`` 重跑安装脚本。
         * - 机械臂未就绪或 FCI 被占用
           - 确认 Franka Desk 已激活 FCI 并关闭其他控制程序。崩溃后先 ``ray stop`` 再 ``pkill -f roslaunch``，可用 ``rostopic echo -n 1 /franka_state_controller/franka_states`` 验证状态话题。

.. _franka-motion-settings:

.. dropdown:: 配置机械臂运动（Franky 柔顺控制默认值）

   数据采集、训练、评估、PICO / SpaceMouse 遥操作，以及下发笛卡尔位姿的 GELLO 遥操作，在硬件条目未配置 ``compliance`` 时共用以下 Franky 笛卡尔控制器默认值：

   .. list-table::
      :header-rows: 1
      :widths: 30 15 55

      * - 参数
        - 默认值
        - 含义
      * - ``translational_stiffness``
        - 1000 N/m
        - 位置误差初始刚度。
      * - ``rotational_stiffness``
        - 50 Nm/rad
        - 姿态误差初始刚度。
      * - ``translational_clip``
        - 0.008 m
        - 计算恢复力矩时的初始位置误差上限。
      * - ``rotational_clip``
        - 0.04 rad
        - 计算恢复力矩时的初始姿态误差上限。
      * - ``max_step``
        - 0.03 m
        - 每次调用目标位置相对上一次指令目标的最大变化量。
      * - ``max_step_rad``
        - 0.10 rad
        - 每次调用目标姿态相对上一次指令目标的最大变化量。

   首次目标以实测位姿为基准限幅。目标变化上限按单次调用生效，误差限幅仅约束控制器力矩计算误差；环境仍会先应用动作缩放与工作空间限制。

   复位时任务可通过 ``compliance_param`` 请求新参数，Franky 默认将刚度上限限制在 1200 N/m 和 80 Nm/rad、误差限幅下限限制在 5 mm 和 0.02 rad（插孔任务复位后采用 5 mm x/y、10 mm z 和 0.02 rad 限幅；``max_step`` 与 ``max_step_rad`` 复位后继续生效）。

   如需覆盖局部参数，在 ``robot_ip`` 同级添加 ``compliance`` 映射（键名拼写错误会在连接前直接报错）：

   .. code-block:: yaml

      compliance:
        max_step: 0.02

   关节位置指令（如双臂 GELLO 关节遥操作）走关节控制通道，不使用这些笛卡尔参数。

.. dropdown:: 通过键盘标注奖励（可选）

   没有自动位姿成功信号的任务，可在训练配置中启用键盘奖励标注：

   .. code-block:: yaml

      env:
        train:
          keyboard_reward_wrapper: single_stage  # 或 multi_stage

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 模式
        - 按键说明
      * - ``single_stage``
        - ``a`` 失败，``b`` 中性，``c`` 成功。
      * - ``multi_stage``
        - ``a`` / ``b`` / ``c`` 切换奖励阶段，``q`` 输出负奖励。

   在机器人主机上查找键盘设备、赋权并在启动前导出 ``RLINF_KEYBOARD_DEVICE``\ （多机部署须在 ``ray start`` 前导出）：

   .. code-block:: bash

      ls -l /dev/input/by-id/*-event-kbd
      sudo chmod 666 /dev/input/event20
      export RLINF_KEYBOARD_DEVICE=/dev/input/event20

.. dropdown:: 其他 Franka 工作流（VLA 模型与相关指南）

   将 Franka 依赖（含灵巧手依赖）与各 VLA policy 安装到独立虚拟环境：

   .. code-block:: bash

      # OpenPI
      bash requirements/install.sh embodied --model openpi --env franka --venv openpi && source openpi/bin/activate

      # OpenVLA
      bash requirements/install.sh embodied --model openvla --env franka --venv openvla && source openvla/bin/activate

      # OpenVLA-OFT
      bash requirements/install.sh embodied --model openvla-oft --env franka --venv openvla-oft && source openvla-oft/bin/activate

      # GR00T
      bash requirements/install.sh embodied --model gr00t --env franka --venv gr00t && source gr00t/bin/activate

   相关指南：

   - :doc:`franka_gello` 与 :doc:`franka_vr`：GELLO 或 PICO 遥操作。
   - :doc:`franka_pi0_sft_deploy` 与 :doc:`hg-dagger`：OpenPI policy。
   - :doc:`franka_reward_model`：学习奖励模型。
   - :doc:`franka_zed_robotiq` 与 :doc:`franka_dexhand`：其他相机与末端执行器。
   - :doc:`dual_franka`：双臂控制；:doc:`/rst_source/guides/rtc`：重叠执行动作与推理。

结果与常见问题
--------------

在 TensorBoard（``tensorboard --logdir ./logs --port 6006``）中监控 ``env/success_once``、``env/return`` 以及 SAC actor / critic loss；完整日志选项见 :doc:`训练指标 <../../reference/metrics>` 与 :doc:`/rst_source/guides/logger`。下方曲线和视频展示了插孔与充电器任务的代表性训练效果。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld-curve.png
   :align: center
   :width: 100%

   真机训练曲线。

.. raw:: html

   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
   </video>
   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/charger-compressed.mp4" type="video/mp4">
   </video>

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 现象
     - 解决方法
   * - ``torch.cuda.is_available()`` 输出 ``False``
     - 确认当前内核下 ``nvidia-smi`` 正常（PREEMPT_RT 内核需带 ``IGNORE_PREEMPT_RT_PRESENCE=1`` 重新编译驱动），然后重跑 ``requirements/install.sh``。
   * - ``Running kernel does not have realtime capabilities``
     - 切换至 PREEMPT_RT 内核，或使用 Franky 默认的 ``realtime_config: ignore``。
   * - libfranka 版本不兼容报错
     - 对照 Franka 兼容性表将 ``LIBFRANKA_VERSION``\ （0.19.0 或 0.15.0）匹配到 Franka Desk 的 ``Control`` 固件版本。
   * - 触发 ``communication_constraints_violation`` reflex
     - 标准内核下训练负载导致 1 kHz 控制超时，建议改用 PREEMPT_RT 内核或将机械臂控制拆至独立控制机。
   * - Franky 阻抗控制器意外停止
     - RLinf 会上报运动错误并安全停止；排查故障后再重启训练（直接调用 Python 接口时需先断开再重连，仅清除错误不会恢复跟踪）。
