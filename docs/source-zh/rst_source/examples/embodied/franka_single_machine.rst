Franka 真机强化学习
===================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%

   在一台带有 GPU 的主机上运行 Franka 真机强化学习流程。

你可以在一台带 GPU 的主机上同时运行计算节点（训练 / rollout）和控制节点（Franka 控制），
完成真机 SAC / RLPD / PPO 训练。该流程使用两个独立的 RLinf 环境，已在标准 Ubuntu 20.04、
Franka System Image 5.9.2 和 libfranka 0.19.0 的组合上验证。如果没有实时内核，
必须按照下文显式关闭 libfranka 的实时检查；仅安装较新的固件和 libfranka 并不会自动关闭该检查。

.. note::

   当前的单主机流程请使用本页。如果你使用较旧的 firmware/libfranka 组合，
   或者希望使用独立的实时控制主机，请参考已归档的多机流程 :doc:`franka`。

.. note::

   单机部署把计算节点（actor / rollout / reward）作为 Ray head（rank 0），
   控制节点（Franka env）作为 rank 1 **加入同一台机器的集群**。
   因为两个角色使用不同的 Python 环境，需要在两个终端分别激活脚本、分别加入集群。

概览
----------------------------------------

从相机观测和机器人反馈中训练真机操作策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN policy · OpenPI π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      SAC · Cross-Q · RLPD · PPO

   .. grid-item-card:: 任务
      :text-align: center

      Peg insertion · charger · PnP

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · RealSense/ZED · gripper

| **你将完成:** 安装控制端依赖 → 采集示教 → 启动 Ray → 发起真机训练 → 观察 ``env/reward`` 和视频.
| **前置条件:** :doc:`安装 </rst_source/start/installation>` · Franka firmware/libfranka 匹配 · 局域网 · 安全操作员.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - 任务
     - 配置 / 入口
     - 说明
   * - Peg insertion
     - ``realworld_peginsertion_rlpd_cnn_async``
     - 在目标末端位姿完成插块插入。
   * - Charger
     - ``realworld_charger_sac_cnn_async``
     - 通过真机奖励反馈完成充电器对齐与插入。
   * - PnP / eval
     - ``realworld_pnp_*``
     - 采集或部署 pick-and-place 类策略。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - RGB 相机帧，以及可选机器人状态。
   * - Action
     - 6D/7D 连续笛卡尔增量动作，可包含夹爪控制。
   * - Reward
     - 任务成功、键盘标注或任务特定稠密反馈。
   * - Prompt
     - 使用 VLA 策略时由 env config 提供真机任务文本。

硬件环境搭建
----------------------------------------

单机部署需要如下硬件组件：

- **机械臂**：Franka Emika Panda 机械臂。
- **相机**：Intel RealSense 相机（默认）或 Stereolabs ZED 相机。
- **夹爪**：Franka 夹爪（默认）或 Robotiq 2F-85/2F-140。
- **主机**：一台带有 GPU 的计算机，同时承担计算节点（训练 CNN 策略）与机器人控制节点（控制 Franka 机械臂）。
- **空间鼠标（可选）**：用于远程操控数据采集或在训练过程中进行人工干预。
- **GELLO（可选）**：一种关节级遥操作设备，可替代空间鼠标，操控更直观，并原生支持夹爪控制。
- **VR / PICO（可选）**：通过 PICO 头显和手柄进行 6D 末端遥操作，可替代空间鼠标进行数据采集。

.. warning::

  主机需要与机械臂处于同一局域网内。

.. note::

   **使用 ZED 相机或 Robotiq 夹爪？** 请参考专门的指南
   :doc:`franka_zed_robotiq`，了解 SDK 安装、串口设备配置、
   YAML 配置字段以及数据采集。

   **使用 VR / PICO 遥操作？** 请参考 :doc:`franka_vr`，了解
   XRoboToolkit、ZeroMQ、PICO wrapper 配置以及操作步骤。

检查 Franka 固件版本
----------------------------------------

在机器人管理网页（一般为 ``http://<robot_ip>/desk``）中，点击 ``SETTINGS`` 选项卡，在 ``DashBoard`` 中查看 ``Control`` 后面的版本号，如下所示。
请记录该固件版本号，后续设置 ``LIBFRANKA_VERSION`` 时会用到。

.. raw:: html

  <div style="flex: 1; text-align: center;">
      <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_firmware_single_machine.png" style="width: 60%;"/>
  </div>

.. note::

   请依据你的固件版本，参考 `Franka 兼容性矩阵 <https://frankarobotics.github.io/docs/compatibility.html>`_
   选择匹配的 libfranka 版本，并通过环境变量 ``LIBFRANKA_VERSION`` 与 ``FRANKA_ROS_VERSION`` 指定。

环境安装
----------------------------------------

单机方案需要同一台主机上克隆两份 RLinf 仓库，对应两个角色：

- `RLinf-franka`：控制 / 数据采集环境（Franka 控制依赖：ROS Noetic、libfranka、franka_ros、serl_franka_controllers）。
- `RLinf-compute`：计算 / rollout 环境（RLinf 框架及真机 RL 训练所需 Python 依赖）。

这样可以把两套松耦合的依赖隔离在各自独立的虚拟环境中，避免相互污染。

A. 克隆仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  # 为了提高国内下载速度，也可以使用：
  # git clone https://ghfast.top/github.com/RLinf/RLinf.git
  git clone https://github.com/RLinf/RLinf.git RLinf-franka
  git clone https://github.com/RLinf/RLinf.git RLinf-compute

B. 安装控制环境（RLinf-franka）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `RLinf-franka` 目录下安装 Franka 控制依赖。
依据你的固件版本设置 libfranka 与 franka_ros 版本（例如固件 5.9.2 对应 ``LIBFRANKA_VERSION=0.19.0``）：

.. code:: bash

  cd RLinf-franka
  # 依据固件版本指定 libfranka / franka_ros 版本
  export LIBFRANKA_VERSION=0.19.0
  export FRANKA_ROS_VERSION=0.10.0

  # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令
  bash requirements/install.sh embodied --env franka
  source .venv/bin/activate

C. 配置实时行为
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libfranka 默认强制检查实时调度能力。安装 System Image 5.9.0+ 和 libfranka 0.18.0+
并不会让标准 Linux 内核自动通过该检查。如果主机没有使用 PREEMPT_RT 内核，
必须显式配置控制后端忽略实时检查。

对于本流程使用的 ``franka_ros`` 后端，请修改安装后的配置文件：

.. code-block:: yaml

   # .venv/franka_catkin_ws/src/franka_ros/franka_control/config/franka_control_node.yaml
   realtime_config: ignore  # 默认值：enforce

对于 ``franky`` 后端，请在创建机器人时传入 ``RealtimeConfig.Ignore``：

.. code-block:: python

   import franky

   robot = franky.Robot(
       "172.16.0.2",
       relative_dynamics_factor=0.2,
       realtime_config=franky.RealtimeConfig.Ignore,
   )

.. note::

   RLinf 当前的 ``FrankyController`` 使用默认的 ``RealtimeConfig.Enforce``
   创建 ``franky.Robot``。因此，该后端在非实时内核上运行时仍需相应修改构造调用，
   目前还不能通过 RLinf YAML 选项启用。

.. warning::

   ``ignore`` 只会关闭 libfranka 启动时的实时检查，并不会让标准内核获得实时能力。
   训练可能使主机处于高负载状态，操作系统调度延迟可能导致错过控制周期或发生通信错误。
   为了获得可靠的控制，请尽可能安装并使用 PREEMPT_RT 内核，并参考 Franka 官方的
   `实时内核指南 <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_。

D. 安装计算环境（RLinf-compute）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `RLinf-compute` 目录下安装 RLinf 框架与训练所需依赖（对应训练所用的模型与仿真环境）：

.. code:: bash

  cd RLinf-compute
  # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令
  bash requirements/install.sh --model openpi --env libero
  source .venv/bin/activate

.. note::

   计算环境的模型 / 仿真 env 参数请与你的配置对应（例如 CNN 策略使用 openpi，实际以你的训练模型为准）。

.. note::

   两个克隆仓库都包含 ``ray_utils/realworld/`` 下的启动脚本
   （``setup_compute_node.sh``、``setup_franka_node.sh``、``setup_franka_collect.sh``、``cleanup.sh``）。
   每个脚本会自动定位到它所在的仓库根并使用该仓库的虚拟环境，
   因此请在各自仓库目录下（``cd RLinf-compute`` / ``cd RLinf-franka``）执行对应的脚本。

下载模型
----------------------------------------

在开始训练之前，需要先下载对应的预训练模型：

.. code:: bash

   # 下载模型（两种方式二选一）
   # 方式 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 方式 2：使用 huggingface-hub
   # 为了提高国内下载速度，可以添加以下环境变量：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

下载完成后，请在对应的配置 YAML 文件中正确填写模型路径。

运行
----------------------------------------

前置准备
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**获取任务的目标位姿**

对于 Peg-insertion 任务，可以使用脚本 `toolkits.realworld_check.test_franka_controller` 获取目标末端位姿。

首先，需要将 Franka 机器人切换到可编程模式，然后手动将机械臂移动到希望的目标位姿。

随后，在运行脚本之前，在 **控制环境（RLinf-franka）** 中激活环境，并设置环境变量 ``FRANKA_ROBOT_IP`` 为机器人 IP 地址：

.. code-block:: bash

   cd RLinf-franka
   export FRANKA_ROBOT_IP=<your_robot_ip_address>
   source ray_utils/realworld/setup_franka_node.sh   # 仅激活控制环境，不启动 Ray

然后运行脚本：

.. code-block:: bash

   python -m toolkits.realworld_check.test_franka_controller

脚本会提示你输入命令，可以输入 `getpos_euler` 来获取当前末端执行器以欧拉角形式表示的位姿。

数据采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于 RLPD 实验，需要先收集一部分初始数据，
该过程只需在 **控制环境（RLinf-franka）** 中单机运行，采集节点作为唯一的 Ray head。

1. 在控制仓库中使用数据采集启动脚本，它会激活环境并启动一个单节点 Ray head（rank 0）：

.. code-block:: bash

   cd RLinf-franka
   export FRANKA_ROBOT_IP=<your_robot_ip_address>
   export RLINF_HEAD_IP=<this_host_ip_address>
   # 可选：export RLINF_COMM_NET_DEVICES=<network_device>  # 默认为 eth0
   source ray_utils/realworld/setup_franka_collect.sh start

.. note::

   该脚本会依次 source ROS 与 catkin 的 setup 脚本，再激活虚拟环境。
   使用 ``install.sh`` 安装的环境，激活时通常已自动完成这些 source（见脚本注释）。

2. 修改配置文件 ``examples/embodiment/config/realworld_collect_data.yaml``，
   将其中 ``robot_ip`` 字段填为你的机器人 IP 地址。

.. code-block:: yaml

  cluster:
    num_nodes: 1
    component_placement:
      env:
        node_group: franka
        placement: 0
    node_groups:
      - label: franka
        node_ranks: 0
        hardware:
          type: Franka
          configs:
            - robot_ip: ROBOT_IP
              node_rank: 0

将配置中的 `target_ee_pose` 字段改为前面步骤中获取到的目标末端位姿：

.. code-block:: yaml

  env:
    eval:
      override_cfg:
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]

4. 运行数据采集脚本：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh

在采集过程中，可以使用空间鼠标对机器人进行人工干预，以获得更丰富的数据。

脚本默认在收集 20 个 episode 后结束（可以通过配置中的 `num_data_episodes` 字段修改），
采集到的数据会保存在 ``logs/[running-timestamp]/data.pkl`` 路径下。

5. 数据采集完成后，将收集到的数据路径用于后续训练（计算环境下读取）。

.. note::

   **使用 ZED 相机和 Robotiq 夹爪？** 我们提供了专用的数据采集脚本和配置文件。
   请参考 :doc:`franka_zed_robotiq` 中的
   :ref:`数据采集 <franka-zed-robotiq-data-collection-zh>` 章节。

使用 GELLO 进行数据采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

除空间鼠标外，RLinf 还支持使用 `GELLO <https://github.com/wuphilipp/gello_software>`_ 进行遥操作数据采集。
GELLO 是一种关节级遥操作设备，其运动学结构与 Franka 机械臂一致，操控更直观、精确，并原生支持夹爪控制。

**前置条件**

- 安装 ``gello`` 和 ``gello-teleop`` 软件包。详细安装说明请参考 :doc:`franka_gello`。
- GELLO 设备通过 USB 串口连接到主机。
- 确认 GELLO 串口路径（例如 ``/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0``）。
  可通过以下命令列出可用串口：

  .. code-block:: bash

     ls /dev/serial/by-id/

**配置**

使用配置文件 ``examples/embodiment/config/realworld_collect_data_gello.yaml``。
与空间鼠标配置的关键区别如下：

.. code-block:: yaml

   env:
     eval:
       use_spacemouse: False
       use_gello: True
       gello_port: "/dev/serial/by-id/usb-FTDI_..."  # 替换为你的 GELLO 串口路径

**运行**

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data_gello

整体流程与空间鼠标采集相同：使用 GELLO 设备操控机器人完成任务，脚本会自动保存成功的 episode。

集群设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在正式开始实验之前，需要先正确地搭建 ray 集群。

.. warning::
  这一步非常关键，请谨慎操作！任何细微的配置错误，都可能导致依赖缺失或无法正确控制机器人。

RLinf 使用 ray 来管理分布式环境，这意味着：
当你在某个终端执行 `ray start` 时，ray 会记录当时的 Python 解释器路径和相关环境变量；
之后在该节点上由 ray 启动的所有进程都会继承同一套 Python 环境与环境变量。

单机部署中，**计算节点（RLinf-compute）** 作为 Ray head（rank 0），
**控制节点（RLinf-franka）** 作为 rank 1 加入该 head。由于两个角色使用不同的 Python 环境，
需要**在两个终端**分别激活对应环境后再加入集群。

仓库提供了以下启动脚本（位于 ``ray_utils/realworld/``）：

- ``setup_compute_node.sh``：计算节点（rank 0，head），``source`` 该脚本后传入 ``start`` 启动 Ray。
- ``setup_franka_node.sh``：控制节点（rank 1，worker），``start`` 时加入计算 head。
- ``setup_franka_collect.sh``：数据采集专用（单节点，rank 0），见数据采集一节。
- ``cleanup.sh``：清理残留的 Ray / ROS / FrankaController 进程（直接执行 ``bash cleanup.sh``）。

前三个脚本均支持 ``source <script>`` / ``source <script> start`` / ``source <script> stop`` 三种用法，
并在启动前校验关键依赖是否可导入。

请在 source 相应脚本 **之前** 设置必需的环境变量：

- ``setup_compute_node.sh start`` 需要 ``RLINF_NODE_IP``\ （计算 head 可被访问的 IP）。
- ``setup_franka_node.sh start`` 需要 ``FRANKA_ROBOT_IP`` 和 ``RLINF_HEAD_IP``\ （计算 head 的 IP）。
- ``setup_franka_collect.sh start`` 需要 ``FRANKA_ROBOT_IP`` 和 ``RLINF_HEAD_IP``\ （数据采集 Ray head 所在主机的 IP）。
- ``cleanup.sh`` 不需要环境变量，但 ``ray`` 必须位于 ``PATH`` 中（请先激活任一 RLinf 环境）。

可选变量 ``RLINF_VENV``、``RLINF_COMM_NET_DEVICES`` 和 ``RAY_TEMP_DIR``
保留脚本头部说明的默认值。

**终端 1：计算环境（rank 0，head）**

.. code-block:: bash

   cd RLinf-compute
   export RLINF_NODE_IP=<this_node_reachable_ip>
   # 可选：export RLINF_COMM_NET_DEVICES=<network_device>  # 默认为 eth0
   source ray_utils/realworld/setup_compute_node.sh start

**终端 2：控制环境（rank 1，worker）**

.. code-block:: bash

   cd RLinf-franka
   export RLINF_HEAD_IP=<compute_head_ip_address>
   export FRANKA_ROBOT_IP=<your_robot_ip_address>
   source ray_utils/realworld/setup_franka_node.sh start

.. note::

   各脚本会先激活各自的虚拟环境。对于控制节点，脚本会依次 ``source`` ROS 与 catkin 的
   setup 脚本，再激活虚拟环境

.. warning::

   两个脚本默认使用不同的 ``--temp-dir`` （``/tmp/rlinf_compute`` 与 ``/tmp/rlinf_control``），
   Ray 才会把它们识别为两个独立的节点；切勿在同终端复用同一份环境变量。

可以通过执行 `ray status` 来检查集群是否已正确启动（应有 2 个节点）。

配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

正式运行实验前，需要根据实际集群与机器人设置修改配置文件
``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``。

首先，在配置文件中将 ``robot_ip`` 字段设置为机器人 IP 地址，将 ``target_ee_pose`` 字段设置为目标末端位姿。

接着，在 ``rollout`` 与 ``actor`` 部分，将 ``model_path`` 字段修改为前面下载好的预训练模型路径；
同时，将 ``data.path`` 字段设置为你上传 demo 数据的位置。

无显示器键盘奖励包装器（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果你希望通过人工使用物理键盘给奖励打标，可以在 real-world env 配置中启用键盘包装器。

例如，在 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml`` 中加入：

.. code-block:: yaml

   env:
     train:
       keyboard_reward_wrapper: single_stage  # 或 multi_stage

可用模式如下：

- ``single_stage``：按 ``a`` 记失败奖励，按 ``b`` 记中性奖励，按 ``c`` 记成功奖励。
- ``multi_stage``：按 ``a`` / ``b`` / ``c`` 在不同奖励阶段之间切换，按 ``q`` 输出负奖励。

新的键盘监听器会直接读取 Linux 输入设备，因此需要在控制节点上、执行 ``ray start`` 之前导出 ``RLINF_KEYBOARD_DEVICE``。

首先，查看当前机器上的键盘设备：

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd

该命令会显示稳定的键盘名称以及其对应的 ``eventX`` 设备。例如，``usb-Logitech_USB_Keyboard-event-kbd -> ../event20`` 表示对应的键盘设备是 ``/dev/input/event20``。

开始训练前，先给该 event 设备开放访问权限：

.. code-block:: bash

   chmod 666 /dev/input/event20

然后在启动 ``ray`` 之前，于 shell 或 setup 脚本中导出这个 event 设备：

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/event20

检查环境（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在启动正式实验前，我们推荐先通过若干测试脚本验证整体环境配置是否正确。

首先，测试相机连接：

.. code-block:: bash

   python -m toolkits.realworld_check.test_franka_camera

然后，通过运行一个 dummy 版本配置来测试基础集群配置。请参照 ``examples/embodiment/config/realworld_dummy_franka_sac_cnn.yaml`` 文件添加 `env.eval.override_cfg`。
可以在配置文件中同时将 `env.train.override_cfg` 与 `env.eval.override_cfg` 部分的 `is_dummy` 字段设置为 `True`，
以启用 dummy 模式。请注意如果启用dummy模式，需要将上面运行 ``toolkits.realworld_check.test_franka_camera.py`` 得到的camera序列号
填补在 `env.train.override_cfg` 与 `env.eval.override_cfg` 部分的 `camera_serials` 字段。

在 **计算环境（head）** 终端中运行测试脚本：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在完成上述检查之后，即可在 **计算环境（head）** 终端中启动真实世界训练实验：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

进阶：多机器人配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 支持对多台 Franka 机器人进行统一管理，实现并行数据采集与训练。
要启用多机器人设置，需要在配置文件的 `node_groups` 部分为每个机器人添加独立的配置。

一个包含两台 Franka 机器人的配置示例位于
``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_2arms.yaml``，如下所示：

.. code-block:: yaml

  cluster:
    num_nodes: 3 # 1 个训练 / rollout 节点 + 2 个机器人控制节点
    component_placement:
      actor:
        node_group: "4090"
        placement: 0 # 运行在训练 / rollout 节点的第一个 GPU 上
      env:
        node_group: franka
        placement: 0-1 # 两个 env 分别绑定两个机器人，rank 0 和 rank 1
      rollout:
        node_group: "4090"
        placement: 0:0-1 # 在训练 / rollout 节点第一个 GPU 上运行两个 rollout 进程
    node_groups:
      - label: "4090"
        node_ranks: 0 # 节点 rank 0 为训练 / rollout 节点
      - label: franka
        node_ranks: 1-2 # 节点 rank 1 和 2 为两个机器人控制节点
        hardware:
          type: Franka
          configs:
            - robot_ip: ROBOT_IP_FOR_RANK1
              node_rank: 1 # 第一个机器人控制节点的 rank
            - robot_ip: ROBOT_IP_FOR_RANK2
              node_rank: 2 # 第二个机器人控制节点的 rank

在单机方案中，每个机器人控制角色同样作为独立的 Ray 节点（独立 ``--temp-dir``）加入同一集群，
对应配置中不同的 ``node_ranks``。

自然的，你可以按照同样的方式扩展到更多的机器人。
关于此类异构硬件配置语法的更多细节，请参考 :doc:`../../guides/hetero`。

可视化与结果
----------------------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **环境指标**:

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）
  - ``env/return``：回合总回报
  - ``env/reward``：环境的 step-level 奖励
  - ``env/success_once``：回合中至少成功一次标志（0或1）

- **Training Metrics**:

  - ``train/sac/critic_loss``: Q 函数的损失
  - ``train/critic/grad_norm``: Q 函数的梯度范数

  - ``train/sac/actor_loss``: 策略损失
  - ``train/actor/entropy``: 策略熵
  - ``train/actor/grad_norm``: 策略的梯度范数

  - ``train/sac/alpha_loss``: 温度参数的损失
  - ``train/sac/alpha``: 温度参数的值
  - ``train/alpha/grad_norm``: 温度参数的梯度范数

  - ``train/replay_buffer/size``: 当前重放缓冲区的大小
  - ``train/replay_buffer/max_reward``: 重放缓冲区中存储的最大奖励
  - ``train/replay_buffer/min_reward``: 重放缓冲区中存储的最小奖励
  - ``train/replay_buffer/mean_reward``: 重放缓冲区中存储的平均奖励
  - ``train/replay_buffer/std_reward``: 重放缓冲区中存储的奖励标准差
  - ``train/replay_buffer/utilization``: 重放缓冲区的利用率

真实世界结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
以下提供插块插入任务和充电器任务的演示视频和训练曲线。在 1 小时的训练时间内，机器人能够学习到一套能够持续成功完成任务的策略。

.. raw:: html

  <div style="flex: 0.8; text-align: center;">
      <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld-curve.png" style="width: 100%;"/>
      <p><em>训练曲线</em></p>
    </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>插块插入（Peg Insertion）</em></p>
  </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/charger-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>充电器插电（Charger）</em></p>
  </div>

排障
----------------------------------------

**相机中途断联**

如果在训练 / 采集运行到一半时出现相机断联问题，可以在 **控制环境（RLinf-franka）** 下重装 opencv：

.. code-block:: bash

   cd RLinf-franka
   source ray_utils/realworld/setup_franka_node.sh   # 激活控制环境
   pip uninstall -y opencv-python-headless
   pip install --force-reinstall --no-deps opencv-python

**运行异常导致的夹爪 / ROS 进程残留**

如果运行过程中抛出异常，或通过 ``Ctrl-C`` 手动中断，可能出现 Franka 夹爪（gripper）断联、
以及 Ray / ROS / FrankaController 等进程残留的情况。可以运行清理脚本停掉残留进程后再重新启动：

.. code-block:: bash

   bash ray_utils/realworld/cleanup.sh
