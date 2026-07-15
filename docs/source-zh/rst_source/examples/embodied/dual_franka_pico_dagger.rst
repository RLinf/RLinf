.. _dual-franka-pico-dagger-zh:

双 Franka 使用 PICO 采集与 DAgger
================================================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%

   双 Franka 真实世界任务中，PICO 可用于双手遥操作采集和在线 HG-DAgger 接管。

本指南介绍如何在双 Franka TCP-rot6d 环境中使用 PICO 进行示教数据采集，并以
PICO 人工接管运行在线 Human-Gated DAgger。双臂硬件、实时内核和相机检查请先参考
:doc:`dual_franka`；PICO / XRoboToolkit 数据发布链路请先参考 :doc:`franka_vr`；
HG-DAgger 的单臂流程可参考 :doc:`hg-dagger`。

概览
----------------------------------------

用 PICO 左右手柄分别控制双臂 Franka，先采集 tcp_rot6d 格式 LeRobot 数据，
再参考 :doc:`dual_franka` 准备 OpenPI π₀.₅ student checkpoint，最后在真机上启动在线 HG-DAgger。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenPI π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      SFT · HG-DAgger

   .. grid-item-card:: 任务
      :text-align: center

      Dual-arm manipulation

   .. grid-item-card:: 硬件
      :text-align: center

      2× Franka · PICO · 3 cameras

| **你将完成:** 启动 PICO publisher → 采集双臂 tcp_rot6d 示教 → 复用双臂 SFT checkpoint 流程 → 运行在线 HG-DAgger.
| **前置条件:** :doc:`dual_franka` · :doc:`franka_vr` · OpenPI π₀.₅ checkpoint · Ray cluster.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - 任务
     - 配置 / 入口
     - 说明
   * - PICO stream
     - ``vr_data_publisher``
     - 从 PICO / XRoboToolkit 发布左右手柄和按键数据。
   * - Collection
     - ``realworld_dual_franka_collect_data_pico``
     - 用 PICO 双手遥操作采集 tcp_rot6d LeRobot 数据。
   * - HG-DAgger
     - ``realworld_dual_franka_dagger_openpi``
     - 策略自主执行，PICO 按臂接管；完整成功轨迹进入 online LeRobot dataset。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - 左腕、右腕、全局相机，加双臂 TCP / 夹爪状态。
   * - Action
     - 双臂 tcp_rot6d：``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``。
   * - Reward
     - 脚踏人工标记的成功 / 失败信号。
   * - Prompt
     - ``task_description`` 写入数据并作为 OpenPI 语言条件。


安装与节点布局
----------------------------------------

软件环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

机器人节点
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在每台直接与 Franka 通信的机器人节点上分别执行机器人节点安装。根据 Franka 官方
`compatibility matrix <https://frankarobotics.github.io/docs/compatibility.html>`_
选择 ``LIBFRANKA_VERSION``；避免使用 libfranka ``0.18.0``。

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   export LIBFRANKA_VERSION=0.15.0       # 替换为与固件兼容的版本
   bash requirements/install.sh embodied --env franka-franky --use-mirror
   source .venv/bin/activate

``franka-franky`` 环境会安装 ``franka`` extra，其中包含 PICO consumer 侧所需的
``pyzmq``。PICO 头显、XRoboToolkit PC Service 和 ``vr_data_publisher`` 的安装与验证
流程见 :doc:`franka_vr`。

推理节点
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行在线 DAgger actor / rollout 的 GPU 推理节点使用 OpenPI 环境：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Ray 节点布局
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

采集配置使用两个 Franka 节点：

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Rank
     - 角色
     - 注意事项
   * - ``0``
     - 左臂控制、env worker、三路相机、PICO consumer
     - 需要脚踏设备和 PICO ZeroMQ 地址可达。
   * - ``1``
     - 右臂控制
     - 只需要右臂 Franka / Robotiq 控制链路。

在线 DAgger 配置使用三个节点：

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Rank
     - 角色
     - 注意事项
   * - ``0``
     - inference / rollout / actor
     - 通常为 GPU 节点，运行 OpenPI。
   * - ``1``
     - 左臂控制、env worker、三路相机、PICO consumer
     - 需要脚踏设备和 PICO ZeroMQ 地址可达。
   * - ``2``
     - 右臂控制
     - 只需要右臂 Franka / Robotiq 控制链路。

.. warning::

   Ray 会在 ``ray start`` 时捕获 Python 解释器和环境变量。请在启动 Ray 前完成
   ``source .venv/bin/activate``、``PYTHONPATH``、``RLINF_NODE_RANK``、
   ``RLINF_KEYBOARD_DEVICE`` 和 ROS / Franka 相关环境变量配置。

集群设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在正式开始实验之前，需要先正确地搭建 ray 集群。

.. warning::
  这一步非常关键，请谨慎操作！任何细微的配置错误，都可能导致依赖缺失或无法正确控制机器人。

RLinf 使用 ray 来管理分布式环境，这意味着：
当你在某个节点上执行 `ray start` 时，ray 会记录当时的 Python 解释器路径和相关环境变量；
之后在该节点上由 ray 启动的所有进程都会继承同一套 Python 环境与环境变量。

我们提供了脚本 ``ray_utils/realworld/setup_before_ray.sh``，
用于在每个节点启动 ray 之前帮助你统一设置环境。你可以根据自己的环境修改该脚本，并在每个节点上 source 它。

该脚本主要负责以下内容：

1. 在使用自定义环境安装方式时，source 正确的虚拟环境；

2. 在 Franka 控制节点上，加载 Franka / Robotiq / 相机所需的运行环境；

3. 在所有节点上设置 RLinf 相关环境变量：

.. code-block:: bash

   export PYTHONPATH=<path_to_your_RLinf_repo>:$PYTHONPATH
   export RLINF_NODE_RANK=<node_rank_of_this_node>
   export RLINF_COMM_NET_DEVICES=<network_device_for_communication> # 如果只有一个网卡可以省略

其中 ``RLINF_NODE_RANK`` 应在集群的 ``N`` 个节点之间设置为 ``0 ~ N-1``，
用来在配置文件中唯一标识每个节点。PICO consumer / env worker 所在节点还需要在
``ray start`` 前导出脚踏设备：

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

采集配置中 ``N=2``，rank ``0`` 为左臂 / env / PICO consumer，rank ``1`` 为右臂。
DAgger 配置中 ``N=3``，rank ``0`` 为 OpenPI inference / actor，rank ``1`` 为左臂 /
env / PICO consumer，rank ``2`` 为右臂。

在完成上述环境设置后，可以按如下方式在各节点上启动 ray：

其中 `<head_node_ip_address>` 为 head 节点的 IP 地址，**必须** 能被集群中其他节点访问。

.. code-block:: bash

   # 在 head 节点（节点 rank 0）上
   ray start --head --port=6379 --node-ip-address=<head_node_ip_address>

   # 在 worker 节点（节点 rank 1 ~ N-1）上
   ray start --address='<head_node_ip_address>:6379'

可以通过执行 `ray status` 来检查集群是否已正确启动。


配置
----------------------------------------

主要配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - 配置
     - 用途
   * - ``examples/embodiment/config/realworld_dual_franka_collect_data_pico.yaml``
     - PICO 双臂 tcp_rot6d 数据采集。
   * - ``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml``
     - PICO 人工接管的在线 HG-DAgger。
   * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
     - 双臂 TCP-rot6d 真机环境默认配置。

硬件占位符
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在采集和 DAgger 配置中替换以下字段：

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``：左右臂 FCI IP。
* ``BASE_CAMERA_SERIAL``、``LEFT_CAMERA_SERIAL``、``RIGHT_CAMERA_SERIAL``：
  RealSense / Lumos 相机 serial 或稳定 ``/dev/v4l/by-id`` 路径。
* ``base_camera_type``、``left_camera_type``、``right_camera_type``：相机类型，
  通常为 ``realsense``、``lumos``、``lumos``。
* ``left_gripper_type`` / ``right_gripper_type``：夹爪类型，Robotiq 夹爪填
  ``robotiq``。
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION``：Robotiq 串口路径。
* ``left_controller_node_rank`` / ``right_controller_node_rank``：左右臂控制节点
  rank。采集配置通常为 ``0`` / ``1``；DAgger 三节点配置通常为 ``1`` / ``2``。
* ``node_rank``：DualFranka 硬件配置所在的 env / PICO consumer 节点 rank。采集配置
  通常为 ``0``；DAgger 三节点配置通常为 ``1``。
* ``TASK_DESCRIPTION``：采集和 DAgger 使用的任务文本，应与 checkpoint 训练时一致。
* ``joint_reset_qpos``：根据采集数据首帧关节均值或安全 home pose 设置。
* ``target_ee_pose`` 和 ``ee_pose_limit_min/max``：按工作空间重新确认。

PICO 配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

采集配置中 PICO consumer 地址位于 ``env.eval.pico.zmq_addr``；DAgger 配置中位于
``env.train.pico.zmq_addr``。该地址必须与 publisher 绑定地址匹配：

ZMQ 数据流由 env worker / PICO intervention 所在节点订阅；在本文配置中，采集时为
rank ``0`` 的左臂控制节点，DAgger 时为 rank ``1`` 的左臂控制节点。

使用双臂 PICO 遥操作时，必须将 ``pico.hand`` 设置为 ``"dual"``，这样左 / 右
PICO 手柄才会分别绑定到左 / 右机械臂。

.. code-block:: yaml

   env:
     train:
       use_pico: True
       pico:
         zmq_addr: "tcp://<vr_publisher_ip>:<port>"
         hand: "dual"
         control_trigger: "grip"
         calibration:
           button: "trigger"

如果 publisher 和 env worker 在同一台机器，可以使用 ``ipc:///tmp/vr_data.ipc``。
如果跨机器运行，publisher 侧绑定 ``tcp://0.0.0.0:<port>``，RLinf consumer
侧填写 ``tcp://<vr_publisher_ip>:<port>``，不要把 ``0.0.0.0`` 写到 consumer 配置里。

默认手柄语义：

.. code-block:: text

   左手 grip -> 接管左臂
   右手 grip -> 接管右臂
   左手 X/Y   -> 关 / 开左夹爪
   右手 A/B   -> 关 / 开右夹爪
   trigger    -> 以当前头显朝向重新标定 operator base

采集和 DAgger 的 ``hold_current_when_inactive`` 语义不同：

* 采集配置为 ``True``：未按 ``grip`` 的手臂保持当前 TCP，适合纯遥操作数据采集。
* DAgger 配置为 ``False``：未按 ``grip`` 的手臂保留 rollout action；按下任意一侧
  ``grip`` 时，只用对应侧的 PICO action 覆盖策略动作，另一侧仍保留 rollout action。

双臂 DAgger 的接管与记录按臂合成。例如只接管左臂时，实际执行并写入
``intervene_action`` 的 20D 动作为：

.. code-block:: text

   [左臂 PICO 10D action, 右臂 rollout 10D action]

只接管右臂时则相反。任意一臂发生替换都会设置 ``intervene_flag=True``；只有两臂
都未接管时才不会产生 intervention 记录。完整成功 episode 仍会由 online LeRobot
collector 保存，``intervene_flag`` 用于区分其中的人工纠正帧。


启动 PICO 数据流
----------------------------------------

先在 PICO publisher 机器启动 XRoboToolkit PC Service，再启动 VR 数据发布进程。
示例命令如下，具体安装路径见 :doc:`franka_vr`：

.. code-block:: bash

   cd /opt/apps/roboticsservice
   bash runService.sh

.. code-block:: bash

   cd /path/to/pico_software/XRoboToolkit-Teleop-Sample-Python
   source .venv/bin/activate
   cd /path/to/pico_software
   python -m vr_data_publisher --config configs/vr_bridge.yaml

在运行 env worker 的节点上验证 PICO 数据可达：

.. code-block:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr tcp://<vr_publisher_ip>:<port>

输出持续刷新，并且按下 ``grip``、``trigger``、``A/B``、``X/Y`` 时数值变化，
才继续启动真机采集或 DAgger。


采集 PICO 示教数据
----------------------------------------

执行采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

确认左右 Franka、Robotiq、相机、脚踏、PICO 数据流和采集 Ray 集群都正常后，在
head 节点执行：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_dual_franka_collect_data_pico

脚踏按键：

* ``a``：开始录制；录制过程中再次按下将中止录制并丢弃当前 buffer。
* ``b``：递增 ``segment_id``，用于标记子任务边界。
* ``c``：标记成功，写入 LeRobot shard，并结束当前 episode。

PICO 操作：

1. 戴好头显并面向工作台正前方。
2. 扣下 ``trigger`` 完成 PICO base 标定。
3. 按住左 / 右手 ``grip`` 分别接管左 / 右臂。
4. 用 ``X/Y`` 控制左夹爪，用 ``A/B`` 控制右夹爪。
5. 松开某只手的 ``grip`` 后，对应手臂保持当前 TCP。

采集脚本会在 ``logs/<timestamp>/`` 下写出：

* replay-buffer 轨迹：``demos/``
* LeRobot 数据：``collected_data/rank_0/id_0/``，后续 shard 为 ``id_1``、``id_2``

PICO 双臂采集已经使用 ``realworld_dual_franka_tcp_rot6d`` 环境，数据动作就是
tcp_rot6d；因此不需要执行 GELLO 流程中的 ``backfill_tcp_rot6d.py``。

.. note::

   配置中 ``data_collection.resume: True`` 只会在相同 ``save_dir`` 下续写。
   ``collect_data.sh`` 默认每次创建新的 ``logs/<timestamp>``，若要多次追加到同一数据集，
   请把 ``data_collection.save_dir`` 改为固定路径。


准备 checkpoint
----------------------------------------

在线 DAgger 需要一个可部署的 OpenPI checkpoint。数据整理、norm stats、SFT
和 checkpoint 目录准备请直接参考 :doc:`dual_franka` 中的 SFT 与部署 checkpoint
流程。

使用本页 PICO 采集的数据时，数据已经来自 ``realworld_dual_franka_tcp_rot6d`` 环境，
不需要再执行 GELLO 关节数据流程中的 ``backfill_tcp_rot6d.py``。

完成 checkpoint 准备后，在
``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml`` 中设置：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/deploy/global_step_<N>

   actor:
     model:
       openpi_data:
         repo_id: <repo_id>/tcp_rot6d_v1


运行在线 HG-DAgger
----------------------------------------

检查 DAgger 关键配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

启动前确认以下字段：

.. code-block:: yaml

   algorithm:
     dagger:
       only_save_expert: False
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
       use_pico: True
       keyboard_reward_wrapper: eval_control
       pico:
         zmq_addr: "tcp://<vr_publisher_ip>:<port>"
         hand: "dual"
         hold_current_when_inactive: False
     eval:
       use_pico: False

``online_lerobot.enabled: True`` 表示启用在线 LeRobot 数据链路。env worker 按 episode 收集 rollout，并将满足过滤条件的 episode 发送给 actor；actor 将其加入 ``RollingLeRobotDataset`` 进行训练，因此在线训练不再使用 trajectory replay buffer。

``only_success: True`` 表示失败 rollout 会被丢弃，只保存成功 episode；
``only_save_expert: False`` 表示成功 episode 的整段轨迹都参与训练，而不是只采样
PICO 接管帧。每个成功 episode 会立即归档到
``${runner.logger.log_path}/online_lerobot/rank_0/id_<N>/``。
``env.eval.use_pico: False`` 表示评测阶段只看策略本身，不混入人工接管。

真机 DAgger 配置不包含 beta 相关字段，因为没有配置 ``rollout.expert_model``。Beta 只用于模型 expert 和 student 之间的动作混合；这里的人工接管由 PICO intervention wrapper 决定。

执行 DAgger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

确认 DAgger Ray 集群已经启动后，在 head 节点启动在线训练：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dual_franka_dagger_openpi

运行中的操作方式：

* ``a``：从 idle 状态启动一次策略 rollout。
* 左 / 右手 ``grip``：接管对应机械臂；未接管的手臂继续执行策略动作。
* ``b``：标记失败并结束当前 rollout。
* ``c``：标记成功并结束当前 rollout。

每次 episode 结束后，env 会 reset 并再次等待 ``a``。策略执行中只在需要纠正时按住
``grip``，松开后让策略继续运行。按住任意一侧 ``grip`` 时，该侧 PICO action 与
另一侧 rollout action 会合成为完整 20D ``info["intervene_action"]``。成功结束后，
整条 episode 经内存发送给 actor，并写入 online LeRobot shard；失败 episode 被丢弃。

监控
----------------------------------------

启动 TensorBoard：

.. code-block:: bash

   tensorboard --logdir ./logs

推荐关注：

* ``train/dagger/actor_loss``：基于完整成功轨迹的监督损失。
* ``train/lerobot_dataset/total_episodes``：actor 已接收的成功 episode 数量。
* ``train/lerobot_dataset/physical_frames``：已接收的 LeRobot 物理帧数量。
* ``train/lerobot_dataset/logical_samples``：rolling window 内可采样的训练样本数。
* ``train/lerobot_dataset/num_sub_datasets``：当前加载的 LeRobot shard 数量。
* ``train/actor/lr`` 和 ``train/actor/grad_norm``：训练稳定性。

采集和在线 DAgger 阶段都可以查看 ``logs/<timestamp>/run_embodiment.log``，确认成功
episode 计数和 LeRobot 写出路径。在线 DAgger shard 位于
``logs/<timestamp>-realworld_dual_franka_dagger_openpi/online_lerobot/rank_0/``。


故障排查
----------------------------------------

**DAgger 等待时间过长未启动**
   这是 ``keyboard_reward_wrapper: eval_control`` 的预期行为。若 DAgger 运行后长时间
   等待未开始，可踩一下映射到键盘 ``a`` 的脚踏启动 rollout。
