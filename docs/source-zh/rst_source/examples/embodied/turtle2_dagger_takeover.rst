Turtle2 接管式 Raw 数据采集
===========================

本示例展示如何在 XSquare Turtle2 上运行 **policy rollout + master takeover**
的 raw episode 采集流程。它的目标是把 policy 自主执行、人工接管、接管后的
hold/recovery，以及真实下发到机器人的动作一起保存下来，供调试、审计和后续
离线清洗使用。

这个示例 **不是在线 DAgger 训练配置**。它不会启动 SAC、DAgger actor update，
也不会修改 OpenPI loss。训练数据如果需要进入 SFT 或 DAgger，应当在采集后从
pickle episode 中再清洗出需要的 expert segments。

整体链路
--------

运行时只保留一条机器人执行链路：

.. code-block:: text

   OpenPI policy
      -> 14D absolute pose + gripper action
      -> DualAbsolutePoseActionWrapper
      -> MasterTakeoverIntervention
      -> Turtle2 direct pose backend
      -> /follow_pos_cmd_1, /follow_pos_cmd_2
      -> CollectEpisode pickle export

``MasterTakeoverIntervention`` 只负责选择当前执行哪一种 action：

- normal mode 下直接放行 policy action；
- takeover mode 下，如果收到 fresh master pose，则用 master pose 覆盖 policy action；
- takeover mode 下如果 master pose 还没准备好，则保持当前 slave pose；
- 从 takeover 退出后，在当前 action chunk 结束前继续 hold，避免执行旧 policy chunk
  中尚未执行的 stale action。

最终动作始终进入同一个 Turtle2 direct pose backend。也就是说，policy action、
master takeover action 和后续 debug/replay action 只要都是 14D absolute pose，
就不会因为来源不同而走不同执行器。

动作与控制语义
--------------

本示例固定使用双臂 14D 动作：

.. code-block:: text

   [left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper,
    right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper]

需要注意：

- ``action_mode`` 必须是 ``absolute_pose``。master 发送的是绝对末端位姿，
  不能被解释成 relative/delta action。
- ``pose_control_backend`` 设置为 ``direct``。direct backend 会尽量按
  ``PosCmd`` 语义直接发布到 ``/follow_pos_cmd_1`` 和 ``/follow_pos_cmd_2``，
  不走 RLinf 的 smooth interpolation 路径。
- direct backend 会先做 shape、finite 和安全边界检查。被接受的动作满足
  ``executed_action == raw_action``；被拒绝的动作不会发布，机器人保持上一条
  accepted action，并在 info 中记录 ``action_rejected`` 和 ``rejection_reason``。

纯部署场景仍然可以使用 Turtle2 deploy 默认的 ``relative_pose``，前提是策略本身
就是按 delta action 训练的。takeover raw collect 必须使用 ``absolute_pose``，
因为 master pose 已经是绝对位姿。如果 checkpoint 是用 relative/delta action
训练得到的，它可以用于普通 ``relative_pose`` 部署，但不能直接用于本示例的
takeover collect；否则 policy action 和 master action 的语义会不一致。

Takeover 协议
-------------

RLinf env 侧会启动 TCP server，master 端作为 client 连接到该 server。协议保留
X2Robot takeover 的三类 frame：

- ``MSG_MODE``：同步当前模式，例如 normal/policy mode 与 takeover mode。
- ``MSG_JOINT``：RLinf 将 slave 当前 joint snapshot 发给 master，用于 master
  在接管前对齐到 slave 当前姿态。
- ``MSG_POSE``：master 向 RLinf 发送 14D absolute pose。

``MSG_JOINT`` 只用于接管前同步。机器人实际执行的动作来自 policy 或 master
发送的 14D pose，并由同一个 direct pose backend 下发。

模式切换通过 ROS 参数完成：

.. code-block:: bash

   # policy rollout
   rosparam set /running_mode 1

   # master takeover
   rosparam set /running_mode 2

   # return to policy at the next action chunk boundary
   rosparam set /running_mode 1

当 ``running_mode_source`` 配置为 ``ros_param`` 时，真实运行环境中必须可以导入
``rospy``，且 ``/running_mode`` 必须能转成整数。否则 RLinf 会显式报错，而不是
静默回退到 policy mode。

运行前提
--------

运行本示例前，需要先确认以下条件：

- Turtle2 ROS、SDK、相机和 follower controller 已经启动并能读到状态。
- ``/follow_pos_cmd_1`` 和 ``/follow_pos_cmd_2`` 只能由 RLinf direct backend 发布；
  不要同时启动旧的 slave 执行器或其他会写 follower pose command 的进程。
- master 端程序已经连接到 RLinf takeover TCP server，并能发送 ``MSG_POSE``。
- ``/running_mode`` 使用整数模式值；本示例默认 ``1`` 为 policy，``2`` 为 takeover。
- OpenPI checkpoint 路径通过 ``RLINF_OPENPI_MODEL_PATH`` 或配置中的
  ``rollout.model.model_path`` 指定。

多机启动流程
------------

下面给出一个通用的三端启动顺序。具体的容器名、网卡名、IP、ROS setup 文件和
master 程序路径需要按现场部署替换。

1. Turtle2 控制节点：启动底层 ROS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Turtle2 控制节点负责 ROS master、机器人 bring-up、相机和 follower controller。
如果使用双容器部署，控制容器负责底层 ROS，RLinf 容器只负责 Ray worker 和
RLinf env。

.. code-block:: bash

   # 在 Turtle2 控制节点执行
   ssh <turtle2_worker_host>

   # 进入 Turtle2 控制容器或等效 ROS 环境
   <enter_turtle2_ros_env>

   # 启动稳定 ROS master
   tmux kill-session -t stable_roscore 2>/dev/null || true
   tmux new-session -d -s stable_roscore \
     'source /opt/ros/noetic/setup.bash; while true; do roscore; sleep 1; done'

   # 启动 Turtle2 底层 bring-up。具体命令以现场 XSquare/Turtle2 安装为准。
   bash <path_to_turtle2_bring_up>/run.sh S

   # 设置默认 policy mode
   rosparam set /running_mode 1

   # 基础检查：状态、相机和 command topic 必须可见
   rostopic info /follow_pos_cmd_1
   rostopic info /follow_pos_cmd_2
   timeout 10 rostopic echo -n 1 /follow1_pos_back
   timeout 10 rostopic echo -n 1 /follow2_pos_back
   rostopic info /camera1/usb_cam1/image_raw
   rostopic info /camera2/usb_cam2/image_raw
   rostopic info /camera3/usb_cam3/image_raw

如果临时使用 UI 做校准，校准后请关闭会发布 ``/follow_pos_cmd_*`` 的 UI 或旧执行器。
运行本示例时，这两个 command topic 应由 RLinf direct backend 独占。

2. GPU / Ray head 节点：启动 Ray head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU 节点负责 OpenPI policy 推理、rollout 入口和 Ray head。

.. code-block:: bash

   # 在 GPU / Ray head 节点执行
   ssh <gpu_head_host>
   cd <path_to_rlinf_repo>
   source .venv/bin/activate

   export RLINF_NODE_RANK=0
   export RLINF_COMM_NET_DEVICES=<head_network_interface>
   export PYTHONPATH=<path_to_rlinf_repo>:$PYTHONPATH
   export RLINF_OPENPI_MODEL_PATH=<path_to_openpi_checkpoint>

   ray stop --force || true
   ray start --head \
     --node-ip-address=<head_ip> \
     --port=6379 \
     --disable-usage-stats

3. Turtle2 / Ray worker 节点：启动 RLinf worker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf worker 必须在能访问 Turtle2 ROS 环境的 shell 或容器中启动。Ray 启动时会记录
当前 Python 和环境变量，后续 EnvWorker 会继承这些配置。

.. code-block:: bash

   # 在 Turtle2 worker 节点执行
   ssh <turtle2_worker_host>

   # 进入 RLinf 运行环境，并 source Turtle2 ROS / SDK setup
   <enter_rlinf_env>
   cd <path_to_rlinf_repo>
   source /opt/ros/noetic/setup.bash
   source <path_to_turtle2_ros_setup>
   source .venv/bin/activate

   export RLINF_NODE_RANK=1
   export RLINF_COMM_NET_DEVICES=<worker_network_interface>
   export PYTHONPATH=<path_to_rlinf_repo>:$PYTHONPATH

   ray stop --force || true
   ray start --address=<head_ip>:6379 \
     --node-ip-address=<worker_ip> \
     --disable-usage-stats

   rosparam set /running_mode 1

回到 head 节点，用 ``ray status --address=<head_ip>:6379`` 确认 GPU 节点和 Turtle2
worker 都已加入集群。

4. Master 节点：启动 master 端程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

master 节点负责采集操作者输入，并作为 TCP client 连接到 RLinf env 侧的 takeover
server。请先启动 master 双臂的 ROS 节点，再启动发送 takeover frame 的 client。
如果夹爪输入依赖额外的通信节点，也需要在 master 端先启动。

.. code-block:: bash

   # 在 master 节点执行
   ssh <master_host>
   source /opt/ros/noetic/setup.bash
   source <path_to_master_ros_setup>

   # 启动 master 双臂，具体 launch 文件按现场安装替换
   roslaunch <master_robot_launch>

   # 如果夹爪 / 手柄输入依赖单独通信节点，先启动它
   roslaunch <master_input_launch>

   # 启动 master TCP client，连接 Turtle2 worker 上的 RLinf takeover server
   cd <path_to_x2robot_master_scripts>
   python3 bi_teleop_master.py \
     --server-host <worker_ip> \
     --server-port <master_takeover_port> \
     --send-rate 100

``master_takeover.port`` 由 YAML 中的 ``env.eval.master_takeover.port`` 决定。
如果现场已有旧 slave server 占用默认端口，请在 YAML 和 master 启动命令中使用同一个
未占用端口。

5. Head 节点：启动 eval 和 raw collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 Ray head 节点运行：

.. code-block:: bash

   cd <path_to_rlinf_repo>
   source .venv/bin/activate

   export RLINF_NODE_RANK=0
   export RLINF_COMM_NET_DEVICES=<head_network_interface>
   export PYTHONPATH=<path_to_rlinf_repo>:$PYTHONPATH
   export RLINF_OPENPI_MODEL_PATH=<path_to_openpi_checkpoint>
   export RLINF_REALWORLD_DATA_DIR=<path_to_save_pickle_episodes>

   bash examples/embodiment/run_realworld_eval.sh realworld_turtle2_dagger_takeover_collect_openpi

示例配置文件是：

.. code-block:: text

   examples/embodiment/config/realworld_turtle2_dagger_takeover_collect_openpi.yaml

关键字段包括：

.. code-block:: yaml

   runner:
     only_eval: True

   rollout:
     collect_transitions: False

   env:
     eval:
       use_master_takeover: True
       action_mode: absolute_pose
       data_collection:
         enabled: True
         export_format: "pickle"
         record_executed_action: True
       master_takeover:
         running_mode_source: "ros_param"
         running_mode_param: "/running_mode"
         normal_mode_value: 1
         takeover_mode_value: 2
       override_cfg:
         pose_control_backend: direct
         use_arm_ids: [0, 1]

``algorithm.adv_type`` 和 ``algorithm.loss_type`` 在该配置中只使用已注册的占位值，
用于通过通用配置校验。因为 ``runner.only_eval`` 为 true，本示例不会实例化训练
loss，也不会执行在线训练更新。

6. 切换 takeover mode
~~~~~~~~~~~~~~~~~~~~~

等 eval 日志中出现 master client 已连接的信息后，在 Turtle2 控制节点切换
``/running_mode``：

.. code-block:: bash

   # policy rollout
   rosparam set /running_mode 1
   sleep 5

   # master takeover
   rosparam set /running_mode 2
   sleep 5

   # return to policy
   rosparam set /running_mode 1

切回 ``1`` 后，RLinf 会在 action chunk 边界恢复 policy 推理。

7. 验证保存结果
~~~~~~~~~~~~~~~

运行结束后，检查 ``RLINF_REALWORLD_DATA_DIR`` 或配置中的 ``data_collection.save_dir``。
pickle episode 中应包含 ``observations``、``actions``、``infos``、``success`` 和
``intervened`` 等字段。若发生接管，``infos`` 中可以看到 ``intervene_flag``、
``intervene_action``、``takeover_active`` 等 metadata。

数据保存语义
------------

本示例推荐使用 ``pickle`` 导出 raw episode，而不是直接导出 LeRobot 数据集。

原因是 takeover raw episode 通常需要保留完整上下文：

- policy 自主执行段；
- master 接管段；
- 接管前同步和 hold；
- 退出接管后的 chunk-boundary recovery；
- 被拒绝动作的 ``raw_intervene_action``、``action_rejected``、
  ``rejection_reason`` 等 metadata；
- 真实执行或保持的 ``executed_action``。

配置中设置了：

.. code-block:: yaml

   data_collection:
     export_format: "pickle"
     record_executed_action: True

因此 ``episode["actions"]`` 会优先记录真实执行动作：

- 如果 backend 接受 action，则记录 accepted ``executed_action``；
- 如果 takeover action 被接受，则 ``intervene_action`` 与 ``executed_action`` 对齐；
- 如果 takeover action 被拒绝，则 rejected candidate 不会写成 action label，只保留在
  ``episode["infos"]`` 中，``episode["actions"]`` 记录机器人实际保持的动作。

``CollectEpisode`` 仍然支持 LeRobot 导出，但 LeRobot 更适合已经清洗过的干净示教
数据集。本示例保存的是 raw takeover episode。如果后续要训练 OpenPI，建议先从
pickle episode 中清洗出可用 expert segments，再单独转换为 LeRobot。

注意事项
--------

- 本示例是 eval-only raw collection。采集到的 pickle episode 是否用于训练，
  由后续离线清洗和训练配置决定。
- 不要同时启动其他会发布 ``/follow_pos_cmd_1`` 或 ``/follow_pos_cmd_2`` 的执行器。
- 如果 policy checkpoint 是 relative/delta action 训练得到的，请使用普通
  ``relative_pose`` 部署配置；不要直接套用本 takeover collect 配置。
