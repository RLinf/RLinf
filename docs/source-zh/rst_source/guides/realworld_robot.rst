真机机器人训练启动
====================

本页说明如何将 **多台 Franka 真机** 与 **GPU 训练节点** 接入同一 Ray 集群，完成 YAML 配置并启动真机强化学习训练。

更完整的硬件安装、依赖、数据采集与任务说明，请参阅：

- :doc:`../examples/embodied/franka` — RealSense + Franka 夹爪的标准真机流程
- :doc:`../examples/embodied/franka_zed_robotiq` — ZED 相机 + Robotiq 夹爪的异构双节点部署

Ray 集群的基础启动步骤（``RLINF_NODE_RANK``、``ray start``、代码同步等）见 :doc:`multi_node`；
``node_groups`` / ``component_placement`` 的通用语义见 :doc:`hetero`。


典型拓扑
--------

真机训练一般采用 **「1 个 GPU 训练节点 + N 个机器人控制节点」** 的异构布局：

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``RLINF_NODE_RANK``
     - 角色
     - 说明
   * - rank 0（head）
     - GPU 训练 / rollout
     - 运行 ``actor`` 、``rollout`` （及可选 ``reward``）；仅在此节点提交训练入口脚本
   * - rank 1 至 N
     - 机器人控制
     - 运行 ``env`` worker 与 ``FrankaController``；每台机械臂对应一个控制节点 rank（或多机共用一台控制节点时需单独配置，见示例文档）

所有节点须处于 **同一局域网** （或 overlay 网络，见 :doc:`cloud_edge`），且 ``cluster.num_nodes`` 等于实际加入 Ray 的节点总数。

.. important::

   - 控制节点需预先安装 Franka 相关依赖（ROS、libfranka 等），详见 :doc:`../examples/embodied/franka`。
   - Ray 在 ``ray start`` 时冻结 Python 与环境变量；请在 **每个节点** 安装好依赖后再启动 Ray。
   - 可使用 ``ray_utils/realworld/setup_before_ray.sh`` 统一设置各节点环境，再执行 ``ray start``。


将机器人节点接入 Ray 集群
---------------------------

步骤 1：在各节点准备环境
~~~~~~~~~~~~~~~~~~~~~~~~

在 **每个节点**、执行 ``ray start`` **之前**：

.. code-block:: bash

   # 建议在各节点 source 仓库脚本并按本机修改
   source ray_utils/realworld/setup_before_ray.sh

   export RLINF_NODE_RANK=<0..N-1>          # 集群内唯一；GPU 节点通常为 0
   # 多网卡时指定对外可达的网卡，例如：
   # export RLINF_COMM_NET_DEVICES=eth0

控制节点还需 source ROS / franka 工作空间（若未写入 ``setup_before_ray.sh``）：

.. code-block:: bash

   source <your_catkin_ws>/devel/setup.bash

步骤 2：启动 Ray
~~~~~~~~~~~~~~~~

记 GPU head 节点对外 IP 为 ``<head_ip>``。

**GPU 节点（rank 0，head）：**

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

**各机器人控制节点（rank 1, 2, …）：**

.. code-block:: bash

   export RLINF_NODE_RANK=1   # 第二台机器人则为 2，以此类推
   ray start --address='<head_ip>:6379'

在任意节点执行 ``ray status``，确认节点数与 ``cluster.num_nodes`` 一致。


YAML 配置
---------

真机训练的核心在 ``cluster`` 段：用 ``node_groups`` 区分 GPU 与 Franka 硬件，用 ``component_placement`` 把 ``actor`` 、``rollout`` 、``env`` 放到对应资源上。

配置前请根据实际硬件修改：

- 任务目标位姿：``robot_ip`` 、``target_ee_pose``
- 预训练模型路径：``actor.model.model_path``
- 离线 demo 数据（RLPD 等）：``algorithm.demo_buffer`` 、``data.path``

单机器人（1 GPU + 1 机械臂）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

参考 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0          # 该节点组内第 1 台机器人
       rollout:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0        # GPU 训练节点
       - label: franka
         node_ranks: 1        # 机器人控制节点
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP>
               node_rank: 1   # 与 node_ranks 中的控制节点 rank 一致

若你的 GPU 节点不是 rank 0、控制节点不是 rank 1，请同步修改 ``node_ranks`` 与 ``hardware.configs[].node_rank``。

多机器人（1 GPU + 2 台机械臂）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

参考 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_2arms.yaml``：

.. code-block:: yaml

   cluster:
     num_nodes: 3
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0-1       # 两台机器人各一个 env worker
       rollout:
         node_group: "4090"
         placement: 0:0-1     # 同一 GPU 上两个 rollout 进程
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1-2
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP_1>
               node_rank: 1
             - robot_ip: <ROBOT_IP_2>
               node_rank: 2

按相同方式为第三台、第四台机械臂扩展 ``num_nodes``、``node_ranks``、``placement`` 与 ``hardware.configs``。

相机与机械臂分机部署（ZED + Robotiq）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当 **相机在 GPU 服务器**、**机械臂与夹爪在 NUC** 时，需在 ``hardware.configs`` 中指定相机/夹爪类型及控制器所在节点。
字段说明与采集配置示例见 :doc:`../examples/embodied/franka_zed_robotiq`。

训练时可将 ``env`` 放在 GPU 节点组（负责相机采集），并通过 ``controller_node_rank`` 将 ``FrankaController`` 固定到 NUC：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: gpu
         placement: 0
       env:
         node_group: gpu
         placement: 0
       rollout:
         node_group: gpu
         placement: 0
     node_groups:
       - label: gpu
         node_ranks: 0
       - label: franka
         node_ranks: 0-1
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP>
               node_rank: 0
               camera_serials:
                 - "<ZED_SERIAL>"
               camera_type: zed
               gripper_type: robotiq
               gripper_connection: "/dev/ttyUSB0"
               controller_node_rank: 1   # 控制器运行在 rank 1（NUC）

.. note::

   ZED SDK 须在 GPU 节点、在 ``ray start`` **之前** 安装到 Ray 使用的同一虚拟环境中；
   Robotiq 串口权限须在 NUC 控制节点上配置好。详见 :doc:`../examples/embodied/franka_zed_robotiq`。

从环境变量自动配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

各机器的硬件取值（``robot_ip`` 、``camera_serials`` 、``gripper_connection`` 等）不必写死在 YAML 中。
在枚举（enumeration）阶段，``hardware.configs`` 中任何 **未设置** 的字段都会从与字段同名、**全大写** 的环境变量自动填充
（``robot_ip`` → ``ROBOT_IP`` 、``camera_serials`` → ``CAMERA_SERIALS``），从而把 IP、序列号等从提交的配置中剥离出来。

规则：

- YAML 中写明的值始终优先；环境变量只填充未设置的字段。
- 变量在 **拥有该配置的节点上** （即其 ``node_rank``）读取，因此须在该节点 ``ray start`` **之前** 导出（Ray 在启动时冻结环境变量）。
- ``node_rank`` 与 ``controller_node_rank`` 不会从环境变量读取。

**每个节点一台机器人。** 直接使用整个取值；列表字段按逗号拆分：

.. code-block:: yaml

   - label: franka
     node_ranks: 1
     hardware:
       type: Franka
       configs:
         - node_rank: 1          # robot_ip / 相机序列号来自环境变量

.. code-block:: bash

   # 在 rank 1 控制节点、`ray start` 之前
   export ROBOT_IP=<ROBOT_IP>
   export CAMERA_SERIALS=<serial_1>,<serial_2>

**同一节点多台机器人。** 为每台机器人提供一个逗号分隔的取值，按顺序赋给各配置（因此列表字段在此模式下每台机器人取一项）。
取值个数须与该节点上机器人配置的数量一致：

.. code-block:: bash

   # 一台控制节点上两台机械臂
   export ROBOT_IP=<ip_arm0>,<ip_arm1>
   export CAMERA_SERIALS=<serial_arm0>,<serial_arm1>

**完全由环境变量创建机器人。** 将 ``configs`` 留空，机器人将完全从环境变量创建——机器人数量由标识变量 ``ROBOT_IP`` 的逗号个数决定，
每个字段都从对应变量读取：

.. code-block:: yaml

   - label: franka
     node_ranks: 1
     hardware:
       type: Franka
       configs: []              # 机器人由 ROBOT_IP、CAMERA_SERIALS… 创建

.. code-block:: bash

   export ROBOT_IP=<ip_arm0>,<ip_arm1>     # 两台机器人
   export DISABLE_VALIDATE=true,true       # 每台机器人一个取值

此模式下，所设置的每个变量都必须提供与 ``ROBOT_IP`` 相同数量的逗号分隔取值；数量不一致会使枚举报错并中止。
其他机器人类型使用各自的标识字段（``GimArm`` → ``CAN_INTERFACE`` 、``DOSW1`` → ``ROBOT_URL``）。


启动真机训练
------------

完成 Ray 集群搭建、YAML 修改与（若使用 RLPD）demo 数据准备后，在 **head 节点（通常为 rank 0 的 GPU 机器）** 进入 RLinf 仓库目录执行。
若 GPU 节点与机器人控制节点 **未共享同一 RLinf 代码目录**，可在运行下方训练命令 **之前** 于同一终端执行 ``export RLINF_CODE_WORKING_DIR=auto`` 开启代码同步（详见 :doc:`multi_node` 中「步骤 3：开启代码同步」）。

**标准单臂训练（插块插入，RLPD + 异步 SAC）：**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

**双臂并行训练：**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async_2arms

**其他任务示例（按需替换配置名）：**

.. code-block:: bash

   # 充电器任务
   bash examples/embodiment/run_realworld_async.sh realworld_charger_sac_cnn_async

   # 异步 PPO
   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_async_ppo_cnn

``<config_name>`` 对应 ``examples/embodiment/config/<config_name>.yaml``，也可传入自定义配置名。

**可选：启动前用 dummy 配置验证集群（单臂）：**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dummy_franka_sac_cnn

正式训练前可在控制节点验证相机、在 head 节点用 dummy 配置确认 Ray 与 placement 是否正确；完整检查项见 :doc:`../examples/embodied/franka`。
