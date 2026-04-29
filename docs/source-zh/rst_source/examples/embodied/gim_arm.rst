GimArm真机强化学习
============================

本文档介绍如何在 RLinf 框架中集成 GimArm 六自由度机械臂，内容涵盖硬件配置、依赖安装以及实验运行步骤。

环境
-----------

**真实世界环境**

- **Environment**: 真机设置

  - GimArm 六自由度机械臂（``gim_arm`` 或 ``gim_arm_xl`` 变体）
  - 达妙（Damiao）伺服电机（J1-3 使用 DM4340 / DM6248P，J4-6 使用 DM4310）
  - CAN-USB 适配器（通过 SocketCAN 接口通信）
  - Intel RealSense 相机（默认）或 Stereolabs ZED 相机
  - 可选夹爪（平行夹爪或单侧夹爪，内置达妙电机）

- **Task**: 目前支持 Peg-Insertion（插块）任务（``GimArmPegInsertionEnv-v1``）。
- **Observation**:

  - 腕部相机的 RGB 图像（128×128）。
  - 状态字典，包含：``tcp_pose`` (7,)、``tcp_vel`` (6,)、``arm_joint_position`` (6,)、``gripper_position`` (1,)、``tcp_force`` (3,)、``tcp_torque`` (3,)。

- **Action Space**: 7 维连续动作：

  - 6 维绝对关节角度（弧度），受配置的关节限位约束。
  - 1 维二值夹爪指令（``[-1, 1]`` 范围，代表开/合）。

- **Reward**: 在笛卡尔空间中通过 FK（正运动学）计算当前 TCP 位姿与 ``target_ee_pose`` 的差值。默认使用稀疏奖励（0/1），也可选用指数衰减的稠密奖励。

Peg Insertion 任务
~~~~~~~~~~~~~~~~~~~~

Peg Insertion 任务（``GimArmPegInsertionEnv``，注册名为 ``GimArmPegInsertionEnv-v1``）实现在
``rlinf/envs/realworld/gim_arm/tasks/peg_insertion.py``。它继承自 ``GimArmEnv``，额外定义了任务相关的
reset 与奖励逻辑：

- **Reset**: 夹爪先夹住 peg，随后机械臂回缩到 ``safe_retract_qpos`` 以远离插孔，再移动至 ``reset_joint_qpos``。
  当 ``enable_random_reset`` 打开（默认开启）时，会在 reset 关节角上施加由 ``random_joint_noise``
  （默认 0.02 rad）控制的小幅扰动，以增加数据多样性。

- **Reward**: 在笛卡尔空间中，将 FK 推出的 TCP 位姿与 ``target_ee_pose`` 逐轴比较。成功判定依据
  ``reward_threshold``\ （默认：位置 1 cm）。``reward_threshold`` 配置接受
  6 元素数组 ``[x, y, z, rx, ry, rz]``\ （与 Franka API 对齐），但当前仅使用 XYZ 位置分量；
  姿态分量保留给未来使用。

硬件准备
----------------

.. warning::

  请确保控制节点与训练节点位于同一本地网络中。
  GimArm 机械臂通过 CAN 总线连接至控制节点，而不是以太网。


依赖安装
-------------------------

控制节点与训练/采样节点需要安装不同的软件依赖。

机器人控制节点
~~~~~~~~~~~~~~~~~~~~~~

1. 安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^

a. 克隆 RLinf 仓库
__________________________

.. code:: bash

   # 中国大陆用户可使用下面的镜像以获得更好的下载速度：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. 安装 RLinf 依赖
________________________________

.. code:: bash

   # 中国大陆用户可以追加 `--use-mirror` 参数以获得更好的下载速度。

   bash requirements/install.sh embodied --env gim_arm
   source .venv/bin/activate

c. 安装 gim_arm_control SDK
________________________________

``gim_arm_control`` 包提供了用于控制 GimArm 机械臂的底层 CAN 通信驱动与 Python 绑定。
它同时附带了下一步需要使用的辅助脚本（``sh/init_can.sh``、``sh/set_zero.sh``），
因此请先完成本步骤的安装再继续。

.. code:: bash

   # 将 SDK 克隆到 RLinf 旁边（下面示例假设路径为 ~/gim_arm_control）。
   cd ~
   git clone https://github.com/RLinf/gim_arm_control.git
   cd ~/gim_arm_control/python
   pip install -e .

脚本会通过 CMake 构建 C++ 核心，并使用 nanobind 安装 Python 绑定。

**构建依赖**：``scikit-build-core>=0.5``、``nanobind>=2.0``、C++17 编译器（GCC >= 7 或 Clang >= 5）。

**运行依赖**：``numpy``、``pinocchio``（以 ``pin`` 名称导入）。

.. note::

   ``pinocchio`` 是控制器做正运动学和 Jacobian 计算所必需的依赖。
   它会随 SDK 自动安装。
   对于需要兼容 NumPy 1.x 的系统，可改用以下命令安装：

   .. code:: bash

      pip install -e ".[pin270]"

2. CAN 接口初始化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CAN 总线需要在使用机械臂之前完成初始化。
``gim_arm_control`` SDK（在上一步中安装）提供了方便脚本，或者也可以手动执行相关命令。

使用 ``gim_arm_control`` 仓库自带的脚本：

.. code:: bash

   bash sh/init_can.sh can0

或者手动执行：

.. code:: bash

   sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
   sudo ip link set can0 txqueuelen 1000
   sudo ip link set can0 up

命令将标准 bitrate 设为 1 Mbps、CAN FD 数据 bitrate 设为 5 Mbps。

.. warning::

  每次系统重启后，CAN 接口都需要重新初始化。
  可以通过以下命令确认接口是否已经启用：

  .. code:: bash

     ip link show can0

3. 电机零点校准
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首次使用（或更换电机后）需要对达妙电机进行零点校准。
校准会把电机当前的物理位置设为零参考点。

通过 ``gim_arm_control`` 仓库的脚本：

.. code:: bash

   # 对单个电机进行零点校准（CAN ID 使用十六进制）
   bash sh/set_zero.sh can0 001

   # 对所有电机（001-008）进行零点校准
   bash sh/set_zero.sh can0 --all

.. warning::

  校准必须在机械臂处于机械零位时进行。
  错误的校准会导致机械臂执行意料之外的动作。
  该操作依赖 ``can-utils``（使用 ``sudo apt install can-utils`` 安装）。

训练/采样节点
~~~~~~~~~~~~~~~~~~~~~~~~~~

首先像上面一样克隆 RLinf 仓库，然后安装相关依赖。

安装依赖
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**选项 1：Docker 镜像**

使用 Docker 镜像来运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 中国大陆用户可使用下面的镜像以获得更好的下载速度：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

**选项 2：自建环境**

直接在你的环境中安装依赖，运行下列命令：

.. code:: bash

   # 中国大陆用户可以追加 `--use-mirror` 参数以获得更好的下载速度。

   bash requirements/install.sh embodied --env gim_arm
   source .venv/bin/activate

   # 如需安装特定模型相关依赖（如 OpenVLA），可添加 --model 参数：
   # bash requirements/install.sh embodied --model openvla --env maniskill_libero


运行实验
-----------------------

前置条件
~~~~~~~~~~~~~~~

**获取任务的目标位姿**

若要获取 Peg-Insertion 任务所需的目标末端执行器位姿，可以使用真机测试脚本。

首先初始化 CAN 接口（参见上文），然后运行：

.. code-block:: bash

   python toolkits/realworld_check/test_gim_arm_env.py --can can0 --variant gim_arm_xl

脚本会启动控制器，对当前关节角度做正运动学计算，并打印 TCP 位置与四元数。
手动将机械臂移动至目标位姿，然后记录打印出的值。
把四元数转换为 Euler XYZ 角度后，即可用于 ``target_ee_pose`` 配置项。

数据采集
~~~~~~~~~~~~~~~~~

可参考我们提供的 VR 遥操作代码（改编自 `XRoboToolkit <https://github.com/XR-Robotics>`_），
用于 GimArm 机械臂的数据采集。

配置文件
~~~~~~~~~~~~~~~~~~~~~~

在开始实验之前，需要创建或修改一个配置 YAML 文件。
关键部分是 cluster 硬件配置，用于指定 GimArm 机器人：

.. code-block:: yaml

  cluster:
    num_nodes: 2
    component_placement:
      actor:
        node_group: "4090"
        placement: 0
      env:
        node_group: gim_arm
        placement: 0
      rollout:
        node_group: "4090"
        placement: 0
    node_groups:
      - label: "4090"
        node_ranks: 0
      - label: gim_arm
        node_ranks: 1
        hardware:
          type: GimArm
          configs:
            - can_interface: can0
              arm_variant: gim_arm_xl
              camera_serials: ["YOUR_CAMERA_SERIAL"]  # 若无可用相机，可使用 []
              camera_type: realsense
              enable_gripper: true
              gripper_type: parallel
              node_rank: 1

在环境覆盖配置里设置 ``target_ee_pose``：

.. code-block:: yaml

  env:
    train:
      override_cfg:
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
        reset_joint_qpos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        safe_retract_qpos: [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
        is_dummy: false

    eval:
      override_cfg:
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]

关键配置字段：

- ``target_ee_pose``：目标末端位姿 ``[x, y, z, rx, ry, rz]``\ （单位：米 / Euler XYZ 弧度）。
- ``reset_joint_qpos``：每一回合起始时的关节角配置。
- ``safe_retract_qpos``：Peg-Insertion reset 期间的安全回缩关节角配置。
- ``is_dummy``：设为 ``true`` 可在无硬件情况下测试流程。

.. note::

   相机是可选的。若 ``camera_serials`` 设为空列表 ``[]`` 或省略，
   环境将在无相机观测的情况下运行，观测空间中的 ``frames`` 键会是一个空字典。
   这适用于纯状态策略，或尚未完成相机配置的情形。

安装后测试（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们提供了若干测试脚本，便于在正式实验前验证硬件与环境是否就绪。
这一步是可选的，但推荐执行。

1. 确认 CAN 接口已启用：

.. code-block:: bash

   ip link show can0

2. 测试机器人控制器：

.. code-block:: bash

   python toolkits/realworld_check/test_gim_arm_env.py --can can0 --variant gim_arm_xl

该脚本会测试：控制器启动、``is_robot_up()``、``get_state()`` 返回形状、``move_joints()``、``reset_joint()``
以及夹爪开/合动作。

.. note::

   相机相关逻辑尚未完全测试。若要运行 Peg-Insertion 实验，请准备好相机并通过
   硬件配置中的 ``camera_serials`` 进行设置。
