UR5 真机强化学习
================

.. figure:: https://upload.wikimedia.org/wikipedia/commons/5/57/Cobot.jpg
   :align: center
   :width: 60%

   Universal Robots UR5 协作机械臂。图片来源：Wikimedia Commons，GrowSkills Robotics。

在真机 Universal Robots UR5 上训练策略，可选择两种机械臂后端：直接通过以太网驱动控制器的
``ur_rtde``，或者通过已运行的
`Universal_Robots_ROS2_Driver <https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver>`_
下发命令的 ``ur_ros2``。本文附带的 reach 任务只需保持一个六关节目标姿态，是打通「组装机器人
→ 得到评分」这条完整链路所需的最小任务，可用于在为 UR5 编写具体操作任务之前，先验证硬件接入是否成功。

概览
----

在 UR5 上训练 MLP SAC 策略，使其到达指定的关节空间目标。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP policy

   .. grid-item-card:: 算法
      :text-align: center

      SAC

   .. grid-item-card:: 任务
      :text-align: center

      Joint reach

   .. grid-item-card:: 硬件
      :text-align: center

      UR5 · ``ur_rtde`` / ``ur_ros2`` · 相机可选

| **你将完成:** 硬件检查 → 安装 → 运行硬件检查脚本 → 启动训练。
| **前置条件:** :doc:`安装 </rst_source/start/installation>` · 可访问的 UR5 控制器或 URSim。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - 任务
     - 配置 / 入口
     - 说明
   * - Joint reach
     - ``ur5_ursim_reach``
     - 保持一个六关节目标姿态，端到端验证控制链路与关节空间运动。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - Observation
     - 6 个关节位置（弧度），以及每台已配置腕部相机提供的一帧 RGB 图像。
   * - Action
     - 6 维，每个关节一维。策略看到的是 ``[-1, 1]``（``normalize_actions``
       会在动作送达机械臂前，将其映射到配置的绝对弧度活动范围内）；将
       ``normalize_actions`` 设为 ``False`` 可改为直接下发绝对弧度目标。
   * - Reward
     - 当每个关节都落在 ``reward_threshold`` 弧度范围内时为 ``1.0`` （稀疏奖励）；
       启用 ``use_dense_reward`` 时为关节距离的负值。
   * - Prompt
     - ``reach a joint configuration``

硬件环境搭建
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 要求
   * - 机械臂
     - Universal Robots UR5，需要与控制节点在同一网络内可达。``ur_rtde`` 后端直接连接控制器的
       RTDE 接口；``ur_ros2`` 后端则通过 ROS 计算图连接已运行的
       ``Universal_Robots_ROS2_Driver`` （或代替真实硬件的 URSim），不会自行建立到机械臂 IP
       的连接。
   * - 控制节点
     - 与机械臂（或 URSim）处于同一网络、运行 env worker 的 Linux 计算机，不需要 GPU。
   * - 训练 / Rollout 节点
     - 带 GPU、用于运行 MLP 策略的计算机；单节点场景下可与控制节点共用同一台机器。
   * - 相机（可选）
     - 一个或多个腕部相机（``realsense``、``zed`` 或 ``lumos``）；reach 任务默认不配置相机。

将机械臂固定在稳固的台面上，并确保其整个活动范围内没有障碍物；连接前先清空工作空间：
``ur_ros2`` 后端在连接时会重启驱动的外部控制程序，而只要环境配置保留 ``reset_on_init``
的默认值 ``True``，两种后端都会在环境构造时把机械臂移动到 ``reset_joint_qpos``。

.. warning::

   构造环境时默认会触发这次初始移动，移动轨迹按 ``reset_duration``
   （最短时长，默认 3 秒）和 ``reset_joint_speed`` （最大命令速度，默认 30 度/秒）控制节奏。
   如需跳过这次移动，可在环境配置中设置 ``reset_on_init: False``，或者在第一次构造环境前，
   先把 ``reset_joint_qpos`` 设置为机械臂当前的姿态。

任务相关的安全约束
------------------

``joint_limit_low`` 与 ``joint_limit_high`` （默认每个关节均为 ``±pi``）限定了动作空间，
并会与机械臂驱动自身的活动范围做校验，但它们只是活动范围限制，并不等同于「无碰撞」的
工作空间保证：即便目标姿态落在这个范围内，机械臂仍可能撞到自身、相机支架或工作空间内
的其他物体。请自行验证并限定实际可用的工作空间和任务范围，例如将环境配置中的
``joint_limit_low``/``joint_limit_high``、``reset_joint_qpos`` 和 ``target_joint_qpos``
收紧到已确认安全的范围内。构造环境时，若 ``reset_joint_qpos`` 或 ``target_joint_qpos``
超出配置的范围会直接报错拒绝，而不是被静默裁剪。

安装
----

在控制节点上按下方自定义环境流程安装机器人依赖；后续的硬件检查和训练均在同一仓库目录和
已激活的环境中运行。

机器人控制节点
~~~~~~~~~~~~~~

在连接机器人的主机上克隆 RLinf，然后安装对应环境。

.. include:: _setup_common.rst
   :end-before: 然后，使用

在仓库根目录安装机器人依赖：

.. code-block:: bash

   # 为提高国内下载速度，可以添加 --use-mirror。
   bash requirements/install.sh embodied --env ur5
   source .venv/bin/activate
   export REPO_PATH="$PWD"

后续命令均在此仓库目录中运行，并保持环境已激活。该安装脚本会同时装好两种机械臂后端；
``ur_rtde`` 无需其他依赖。平台要求见 :doc:`安装 </rst_source/start/installation>`。

ROS2 后端（``ur_ros2``）
^^^^^^^^^^^^^^^^^^^^^^^^

``ur_ros2`` 后端通过一个已运行的 ``Universal_Robots_ROS2_Driver`` 来接入机械臂，它本身不会
启动驱动或仿真器。请先启动 URSim 或真实控制器，再启动驱动，之后 env worker 才能连接：

.. code-block:: bash

   ros2 launch ur_robot_driver ur_control.launch.py \
     ur_type:=ur5 robot_ip:=<robot-ip> \
     launch_rviz:=false headless_mode:=true

这会启动该后端实际下发命令的 ``scaled_joint_trajectory_controller``，并开始发布
``/joint_states``。请让环境配置中的 ``headless_mode`` 与驱动的启动方式保持一致：
在（默认的）headless 模式下，连接时会通过 ``resend_robot_program`` 服务重启驱动的
外部控制程序；若不是 headless 模式，需要先在示教器上手动启动 External Control，
并将配置设为 ``headless_mode: False``，否则这次重启调用会报告一个虚假的失败。

ROS2 Jazzy 的 ``rclpy`` 面向 Python 3.12，而上面的 UR5 安装默认使用 Python 3.11.14，
无法导入它。请为 env worker 进程准备一个同时具备 ``rclpy`` 和完整 UR5 安装内容的
Python 3.12 环境，而不是只装了 ``rclpy`` 的普通 ``venv``：

.. code-block:: bash

   python3.12 -m venv --system-site-packages .venv-ros2
   bash requirements/install.sh embodied --env ur5 \
     --python 3.12 --venv .venv-ros2

运行 env worker 前，激活这个环境并加载 ROS2 Jazzy 的 setup 脚本，而不是默认的 ``.venv``：

.. code-block:: bash

   source .venv-ros2/bin/activate
   source /opt/ros/jazzy/setup.bash

UR e-Series 控制器会按示教器实时速度滑块的比例，限制 ``scaled_joint_trajectory_
controller`` 的实际运行速度；如果你以低于 100% 的速度运行，请把环境配置中的
``speed_scaling`` 设为对应比例（例如 10% 速度对应 ``0.1``），这样一步动作的完成等待
时间会按相同比例放大，避免把一条在降速下仍正常进行的轨迹误判为卡死并超时取消。

RTDE 后端（``ur_rtde``）
^^^^^^^^^^^^^^^^^^^^^^^^

``ur_rtde`` 后端直接对控制器 IP 打开 ``RTDEControlInterface`` 和
``RTDEReceiveInterface``，不需要启动额外的驱动程序。需要把控制器切换到 Remote Control
模式（e-Series 示教器上的 ``Settings`` > ``System`` > ``Remote Control``），外部 RTDE
客户端才能下发运动命令，而不是只能通过示教器操作。

运行
----

完成接线并选定后端后，先确认机械臂能够组装、连接、读取状态并正常断开，且不会产生任何运动：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5 \
     --arg robot_ip=<robot-ip> --arg node_rank=0 --arg backend=ur_rtde

要检查 ROS2 路径，先按上文启动驱动（或 URSim）并激活对应环境，再把
``--arg backend=ur_ros2`` 换上即可。该命令会打印每个部件观测到的内容，然后断开连接并确认
没有部件仍处于连接状态；它会读取关节位置（``ur_rtde`` 下还包括工具位姿），但不会下发
reset 或目标命令，因此这项检查不会使机械臂产生运动。

硬件检查通过后，就可以在仓库根目录对 URSim 或真实控制器启动本文附带的 reach 任务：

.. code-block:: bash

   bash examples/embodiment/run_realworld.sh ur5_ursim_reach

``examples/embodiment/config/ur5_ursim_reach.yaml`` 默认指向 URSim 的桥接地址；若要连接
真实硬件，请把 ``robot_ip`` 改为你的控制器地址，并与上文配置的后端和 ROS 计算图保持一致。
启动前请清空工作空间：如前文所述，构造环境会把机械臂移动到 reset 姿态。

可视化与结果
------------

通过 TensorBoard 和终端日志查看训练进度：

.. code-block:: text

   logs/<timestamp>-ur5_ursim_reach/run_embodiment.log

关注 ``env/success_once`` 以跟踪 reach 任务的成功情况；完整指标说明见
:doc:`训练指标 </rst_source/reference/metrics>`。

新增真机任务
------------

上面的 reach 任务是接入一台新 UR5 所需的最小任务；如果要在同一台机械臂上定义其他任务的
reset 流程、观测、动作、奖励和成功条件，请参考
:doc:`新增真机任务 </rst_source/extending/new_task>`。
