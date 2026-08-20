机器人模型
==========

在 RLinf 中，机器人表示为一棵按名称组织的部件树。机械臂、夹爪、相机和移动底盘分别对应树中的一个位置。每个部件都能返回观测；可控部件还可以接收动作。

如果仅需编写任务、接入 policy 或调试现有机器人，了解上述模型即可。以下以 Franka 为例，说明如何读取部件树，以及任务代码与机器人代码之间的边界。关于 worker、远程调用及硬件连接的管理机制，请在接入新设备或排查底层问题时参阅架构文档。

使用现有机器人
--------------

通过注册名构建机器人后，可先调用 ``describe()`` 检查部件名称和部署位置。该方法不会连接硬件，因此可以在机械臂上电前运行。在实际运行中，建议将 ``disconnect()`` 置于 ``finally`` 块中，确保 rollout 或调试命令发生异常时仍能关闭硬件连接。

.. code-block:: python

   from rlinf.robotics import build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # 这一步不会打开任何硬件。
   print(robot.describe())

   robot.connect()
   try:
       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_width = observation["end_effector"]["state"]

       robot.send_action(
           {"arm": {"tcp_pose": target}, "end_effector": {"target": width}}
       )
   finally:
       robot.disconnect()

``get_observation()`` 返回一个嵌套字典，其层级与部件树一致。调用 ``send_action()`` 时沿用相同的路径，只需指定本次需要控制的部件，无需补齐整棵树。

以部件名称访问观测和动作
------------------------

单臂 Franka 的顶层有 ``arm`` 和 ``end_effector`` 两个部件：

.. code-block:: text

   FrankaRobot
   ├── arm           declared      node=1     via FrankaROSArm#1
   └── end_effector  declared      node=1     via FrankaROSArm#1

两行输出中的 ``via`` 值相同，表示机械臂和夹爪共用一条 Franka 连接。读取观测和发送动作时无需处理这条连接；排查重复连接或节点配置时，可结合 ``node`` 和 ``via`` 确认资源归属。

双臂机器人在顶层增加 ``left`` 和 ``right`` 两个分支，部件名称保持不变：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

部件树可以继续向下嵌套。RLinf 不预设 ``arms``、``cameras`` 等固定分组；新增升降机构、云台或第三条机械臂时，只需为新部件命名并将其加入树中。

调用方式与部署位置解耦
----------------------

部件路径只表示其在机器人结构中的位置，不包含部署信息。本机相机和远端机械臂仍会出现在同一棵观测树中。RLinf 会并行访问相互独立的硬件连接；共用同一连接的部件则按声明顺序访问。

任务和 policy 代码只依赖部件名称与数据，不需要区分底层使用的是 Ray actor、RPC、串口还是厂商 SDK。现有 policy 如果仍接收扁平向量，可在环境边界使用 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 完成格式转换。

分离部件与任务逻辑
------------------

部件负责读取传感器并控制执行器。rollout 的成功条件、奖励、终止条件、复位流程和 Gymnasium space 由 ``RobotTask`` 定义；``RobotTaskEnv`` 再将任务与机器人组合。为同一台机器人切换任务时，无需修改底层设备代码。

后续阅读
--------

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机上添加任务。
- :doc:`添加机器人 <../extending/new_robot>`：接入本地传感器、执行器或整台机器人。
- :doc:`机器人架构 <robotics_architecture>`：了解共享连接、``Endpoint``、远程句柄、失败回滚和 worker 放置。
- :doc:`遥操作 <../guides/teleoperation>`：组合多种操作者设备。
