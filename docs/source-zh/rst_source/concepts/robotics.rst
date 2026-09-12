机器人接口
============

本页说明任务和 env 如何通过机器人接口检查、连接、读取并控制真机。内容先按照实际调用顺序介绍如何构建和检查机器人、按类型获取零部件、管理连接和读写数据，再解释访问路径如何产生、组合为何不改变接口，以及 env 应在哪一层转换机器人数据。硬件 session、资源归属和 worker placement 等实现机制见 :doc:`机器人架构 <robotics_architecture>`。

机器人接口为机械臂、末端执行器、相机和移动底盘等零部件提供稳定的名称和访问路径，观测和动作沿用同一组路径。任务侧无需直接依赖具体 driver；只要路径保持不变，更换 backend 或调整部署节点也无需修改调用代码。

使用机器人接口
------------------

一次完整调用可以分为四个阶段：构建并检查机器人，取得初始化所需的零部件，连接后读写数据，最后释放硬件。下面的示例将这四个阶段放在同一段代码中：

.. code-block:: python

   from rlinf.robotics import Arm, Camera, build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # 机器人未上电或尚不可达时也可以执行。
   print(robot.describe())

   arm = robot.child("arm", Arm)
   cameras = robot.parts_of_type(Camera)

   robot.connect()
   try:
       if not arm.is_robot_up():
           raise RuntimeError("The arm is connected but not ready.")
       if not all(camera.is_ready() for camera in cameras.values()):
           raise RuntimeError("A camera is not delivering frames.")

       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_state = observation["end_effector"]["state"]

       applied = robot.send_action(
           {
               "arm": {"tcp_pose": target},
               "end_effector": {"target": width},
           }
       )
   finally:
       robot.disconnect()

``build_robot()`` 根据注册名找到对应的机器人 builder，并返回尚未连接的 ``Robot``。构建阶段只记录硬件参数和 placement，不导入厂商 SDK，也不打开设备。``describe()`` 读取这些声明，因此机器人尚未上电或网络不可达时，也能检查访问路径、部署节点和连接归属。env 拿到的是 scheduler 分配的硬件配置，而不是 builder 参数，因此它调用已注册机器人类的 ``from_config()``：该方法把配置字段逐一传给 ``build()``，并把零部件部署到 env 所在节点，配置另行指定节点时除外。

初始化代码随后取得所需能力。``child("arm", Arm)`` 返回 ``arm`` 路径上的零部件，并立即检查它是否属于 ``Arm``；编辑器也会将返回值推断为 ``Arm``。``parts_of_type(Camera)`` 则遍历完整组合，返回以完整路径为 key 的所有相机。任务依赖固定路径时使用 ``child()``；只关心设备类别而不依赖相机名称时，使用 ``parts_of_type()``。

``connect()`` 会为每条 owner connection 打开一次资源；如果后续 connection 打开失败，它会回滚此前已经打开的资源。连接完成后，可通过 ``Arm.is_robot_up()`` 和 ``Camera.is_ready()`` 检查设备是否可用；``clear_errors()`` 与 ``reset_joint()`` 等初始化操作位于单步动作流之外。接受 ``tcp_pose`` 命令的机械臂还提供 ``move_to()``，它以均匀间隔的目标位姿移动到指定工具位姿，适合在两个 episode 之间回零。末端执行器通过 ``open()`` 与 ``close()`` 完全张开或闭合，并由 ``is_open`` 报告当前状态。每个机械臂类都声明 ``DOF``，即其驱动的关节数，因此在打开任何硬件之前就能检查关节目标的维度。

step 循环只需两个接口。``get_observation()`` 对组合后的机器人执行一次完整读取，并按访问路径返回嵌套字典。``send_action()`` 接收层级相同的动作，只下发本次提供的分支，并将各零部件实际发送的动作作为 ``applied`` 返回。``disconnect()`` 最后按相反顺序关闭 connection；将它置于 ``finally`` 中，可以保证读取或控制失败后仍完成清理，而且重复调用不会产生额外影响。

所有机器人都遵循这条调用链。不同机器人之间的差异，体现在观测和动作字典包含哪些路径，以及这些路径如何嵌套。

episode 之间，任务只提出要求，而不编写具体动作。``hold()`` 让机械臂停在当前位置，不再追踪 policy 最后下发的目标；``clear(distance=..., qpos=...)`` 让末端脱离正在接触的物体：以末端位姿驱动的机械臂抬起相应距离，以关节目标驱动的机械臂则转到为此指定的关节配置；``go_home(Home(...))`` 前往任务指定的等待位置，并在未到达容差范围内时重新下发；``unwind(qpos)`` 把关节回到一个已知配置。任务会同时给出两种等待位置的写法，每条机械臂各取自己能用的那一种，因此插销任务在 Franka 和 GimArm 上是同一个任务。

按路径访问零部件
--------------------

理解前面的观测和动作字典，需要先明确访问路径从何而来。组合时使用的名称构成第一段路径；某个零部件承载的下级零部件会继续增加路径层级。Franka 与 SO-101 分别对应这两种情况。

单臂 Franka 的 ``arm`` 和 ``end_effector`` 位于同一层级，因为机械臂和 Franka Hand 分别打开独立端点：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

``via`` 不同表示两个零部件各自持有 connection；两个顶层名称因此对应两项可以独立管理的能力。

SO-101 则共用一条伺服总线。五个机械臂关节和夹爪由同一个 connection 驱动，因此末端执行器位于机械臂下一级：

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

观测和动作也遵循这一层级：

.. code-block:: python

   joint_position = observation["arm"]["arm_joint_position"]
   gripper_state = observation["arm"]["end_effector"]["state"]

   robot.send_action(
       {
           "arm": {
               "joint_position": joint_target,
               "end_effector": {"target": gripper_target},
           }
       }
   )

嵌套动作并不是另一套 API，只是在同一个 ``send_action()`` 调用中增加了一层路径。``describe()`` 中的 ``via`` 用于解释层级和资源归属，任务代码仍只使用路径和对应数据。完整输出属于诊断信息，不是稳定的序列化格式。

字段名并不等于完整的约定。两条机械臂可以都上报 ``tcp_pose``，含义却不同；按其中一种评分的任务会悄悄误读另一种。因此 ``rlinf/robotics/fields.py`` 规定了每个规范字段的含义：``tcp_pose`` 是 7 个数，先是 base 坐标系下以米为单位的位置，再是 ``xyzw`` 四元数；``arm_joint_position`` 是每个关节一个弧度值。上报这些数值的零部件无需额外说明；数值含义不同的零部件则声明自己的含义，此时把任务绑定到它会直接失败，并同时给出两种含义，而不是让任务把夹爪开度当成旋转的一部分：

.. code-block:: text

   RequirementError: CartesianTarget cannot run on ExampleRobot: arm
   (MethodArm) reports 'tcp_pose' as xyz+rpy+gripper_width in m,rad in the
   base frame, not xyz+quat_xyzw in m in the base frame

这个报错正是提示：应当在 driver 内部完成转换，厂商约定本就属于那一层。``MethodArm`` 的 ``decode`` 和 ``encode`` 正是为此准备：每个字段两个方向各写一次转换，之后该零部件即按规范含义上报，所有任务都能在它上面运行。Turtle2 的控制器使用欧拉角，并把夹爪开度放在它称为 ``tcp_pose`` 的向量里；在 driver 内完成转换之后，它就能运行与 Franka 相同的任务，而开度仍然是夹爪自身的状态。

组合零部件时保持接口一致
----------------------------

上一节从调用方角度说明了访问路径，本节进一步说明这些路径如何在组合阶段产生。组合只负责将可读零部件放到稳定名称下，不会改变后续读写方式。

``RobotPart`` 可以直接加入机器人。例如，``Robot(base=base, arm=arm, end_effector=hand)`` 中的三个关键字参数名会分别成为顶层路径。裸 ``Connection`` 只管理共享硬件 session，本身没有可供任务读取的观测；遇到这种情况，应先调用 ``session.part("left")``，再将返回的 ``RobotPart`` 加入机器人。

零部件还可以承载共用同一 connection 的下级零部件，它们会自动出现在下一层，因此 SO-101 的夹爪使用 ``arm.end_effector``。多项零部件需要组成可复用单元时，使用 ``PartGroup`` 再增加一层名称。双臂机器人可以在外层加入 ``left`` 和 ``right`` 两个 group，内部仍沿用同一套机械臂接口：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

RLinf 不预设 ``arms`` 或 ``cameras`` 等固定字段。新增移动底盘、升降机构、云台或第三条机械臂时，只需确定稳定名称，调用方仍使用 ``get_observation()`` 与 ``send_action()``。组合结构确定后，env 就可以将整台机器人作为一个硬件边界使用。

在 env 中统一读写机器人
------------------------

env 需要把具名机器人数据转换为 policy 使用的观测和动作格式。它应从完整机器人接口进入，而不是在旁路直接读取 driver：每个 step 通过一次 ``robot.get_observation()`` 同时构造 ``state`` 和 ``frames``，再通过 ``robot.send_action()`` 将动作送回硬件。共用 connection 的零部件会在这次读取中复用同一份底层状态快照。

连接前取得的具类型零部件仍用于 step 循环之外的初始化。当前真机 env 通过 ``Arm`` 检查就绪状态、恢复错误并复位关节，通过 ``parts_of_type(Camera)`` 查找画面来源。相机的 placement 和生命周期仍由机器人管理，env 只消费观测结果。

``TaskEnv`` 按照 ``ObservationSpec`` 的声明，从这一次读取中逐个 key 构造 policy 的 ``state`` 与 ``frames``，其中也包含 checkpoint 训练时使用的编码方式和颜色顺序。policy 需要的表示形式集中在这里，机器人接口仍保留具名嵌套结构。正因为任务只依赖这套接口，placement 才能独立调整。

任务代码不依赖部署位置
----------------------

访问路径标识机器人能力，不表示承载它的进程。相机可以位于 env 进程，机械臂可以运行在其他节点，两者仍出现在同一份观测中；具体位置记录在各自的 owner connection 上。使用不同 connection 的零部件可以并行读写，共享 connection 的分支则按声明顺序执行，避免并发访问不支持该模式的厂商 SDK。

任务和 policy 代码只依赖零部件名称及其数据，无需感知 Ray actor、RPC、串口或厂商 session。调整 connection 的部署位置时，前述调用和数据结构保持不变。机器人接口还需要划清最后一项职责：它说明硬件如何工作，但不定义 rollout 要完成什么目标。

分离硬件能力与任务逻辑
----------------------

零部件说明如何读取传感器或控制执行器，任务则解释这些读数和动作为什么重要。因此，奖励、终止条件和任务特有的复位流程由 ``Task`` 定义，policy 数字的含义由 ``ActionLayout`` 的各个通道定义。``TaskEnv`` 把任务、动作布局和机器人组合在一起，核对机器人是否满足它们的要求，并负责生命周期。尚未迁移到 ``TaskEnv`` 的专用 env 仍直接使用同一套机器人调用。

最终形成两项相互独立的 contract：机器人路径在任务和 placement 变化时保持稳定，任务则可以调整 policy 侧 schema，而无需修改 driver。

后续阅读
--------

根据下一步需要修改的边界选择文档：

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机上添加任务。
- :doc:`添加机器人 <../extending/new_robot>`：接入本地传感器、执行器或整台机器人。
- :doc:`机器人架构 <robotics_architecture>`：了解 ``parts`` 与 ``children`` 的区别，以及共享 connection、``PartGroup``、资源生命周期和 worker placement。
- :doc:`遥操作 <../guides/teleoperation>`：组合多种操作者设备。
