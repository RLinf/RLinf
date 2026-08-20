机器人架构
============

编写任务时，只需将机器人理解为部件树。当需要接入设备、让多个部件共享硬件连接，或排查远程部署与资源回收问题时，再参考本页的实现模型。若尚未了解部件树，请先阅读 :doc:`robotics`。

区分部件树与硬件连接
--------------------

以机械臂和夹爪共用一个 ROS 会话为例，policy 仍会看到两个独立部件：

.. code-block:: text

   robot
   ├── arm
   └── end_effector

这两种结构分别回答不同的问题：

- 机器人树描述 policy 可以观测和控制哪些部件。
- 硬件连接描述哪些资源只能打开一次、部署在哪个节点，以及由谁负责释放。

代码使用两个 mapping 分别保存这两组名称：

- ``Group`` 或 ``Robot`` 将对外公开的部件树保存在 ``children`` 中。每个 key 都会成为观测和动作路径，例如 ``left.arm``；任务、policy 和数据集使用这些名称访问部件。
- ``Endpoint`` 将同一硬件连接能够提供的部件保存在 ``exports`` 中。其中的 key 只用于从连接中选择部件，不会自动成为机器人路径。

组合过程将两个 mapping 关联起来。``connection.export("arm")`` 从连接中选择 ``arm``，再将结果传给 ``Robot(arm=...)``；此时，该部件才会以公开名称 ``arm`` 进入机器人的 ``children``。``exports`` 表示一条连接能够提供的部件，``children`` 表示机器人最终公开的部件结构。

核心类型
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 类型
     - 用途
   * - ``Endpoint``
     - RLinf 在某台机器上打开、之后再关闭的对象。它有运行位置和生命周期，但不一定能读取观测。
   * - ``RobotPart``
     - 可读取观测的 ``Endpoint``。它实现 ``get_observation()``，并使用 ``observation_features`` 描述返回的数据。
   * - ``ControllablePart``
     - 可接收动作的 ``RobotPart``。它还实现 ``send_action()`` 和 ``action_features``。
   * - ``Connection``
     - 一次硬件会话，可以支撑多个部件。其本身不是部件，因此不会出现在观测树中。
   * - ``Group``
     - 由一组具名 ``children`` 组成，可表示一套机械臂、一个躯干或整台机器人。
   * - ``Robot``
     - 最外层的 ``Group``。除了组合部件，它还管理注册、放置和 ``connect()`` 期间创建的句柄。
   * - ``PartSpec``
     - 调用部件类的 ``at()`` 后生成的延迟声明，记录部件类、构造参数、目标节点和可选的 worker 名称。
   * - ``PartHandle``
     - 指向已连接部件的统一句柄。本地部件和 worker 中的远端部件使用同一套调用方式。

``Camera``、``EndEffector``、``MobileBase`` 和 ``LeggedBase`` 表示具有明确设备类别的部件。远程代理根据这些类别恢复对应接口。

将共享硬件连接映射到机器人树
----------------------------

当一条硬件连接提供多个部件时，使用 ``exports`` mapping 声明这些部件。以下机械臂 endpoint 只打开一条连接，但同时提供机械臂和夹爪：

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def exports(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(
                   self, state_field="gripper_position"
               ),
           }

此时，``arm`` 和 ``end_effector`` 只是连接内部的 selector，尚未成为机器人路径。组合机器人时，再指定对外公开的名称：

.. code-block:: python

   connection = ExampleArm.at("10.0.0.2", node_rank=1)
   robot = Robot(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )

传给 ``Robot`` 的 keyword arguments 会进入 ``robot.children``，因此上述机器人最终包含 ``arm`` 和 ``end_effector``。``Connection`` 不属于公开的机器人树，因此没有 ``children``；``Group`` 中的部件已经位于树中，因此不再通过 ``exports`` 提供。两个 mapping 只在调用 ``connection.export(...)`` 时关联。

同一规则也适用于联动控制器。控制器本身不是机械臂，因此不应直接加入部件树：

.. code-block:: python

   connection = Turtle2Connection.at(50, camera_ids, node_rank=0)
   robot = Turtle2Robot(
       left=Group(
           arm=connection.export("left"),
           end_effector=connection.export("left_end_effector"),
       ),
       right=Group(
           arm=connection.export("right"),
           end_effector=connection.export("right_end_effector"),
       ),
       wrist=connection.export("wrist_1"),
   )

这些引用来自同一个 ``PartSpec``，因此底层控制器只会打开一次。

先声明部署位置，再打开硬件
--------------------------

调用 ``at()`` 不会创建对象或打开硬件，只会记录构造参数和目标节点：

.. code-block:: python

   arm_connection = ExampleArm.at("10.0.0.2", node_rank=1)
   robot = Robot(
       arm=arm_connection.export("arm"),
       end_effector=arm_connection.export("end_effector"),
       scene=RealSenseCamera.at(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

执行 ``connect()`` 时，RLinf 才会解析 ``PartSpec``。本地部件使用 ``LocalPartHandle``；远端部件由调度器部署到 worker，并使用 ``RemotePartHandle``。两种句柄提供相同的公有方法，调用方无需区分本地和远端部件。

同一份 ``PartSpec`` 无论被引用多少次，都只会创建一个句柄，并在断开时释放一次。不同句柄可以并行调用；同一句柄上的部件按声明顺序调用，避免并发访问不支持该模式的厂商 SDK。

``spawn()`` 会立即部署部件，并将句柄交由调用方管理，适用于独立的工装脚本。机器人内部应使用 ``at()``，由 ``Robot`` 统一处理启动失败和资源清理。

连接前检查组合结果
------------------

``Robot.describe()`` 读取最初保存的部件声明，不依赖已连接的代理，因此连接前后显示的节点和资源归属保持一致：

.. code-block:: text

   FrankaRobot
   ├── arm           declared      node=1     via FrankaROSArm#1
   └── end_effector  declared      node=1     via FrankaROSArm#1

``via`` 值相同的行共用一个 ``Endpoint``。直接声明的部件可以在连接前显示类别；通过 ``Connection`` 引用某个 export 的部件则显示为 ``declared``，因为连接尚未打开，具体部件也尚未创建。

``describe()`` 目前只显示组合结构、节点和资源归属，不显示 observation/action schema。若需检查字段和 shape，请使用 :doc:`添加机器人 <../extending/new_robot>` 中的 conformance 检查；该检查会通过 mock SDK 或真机打开连接后进行验证。

资源生命周期
------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 阶段
     - 执行内容
   * - 声明
     - ``at()`` 记录构造参数和放置节点。这时不导入厂商 SDK，也不打开硬件。
   * - 连接
     - 机器人部署每份声明，打开 ``Endpoint``，解析其公开的部件，并将句柄写入 ``robot.handles``。
   * - 使用
     - 机器人按部件名读取、复位和发送动作，再根据资源是否共享决定能否并行。
   * - 断开
     - 部件释放实际打开的设备，关闭句柄，机器人再恢复最初的声明树。

``_open()`` 返回的厂商对象会原样传给 ``_release(device)``。释放资源时必须使用传入的 ``device``，不应再从 ``self`` 读取。

连接过程要么全部成功，要么全部回滚。如果后续部件启动失败，``Robot.connect()`` 会关闭此前已打开的资源，并恢复原始声明。排除故障后，可以对同一对象再次调用 ``connect()``；``disconnect()`` 也支持重复调用。

通过句柄调用设备专有方法
------------------------

部署层根据部件类生成 worker 接口，无需为每种设备额外实现一层 worker。标准接口之外的公有方法可以通过句柄直接调用：

.. code-block:: python

   robot.handles["arm"].is_robot_up().wait()[0]
   robot.handles["arm"].reset_joint(home_qpos).wait()

本地与远端部件使用相同的调用方式。句柄适用于初始化、诊断和厂商专有操作；任务中的观测与动作仍应通过部件树访问。

保持导入边界
------------

部件模块不能导入 Ray、Gymnasium 或 ``rlinf.scheduler``，以便在不启动集群的机器上直接导入和测试驱动。``rlinf/robotics/placement/handles.py`` 是唯一的桥接层，并且只在远程部署时加载。

该依赖关系保持单向：调度器不导入 robotics。``tests/unit_tests/test_robotics.py`` 会检查两个方向的导入边界。

代码位置
--------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 路径
     - 内容
   * - ``robotics/parts/base.py``
     - ``Endpoint``、各类部件、``Group`` 和 ``Connection``。
   * - ``robotics/parts/arms/``
     - 机械臂与联动控制器实现。
   * - ``robotics/parts/cameras/``
     - 相机生命周期，以及 RealSense、ZED 和 Lumos 实现。
   * - ``robotics/parts/end_effectors/``
     - 夹爪和灵巧手。
   * - ``robotics/parts/views.py``
     - 将共享厂商会话呈现为部件的 ``MethodArm``、``MethodGripper`` 和 ``MethodCamera``。
   * - ``robotics/placement/``
     - 延迟声明、本地与远端句柄、worker 放置。
   * - ``robotics/robot.py``
     - 最外层组合、声明快照、``describe()`` 和生命周期。
   * - ``robotics/discovery/``
     - 机器人注册、发现与配置查找。

后续阅读
--------

- :doc:`添加机器人 <../extending/new_robot>`：按实际接入顺序应用本页介绍的类型和机制。
- :doc:`placement <placement>`：了解调度器资源如何映射到节点和 GPU。
- :doc:`遥操作 <../guides/teleoperation>`：了解环境侧如何组合操作者设备与 binding。
