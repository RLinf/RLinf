机器人接口
============

本页介绍如何在任务和 ``env`` 代码中使用真机接口连接、读取和控制物理机器人。全文按从单机调用到系统集成的顺序展开：

1. 如何连接单台机器人、读取观测并下发控制动作。
2. 如何用真机接口组装多部件机器人，并按路径访问各个部件。
3. 如何在底层自动平滑大幅度运动，并将多部件控制指令合并下发。
4. 如何在运行中检测硬件报错并自动恢复，避免机械臂失控或跳变。
5. 如何在强化学习 ``env`` 和多节点集群中使用组装好的机器人。

硬件会话管理、资源归属与进程部署的内部实现见 :doc:`机器人架构 <robotics_architecture>`。

连接、读取与控制机器人
----------------------

控制一台机器人通常分为四步：构建并查看机器人结构、获取初始化所需的部件、连接硬件并运行 ``step`` 循环，最后断开连接。下面的示例展示了这一完整流程：

.. code-block:: python

   from rlinf.robotics import Arm, Camera, build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # 机器人未上电或网络未连通时也可以安全调用。
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

构建阶段只记录硬件参数和部署节点，不会导入厂商 SDK，也不会打开硬件设备。调用 ``build_robot()`` 会返回一个尚未连接的 ``Robot`` 对象；调用 ``describe()`` 可以直接查看各部件的访问路径、所在节点和底层连接归属，即使机器人还没上电也能正常执行。

连接硬件前，可以先获取初始化所需的部件引用。如果明确知道部件路径（例如 ``arm``），用 ``child("arm", Arm)`` 获取并校验其类型；如果只想找出机身上安装的所有相机而不关心具体命名，用 ``parts_of_type(Camera)`` 遍历整棵机器人结构树，返回以路径为 key 的字典。

调用 ``connect()`` 会依次打开底层硬件连接；如果某条连接打开失败，会自动关闭前面已经打开的连接。连接成功后，可以先用 ``is_robot_up()`` 和 ``is_ready()`` 检查机械臂与相机是否就绪，或调用 ``clear_errors()`` 和 ``reset_joint()`` 完成清错与初始姿态复位。

进入 ``step`` 循环后，只需一读一写两个接口：``get_observation()`` 一次性读取所有部件，返回以路径为 key 的嵌套字典；``send_action()`` 接收相同层级结构的动作字典，只下发本次传入的分支，并将各部件实际执行的动作作为 ``applied`` 返回。把 ``disconnect()`` 放在 ``finally`` 块中，可以保证哪怕中途报错也能按相反顺序安全释放所有硬件连接，且重复调用不会报错。

所有机器人都使用这套相同的调用流程，不同机器人的区别只在于观测和动作字典里的路径结构。

组装机器人并按路径访问部件
--------------------------

``get_observation()`` 和 ``send_action()`` 字典里的路径，直接由组装机器人时的部件命名和硬件连接方式决定。增加部件或对部件分组只会增加路径层级，不会改变读写方法。

独立连接与共享总线的路径差异
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

组装时赋给部件的名称成为第一段路径；如果某个部件下面还挂载了共用同一条通信总线的子部件，路径就会再往下延伸一层。对比单臂 Franka 和 SO-101 可以看清这两种结构。

单臂 Franka 的机械臂控制器与 Franka Hand 夹爪分别使用独立的网络端点通信，因此 ``arm`` 和 ``end_effector`` 处于同一层级：

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

``describe()`` 输出中不同的 ``via`` 编号说明它们各自拥有独立的底层连接。

SO-101 的五个机械臂关节和夹爪则串在同一条舵机总线上。由于夹爪通过机械臂的连接收发指令，它挂在 ``arm`` 下面成为 ``arm.end_effector``，两者的 ``via`` 都是 ``SO101Arm#1``：

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

读写数据时的字典层级与路径结构完全一致：

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

组装多部件机器人
^^^^^^^^^^^^^^^^

组装一台新机器人时，每个可直接读写的部件（``RobotPart``）都可以作为关键字参数直接传给 ``Robot``，例如 ``Robot(base=base, arm=arm, end_effector=hand)`` 会生成 ``base``、``arm`` 和 ``end_effector`` 三个顶层路径。如果底层是一条同时驱动双臂或全身的共享连接（``Connection``），连接本身并不直接对应单个部件，此时先调用 ``session.part("left")`` 取出对应的部件，再传给 ``Robot``。

当多个部件共同组成一个逻辑单元（例如左臂加上左夹爪）时，可以用 ``PartGroup`` 将它们组合在一起，在外层加上 ``left`` 或 ``right`` 等路径前缀：

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

真机接口不限制机器人必须有哪些固定槽位。无论是接入移动底盘、腰部、头部还是第三条机械臂，都只需赋予一个路径名，上层依然通过 ``get_observation()`` 和 ``send_action()`` 读写。不过，当双臂或全身多个部件共用同一条底层连接时，要让硬件运动平稳且不互相冲突，还需要由底层连接统一处理运动平滑与多部件指令合并。

平滑运动与多部件协同控制
------------------------

高频流式控制器（通常为 50–100 Hz）要求持续收到小幅、连续的目标值，但 ``policy`` 或遥操作设备发来的指令往往做不到这一点。实际运行中，以下三种情况最容易造成机身猛冲、动作卡顿，或让控制器退出流式模式：

1. 单条指令的目标跨度过大，超过电机在一帧内可以安全移动的距离。
2. 指令到达的频率低于控制器，或者间隔忽长忽短。如果每条指令都直接作为一帧目标下发，运动就会变成阶梯状；指令循环中的任何停顿（例如一次较慢的 IK 求解、一个迟到的遥操作采样）都会让机械臂先停住、再猛地追上去。
3. 在双臂或全身共用一条底层连接的机器人上，先后调用左右臂的运动方法，会串行执行两段独立轨迹。

底层连接统一处理了这三种情况，任务代码不需要自己编写插值循环。

固定频率的目标流
^^^^^^^^^^^^^^^^

流式下发关节位置的底层连接，把“指令何时到达”和“目标何时下发”拆成了两件事。指令只负责更新连接内部的目标，调用会立即返回；后台循环则按固定的控制周期每次下发一帧，并在关节速度和加速度上限之内把这一帧逐步推向目标。上限按关节分别生效，但所有关节会按同一比例缩放，因此机械臂在关节空间中的运动方向保持不变。

由于指令通常比控制周期慢、间隔也不均匀，后台循环会在两条指令之间按最近两次目标隐含的速度，把最新目标继续向前推，最多推进一个半指令间隔。这样一来，遥操作时机械臂会平滑地跟随操作者的手，而不是一格一格地跳；指令循环出现短暂停顿时，运动也能平稳衔接。操作者突然停手时，机械臂最多会多走出这一个间隔对应的位移，随后回到最后的目标。如果目标的变化速度超过了速度上限，说明这是一次跳变而不是连续的指令流，后台循环会在速度和加速度上限内直接逼近它，不做外推，因此不会冲过目标。

``reset_joint()`` 这类点到点运动，会从当前目标值出发执行一段首尾速度均为零的余弦平滑轨迹。运动时长至少为传入的 ``duration``，如果关节上限或 ``max_velocity`` 参数要求更慢，时长会相应延长；调用会一直阻塞到运动结束。

移动底盘的速度指令同样需要连续性：许多底盘带有硬件看门狗，速度帧一旦中断就会急停。底盘速度脉冲会按设定频率连续发送平滑加速、匀速巡航和余弦减速的速度流，并在结束时补发零速帧让底盘平稳停下。

这套机制给控制回路带来的开销很小：``send_action()`` 更新完目标就会返回，后台循环每个周期只检查一次内存中的故障状态并下发一帧。控制频率和上限保存在 ``MotionProfile`` 中（``control_rate_hz``、以 rad/s 为单位的 ``max_joint_velocity``、以 rad/s² 为单位的 ``max_joint_acceleration``，以及 ``max_extrapolation_s``），惯量更大的关节组还可以通过 ``JointGroupSpec`` 设置更严格的上限。R2 的底层连接以 100 Hz 下发，双臂上限为 2.0 rad/s 和 20.0 rad/s²；腰部承载着双臂，上限收紧到 0.5 rad/s 和 2.0 rad/s²。需要调整时，可以在硬件配置中覆盖其中任意一项，或在运行时调用 ``configure_motion_profile()``：

.. code-block:: yaml

   hardware:
     type: R2
     config:
       motion_profile:
         max_joint_velocity: 1.5
         max_joint_acceleration: 15.0

其他机器人的驱动不需要自己再写一套下发循环，只要持有一个 ``rlinf.robotics.parts.motion`` 中的 ``JointStream``，就能获得同样的行为。驱动需要交给它五个可调用对象：下发一帧目标值、读取实测关节角、从内存中报告已锁存的故障、返回各关节的速度与加速度上限，以及返回连接的 ``MotionProfile``。此后，驱动的控制指令只需调用 ``track()`` 更新目标，或调用 ``move()`` 执行一段定时运动；故障恢复后调用 ``hold()``，下一条指令就会从实测关节角重新开始，而不会跳回过时的目标。如果需要提前停止运动（例如操作员或规划器取消了当前动作），可以调用 ``brake()``：它结束正在执行的运动，但会继续下发目标值，让关节在加速度上限内减速停下，而不是停在最后发出的那一帧上突然刹住。``R2Connection.hold()`` 就基于它实现，可以在运动执行期间从其他线程调用，同时也会结束正在进行的底盘脉冲。

合并多部件控制指令
^^^^^^^^^^^^^^^^^^

在双臂或全身共用同一个底层连接的机器人上，如果依次调用 ``left_arm.reset_joint(...)`` 和 ``right_arm.reset_joint(...)``，两段轨迹会串行执行：左臂运动时右臂原地等待，左臂结束后右臂才开始动。

把这些调用放进 ``with robot.batch_commands() as batch:`` 上下文，就能合并为同一条同步轨迹。在 ``with`` 块内，每个部件只更新自己那一部分的目标关节角，并在目标暂存后立即返回；退出 ``with`` 块时，底层连接让所有更新过的关节一起运动。由于块内的调用在机器人真正运动之前就已返回，需要在块结束后读取 ``batch.ok``，才能知道合并后的运动是否完成：

.. code-block:: python

   # 按需在运行时修改控制频率或关节速度上限
   robot.configure_motion_profile(control_rate_hz=100.0, max_joint_velocity=1.0)

   # 将左右臂复位合并为一次同步全身轨迹
   with robot.batch_commands() as batch:
       robot.child("left_arm").reset_joint(left_home_qpos)
       robot.child("right_arm").reset_joint(right_home_qpos)
   if not batch.ok:
       raise RuntimeError(f"复位未完成：{robot.diagnose().error_message}")

在正常的 ``step`` 循环中，``robot.send_action({...})`` 内部已经自动开启了多部件指令合并，因此 ``policy`` 输出的双臂或全身动作天然会同步下发；只有在初始化或复位阶段直接跨多个部件调用 ``reset_joint()`` 等方法时，才需要显式加上 ``with robot.batch_commands():``。

硬件故障诊断与自动恢复
----------------------

运动平滑避免了软件指令跳变，但真机交互中的碰撞或关节超限仍可能触发硬件保护，导致电机掉使能或控制器退出流式模式。你可以在 ``Robot`` 或任意部件上手动调用 ``diagnose()`` 检查健康状态、调用 ``recover()`` 清除故障，也可以保持自动恢复开启，让底层连接在每次发指令前自动检查并恢复：

.. code-block:: python

   status = robot.diagnose()
   if not status.healthy:
       robot.recover()

不阻塞控制回路的故障检测
^^^^^^^^^^^^^^^^^^^^^^^^

如果在每次下发 50–100 Hz 控制指令前都额外发起一次同步远程调用或串口查询去问控制器状态，通信延迟会直接卡住控制循环。因此，底层连接在内存中维护一份健康状态缓存（``HealthStatus``），并通过两条非阻塞途径更新它：

1. **后台状态回调**：当后台监听线程（如 ROS/RFrame 状态话题、UDP 广播或 SocketCAN 线程）收到错误码或控制模式变化时，立即更新内存中的健康状态。
2. **随关节状态读取一并检查**：对于每次读取关节角时已经在返回包里自带错误位图的硬件（如串口舵机或 CANopen 报文），底层在读取观测的同时直接检查该字段并更新内存缓存，不需要发起第二次总线请求。

每次发指令前，底层只读取内存里的健康状态，正常运行时没有任何额外通信开销。当检测到故障或主动调用 ``robot.diagnose()`` 时，返回的 ``HealthStatus`` 会用 ``FaultLevel`` 标明严重程度：

- ``FaultLevel.OK``：控制器健康，正常接收指令。
- ``FaultLevel.MODE_DROP``：没有硬件报错，但控制器退出了流式控制模式。
- ``FaultLevel.RECOVERABLE_FAULT``：触发了关节限位、速度限制或自碰撞等保护，或存在关节轴掉使能，可通过软件清错恢复。
- ``FaultLevel.UNRECOVERABLE_FAULT``：急停按下、安全锁触发或连接断开，无法由软件恢复，需人工介入。

自动故障恢复（``enable_fault_recovery``）默认开启，可在硬件 YAML 配置或运行时通过 ``robot.configure_fault_recovery(...)`` 调整：

.. code-block:: yaml

   hardware:
     type: R2
     config:
       enable_fault_recovery: true
       max_recovery_attempts: 3
       recovery_cooldown_window: 10.0

按故障等级恢复并防止位置跳变
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

触发恢复时，``recover()`` 会根据故障等级只执行必要的恢复步骤：

- **流式模式退出（``FaultLevel.MODE_DROP``）**：直接切回流式控制模式，不重置电机驱动器。
- **可恢复的驱动故障（``FaultLevel.RECOVERABLE_FAULT``）**：依次清除控制器错误码、重新为掉使能的关节轴上使能，再切回流式控制模式。
- **不可恢复故障或频繁复发**：遇到 ``FaultLevel.UNRECOVERABLE_FAULT`` 时直接抛出 ``RuntimeError``；对于可恢复故障，如果在 ``recovery_cooldown_window`` 秒内连续触发恢复超过 ``max_recovery_attempts`` 次，说明外部故障原因仍未消除，熔断机制也会停止重试并抛出 ``RuntimeError``。

急停即使已被人工松开，也仍按不可恢复故障处理。R2 的底层连接会根据机器人的 ``/system_state`` 和运动错误码 401 锁存急停状态，因此自动恢复永远不会在急停之后自行恢复运动；确认机器人可以安全运动后，调用 ``clear_errors()`` 才能继续。

为了防止电机重新上使能瞬间发生位置跳变，底层连接在重新上使能或切回流式模式前，会先用当前的实测关节角覆盖内部缓存的目标关节角。这样即使故障停机期间上层 ``policy`` 已经把目标角推到了远处，恢复后的第一帧指令也会让机器人平稳停留在当前姿态。

正常运行时这套机制零延迟；只有在某个 ``step`` 实际触发故障恢复时，清错、上使能和切模式会使控制流暂停约 50–300 ms，恢复完成后后续指令继续正常执行。

在新驱动中接入故障恢复
^^^^^^^^^^^^^^^^^^^^^^

为新机器人驱动（继承自 ``Connection`` 或 ``ControllablePart`` 的类）接入故障恢复时，只需按硬件通信方式二选一上报状态，并实现 ``clear_errors()``：

- **方式一（后台状态回调）**：在异步回调中调用 ``self.report_health(HealthStatus(...))``。
- **方式二（随状态读取检查）**：重写 ``_inspect_state_health(self, state) -> HealthStatus | None``，检查 ``get_state()`` 返回字典中的错误字段。
- **清错动作**：实现 ``clear_errors(self) -> bool``，在其中将缓存目标同步到实测角、清除硬件报错并恢复控制模式（若模式退出可单独恢复，还可按需重写 ``_recover_mode(self) -> str``）。

.. code-block:: python

   from rlinf.robotics import ControllablePart, FaultLevel, HealthStatus


   class MyArmDriver(ControllablePart):
       # 方式二：检查每次 get_state() 返回字典中自带的错误码
       def _inspect_state_health(self, state: dict) -> HealthStatus:
           err = int(state.get("error_code", 0))
           if err != 0:
               return HealthStatus(
                   healthy=False,
                   level=FaultLevel.RECOVERABLE_FAULT,
                   error_code=err,
                   error_message=f"drive fault {err}",
               )
           return HealthStatus(healthy=True)

       # 或方式一：在 SDK 异步状态回调中上报健康状态
       def _on_sdk_error_callback(self, err_code: int, msg: str) -> None:
           self.report_health(
               HealthStatus(
                   healthy=(err_code == 0),
                   level=FaultLevel.RECOVERABLE_FAULT
                   if err_code
                   else FaultLevel.OK,
                   error_code=err_code,
                   error_message=msg,
               )
           )

       def clear_errors(self) -> bool:
           self._sync_cmd_to_actual()
           self._sdk.reset_faults()
           return True

定义上述方法后，``robot.diagnose()``、``robot.recover()``、下发指令前的自动恢复、冷却窗口熔断以及恢复事件记录都会自动生效。

在强化学习环境与多节点集群中使用
--------------------------------

路径寻址、运动平滑与故障恢复都由机器人接口内部处理后，上层的强化学习 ``env`` 与集群调度器就可以用统一方式驱动任意组合的机器人。

对接强化学习环境与处理故障截断
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

机器人部件只负责读写传感器与驱动执行器，而具体的强化学习任务环境（继承自 ``RealWorldEnv``）负责定义奖励函数、终止条件、``episode`` 复位流程以及 Gymnasium spaces：

- **观测与动作转换**：每个 ``step`` 调用一次 ``robot.get_observation()`` 构造 ``state`` 和相机 ``frames``，共用底层连接的部件在同一次读取中复用同一份硬件状态快照，并通过 ``robot.send_action()`` 下发动作。如果 ``policy`` 使用扁平向量，可通过 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter`` 在扁平数组与嵌套字典之间自动转换。
- **episode 初始化与复位**：连接前通过 ``child()`` 或 ``parts_of_type()`` 获取的部件引用，在 ``reset()`` 中用于检查就绪状态、清除报错以及将机械臂复位到初始姿态。
- **故障打断时的 episode 截断**：在遥操作或独立评估脚本中，经历 50–300 ms 的自动恢复后可以直接继续操作；但在强化学习数据采集中，中间这几百毫秒的停顿会破坏当前 ``step`` 状态转移的连续性。因此 ``env`` 会在每个 ``step`` 结束时通过 ``robot.consume_recovery_event()`` 检查是否发生过自动恢复，把故障信息记录到 ``info["robot_fault"]``，并在开启 ``truncate_on_fault_recovery`` 时将该 ``step`` 标记为 ``truncated=True``，通知 rollout runner 结束当前 ``episode`` 并在下一轮采样前复位机器人。

跨进程与跨节点部署
^^^^^^^^^^^^^^^^^^

访问路径只表示机器人具备什么部件，不绑定驱动运行在哪个进程。相机可以运行在 ``env`` 进程，机械臂可以运行在另一台机器的 ``worker`` 上，但两者依然出现在同一个 ``robot.get_observation()`` 字典里，因为 ``node_rank`` 和 ``worker_name`` 配置在各自的底层连接上。不同连接之间会并行读写，而共用同一条连接的部件则按声明顺序依次执行，避免并发调用底层 SDK。

把某个部件迁移到其他进程或节点时，只需修改其连接声明中的部署参数，``env`` 和 ``policy`` 代码完全不需要改动。

后续阅读
--------

- :doc:`新增真机任务 <../extending/new_task>`：在已支持的真机硬件上定义新的 Gymnasium 任务。
- :doc:`添加机器人 <../extending/new_robot>`：实现并注册新的传感器、执行器或整机组合。
- :doc:`机器人架构 <robotics_architecture>`：了解 ``parts``、``children``、共享 ``Connection``、生命周期与 worker placement 的实现细节。
- :doc:`遥操作 <../guides/teleoperation>`：将操作员输入设备与机器人组合使用。
