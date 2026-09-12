真机任务与环境
==============

本页说明真机环境如何组合机器人与任务，并在外层加入遥操作、人工结果标记和数据转换等 wrapper。新增行为前，可据此判断它属于硬件能力、任务目标还是 rollout 外层流程，避免把不同职责继续堆入 env subclass。

本页沿数据在环境栈中的流向展开：先定义任务配置与 env 类，再通过 Gymnasium ID 注册；随后说明 env 如何持有并读写一台组合机器人，最后按照执行顺序介绍 wrapper、遥操作设备、动作仲裁和 episode 控制。若尚未了解机器人访问路径，请先阅读 :doc:`机器人接口 <robotics>`。

定义任务数据与行为
------------------

先处理任务之间真正不同的内容：目标、成功条件和任务特有的复位运动。``rlinf/envs/real/tasks/`` 中的任务只保存这些内容，并且只编写一次，供所有能运行它的机器人使用。插销任务由一个配置 dataclass 和一个 class 组成，后者在 ``CartesianTarget`` 的基础上加入自己的复位运动：

.. code-block:: python

   @dataclass
   class PegInsertionConfig(FixtureConfig):
       random_xy_range: float = 0.05
       clip_z_range_high: float = 0.1
       ...


   class PegInsertion(CartesianTarget):
       CONFIG = PegInsertionConfig
       DESCRIPTION = "peg and insertion"

       def reset(self, parts, context):
           # 先夹紧插销并抬离插孔，再返回初始位姿。
           context.action.grasp(parts)
           hold(parts)
           lift(parts, context, 0.10)
           self.go_to_rest(parts, context)

``PegInsertionConfig`` 为任务统一提供目标、目标周围的工作空间和复位随机范围。任务只声明它对机器人的要求，即一个上报 ``tcp_pose`` 的机械臂，从不构造动作：无论机器人使用哪些通道，``context.action.grasp()`` 都通过 policy 所驱动的同一通道闭合夹爪。

``rlinf/envs/real/franka/base.py`` 中的 ``FrankaEnv`` 等机器人 preset 提供另一半：机器人 class、把 policy 的数字转换为零部件命令的动作通道，以及 policy 读取的观测布局。任务 ID 于是只需一次注册，指定任务以及该任务默认需要的机器人侧设置：

.. code-block:: python

   class PegInsertionEnv(FrankaEnv):
       TASK = PegInsertion
       DEFAULTS = {
           "compliance_param": compliance(translational_stiffness=2000),
           "action_scale": (0.02, 0.1, 1.0),
       }

``compliance()`` 将任务参数合并到 ``COMPLIANCE_DEFAULTS``；字段名错误或控制器不支持相应参数时，该函数会立即报错。插销任务只覆盖一项参数，bin relocation 覆盖十一项。运行中的设置优先于这些默认值。

将硬件设置保留在机器人描述中
----------------------------

任务决定目标、复位动作、奖励、动作限制和图像处理方式。机器人的地址、相机序列号与 backend、夹爪配置及 placement 则描述真机设备，应写在 ``cluster.node_groups[].hardware.configs`` 中。``env.train.override_cfg`` 和 ``env.eval.override_cfg`` 只接受环境与任务设置。例如，``enable_camera_player`` 控制图像显示，仍属于环境配置。省略 ``camera_serials`` 时，系统通过选定的 backend（默认为 RealSense）发现相机，并按序列号排序。只有需要选择部分相机或指定顺序时才填写序列号；对于支持无相机运行的机器人，显式空列表表示不使用相机。

独立检查工具与 scheduler 启动使用相同的枚举流程。下面的示例从节点本地环境变量读取 SO-101 设置，解析相机序列号并验证相机设备，然后创建环境：

.. code-block:: python

   from rlinf.envs.real.so101 import SO101ReachEnv
   from rlinf.robotics.discovery import RobotDiscovery

   # Set SERIAL_PORT and CALIBRATION_ID for this rig.
   discovery = RobotDiscovery.registry["SO101"].discovery_cls
   resources = discovery.enumerate(node_rank=0, configs=[])
   if resources is None or len(resources.infos) != 1:
       raise ValueError("Configure one SO-101 for this check.")

   env = SO101ReachEnv(
       {"enable_camera_player": False, "target_joint_qpos": [0.0] * 5},
       robot_info=resources.infos[0],
   )
   try:
       observation, info = env.reset()
       print(observation["state"])
   finally:
       env.close()

``enumerate()`` 返回硬件描述，不会打开机械臂的控制连接。每个 ``RobotInfo`` 的 ``config`` 保存解析后的类型化配置。环境构造函数连接机器人，并可能将其移动到复位姿态；``reset()`` 开始一个 episode，返回任务观测和 info 字典；``close()`` 释放机器人的连接。SO-101 检查工具也使用这一流程。``SO101Config.serial_port`` 和 ``calibration_id`` 通过共享 resolver 从 ``SERIAL_PORT`` 和 ``CALIBRATION_ID`` 解析，因此无关的 ``PORT=8080`` 不会被当作串口地址。已有机械臂配置时，``--port`` 必须精确匹配其中一个串口，并保留该机械臂的 calibration；找不到匹配项会报错。没有机械臂配置时，``--port`` 和 ``--id`` 用于描述本地机械臂。``--id`` 显式设置选中机械臂的 calibration 标识。SO-101 硬件 YAML 条目中的 ``port`` 应改为 ``serial_port``。检查工具的 ``--port`` 和 ``--id`` 分别选择串口和 calibration 标识。

通过 scheduler 运行时，节点 probe 完成枚举，worker placement 分配 ``RobotInfo``，再由 ``RealWorldEnv`` 将其传给任务构造函数。环境只读取其中的硬件配置，不修改原对象。``camera_serials``、``robot_ip`` 等字段不再允许出现在任务 override 中，应移至硬件条目。若硬件默认值已能描述所需的观测空间，dummy 环境可以省略 ``robot_info``；需要其他相机布局或末端执行器时，也应传入对应布局的描述。Franka 在 dummy 模式下也要求至少一个相机，因此构造时始终需要带有相机序列号的描述。离线运行可以使用虚拟序列号；dummy 构造过程不会打开或探测设备。

唯一仍然拥有独立 env 的机器人是 DOSW1，其任务 dataclass 保持原名 ``DOSW1EnvConfig``，对应的硬件配置仍位于 ``rlinf.robotics.robots``。Turtle2 的相机通道从任务字段 ``use_camera_ids`` 移至硬件字段 ``camera_ids``；本次运行驱动哪几条机械臂，也从 ``use_arm_ids`` 的编号改为任务的 ``roles``，直接写出机械臂的名字。

Turtle2、双臂 Franka、单臂 Franka、Piper、SO-101 和 GimArm 的任务运行在 ``TaskEnv`` 上。运行时仍然只传入一个扁平的 ``override_cfg``，其中每个 key 交给声明它的那一个配置：``RegisteredTaskEnvConfig`` 负责 episode 如何运行、相机和 reward model；动作通道的配置（``PoseActionConfig`` 或 ``JointActionConfig``）负责动作缩放、增益和关节范围；任务配置（例如 ``PegInsertionConfig`` 或 ``JointReachConfig``）负责目标与奖励。有自身设置的 preset 会再增加一个配置，例如 GimArm 用于控制模式的 ``GimArmOptions``。这些配置都没有声明的 key 会被拒绝，``hand_target_state`` 等已停用的 key 会被丢弃并给出警告。Piper 的硬件字段 ``with_gripper`` 决定 action 包含 6 个关节值，还是包含夹爪开度的 7 个值。

注册任务
--------

任务的配置和行为确定后，需要用稳定 ID 供配置文件和数据集引用。``TASKS`` 只记录 Gymnasium ID 与 env class 的对应关系；wrapper 由 env 自行声明，不进入第二套注册流程：

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": FrankaEnv,
       "PegInsertionEnv-v1": PegInsertionEnv,
       "DualFrankaTcpEnv-v1": DualFrankaTcpEnv,
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` 将每一项映射转换为 Gymnasium entry point，并把生成结果保存在 ``_ENTRY_POINTS``。用户配置和数据集元数据都会保存 Gym ID，因此数据采集开始后不应随意修改 ID。

以这种方式注册到机器人上的每个任务，还会得到一个以任务命名、与机器人无关的 ID，例如 ``PegInsertion-v1`` 或 ``JointReach-v1``。它在运行所分配的机器人上执行该任务，并根据 ``robot_info.type`` 选择这台机器人的 preset，因此配置只需指定任务，机器人由集群的硬件配置决定。与机器人绑定的 ID 仍然保留。没有机器人描述的 dummy 运行无从选择机器人，仍需使用绑定机器人的 ID。

与机器人无关的 ID 要求每台机器人只对应一个 preset；如果同一台机器人用两种方式驱动同一个任务，这个前提就不成立。双臂 Franka 注册了两次 ``MultiArmTarget``，一次按关节目标驱动，一次按末端位姿点驱动，两者的动作布局宽度不同，因此运行时必须指定 checkpoint 所对应的那个 ID。两者都设置 ``GENERIC_ID = False``，宁可让该任务没有与机器人无关的 ID，也不任意选择其中之一。

通过机器人接口读写硬件
----------------------

注册决定构造哪个 env，``TaskEnv`` 实例随后在整个生命周期内持有同一台组合机器人。初始化时，它绑定任务、动作布局和观测所需的零部件，并连接机器人；``close()`` 再断开连接。每个 step 只从 ``robot.get_observation()`` 取得一份嵌套观测，各个通道通过 ``robot.send_action()`` 下发具名动作，不在旁路直接访问 driver 或厂商 SDK。

不同硬件结构使用同一边界。Franka 的机械臂和末端执行器分别打开连接，因此使用并列路径；SO-101 的夹爪是机械臂总线上的另一个伺服，因此使用 ``arm.end_effector``。绑定在两种结构下都能找到末端执行器，所以 SO-101 的关节通道和夹爪通道都无需知道夹爪的位置，就能把关节目标和夹爪开度放在一条命令中下发。

episode 的结束时机和成败由一层决定：任务 env。它统计自己的步数，按自己的时长上限截断，应用 reward scale 和夹爪惩罚，并在 ``info`` 中报告任务是否认为这一步达成了目标。``RealWorldEnv`` 读取这些结果，而不再重新计算，只补充批量化所需的部分：它的 ``max_episode_steps`` 是在 env 时长上限之上再加一道上限，而不是替代它；``success_once`` 指标来自 env 的报告，而不是把缩放后的 reward 与 1 比较。

单步读写接口保持精简，就绪检查和复位则需要设备类别提供的方法。任务通过绑定到其角色上的零部件访问这些方法，零部件按类别提供类型：

.. code-block:: python

   def home(self, parts, context):
       arm = parts.arm()                  # 填充 "arm" 角色的 Arm
       arm.reset_joint(self.config.reset_joint_qpos)

``parts.arm()`` 返回绑定到该角色的 ``Arm`` 接口，``parts.end_effector()`` 返回它携带的末端执行器。相机的 placement 和生命周期仍由机器人管理；env 从构造状态所用的同一份整机观测中读取画面，因此同一步的数据不会混入后续 SDK 读取的结果。

在不同机器人上运行同一任务
--------------------------

任务只规定自己需要什么，而不指定由哪台机器人提供，因此同一个任务 class 可以在运动学和动作通道都不同的机器人上运行。``PegInsertionEnv-v1`` 和 ``GimArmPegInsertionEnv-v1`` 运行的都是 ``PegInsertion``：

.. code-block:: python

   class PegInsertionEnv(FrankaEnv):         # 笛卡尔增量，灵巧手或夹爪位于机械臂旁
       TASK = PegInsertion


   class GimArmPegInsertionEnv(GimArmEnv):   # 关节目标，夹爪在机械臂总线上
       TASK = PegInsertion
       DEFAULTS = {
           "reset_mode": "joint",
           "safe_retract_qpos": (0.0, -1.5, 1.5, 0.0, 0.0, 0.0),
       }

Franka preset 在任务的工作空间内以笛卡尔增量移动末端。GimArm preset 下发绝对关节目标，末端工作空间对关节命令没有意义，因此它的通道会忽略工作空间。两者的奖励相同，都是末端到插销就位位姿的距离，因为两台机械臂都上报 ``tcp_pose``。唯一的区别在复位：GimArm 无法接收末端位姿，所以 ``reset_mode="joint"`` 通过关节配置回缩和停靠，而不是抬起末端。这样的选项对应机械臂之间真实存在的差异；任务从不根据机器人类型分支。

属于某台机器人、而不属于任务或其通道的设置，例如 GimArm 的控制模式，放在 preset 的 ``OPTIONS`` dataclass 中，由 preset 传给 ``Robot.from_config``。preset 为其任务未声明的设置提供的默认值会被丢弃，因此 preset 的默认值不会妨碍它运行其他任务。

单元测试在一个假关节机械臂上把 ``PegInsertion(PegInsertionConfig(reset_mode="joint"))`` 与一个关节通道、一个二值夹爪通道组合起来，不经过任何 Gymnasium ID。新机器人在拥有 preset 之前，也可以用同样的组合方式试运行现有任务。

按照职责组织 wrapper
---------------------

至此，基础 env 已经定义任务行为和硬件读写。wrapper 只应改变外层 rollout 流程，并根据所转换的内容划分目录：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 包
     - wrapper 的职责
   * - ``teleop/``
     - 修改动作本身。在操作者接管期间，以操作者指令替换 policy 输出。
   * - ``transforms/``
     - 转换观测或动作的表示形式，不改变其语义。例如，将同一运动转换到末端坐标系。
   * - ``episode/``
     - 控制 rollout 的起止和评分。这些信息由现场操作者提供，而非由传感器产生。

``build_stack`` 根据 env 配置和声明依次应用这些 wrapper，并返回最外层 env；不同机器人共用同一套构建流程。

.. code-block:: python

   env = build_stack(PegInsertionEnv(...), cfg)

在环境侧处理动作接管
--------------------

根据前面的职责划分，动作替换属于 ``teleop/``。:doc:`遥操作指南 <../guides/teleoperation>` 介绍设备选择和 binding，本节继续跟踪动作在 env 侧的两个步骤：先在操作者与 policy 之间仲裁，再将选中的具名动作写入 env 使用的扁平 action vector。

``TeleopIntervention`` 会在短时间内保留最近一次操作者动作，避免两次设备采样之间切回 policy。PICO 使用 grip 明确标识接管区间，因此将 ``timeout`` 设为 0，并在松开后立即交还控制权。数据采集器从 ``intervene_action`` 读取仲裁后的动作。

``TeleopGroup`` 按零部件名称返回动作，而环境接收扁平向量。``ComposedTeleop`` 根据环境声明的布局，将各零部件的动作写入对应区间；未由操作者控制的部分保留 policy 输出。

布局还声明动作语义。``FrankaEnv`` 将前六个数解释为位姿增量，``GimArmEnv`` 则将其解释为关节角。两者宽度相同，但 SpaceMouse 只能匹配前者。binding 与环境声明的动作类型不一致时，系统会在构建阶段报错。

单台设备使用一个配置项；组合多台设备时，将配置改为列表：

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse   # 也可以是 gello、pico、none
       gello_port: /dev/serial/by-id/...

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

设备名称来自 ``TeleopDevice`` registry。需要兼容旧布尔配置项的设备，在自己的 ``LEGACY_FLAGS`` mapping 中声明这些别名，由 ``TeleopDevice.legacy_flags()`` 汇总后交给配置读取逻辑。旧配置项仍会触发弃用警告；配置叠加后同时出现两种形式时，以 ``teleop`` 为准。

分离设备读取与动作映射
----------------------

env 侧仲裁能够保持清晰，前提是设备读取与动作映射分开。下面沿一次遥操作采样，从设备、group 到 wrapper 说明各层职责：

- ``robotics/parts/teleop/<device>.py``：读取一台串口设备、HID 设备或头显，并声明其读数对应哪些机器人动作及各自的语义。它本身是 ``RobotPart``，因此拥有一致的连接、观测和断开接口，也可以放置到其他节点。
- ``robotics/parts/teleop/base.py``：保存 registry 和各设备的公共逻辑；``group.py`` 将多台设备的动作合并为一个动作。
- ``real/wrappers/teleop/builder.py``：解析配置中的设备名称；``composed.py`` 再根据零部件名称，将动作写入 env 声明的扁平 action vector。

前两层产生具名动作，第三层才知道这些名称如何写入具体 env 的向量。设备层不依赖 Gymnasium，因此排查线缆或设备权限时可以单独运行设备，无需启动机器人；扁平 action vector 的转换仍由 env 层完成，因为只有它知道对应布局。

连接机器人前，可使用以下命令检查主臂接线：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0

遥操作设备本身也是 :class:`~rlinf.robotics.parts.base.RobotPart`。:class:`~rlinf.robotics.parts.teleop.base.TeleopPart` 直接继承该类，因此沿用标准连接生命周期。构造设备时不会访问硬件；wrapper stack 启动后，``TeleopGroup.connect()`` 才会依次打开设备。

遥操作设备在类型上继承 ``RobotPart``，但不会加入 ``Robot`` 的组合结构。主臂读取操作者输入，而非机器人状态，因此 policy 不会观测该设备。设备控制哪些机器人零部件，由环境侧的 binding 决定。这个边界也影响 placement：内置遥操作构建器在 env 进程中打开设备，不会经过 ``Robot.connect()``。手动部署独立设备前，请先阅读 :doc:`遥操作指南 <../guides/teleoperation>`。

将 episode 控制置于独立层
-------------------------

并非所有操作者输入都属于遥操作。标记成功、放弃当前 rollout 或切换 policy 改变的是 episode 状态，而不是前一节仲裁得到的动作。此类 wrapper 位于 ``episode/``，并共用 :class:`KeyboardSession`。该类负责键盘监听、防抖，以及在 reset 时清空队列，避免机械臂返回初始位姿期间的输入影响下一个 episode。

新增控制模式时，读取 ``presses()`` 并定义各按键的含义：

.. code-block:: python

   class KeyboardRLTPolicySwitchWrapper(KeyboardSession):
       def step(self, action):
           obs, reward, terminated, truncated, info = self.env.step(action)
           for key in self.presses():
               if key == "b":
                   self._rlt_switch_flags = True
           info["rlt_switch_flags"] = self._rlt_switch_flags
           return obs, reward, terminated, truncated, info

代码位置
--------

代码目录沿用前述数据流顺序，从任务构建、机器人读写到三类 wrapper 分别组织：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 路径
     - 内容
   * - ``real/<robot>/``
     - 每个任务对应一个模块；``base.py`` 保存公共逻辑，``__init__.py`` 保存 ``TASKS`` 映射。
   * - ``robotics/parts/teleop/``
     - 每个操作者设备对应一个模块；``base.py`` 保存公共逻辑，``group.py`` 将多个设备合成一个动作。
   * - ``robotics/actions.py``
     - 动作向量中每一段的语义，env 与设备共用。
   * - ``real/wrappers/teleop/``
     - 设备选择、policy 与操作者动作仲裁、扁平动作布局，以及可选的直接推送路径。
   * - ``real/wrappers/transforms/``
     - 相对坐标系、四元数转欧拉角、夹爪维度裁剪。
   * - ``real/wrappers/episode/``
     - 各类键盘会话：奖励与结束、开始与结束、评测控制、policy 切换、主从臂。
   * - ``real/wrappers/__init__.py``
     - 组装三类 wrapper 的构建函数。
   * - ``real/registry.py``
     - ``task_factory`` 与 ``register_tasks``。
   * - ``real/env.py``
     - ``RealWorldEnv``，把一个任务 env 适配到批量化 runner：张量转换、主视角与额外视角的拆分、auto-reset、指标统计和 action chunk。episode 本身由它下层的任务 env 负责。
   * - ``real/task_env.py``
     - ``TaskEnv`` 在一台机器人上运行一个任务；``RegisteredTaskEnv`` 根据运行配置构造它，供 Gymnasium ID 使用。
   * - ``real/tasks/``
     - 只编写一次、可在任何满足要求的机器人上运行的任务，以及把任务绑定到机器人零部件的要求核对。
   * - ``real/policy/``
     - policy 发出和读取的内容：每台机器人布局中的动作通道，以及 checkpoint 所依赖的观测 key、编码方式和颜色顺序。如果某个通道驱动的零部件比控制周期更慢，它会把实际动作放在 ``Command.defer`` 中返回，由布局在该角色自己的队列上执行，而不让 step 等待。

后续阅读
--------

根据需要扩展的层次继续阅读：

- :doc:`新增真机任务 <../extending/new_task>`：按步骤接入新的真机任务。
- :doc:`机器人接口 <robotics>`：了解如何读取和控制底层机器人。
