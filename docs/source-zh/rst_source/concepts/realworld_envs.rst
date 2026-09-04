真机任务与环境
==============

本页说明真机环境如何组合机器人与任务，并在外层加入遥操作、人工结果标记和数据转换等 wrapper。新增行为前，可据此判断它属于硬件能力、任务目标还是 rollout 外层流程，避免把不同职责继续堆入 env subclass。

本页沿数据在环境栈中的流向展开：先定义任务配置与 env 类，再通过 Gymnasium ID 注册；随后说明 env 如何持有并读写一台组合机器人，最后按照执行顺序介绍 wrapper、遥操作设备、动作仲裁和 episode 控制。若尚未了解机器人访问路径，请先阅读 :doc:`机器人接口 <robotics>`。

定义任务数据与行为
------------------

先处理任务之间真正不同的内容：目标、成功条件、控制器参数和任务特有的复位运动。``rlinf/envs/real/franka/`` 中的每个任务对应一个模块；dataclass 保存这些配置值，无法用数据表达的行为则由 env class 覆盖，公共逻辑继续留在 ``base.py``。

.. code-block:: python

   @dataclass
   class PegInsertionConfig(FrankaRobotConfig):
       task_description: str = "peg and insertion"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       random_xy_range: float = 0.05

       def __post_init__(self):
           # 仅覆盖与公共阻抗参数不同的项。
           self.compliance_param = compliance(translational_stiffness=2000)
           ...


   class PegInsertionEnv(FrankaEnv):
       CONFIG_CLS = PegInsertionConfig

       def go_to_rest(self, joint_reset=False):
           # 先抬离插孔，再返回初始位姿，避免插销卡住。
           ...

``PegInsertionConfig`` 为继承的 env 逻辑统一提供目标、随机范围和控制器设置，``CONFIG_CLS`` 则告诉 ``PegInsertionEnv`` 应构造哪一种配置。``go_to_rest()`` 只覆盖与插销任务相关的复位顺序。``compliance()`` 将任务参数合并到 ``COMPLIANCE_DEFAULTS``；字段名错误或控制器不支持相应参数时，该函数会立即报错。插销任务只覆盖一项参数，bin relocation 覆盖十一项，其余任务数据也保存在各自配置中。

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

通过机器人接口读写硬件
----------------------

注册决定构造哪个 env class，env 实例随后在整个生命周期内持有同一台组合机器人。初始化时，它构建机械臂、末端执行器和相机，并调用 ``robot.connect()``；``close()`` 再通过 ``robot.disconnect()`` 释放资源。每个 step 只从 ``robot.get_observation()`` 取得一份嵌套观测，并通过 ``robot.send_action()`` 下发具名动作，不在旁路直接访问 driver 或厂商 SDK。

不同硬件结构使用同一边界。Franka 的机械臂和末端执行器分别打开连接，因此使用并列路径；SO-101 的夹爪是机械臂总线上的另一个伺服，因此使用 ``arm.end_effector``。``SO101ReachEnv-v1`` 仍通过这套嵌套接口读写硬件，再将数据转换为 policy 使用的六维关节与夹爪向量。

单步读写接口保持精简，就绪检查和复位则需要设备类别提供的方法。env 可以从同一机器人中保留具类型零部件，用于完成这些初始化操作：

.. code-block:: python

   from rlinf.robotics import Arm, Camera

   arm = robot.child("arm", Arm)
   cameras = robot.parts_of_type(Camera)

   if not arm.is_robot_up():
       raise RuntimeError("The arm is not ready.")
   arm.reset_joint(reset_qpos)
   ready = all(camera.is_ready() for camera in cameras.values())

``child("arm", Arm)`` 检查任务要求的机械臂路径，并返回用于就绪检查和复位的 ``Arm`` 接口。``parts_of_type(Camera)`` 按完整路径返回所有相机，使 env 无需假设相机名称。相机的 placement 和生命周期仍由机器人管理；env 可以保留引用用于处理画面，但不应为同一设备构建或关闭第二个对象。构造当前步的状态和画面时，还应复用一次整机读取结果，避免混入后续 SDK 读取的数据。

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
     - ``RealWorldEnv``，框架根据 ``env_type: real`` 创建的向量化环境类。
   * - ``real/task_env.py``
     - ``RobotTask`` 和 ``RobotTaskEnv`` 划定任务逻辑与硬件代码的边界。

后续阅读
--------

根据需要扩展的层次继续阅读：

- :doc:`新增真机任务 <../extending/new_task>`：按步骤接入新的真机任务。
- :doc:`机器人接口 <robotics>`：了解如何读取和控制底层机器人。
