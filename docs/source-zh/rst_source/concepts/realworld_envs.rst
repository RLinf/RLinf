真机环境模型
============

真机环境将机器人与任务组合，并在外层添加遥操作、人工结果标记和数据转换等 wrapper。

新增逻辑时，可根据职责确定代码位置：直接读写硬件的逻辑属于部件；奖励、成功条件和复位流程属于任务；只改变 rollout 控制方式或数据表示的逻辑属于 wrapper。若尚未了解部件树模型，请先阅读 :doc:`robotics`。

任务由配置和少量覆盖组成
--------------------------

``rlinf/envs/real/franka/`` 中的每个任务对应一个模块，公共逻辑位于 ``base.py``。通常只需新增一个配置 dataclass；任务需要特殊复位流程时，再覆盖相应的 env 方法。

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

``compliance()`` 将任务参数合并到 ``COMPLIANCE_DEFAULTS``。字段名错误或控制器不支持相应参数时，该函数会立即报错。插销任务只覆盖一项参数，bin relocation 覆盖十一项；位姿、奖励阈值和复位随机范围也分别保存在对应的任务配置中。

注册任务
--------

``TASKS`` 只记录 Gymnasium ID 与 env 类之间的对应关系。wrapper 由 env 自行声明，无需写入注册表：

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": FrankaEnv,
       "PegInsertionEnv-v1": PegInsertionEnv,
       "DualFrankaTcpEnv-v1": DualFrankaTcpEnv,
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` 根据该映射生成 Gymnasium entry point。用户配置和数据集元数据都会保存 Gym ID，因此数据采集开始后不应随意修改 ID。

按职责组织 wrapper
-------------------

wrapper 根据所承担的职责划分目录：

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

``build_stack`` 根据 env 声明组装 wrapper，所有机器人共用同一套构建流程。

.. code-block:: python

   env = build_stack(PegInsertionEnv(...), cfg)

在环境侧处理动作接管
--------------------

:doc:`遥操作指南 <../guides/teleoperation>` 介绍设备选择和 binding。环境侧处理两个问题：在相邻采样之间保持接管状态，以及将按部件名称返回的动作写入环境使用的扁平 action vector。

``TeleopIntervention`` 会在短时间内保留最近一次操作者动作，避免两次设备采样之间切回 policy。PICO 使用 grip 明确标识接管区间，因此将 ``timeout`` 设为 0，并在松开后立即交还控制权。数据采集器从 ``intervene_action`` 读取仲裁后的动作。

``TeleopGroup`` 按部件名称返回动作，而环境接收扁平向量。``ComposedTeleop`` 根据环境声明的布局写入各部件动作；未由操作者控制的部分保留 policy 输出。

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

遥操作代码分为以下四层：

- ``robotics/parts/teleop/readers/``：直接读取串口设备、HID 设备和头显。
- ``robotics/parts/teleop/devices.py``：将 reader 封装为 ``TeleopPart``，并提供与其他部件一致的连接、观测和断开接口。
- ``robotics/teleop/bindings.py``：声明设备读数对应的机器人动作部件。
- ``real/wrappers/teleop/composed.py``：根据部件名称，将动作写入环境声明的扁平 action vector。

排查线缆或设备权限时，可单独运行 reader，无需启动机器人。同一台物理设备也可以通过不同的 binding 接入另一种动作空间。

连接机器人前，可使用以下命令检查主臂接线：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

遥操作设备本身也是 :class:`~rlinf.robotics.parts.base.RobotPart`。:class:`~rlinf.robotics.parts.teleop.devices.TeleopPart` 直接继承该类，因此连接、断开和远程放置方式与机械臂一致。例如，``SpaceMouse.at(node_rank=1)`` 会在节点 1 上打开设备。

遥操作设备虽然继承 ``RobotPart``，但不属于机器人部件树。主臂读取操作者输入，而非机器人状态，因此 policy 不会观测该设备。设备控制哪些机器人部件，由环境侧的 binding 决定。

将 episode 控制置于独立层
-------------------------

标记成功、放弃当前 rollout 或切换 policy 都不会修改机器人动作。此类 wrapper 位于 ``episode/``，并共用 :class:`KeyboardSession`。该类负责键盘监听、防抖，以及在 reset 时清空队列，避免机械臂返回初始位姿期间的输入影响下一个 episode。

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

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 路径
     - 内容
   * - ``real/<robot>/``
     - 每个任务对应一个模块；``base.py`` 保存公共逻辑，``__init__.py`` 保存 ``TASKS`` 映射。
   * - ``robotics/parts/teleop/``
     - 操作者设备及其部件接口；底层 ``readers/`` 不依赖 Gymnasium。
   * - ``robotics/teleop/``
     - binding、动作含义和 ``TeleopGroup`` 组合。
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

- :doc:`新增真机任务 <../extending/new_task>`：按步骤接入新的真机任务。
- :doc:`机器人模型 <robotics>`：了解机器人的部件树及其组合方式。
