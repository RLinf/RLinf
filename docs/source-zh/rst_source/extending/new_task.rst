新增真机任务
============

本页说明在 RLinf 已支持目标真机的前提下，如何新增真机任务，而无需修改机器人实现。完成后将得到一个任务 class、一行机器人上的注册、一个 Gymnasium ID 和一份可直接启动的 YAML 配置。

本页所述真机任务专指 ``rlinf/envs/real`` 下的任务模块。如需为模拟器或 benchmark 新增 task，请参阅 :doc:`new_env`。

任务只规定怎样算完成工作：目标在哪里，机器人必须上报哪些量才能评分，episode 之间如何恢复场景，以及每一步得到多少奖励。任务从不构造动作。policy 的动作如何送到机械臂由机器人的动作通道负责，硬件的连接与 placement 由机器人负责，因此同一个任务可以在所有满足其要求的机器人上运行。如果硬件本身尚未接入，请先按照 :doc:`new_robot` 实现零部件，并确认观测和动作能够正常传递，再添加任务。如果还需要 RLinf 尚未提供的操作者设备或 wrapper，应先完成任务主流程，再将其作为文末所述的独立扩展处理。

核心流程包含四步：编写任务，将其注册到机器人上，为一次运行添加配置，并验证注册结果。后续章节说明哪些能力可直接复用，并介绍新增操作者设备和新增 wrapper 两类可选扩展；大多数任务无需执行这两部分。

核心流程
--------

以下示例新增一个擦拭任务，并以 ``WipeEnv-v1`` 在现有 Franka 上运行。每一步都会产生下一步的输入：任务以一个 ID 注册到机器人上，YAML 选择该 ID，最后的检查则在硬件打开前确认整条解析路径可用。

1. 编写任务
~~~~~~~~~~~

新建 ``rlinf/envs/real/tasks/wipe.py``。按末端位置评分的任务基于 ``CartesianTarget`` 编写，它已经负责目标位姿、目标周围的工作空间、到达目标的奖励，以及把机械臂送回初始位姿的流程。任务的数值放在一个配置 dataclass 中，任务 class 只补充不同的部分：

.. code-block:: python

   from collections.abc import Sequence
   from dataclasses import dataclass

   from rlinf.envs.real.tasks.cartesian import (
       CartesianTarget,
       FixtureConfig,
       hold,
       lift,
       reach_target,
   )
   from rlinf.envs.real.tasks import Evaluation, Needs


   @dataclass
   class WipeConfig(FixtureConfig):
       reward_threshold: Sequence[float] = (0.02, 0.02, 0.02, 0.2, 0.2, 0.2)
       random_xy_range: float = 0.03
       clip_z_range_high: float = 0.05
       contact_force: float = 5.0


   class Wipe(CartesianTarget):
       CONFIG = WipeConfig
       DESCRIPTION = "wipe the surface"

       def requirements(self):
           return {"arm": Needs(observes=frozenset({"tcp_pose", "tcp_force"}))}

       def reset(self, parts, context):
           hold(parts)
           lift(parts, context, 0.05)
           self.go_to_rest(parts, context)

       def evaluate(self, reading, applied):
           arm = reading.arm()
           reached = reach_target(
               arm["tcp_pose"],
               self.config.target_ee_pose,
               self.config.reward_threshold,
               dense=self.config.use_dense_reward,
           )
           pressing = arm["tcp_force"][2] < -self.config.contact_force
           in_zone = reached.in_zone and pressing
           return Evaluation(reward=float(in_zone), in_zone=in_zone)

配置中的字段分别回答不同的问题：``target_ee_pose`` 是工件位姿，``reward_threshold`` 判断位置误差是否足够小，``random_xy_range`` 决定每个 episode 之间初始位姿的随机范围。``FixtureConfig`` 根据 ``clip_*_range`` 字段在目标周围划定工作空间，并让机械臂停在目标上方 ``clip_z_range_high`` 处；运行时若直接设置 ``ee_pose_limit_min``、``ee_pose_limit_max`` 或 ``reset_ee_pose``，则以设置值代替推导值。运行未设置 ``task_description`` 时，``DESCRIPTION`` 就是语言指令。

env 按以下顺序调用任务的方法。``requirements()`` 规定每个角色的零部件必须上报哪些量，这里在 ``CartesianTarget`` 所需的 ``tcp_pose`` 之外增加了 ``tcp_force``，并在机器人连接之前完成检查。``workspace`` 交给通道，通道会把每个下发的位姿限制在其中。``home()`` 在连接后执行一次，``reset()`` 在每个 episode 开始时执行，此时通道已经应用了机械臂的柔顺参数。擦拭任务先把抹布抬离表面，再由 ``go_to_rest()`` 把机械臂送回初始位姿；插销任务在同一位置夹紧插销并将其抬出插孔。``evaluate()`` 根据每一步之后的读数评分，``in_zone`` 计入结束 episode 所需的 ``success_hold_steps`` 连续成功次数，夹爪惩罚由 env 扣除，因此任务只需报告这一步应得的奖励。

机器人无法运行该任务时，系统会在连接任何硬件之前给出原因：

.. code-block:: text

   RequirementError: Wipe cannot run on FrankaRobot: arm (PoseOnlyArm) does not
   report ['tcp_force']; it reports ['tcp_pose']

关节空间机械臂以同样的方式使用关节任务。Piper 和 SO-101 都运行 ``JointReach``：``SO101ReachEnv-v1`` 的注册只有 ``class SO101ReachEnv(SO101Env): TASK = JointReach`` 一行，``PiperReachEnv-v1`` 则在 Piper 上注册同一个 class。preset 的动作布局把 policy 的扁平动作（SO-101 上是五个绝对关节目标加一个连续夹爪值）转换为发给 ``arm`` 和 ``arm.end_effector`` 的命令。运行配置可参考 ``examples/embodiment/config/env/so101_reach.yaml``。

2. 注册到机器人
~~~~~~~~~~~~~~~

任务可以在任何满足其要求的机器人上运行，接下来由机器人的 preset 决定这个 ID 驱动哪一台。新建 ``rlinf/envs/real/franka/wipe.py``：

.. code-block:: python

   from rlinf.envs.real.tasks.wipe import Wipe

   from .base import FrankaEnv, compliance


   class WipeEnv(FrankaEnv):
       TASK = Wipe
       DEFAULTS = {
           "compliance_param": compliance(
               translational_stiffness=800,   # 降低刚度以保持接触
               translational_clip_z=0.02,
           ),
           "action_scale": (0.02, 0.1, 1.0),
       }

``FrankaEnv`` 提供机器人、动作通道、观测布局和遥操作设备；注册只指定任务，以及该任务默认需要的设置。``action_scale`` 限制一次 policy 动作移动末端的幅度，``compliance_param`` 设置执行动作时使用的阻抗控制器。阻抗参数只需声明与默认值不同的部分：``compliance()`` 会将差异项合并到 ``COMPLIANCE_DEFAULTS``，字段名错误或控制器不支持相应参数时会在导入时报错。运行中设置了这些 key 时，以运行的设置为准。

注册只负责配置，不能覆盖 ``step``、``reset`` 或观测，这些由 ``TaskEnv`` 对所有机器人统一执行，并有测试保证这一点。

然后为这个 class 指定供配置和数据集长期引用的稳定 ID。在 ``rlinf/envs/real/franka/__init__.py`` 的 ``TASKS`` mapping 中加入一项：

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": WipeEnv,
   }

``register_tasks`` 根据该映射生成 entry point，并将其注册到 Gymnasium。wrapper 无需在此配置：动作布局声明了与其动作相匹配的 wrapper，``build_stack`` 会读取这项声明。注册 ``WipeEnv-v1`` 的同时也会注册 ``Wipe-v1``，它在运行所分配的机器人上执行 ``Wipe``；其他机器人注册 ``Wipe`` 后会自动加入该 ID，无需新增 ID。

Gym ID 会写入用户配置和数据集元数据，因此数据采集开始后不应再修改 ID。

3. 添加环境配置
~~~~~~~~~~~~~~~

注册 ID 后，YAML 可以为一次具体运行选择该任务，并提供随实验变化的参数。在 ``examples/embodiment/config/env/`` 下新增文件，结构如下：

.. code-block:: yaml

   env_type: real
   init_params:
     id: "WipeEnv-v1"      # 上一步注册的 gym id
     num_envs: null
   teleop: spacemouse
   override_cfg:
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
     random_xy_range: 0.03
     action_scale: [0.01, 0.1, 1.0]

``env_type: real`` 选择 RLinf 的真机 env adapter，``init_params.id`` 选择上一步注册的 Gymnasium 任务，``teleop`` 指定评估或数据采集使用的操作者设备。``override_cfg`` 是一个扁平 mapping，每个 key 交给声明它的那一个配置：env 的配置负责 episode 如何运行，动作通道的配置负责缩放和增益，``WipeConfig`` 负责任务本身。三者都没有声明的 key 会被拒绝。机器人地址和 placement 仍应写在集群硬件配置中。

4. 验证注册结果
~~~~~~~~~~~~~~~

此时，从 YAML 到任务 class 的核心路径已经完整。连接硬件前，先导入真机 env package，并确认 ID 可以解析：

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # 触发全部任务注册
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` 会检查所有内置任务。请将新 ID 加入 ``EXPECTED_IDS``，并在 ``TASK_SCHEMAS`` 中为 policy 将要训练的观测和动作添加一行。若不经过 Gymnasium ID 运行任务，可以手动组合 ``TaskEnv(robot, Wipe(), layout, observation=...)``，单元测试就是这样在假零部件上运行任务的。这项断言只验证注册；如果任务改变了面向机器人的观测或动作路径，还需按照 :doc:`new_robot` 运行 mock 和真机检查。

复用现有基础设施
----------------

符合现有机器人与 wrapper contract 的任务完成以上四步即可。下列职责已经由相应层次负责，任务代码应直接调用或配置这些能力，不应再次实现：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 需求
     - 现有实现
   * - 连接硬件、部署零部件
     - ``Robot.connect``，见 :doc:`../concepts/robotics`。
   * - 把 policy 动作转换为机械臂、夹爪和灵巧手的命令
     - 机器人 preset 的 ``ActionLayout`` 通道，例如 ``PoseDelta``、``JointPositions`` 以及末端执行器通道。
   * - 把下发的位姿限制在范围内
     - 任务的 ``workspace``，位姿通道会把每个位姿裁剪到其中。
   * - 用学习得到的 reward model 评分
     - ``use_reward_model``，由 env worker 根据运行的 ``reward`` 配置设置。
   * - 遥操作
     - 环境配置中的 ``teleop`` 选择设备，wrapper 栈负责组装。
   * - 手动标记奖励、手动结束 episode
     - 环境配置里的 ``keyboard_reward_wrapper``。
   * - 相对坐标系、欧拉角转换、夹爪维度裁剪
     - ``real/wrappers/transforms/``，由 wrapper 栈加载。
   * - 各任务共用的阻抗参数
     - ``franka/base.py`` 中的 ``COMPLIANCE_DEFAULTS``，任务只需在注册的 ``DEFAULTS`` 中声明差异项。

新增遥操作设备
--------------

只有任务需要 RLinf 尚未提供的操作者设备时，才继续这一节。新设备属于可供多个任务复用的独立硬件扩展，应实现为 ``rlinf/robotics/parts/teleop/`` 下的一个 class。它需要回答三个问题：如何连接硬件、操作者正在做什么，以及机器人应当如何响应。

.. code-block:: python

   @TeleopDevice.register("pedal")
   class Pedal(TeleopDevice):
       PRODUCES = {"end_effector": ActionKind.GRIPPER}

       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           from example_pedal_sdk import PedalClient

           return PedalClient(port=self._port)

       def _release(self, device) -> None:
           device.close()

       @property
       def observation_features(self):
           return {"pressed": {"shape": (1,), "dtype": "bool"}}

       def get_observation(self):
           return {"pressed": np.asarray([self._device.is_pressed()])}

       def action(self, reading, context):
           pressed = bool(reading["pressed"][0])
           return TeleopAction(
               parts={"end_effector": np.array([-1.0 if pressed else 1.0])},
               driving=pressed,
           )

按照从配置选择到单次采样的顺序理解这个 class。``register("pedal")`` 定义配置名称，``PRODUCES`` 声明设备会填充夹爪动作；如果 env 不具备相应语义的路径，系统会在打开硬件前拒绝该设备。``__init__`` 只保存端口，因为声明与连接可能发生在不同机器；``_open()`` 创建硬件句柄并将其保存为 ``self._device``，``_release(device)`` 在启动回滚、正常关闭或重连时释放同一个句柄。句柄如果持有轮询线程，其关闭流程还必须停止并等待线程退出。

其余方法共同定义一次采样。``observation_features`` 在连接前声明 ``pressed`` 字段，``get_observation()`` 返回符合该 schema 的值；``action(reading, context)`` 再将读数转换为一个 ``TeleopAction``，同时包含夹爪动作和 ``driving`` 状态。将两者放在同一个返回值中，可以避免再次读取设备或保存隐式中间状态。

``register`` 中的名称就是配置里书写的名称。配置到构造参数的转换由 ``from_config`` 完成，其默认实现直接把设备自身的选项传给构造函数，因此上面的例子无需编写这一部分。若要读取更外层的 env 配置，或根据被驱动的机器人调整行为，可以覆盖它：

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       port = options.get("port") or cfg.get("pedal_port")
       if port is None:
           raise ValueError("teleop device 'pedal' requires a port")
       return TeleopEntry(cls(port=port), drives=options.get("drives"))

最后把 ``pedal`` 加入机器人 preset 的 ``TELEOP`` 元组，声明该 env 能够表示这种设备产生的动作。这一步不会重复注册设备，公共 builder 会通过 ``TeleopDevice`` 查找该名称。如果机器人不包含 ``end_effector``，系统会在构建阶段报错。

如果同一套硬件需要第二种映射方式，例如输出关节角而非笛卡尔量，继承已有设备并覆盖 ``action`` 即可，``GelloJoint`` 与 ``Gello`` 就是这样的关系。

新增 wrapper
------------

另一类可选扩展改变的是 env 边界，而不是硬件设备。如果新逻辑作用于 rollout 周边，应新增 wrapper：动作接管放入 ``teleop/``，表示转换放入 ``transforms/``，rollout 起止和评分放入 ``episode/``。遥操作设备仍按上一节实现为独立设备 class，新的键盘模式则继承 ``KeyboardSession``。:doc:`../concepts/realworld_envs` 在完整运行流程中说明了这两类扩展点的位置。
