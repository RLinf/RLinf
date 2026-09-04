新增真机任务
============

本页说明在 RLinf 已支持目标真机的前提下，如何新增真机任务，而无需修改机器人实现。新任务可以定义特有的目标、奖励和复位方式；完成后将得到任务配置、env 类、Gymnasium ID 和对应的 YAML 配置，后续步骤会依次定义并验证这些内容。

本页所述真机任务专指 ``rlinf/envs/real`` 下的任务模块。如需为模拟器或 benchmark 新增 task，请参阅 :doc:`new_env`。

核心任务流程不修改机器人构造、设备 placement 或遥操作。目标、compliance 参数、成功条件和复位行为属于任务，现有机器人与 wrapper stack 保持不变。如果硬件本身尚未接入，请先按照 :doc:`new_robot` 实现零部件，并确认观测和动作能够正常传递，再添加任务。如果还需要 RLinf 尚未提供的操作者设备或 wrapper，应先完成任务主流程，再将其作为文末所述的独立扩展处理。

核心流程包含五步：定义任务数据，将其绑定到 env class，注册稳定的 Gymnasium ID，为一次运行添加配置，并验证注册结果。后续章节说明哪些能力可直接复用，并介绍新增操作者设备和新增 wrapper 两类可选扩展；大多数任务无需执行这两部分。

核心流程
--------

以下示例为 Franka 添加 ``WipeEnv-v1``。每一步都会产生下一步的输入：dataclass 配置 env，env class 注册为 Gymnasium ID，YAML 选择该 ID，最后的检查则在硬件打开前确认整条解析路径可用。

如果新任务使用关节空间机械臂，实施顺序不变，只需继承对应的 env 基类。``SO101ReachEnv-v1`` 和 ``examples/embodiment/config/env/so101_reach.yaml`` 展示了当前 SO-101 的实现：动作包含五个绝对关节目标和一个连续夹爪值。机器人中的夹爪路径仍为 ``arm.end_effector``，env 负责将这套嵌套接口转换为 policy 使用的六维动作。

1. 定义任务配置
~~~~~~~~~~~~~~~

在 ``rlinf/envs/real/<robot>/<task>.py`` 中新建模块，并继承该机器人的配置 dataclass。配置类只需添加擦拭任务所需的字段：

.. code-block:: python

   import copy
   from dataclasses import dataclass, field

   import numpy as np

   from rlinf.robotics.actions import ActionKind, ActionPart

   from .base import FrankaEnv, FrankaRobotConfig, compliance


   @dataclass
   class WipeConfig(FrankaRobotConfig):
       task_description: str = "wipe the surface"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       reward_threshold: np.ndarray = field(
           default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
       )
       random_xy_range: float = 0.03

       def __post_init__(self):
           self.compliance_param = compliance(
               translational_stiffness=800,   # 降低刚度以保持接触
               translational_clip_z=0.02,
           )
           self.target_ee_pose = np.array(self.target_ee_pose)
           self.action_scale = np.array([0.02, 0.1, 1])

这些字段分别回答任务执行中的不同问题：``task_description`` 提供语言指令，``target_ee_pose`` 定义目标，``reward_threshold`` 判断各项位姿误差是否足够小，``random_xy_range`` 控制复位时的随机范围；``action_scale`` 限制一次 policy 动作的移动幅度，``compliance_param`` 则配置执行动作时使用的控制器。

阻抗参数只需声明与默认值不同的部分。``compliance()`` 会将差异项合并到 ``COMPLIANCE_DEFAULTS``；字段名错误或控制器不支持相应参数时，任务配置在构建阶段就会报错，不会将无效参数继续传给阻抗控制器。

2. 定义 env 类
~~~~~~~~~~~~~~

配置类已经包含任务所需的数据，env class 接下来将这些数据接入现有机器人的执行流程。首先指定前一步定义的配置类型；对于多数任务，这就是全部实现：

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

``CONFIG_CLS`` 告诉继承的构造流程应使用哪个 dataclass 解析 ``override_cfg``。仅当任务的运行行为确有差异时才覆盖相应 hook。最常见的是 ``go_to_rest``：插销任务在返回初始位姿前需要先抬高末端，避免插销卡在插孔中。

.. code-block:: python

       def go_to_rest(self, joint_reset=False):
           reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
           reset_pose[2] += 0.05
           self._interpolate_move(reset_pose, timeout=1)
           super().go_to_rest(joint_reset)

如果任务沿用原有动作空间，则无需实现 ``action_parts``。如果任务修改动作空间，则必须声明每段动作对应的零部件及其语义；遥操作根据这些语义匹配设备，而不能只根据维度判断。

.. code-block:: python

       def action_parts(self):
           return (
               ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
               ActionPart("end_effector", 1, ActionKind.GRIPPER),
           )

各段宽度之和必须与动作空间维度一致，否则系统会在构建阶段报错。

3. 注册任务
~~~~~~~~~~~~

env class 可以执行任务后，还需要一个供配置和数据集长期引用的稳定 ID。在 ``rlinf/envs/real/<robot>/__init__.py`` 的 ``TASKS`` mapping 中加入该 class：

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": WipeEnv,
   }

``register_tasks`` 根据该映射生成 entry point，并将其注册到 Gymnasium。wrapper 由 env 自行声明，无需在此重复配置。Gym ID 会写入用户配置和数据集元数据，因此数据采集开始后不应再修改 ID。

4. 添加环境配置
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

``env_type: real`` 选择 RLinf 的真机 env adapter，``init_params.id`` 选择上一步注册的 Gymnasium 任务，``teleop`` 指定评估或数据采集使用的操作者设备。``override_cfg`` 会传给 ``WipeConfig``，其中每个 key 都应对应任务配置字段；机器人地址和 placement 仍应写在集群硬件配置中。

5. 验证注册结果
~~~~~~~~~~~~~~~

此时，从 YAML 到任务 class 的核心路径已经完整。连接硬件前，先导入真机 env package，并确认 ID 可以解析：

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # 触发全部任务注册
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` 会检查所有内置任务，请将新 ID 加入 ``EXPECTED_IDS``。这项断言只验证注册；如果任务改变了面向机器人的观测或动作路径，还需按照 :doc:`new_robot` 运行 mock 和真机检查。

复用现有基础设施
----------------

符合现有机器人与 wrapper contract 的任务完成以上五步即可。下列职责已经由相应层次负责，任务代码应直接调用或配置这些能力，不应再次实现：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 需求
     - 现有实现
   * - 连接硬件、部署零部件
     - ``Robot.connect``，见 :doc:`../concepts/robotics`。
   * - 遥操作
     - 环境配置中的 ``teleop`` 选择设备，wrapper 栈负责组装。
   * - 手动标记奖励、手动结束 episode
     - 环境配置里的 ``keyboard_reward_wrapper``。
   * - 相对坐标系、欧拉角转换、夹爪维度裁剪
     - ``real/wrappers/transforms/``，由 wrapper 栈加载。
   * - 各任务共用的阻抗参数
     - ``COMPLIANCE_DEFAULTS``，任务只需声明差异项。

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

最后把 ``pedal`` 加入对应 env 的 ``TELEOP`` 元组，声明该 env 能够表示这种设备产生的动作。这一步不会重复注册设备，公共 builder 会通过 ``TeleopDevice`` 查找该名称。如果机器人不包含 ``end_effector``，系统会在构建阶段报错。

如果同一套硬件需要第二种映射方式，例如输出关节角而非笛卡尔量，继承已有设备并覆盖 ``action`` 即可，``GelloJoint`` 与 ``Gello`` 就是这样的关系。

新增 wrapper
------------

另一类可选扩展改变的是 env 边界，而不是硬件设备。如果新逻辑作用于 rollout 周边，应新增 wrapper：动作接管放入 ``teleop/``，表示转换放入 ``transforms/``，rollout 起止和评分放入 ``episode/``。遥操作设备仍按上一节实现为独立设备 class，新的键盘模式则继承 ``KeyboardSession``。:doc:`../concepts/realworld_envs` 在完整运行流程中说明了这两类扩展点的位置。
