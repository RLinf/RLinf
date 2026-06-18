.. _tutorial-new-realworld-robot-zh:

集成新的真实世界机器人
=========================

本教程以 **SO101 机械臂** 为具体示例，逐步讲解如何在 RLinf 中添加对新真实世界机器人的支持。学完本教程后，你将掌握适用于*任何*真实世界机器人的通用集成模式——无论它使用 USB 串口、CAN 总线、ROS 还是自定义 SDK。

.. note::

   SO101 是一款开源的、可 3D 打印的 6 自由度机械臂，使用 Feetech STS3215
   电机。它是 LeRobot 生态中的参考低成本机械臂。RLinf **不会**重新实现 SO101
   的硬件驱动，而是封装 LeRobot 现有的 Python API（``SO101Follower``、
   ``SO101Leader``），因此你只需要编写 RLinf 的集成层代码。

前置条件
---------

* **已组装的 SO101 机械臂**：请按照 `LeRobot SO101 组装指南
  <https://huggingface.co/docs/lerobot/so101>`_ 完成组装。在使用 RLinf 之前，
  请使用 LeRobot 的校准工具（``lerobot-calibrate``）对机械臂进行校准。

* **RLinf 已安装**：参见 :doc:`../../start/installation`。
  ``so101`` 扩展会安装 LeRobot 和相机依赖：

  .. code-block:: bash

     bash requirements/install.sh embodied --env so101

* **熟悉 RLinf 基本概念**：配置、环境、集群、硬件；建议先浏览
  :doc:`../usage/placement` 和 :doc:`../configuration/hetero`。

理解集成架构
--------------

RLinf 的真实世界机器人栈是一个 **5 层架构**。添加新机器人时，你需要为每一层
创建一个组件。SO101 示例展示了最简单的情况：没有 ROS、没有 CAN 总线 SDK、
没有分布式控制器 Worker——只需要一个由 LeRobot 完全处理的 USB 串口连接。

.. list-table:: 5 个集成层
   :header-rows: 1
   :widths: 10 30 60

   * - 层级
     - 需要创建的内容
     - SO101 示例
   * - **1. 机器人状态**
     - 保存机器人硬件快照的冻结数据类
     - :class:`~rlinf.envs.realworld.so101.SO101RobotState`（关节角度、夹爪状态）
   * - **2. 内部环境**
     - 封装硬件 SDK 的 ``gym.Env`` 子类
     - :class:`~rlinf.envs.realworld.so101.SO101Env`（封装 LeRobot 的 ``SO101Follower``）
   * - **3. 任务 + 注册**
     - 特定任务的 env 子类，注册到 Gymnasium
     - :class:`~rlinf.envs.realworld.so101.tasks.SO101PickEnv`，注册为 ``SO101PickEnv-v1``
   * - **4. 硬件注册**
     - 配置类 + 信息类 + 调度器硬件枚举
     - :class:`~rlinf.scheduler.SO101Config`、
       :class:`~rlinf.scheduler.SO101HWInfo`、
       :class:`~rlinf.scheduler.hardware.robots.so101.SO101Robot`
   * - **5. 连接 + 配置文件**
     - ``__init__.py`` 导出和 YAML 配置文件
     - ``realworld/__init__.py``、``hardware/robots/__init__.py``、
       ``realworld_so101*.yaml``

.. tip::

   **核心原则：利用现有库。** RLinf 的理念是封装而非重写。SO101
   环境从 LeRobot 导入 ``SO101Follower``；Franka 环境通过 ROS 导入
   ``libfranka``；GimArm 环境与 CAN 总线 SDK 通信。你只需要编写 RLinf
   的集成代码——硬件驱动留在上游库中。

第 1 步：定义机器人状态
-------------------------

每个真实世界机器人都需要一个状态数据类来记录每个时间步硬件返回的信息。
这是硬件与 RLinf 其余部分之间的**数据契约**。

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/so101_robot_state.py

   from dataclasses import asdict, dataclass, field
   import numpy as np

   @dataclass
   class SO101RobotState:
       """SO101 6 自由度机械臂的状态快照。

       SO101 包含 5 个旋转关节（shoulder_pan、shoulder_lift、elbow_flex、
       wrist_flex、wrist_roll）以及一个 1 自由度的夹爪。

       关节角度以**度**为单位（LeRobot 约定）。
       """

       joint_position: np.ndarray = field(
           default_factory=lambda: np.zeros(5)
       )
       """机械臂关节角度（度），形状 ``(5,)``。"""

       joint_velocity: np.ndarray = field(
           default_factory=lambda: np.zeros(5)
       )
       """机械臂关节速度（度/秒），形状 ``(5,)``。"""

       gripper_position: float = 0.0
       """夹爪位置（度）。"""

       gripper_open: bool = False
       """夹爪是否打开。"""

       is_connected: bool = False
       """电机总线是否处于连接状态。"""

       def to_dict(self):
           return asdict(self)


.. admonition:: 通用模式
   :class: seealso

   对于**任何**机器人，定义一个包含机器人所提供传感器读数的数据类。
   常见字段：``tcp_pose``（末端执行器位姿）、``joint_position``、
   ``gripper_position``、``tcp_force``。
   参见 :class:`~rlinf.envs.realworld.gim_arm.GimArmRobotState` 或
   :class:`~rlinf.envs.realworld.franka.FrankaRobotState` 了解包含正运动学
   计算的 TCP 位姿和雅可比矩阵的更复杂示例。

第 2 步：实现内部环境
-----------------------

内部环境是一个封装机器人硬件 SDK 的 ``gym.Env`` 子类。
其 ``__init__`` 签名必须接受：

- ``config`` —— 类型化配置数据类
- ``worker_info`` —— 调度器提供的 ``WorkerInfo``（可能为 ``None``）
- ``hardware_info`` —— 调度器提供的硬件信息（可能为 ``None``）
- ``env_idx`` —— 此环境在 worker 进程中的索引

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/so101_env.py（关键部分）

   from lerobot.robots.so_follower import SO101Follower
   from lerobot.robots.so_follower.config_so_follower import (
       SO101FollowerConfig,
   )

   class SO101Env(gym.Env):
       """基于 LeRobot 的 SO101 关节空间环境。"""

       def __init__(self, config, worker_info, hardware_info, env_idx):
           self.config = config
           self._state = SO101RobotState()
           self._robot = None

           if not self.config.is_dummy:
               self._setup_hardware()

           self._init_action_obs_spaces()

           if not self.config.is_dummy:
               self._connect_robot()

       def _connect_robot(self):
           """通过 LeRobot 连接——不需要分布式 Worker。"""
           robot_cfg = SO101FollowerConfig(
               port=self.config.port,
               id=self.config.calibration_id,
               use_degrees=self.config.use_degrees,
           )
           self._robot = SO101Follower(robot_cfg)
           self._robot.connect(calibrate=True)

       def step(self, action):
           """action: [q1..q5, gripper]（度）。"""
           # LeRobot 的 SOFollower.send_action() 会通过
           # ``key.endswith(".pos")`` 过滤动作字典的键，未带后缀的键会被
           # 静默丢弃，因此每个电机键（包括夹爪）都必须采用
           # ``<motor>.pos`` 形式。
           robot_action = {
               "shoulder_pan.pos":   float(action[0]),
               "shoulder_lift.pos":  float(action[1]),
               "elbow_flex.pos":     float(action[2]),
               "wrist_flex.pos":     float(action[3]),
               "wrist_roll.pos":     float(action[4]),
               "gripper.pos":        float(action[5]),
           }
           self._robot.send_action(robot_action)
           self._update_state()
           obs = self._get_observation()
           reward = self._calc_step_reward(obs)
           return obs, reward, terminated, truncated, {}

       def _init_action_obs_spaces(self):
           # 动作：[q1..q5, gripper]（度，6 维）
           self.action_space = gym.spaces.Box(
               low=np.append(self._joint_limit_low, self.config.gripper_limit_low),
               high=np.append(self._joint_limit_high, self.config.gripper_limit_high),
           )
           # 观测：5 维机械臂状态 + 1 维夹爪（+ 可选相机图像）
           self.observation_space = gym.spaces.Dict({
               "state": gym.spaces.Dict({
                   "joint_position": Box(-inf, inf, (5,)),
                   "gripper_position": Box(-inf, inf, (1,)),
               }),
           })

.. important::

   **始终支持 dummy 模式。** 设置 ``config.is_dummy = True`` 可跳过所有硬件调用。
   这允许用户在没有物理机器人的情况下进行离线训练、验证配置和运行 CI。
   在 dummy 模式下，``step()`` 返回来自观测空间的随机观测。

.. admonition:: SO101 的设计决策

   **为什么没有控制器 Worker？** LeRobot 的 ``SO101Follower`` 提供了一个同步
   Python API——你调用 ``send_action()``，它会立即写入串口。更复杂的机器人
   （Franka + ROS、GimArm + CAN）需要分布式 :class:`~rlinf.scheduler.Worker`
   子类，因为它们的 SDK 是有状态的、运行在独立进程中，或者必须部署在特定节点上。
   参见 :class:`~rlinf.envs.realworld.gim_arm.GimArmController` 了解 Worker 模式。

   **为什么使用角度制？** LeRobot 对 Feetech 电机原生使用角度制。你可以设置
   ``use_degrees: False`` 使用弧度制，并相应地调整关节限位。

第 3 步：定义任务并注册到 Gymnasium
---------------------------------------

创建一个任务——具有特定目标的 env 子类。然后将其注册到 Gymnasium，
以便 RLinf 可以通过 ID 创建它。

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/tasks/so101_pick.py

   class SO101PickEnv(SO101Env):
       """到达目标关节配置。"""

       def __init__(self, override_cfg, worker_info=None,
                    hardware_info=None, env_idx=0):
           config = SO101PickConfig(**override_cfg)
           super().__init__(config, worker_info, hardware_info, env_idx)

   @dataclass
   class SO101PickConfig(SO101RobotConfig):
       target_joint_qpos = np.array([30.0, -60.0, 120.0, 0.0, 30.0, 60.0])
       reward_threshold_deg: float = 8.0

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/tasks/__init__.py

   from gymnasium.envs.registration import register
   from .so101_pick import SO101PickEnv

   register(
       id="SO101PickEnv-v1",
       entry_point="rlinf.envs.realworld.so101.tasks:SO101PickEnv",
   )

.. note::

   入口点可以是一个**类**（如上）或一个**工厂函数**。当你需要在构建时应用
   wrappers 时可以使用工厂函数。参见
   ``rlinf/envs/realworld/franka/tasks/__init__.py`` 中的工厂函数示例。

第 4 步：向调度器注册硬件
----------------------------

调度器需要了解你的机器人，以便将环境 worker 分配到正确的节点。
你需要注册三样东西：**配置**、**信息类**和**硬件类**。

.. code-block:: python
   :caption: rlinf/scheduler/hardware/robots/so101.py

   @dataclass
   class SO101HWInfo(HardwareInfo):
       config: "SO101Config"

   @Hardware.register()
   class SO101Robot(Hardware):
       HW_TYPE = "SO101"

       @classmethod
       def enumerate(cls, node_rank, configs=None):
           robot_configs = RobotAutoConfig.resolve(
               configs, config_cls=SO101Config,
               node_rank=node_rank, count_fields=("port",),
           )
           if robot_configs:
               return HardwareResource(
                   type=cls.HW_TYPE,
                   infos=[SO101HWInfo(type=cls.HW_TYPE, config=c)
                          for c in robot_configs],
               )
           return None

   @NodeHardwareConfig.register_hardware_config("SO101")
   @dataclass
   class SO101Config(HardwareConfig):
       port: str = "/dev/ttyACM0"
       leader_port: Optional[str] = None
       arm_variant: str = "so101"
       calibration_id: str = "default"
       camera_serials: Optional[list[str]] = None
       camera_type: str = "opencv"
       disable_validate: bool = False

.. tip::

   **RobotAutoConfig** 会将未设置的 ``None`` 字段从同名大写环境变量中填充。
   如果 ``SO101Config.port`` 为 ``None`` 且 ``SO101_PORT=/dev/ttyACM0``
   已设置，``RobotAutoConfig.resolve()`` 会自动填充它。这允许用户无需编辑
   YAML 就能配置硬件。

第 5 步：接入包体系
---------------------

在两个 ``__init__.py`` 文件中添加导入和导出条目：

.. code-block:: python
   :caption: rlinf/envs/realworld/__init__.py（添加以下行）

   from .so101 import SO101Env, SO101RobotConfig, SO101RobotState
   from .so101 import tasks as so101_tasks

   # 添加到 __all__：
   "SO101Env", "SO101RobotConfig", "SO101RobotState", "so101_tasks",

.. code-block:: python
   :caption: rlinf/scheduler/hardware/robots/__init__.py（添加以下行）

   from .so101 import SO101Config, SO101HWInfo
   # 添加到 __all__："SO101Config", "SO101HWInfo",

.. code-block:: python
   :caption: rlinf/scheduler/__init__.py（添加以下行）

   from .hardware import SO101HWInfo
   # 将 SO101HWInfo 添加到 __all__

第 6 步：创建配置文件
-----------------------

三个配置文件覆盖主要工作流。全部放在
``examples/embodiment/config/`` 中。

**环境配置**（``config/env/realworld_so101.yaml``）：

.. code-block:: yaml

   env_type: realworld
   group_size: 1
   max_episode_steps: 150
   init_params:
     id: "SO101PickEnv-v1"

**数据采集配置**（``config/realworld_so101_collect_data.yaml``）：

.. code-block:: yaml

   defaults:
     - env/realworld_so101@env.eval
   cluster:
     num_nodes: 1
     node_groups:
       - label: so101
         node_ranks: 0
         hardware:
           type: SO101
           configs:
             - port: /dev/ttyACM0
               leader_port: /dev/ttyACM1
               node_rank: 0
   env:
     eval:
       override_cfg:
         enable_teleop: True
       data_collection:
         enabled: True
         export_format: "pickle"
         robot_type: "so101"

**RL 训练配置**（``config/realworld_so101_rl.yaml``）：

.. code-block:: yaml

   defaults:
     - env/realworld_so101@env.train
     - model/cnn_policy@actor.model
   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-0
       env: 0-0
       rollout: 0-0
   env:
     train:
       override_cfg:
         is_dummy: True   # 真实硬件时设为 False

第 7 步：添加安装支持
------------------------

将你的机器人添加到安装脚本和 ``pyproject.toml`` 中：

.. code-block:: bash
   :caption: requirements/install.sh

   # 在 SUPPORTED_ENVS=() 中：添加 "so101"
   # 在 install_env_only case) 中：添加
   so101)
       uv sync --extra so101 --active $NO_INSTALL_RLINF_CMD
       ;;

.. code-block:: toml
   :caption: pyproject.toml

   [project.optional-dependencies]
   so101 = [
       "lerobot @ git+https://github.com/huggingface/lerobot.git@...",
       "gymnasium",
       "opencv-python",
       "numpy",
       "imageio[ffmpeg]",
   ]

   [tool.uv.conflicts]
   # 添加到列表中：
   { extra = "so101" },


在 RLinf 中使用 SO101 机械臂
==============================

数据采集（遥操作）
--------------------

SO101 数据采集使用**双边遥操作**：人类物理移动 leader 臂，follower 臂镜像跟随。
两个臂都通过 USB 串口连接。

.. code-block:: bash

   python examples/embodiment/collect_real_data.py \
       --config-name realworld_so101_collect_data

环境创建一个 ``SO101Leader`` 实例，读取 leader 臂的关节角度并转发给 follower。
采集的 episode 以 LeRobot 格式保存在配置的 ``save_dir`` 中。

SFT 训练（行为克隆）
-----------------------

采集演示数据后，通过监督微调训练策略：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_so101_sft \
       algorithm.demo_buffer.dataset_path=/path/to/collected_data

SFT 配置使用 ``is_dummy: True``（训练是离线的），并设置
``adv_type: embodied_sft`` / ``loss_type: embodied_sft``。

RL 训练
----------

对于在线 RL，使用真实或 dummy 机器人的 RL 配置：

.. code-block:: bash

   # Dummy 验证（无需硬件）
   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_so101_rl

   # 真实硬件（用环境变量进行自动配置）
   export SO101_PORT=/dev/ttyACM0
   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_so101_rl \
       env.train.override_cfg.is_dummy=False \
       env.eval.override_cfg.is_dummy=False \
       cluster.node_groups.0.hardware.type=SO101

RL 算法（默认 SAC）在机器人上端到端地训练 CNN 策略。

人机协同（Human-in-the-Loop）
-------------------------------

RLinf 支持在 RL 训练过程中进行人工干预。SO101 环境可以在运行时
在策略控制和人工遥操作之间切换：

.. code-block:: python

   # 在 env step() 中：
   if self.config.enable_teleop and self._leader is not None:
       # 人工遥操作覆盖策略动作
       leader_action = self._leader.get_action()
       robot_action = leader_action
   else:
       # 策略动作
       self._robot.send_action(robot_action)

这可以通过在 env 配置中设置 ``enable_teleop: True`` 或键盘监听器
（参见 ``rlinf/envs/realworld/common/keyboard/``）来触发。
干预事件会被记录用于从人类反馈中学习。

使用 Dummy 模式进行测试
--------------------------

在部署到硬件之前，始终使用 dummy 模式验证你的集成：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_so101_rl \
       max_epochs=1 max_steps=10

这会在没有机器人的情况下运行一个 epoch 的训练（使用随机观测）。
如果通过，说明你的集成在结构上是正确的。

常见问题排查
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 问题
     - 解决方案
   * - ``ImportError: cannot import name 'SO101Follower'``
     - 安装 ``so101`` 扩展：``uv sync --extra so101``
   * - ``串口 '/dev/ttyACM0' 未找到``
     - 检查 ``ls /dev/tty.*``。在 Linux 上，确保用户属于
       ``dialout`` 组。在 macOS 上，使用 ``/dev/tty.usbmodem*``。
   * - 执行 ``send_action`` 后机器人不动
     - 先用 ``lerobot-calibrate`` 校准机械臂。检查电机扭矩是否启用。
   * - ``gym.error.UnregisteredEnv``
     - 必须导入 tasks ``__init__.py``。确认
       ``so101_tasks`` 在 ``realworld/__init__.py`` 中列出。
   * - 策略输出 NaN 或零动作
     - 检查模型配置中的 ``state_dim`` 和 ``action_dim``。
       SO101：``state_dim: 6``（5 个机械臂关节 + 1 个夹爪），
       ``action_dim: 6``（5 个机械臂目标 + 1 个夹爪目标）。


添加你自己的机器人：快速清单
=================================

集成新机器人时使用此清单：

.. list-table::
   :header-rows: 1
   :widths: 5 45 50

   * - #
     - 创建/修改的文件
     - 提供的内容
   * - 1
     - ``realworld/<robot>/*_robot_state.py``
     - 机器人状态数据类
   * - 2
     - ``realworld/<robot>/*_env.py``
     - 封装硬件 SDK 的内部 ``gym.Env``
   * - 3
     - ``realworld/<robot>/tasks/*.py`` + ``__init__.py``
     - 任务 env 子类 + ``gym.register()``
   * - 4
     - ``realworld/<robot>/__init__.py``
     - 包导出
   * - 5
     - ``scheduler/hardware/robots/<robot>.py``
     - ``Config``、``HWInfo``、``Hardware``（带 ``HW_TYPE``）
   * - 6
     - ``scheduler/hardware/robots/__init__.py``
     - 重新导出配置/信息类
   * - 7
     - ``scheduler/__init__.py``
     - 重新导出 ``HWInfo``
   * - 8
     - ``envs/realworld/__init__.py``
     - 导入和 ``__all__`` 条目
   * - 9
     - ``examples/embodiment/config/env/realworld_<robot>.yaml``
     - 环境默认配置
   * - 10
     - ``examples/embodiment/config/realworld_<robot>_*.yaml``
     - 实验配置（采集、SFT、RL）
   * - 11
     - ``requirements/install.sh`` + ``pyproject.toml``
     - 安装支持
   * - 12
     - ``docker/Dockerfile``
     - ``embodied-<robot>-image`` 构建阶段，调用安装脚本
   * - 13
     - ``tests/e2e_tests/embodied/<robot>_dummy_*.yaml``
     - dummy 模式 CI 冒烟测试，端到端执行 env

无需修改 ``SupportedEnvType``、``get_env_cls`` 或 ``action_utils``
（真实世界机器人全部复用 ``SupportedEnvType.REALWORLD``）。

对于需要分布式控制器的更复杂机器人（例如 Franka + ROS、GimArm + CAN），
参见 :class:`~rlinf.envs.realworld.gim_arm.GimArmController`
了解 ``Worker`` 子类模式。
