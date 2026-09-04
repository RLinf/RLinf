添加新的机械臂
================

将新的真实世界机械臂集成到 RLinf 中。RLinf 支持多种机械臂的真实世界强化学习，
包括 Franka、GIM_Arm、XSquare Turtle2 和 DOS-W1。本指南将引导您完成添加自己
机械臂的步骤。

RLinf 真实世界机器人系统由以下组件组成：

- **硬件接口**：一个硬件类，负责注册和控制物理机器人（关节角度、夹爪、相机）。
- **机器人环境**：一个 ``gym.Env`` 子类，封装硬件并提供观测/动作空间、reset 和
  step 逻辑。
- **任务环境**：机器人环境的子类，定义特定任务（奖励函数、成功标准、任务描述）。
- **注册**：在包中暴露新环境，使其可以通过配置文件选择。

.. note::

   机器人相关代码目前正在重构中。本指南反映了 ``rlinf/envs/realworld/`` 下的
   当前架构。如需最新且经过最全面测试的参考，请使用 Franka 实现作为模板。

前置条件
--------

开始之前，请确保您已具备：

- 已安装 RLinf 并配置好真实世界依赖。
- 已安装机器人的 Python SDK 或通信库。
- 了解机器人的关节配置、动作空间和观测输出。

步骤 1：学习 Franka 参考实现
------------------------------

``rlinf/envs/realworld/franka/`` 下的 Franka 实现是经过最全面测试的参考。
首先阅读其目录结构：

.. code-block:: text

   rlinf/envs/realworld/
   ├── common/                  # 通用工具（控制器、相机等）
   ├── franka/                  # Franka 参考实现
   │   ├── hardware.py          # 硬件接口类
   │   ├── franka_env.py        # 基础机器人环境
   │   └── tasks/               # 特定任务环境
   ├── gim_arm/                 # GIM Arm 实现
   ├── xsquare/                 # XSquare Turtle2 实现
   ├── dosw1/                   # DOS-W1 实现
   ├── realworld_env.py         # 真实世界环境基类
   └── __init__.py              # 环境注册

复制 ``franka/`` 目录的结构作为您机器人的起始模板。

步骤 2：定义硬件接口
--------------------

在 ``rlinf/envs/realworld/`` 下为您的机器人创建新目录（例如 ``your_robot/``）。
在其中创建一个硬件类，负责与物理机器人通信。

硬件类负责：

- 初始化机器人连接并回零。
- 读取关节状态、末端执行器位姿和相机图像。
- 向机器人发送关节或笛卡尔命令。
- 控制夹爪（张开/闭合）。
- 干净地关闭连接。

.. code-block:: python

   # rlinf/envs/realworld/your_robot/hardware.py
   class YourRobotHardware:
       """YourRobot 机械臂的硬件接口。"""

       def __init__(self, config):
           self.config = config
           # 初始化机器人 SDK 连接
           # self.robot = YourRobotSDK.connect(...)

       def initialize(self):
           """机器人回零并准备控制。"""
           pass

       def get_observation(self):
           """返回关节状态、末端位姿和相机图像。"""
           pass

       def send_action(self, action):
           """向机器人发送关节或笛卡尔命令。"""
           pass

       def control_gripper(self, width):
           """张开或闭合夹爪。"""
           pass

       def shutdown(self):
           """干净地断开机器人连接。"""
           pass

完整接口请参考 ``rlinf/envs/realworld/franka/hardware.py``。

步骤 3：实现机器人环境
----------------------

为您的机器人创建基础环境类，继承自真实世界环境基类（参见
``rlinf/envs/realworld/realworld_env.py``）。

机器人环境必须定义：

- **观测空间**：关节位置、速度、末端执行器位姿、相机图像（RGB/D）以及任何
  机器人特定传感器。
- **动作空间**：关节位置增量、笛卡尔位置增量或夹爪命令，取决于您的控制模式。
- ``reset()``：机器人回零、重置任务状态并返回初始观测。
- ``step(action)``：向硬件发送动作、读取新观测、计算奖励（委托给任务子类）、
  检查终止条件。

.. code-block:: python

   # rlinf/envs/realworld/your_robot/your_robot_env.py
   import gymnasium as gym
   import numpy as np

   class YourRobotEnv(gym.Env):
       """YourRobot 机械臂的基础环境。"""

       def __init__(self, cfg, rank=0, num_envs=1, ret_device="cpu"):
           self.cfg = cfg
           self.rank = rank
           self.num_envs = num_envs
           self.ret_device = ret_device

           # 初始化硬件
           from .hardware import YourRobotHardware
           self.hardware = YourRobotHardware(cfg.hardware)
           self.hardware.initialize()

           # 定义空间
           self._setup_spaces()

       def _setup_spaces(self):
           """定义观测和动作空间。"""
           # 示例：关节位置 + 速度 + 末端位姿
           obs_dim = self.cfg.obs_dim
           self.observation_space = gym.spaces.Box(
               low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
           )
           # 示例：关节增量动作 + 夹爪
           act_dim = self.cfg.act_dim
           self.action_space = gym.spaces.Box(
               low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
           )

       def reset(self, options=None):
           """重置机器人并返回初始观测。"""
           self.hardware.initialize()
           obs = self.hardware.get_observation()
           return obs, {}

       def step(self, action):
           """执行动作并返回 (obs, reward, terminated, truncated, info)。"""
           self.hardware.send_action(action)
           obs = self.hardware.get_observation()
           # 奖励和终止由任务子类定义
           reward = self._compute_reward(obs, action)
           terminated = self._check_termination(obs)
           truncated = self._check_truncation()
           info = {}
           return obs, reward, terminated, truncated, info

       def _compute_reward(self, obs, action):
           """在任务子类中重写。"""
           raise NotImplementedError

       def _check_termination(self, obs):
           """在任务子类中重写。"""
           return False

       def _check_truncation(self):
           """检查回合长度限制。"""
           return self._step_count >= self.cfg.max_episode_steps

生产环境中使用的完整实现模式请参考
``rlinf/envs/realworld/franka/franka_env.py``。

步骤 4：实现任务环境
--------------------

创建继承自基础机器人环境的特定任务环境。每个任务定义：

- **奖励函数**：基于任务进度的密集或稀疏奖励。
- **成功标准**：任务是否完成。
- **任务描述**：用于 VLA 模型的自然语言字符串。
- **重置逻辑**：任务特定的初始状态设置。

.. code-block:: python

   # rlinf/envs/realworld/your_robot/tasks/pick_and_place.py
   from ..your_robot_env import YourRobotEnv

   class YourRobotPickAndPlace(YourRobotEnv):
       """YourRobot 的抓取放置任务。"""

       def _compute_reward(self, obs, action):
           """计算任务奖励。"""
           # 示例：到达奖励 + 抓取奖励 + 放置奖励
           return 0.0

       def _check_termination(self, obs):
           """检查任务是否完成或失败。"""
           return False

       def get_task_description(self):
           """返回用于 VLA 模型的自然语言任务描述。"""
           return "把方块抓起来放到箱子里"

将任务文件组织在 ``rlinf/envs/realworld/your_robot/tasks/`` 下。

步骤 5：注册环境
----------------

在包的 ``__init__.py`` 中暴露新的机器人环境，使其可以被配置系统发现。

.. code-block:: python

   # rlinf/envs/realworld/__init__.py
   from .your_robot.your_robot_env import YourRobotEnv
   from .your_robot.tasks.pick_and_place import YourRobotPickAndPlace

   __all__ = [
       "YourRobotEnv",
       "YourRobotPickAndPlace",
       # ... 现有条目
   ]

步骤 6：创建配置文件
--------------------

在相应的 ``configs/`` 目录下为您的机器人任务添加 YAML 配置文件。
使用现有的 Franka 配置作为模板，并修改机器人特定字段。

.. code-block:: yaml

   # configs/realworld/your_robot_pick_and_place.yaml
   env:
     env_type: "realworld"
     robot_type: "your_robot"
     task: "pick_and_place"
     total_num_envs: 1
     max_episode_steps: 200
     hardware:
       ip: "192.168.1.100"
       # 机器人特定硬件配置

步骤 7：测试和验证
------------------

提交 PR 之前，请验证您的实现：

1. **硬件冒烟测试**：验证机器人在没有 RL 循环的情况下可以连接、回零并响应命令。
2. **环境冒烟测试**：运行几个随机动作回合，验证 ``reset()`` 和 ``step()`` 正常工作。
3. **观测/动​​作验证**：确认观测和动作空间与实际硬件输出和输入匹配。
4. **安全检查**：确保急停、速度限制和工作空间边界已生效。

.. code-block:: python

   def test_your_robot_env():
       """YourRobot 环境的基础冒烟测试。"""
       cfg = get_test_config()
       env = YourRobotPickAndPlace(cfg)
       obs, info = env.reset()
       assert obs is not None
       for _ in range(10):
           action = env.action_space.sample()
           obs, reward, terminated, truncated, info = env.step(action)
           assert obs is not None
           if terminated or truncated:
               obs, info = env.reset()
       env.hardware.shutdown()

参考实现
--------

如需可运行的代码，请参考以下现有实现：

- **Franka**（最全面）：``rlinf/envs/realworld/franka/``
- **GIM Arm**：``rlinf/envs/realworld/gim_arm/``
- **XSquare Turtle2**：``rlinf/envs/realworld/xsquare/``
- **DOS-W1**：``rlinf/envs/realworld/dosw1/``

常见问题
--------

**机器人不响应命令**
   检查硬件 SDK 连接、IP 地址和控制模式（位置 vs. 速度 vs. 力矩）。

**观测维度不匹配**
   验证您的 ``observation_space`` 定义与硬件返回的实际观测向量匹配。

**夹爪控制不工作**
   有些机器人需要单独的夹爪 SDK 或串口连接。检查硬件类中的夹爪特定初始化。

**相机图像不可用**
   确保已安装相机驱动程序，并且相机已在硬件类中注册。共享相机工具请参考
   ``rlinf/envs/realworld/common/``。

贡献回社区
----------

如果您的机器人实现足够通用，对他人有用，我们欢迎提交 PR 将其添加到 RLinf！
请包括：

- 硬件接口和环境代码。
- 至少一个示例任务。
- 配置文件。
- 文档（本指南是一个很好的起点）。
- 基础冒烟测试（如果 CI 中有可用硬件）。

请先开一个 issue 讨论机器人型号，确保没有正在进行的重叠工作。
