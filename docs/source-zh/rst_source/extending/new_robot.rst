添加机器人
==========

接入新硬件时，建议先在当前进程中验证单个设备，再组合机器人，最后配置远程部署。按照该顺序实施，可以将设备或 SDK 问题与 Ray 部署问题分开排查。

开始前，请先阅读 :doc:`机器人模型 <../concepts/robotics>`。如果 RLinf 已支持该机器人，而变更仅涉及奖励、复位流程或成功条件，请参阅 :doc:`new_task`。

1. 实现本地部件
----------------

首先选择一个可以独立调试的最小设备。传感器继承 ``RobotPart``；需要接收控制命令的设备继承 ``ControllablePart``。此阶段暂不添加集群配置，只验证连接、读取和断开流程。

部件需要实现三个关键操作：``_open`` 打开设备，``get_observation`` 读取数据，``_release`` 释放资源。厂商 SDK 只在 ``_open`` 中导入，使连接硬件的节点按需加载 SDK，同时允许其他节点在未安装 SDK 的情况下导入部件模块。

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint

       def _open(self):
           from example_robot_sdk import Client

           return Client(self.endpoint)

       def _release(self, device) -> None:
           device.close()

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       def reset(self) -> None:
           self._device.move_home()

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": self._device.get_joint_position()}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"joint_position"}:
               raise KeyError("Expected only 'joint_position'.")
           self._device.move_joints(action["joint_position"])
           return action

组合机器人前，先直接运行并验证该部件：

.. code-block:: python

   arm = ExampleArm("tcp://left-arm:5000")
   arm.connect()
   try:
       print(arm.get_observation())
   finally:
       arm.disconnect()

``_open()`` 的返回值保存在 ``self._device``。断开时，同一对象会传给 ``_release(device)``；清理逻辑必须直接使用参数 ``device``。

不应在 ``__init__`` 中打开设备，否则无法在一台机器上声明部件、再由另一台机器创建。机械臂如果在连接后还需上电或回零，可以覆盖 ``connect()`` 和 ``disconnect()``；两个方法都必须支持重复调用，以便在启动失败后回滚并重新连接。

相机、末端执行器、移动底盘和足式底盘分别继承 ``Camera``、``EndEffector``、``MobileBase`` 和 ``LeggedBase``。远程代理会保留这些类型对应的接口。

2. 共享硬件连接
----------------

本地部件验证通过后，需要确认其硬件连接是否同时控制其他部件。一个 socket、CAN 总线或 ROS 节点可能同时控制机械臂、夹爪和相机，但整条连接只应打开一次。

在这种情况下，应在 endpoint 上实现 ``exports`` mapping。key 是传给 ``connection.export(name)`` 的内部名称，value 是返回给调用方的 ``RobotPart``。该 mapping 只声明连接能够提供的部件，不决定部件在机器人树中的最终位置。机械臂本身统一使用 ``"arm"``：

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       @property
       def exports(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

部分 SDK 只提供 ``open_gripper``、``move_left_arm``、``get_camera(id)`` 等方法。使用 ``MethodGripper``、``MethodArm`` 或 ``MethodCamera`` 将其适配为部件，使后续组合代码不再依赖厂商方法名称。

3. 定义公开部件名称
-------------------

上一步声明了连接能够提供的部件，本步骤定义任务和 policy 使用的公开部件名称。可以先手动组合一个最小机器人，无需立即实现 discovery 或 YAML 配置。共享连接只声明一次，再逐个选择部件并指定公开名称：

.. code-block:: python

   from rlinf.robotics import Robot


   class Bench(Robot):
       ROBOT_TYPE = "Bench"


   connection = ExampleArm.at("tcp://left-arm:5000")
   robot = Bench(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )
   print(robot.describe())
   robot.connect()
   try:
       print(robot.get_observation())
   finally:
       robot.disconnect()

独立调试脚本可以继续使用上述写法。只有需要根据配置和类型名称创建机器人时，才需要实现 discovery。

部件名称属于公开 API，并会写入观测、动作和数据集。数据采集开始后修改名称，会同时改变 policy 输入和已有数据格式。

.. code-block:: python

   from rlinf.robotics import Group, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   left = ExampleArm.at("tcp://left-arm:5000")
   right = ExampleArm.at("tcp://right-arm:5000")
   robot = ExampleRobot(
       left=Group(
           arm=left.export("arm"),
           end_effector=left.export("end_effector"),
       ),
       right=Group(
           arm=right.export("arm"),
           end_effector=right.export("end_effector"),
       ),
   )
   robot.connect()
   try:
       observation = robot.get_observation()
       robot.send_action(
           {
               "left": {"arm": {"joint_position": left_target}},
               "right": {"arm": {"joint_position": right_target}},
           }
       )
   finally:
       robot.disconnect()

访问路径完全由组合时指定的名称决定，系统不会自动插入 ``arms`` 或 ``cameras`` 层级。在上述示例中，左臂路径为 ``left.arm``，左夹爪路径为 ``left.end_effector``；名为 ``wrist`` 的相机路径即为 ``wrist``。

系统会并行读取和控制使用独立连接的部件；共用连接的部件则按声明顺序调用。机器人子类无需自行管理线程。

4. 配置远程部署
----------------

确认本地连接、读取和断开流程正常后，在原有声明中加入 ``node_rank``。部件类和机器人树均无需修改：

.. code-block:: python

   connection = ExampleArm.at("tcp://left-arm:5000", node_rank=0)
   robot = Robot(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )
   robot.connect()
   try:
       print(robot.get_observation())
   finally:
       robot.disconnect()

``at()`` 只记录 ``ExampleArm`` 将部署在节点 0。调用 ``connect()`` 时，RLinf 才会创建 worker、导入 SDK 并打开设备。句柄保存在 ``robot.handles`` 中，并由 ``disconnect()`` 负责回收。

相机也可以采用相同方式部署到实际连接设备的机器上::

   scene=RealSenseCamera.at(info, node_rank=2)

当一条连接包含多个部件时，只声明一次连接，再通过 ``export()`` 选择所需部件::

   connection = ExampleConnection.at(node_rank=0)
   Group(arm=connection.export("left"), gripper=connection.export("left_end_effector"))

``spawn()`` 会立即创建远程资源，仅适用于独立调试脚本；此类脚本必须自行关闭句柄。

无需为每个设备单独实现 worker。RLinf 会根据部件类生成 worker，并将公有方法绑定为 RPC。设备专有方法通过句柄调用，本地与远端使用相同的调用方式::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

5. 通过配置构建机器人
---------------------

需要从 YAML 创建机器人时，应实现 ``RobotConfig`` 和 ``build()``。连接地址、``node_rank`` 等部署信息写入机器人配置；复位姿态、奖励和 episode 长度仍属于任务配置。

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Group, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"

       @classmethod
       def build(
           cls,
           *,
           left_endpoint: str,
           right_endpoint: str,
           node_rank: int = 0,
           **_,
       ) -> "ExampleRobot":
           arms = {}
           for side, endpoint in (
               ("left", left_endpoint),
               ("right", right_endpoint),
           ):
               connection = ExampleArm.at(
                   endpoint,
                   node_rank=node_rank,
                   name=f"ExampleArm-{side}",
               )
               arms[side] = Group(
                   arm=connection.export("arm"),
                   end_effector=connection.export("end_effector"),
               )
           return cls(**arms)

单臂型号只需返回一个 ``Group``。``build()`` 仅负责组合 ``PartSpec``，不应打开硬件。设备统一由 ``Robot.connect()`` 启动；如果启动过程中发生失败，已创建的资源会被回收。

.. warning::

   ``build()`` 不会访问硬件。读取观测或发送命令前必须调用 ``connect()``，结束时调用 ``disconnect()``。真机环境会在初始化和清理阶段调用这两个方法。

注册机器人
----------

在机器人模块末尾完成注册，无需修改中央注册表。

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import RobotDiscovery, RobotInfo
   from rlinf.scheduler.hardware import HardwareConfig, HardwareResource


   class ExampleRobotDiscovery(RobotDiscovery):
       HW_TYPE = ExampleRobot.ROBOT_TYPE

       @classmethod
       def enumerate(
           cls,
           node_rank: int,
           configs: Optional[list[HardwareConfig]] = None,
       ) -> Optional[HardwareResource]:
           matching = [
               config
               for config in configs or []
               if isinstance(config, ExampleRobotConfig)
               and config.node_rank == node_rank
           ]
           if not matching:
               return None
           return HardwareResource(
               type=cls.HW_TYPE,
               infos=[
                   RobotInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=config)
                   for config in matching
               ],
           )


   ExampleRobot.register(ExampleRobotConfig, ExampleRobotDiscovery)

注册调用必须位于配置类和发现类之后。注册完成后，调用方可以直接使用 ``build_robot("ExampleRobot", ...)``，无需导入具体机器人类。

如果新硬件只是现有机器人的变体，可以直接继承对应类。例如，``DualFrankaRobot`` 只修改 ``FrankaRobot`` 的 ``build_arms`` 和 ``BACKEND``，``build`` 与生命周期方法继续使用父类实现。

构造 ``Cluster`` 前必须导入注册模块，否则硬件发现无法识别该机器人。每个节点使用的 Python 环境也必须能够导入该模块。

配置集群
--------

将硬件信息写入 ``cluster.node_groups.hardware``。``endpoint``、``node_rank`` 等部署参数应保留在 YAML 中，不应硬编码到 Python 代码中：

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement: {}
     node_groups:
       - label: example_robot
         node_ranks: 0
         hardware:
           type: ExampleRobot
           configs:
             - node_rank: 0
               left_endpoint: tcp://left-arm:5000
               right_endpoint: tcp://right-arm:5000

保持任务逻辑独立
----------------

机器人负责发现和操作硬件，不定义任务逻辑。reset、奖励、成功条件、截断条件和 Gymnasium space 应写入 ``RobotTask`` 或真机环境，再通过 ``RobotTaskEnv`` 与 ``Robot`` 组合。现有 policy 如果使用扁平动作向量和 ``state``/``frames`` 观测，应在环境边界添加 ``LegacyObservationAdapter`` 和 ``VectorActionAdapter``。

.. warning::

   改造已有机器人时，不要改变现有的 Gym ID、动作维度、观测字段、相机名称和数据集字段。需要转换格式时，请使用适配器，并补上回归测试。

检查组合结果
------------

连接硬件前，使用 ``describe()`` 检查部件路径、部署节点和共享连接::

   >>> print(robot.describe())
   ExampleRobot
   ├── left.arm             declared      node=0     via ExampleArm#1
   ├── left.end_effector    declared      node=0     via ExampleArm#1
   ├── right.arm            declared      node=0     via ExampleArm#2
   └── right.end_effector   declared      node=0     via ExampleArm#2

``via`` 值相同的部件共用一条连接，因此只会打开一次。``describe()`` 读取机器人保存的声明，因此在 ``connect()`` 前后会显示相同的路径、节点和资源归属。

测试集成
--------

在 ``tests/unit_tests/`` 中新增测试，并使用 ``PartContract``、``ConnectionContract`` 和 ``RobotContract`` 检查新实现的部件、连接和机器人。测试可以复用工具脚本中的 mock SDK：

.. code-block:: python

   from robot_contracts import ConnectionContract, PartContract, RobotContract


   def test_my_arm_conforms():
       PartContract(
           lambda: MyArm("10.0.0.1"),
           action={"joint_position": np.zeros(6)},
       ).assert_kept()


   def test_my_link_conforms():
       ConnectionContract(MyConnection).assert_kept()


   def test_my_robot_conforms():
       RobotContract(lambda: MyRobot.build(robot_ip="10.0.0.1")).assert_kept()

这些 contract 位于 ``tests/robot_contracts``，mock SDK 位于 ``tests/robot_mocks``。二者仅用于测试，不会随 ``rlinf`` 包发布。

contract 会检查重复连接和断开、观测字段及 shape、未知动作字段，以及连接失败后的资源回收。向 ``PartContract`` 传入示例动作后，contract 还会实际调用一次动作接口。

如果实现违反 contract，错误信息会列出具体原因，例如：

.. code-block:: text

   ConformanceError: MyArm does not keep 2:
     - MyArm: reconnecting raised RuntimeError: threads can only be started
       once; stall recovery closes an endpoint and opens it again
     - MyArm observes tcp_pose with shape (7,), declares (6,)

测试还应覆盖组合路径、句柄生命周期、注册、硬件发现，以及现有 policy 依赖的数据结构：

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py tests/unit_tests/test_real_env.py

上述测试不依赖真实硬件。

使用 mock SDK 验证
~~~~~~~~~~~~~~~~~~

设备 SDK 只在 ``_open()`` 中导入。``tests/robot_mocks`` 中的 mock SDK 允许真实部件类在没有硬件和厂商 SDK 的机器上运行。

首先检查机器人的组合结构和生命周期：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot --mock \
       --arg robot_ip=10.0.0.1 --arg node_rank=0

脚本会列出部件路径、共享连接和部署节点，然后读取观测并断开。未声明的观测、错误的 shape、错误加入部件树的连接，以及断开后仍报告已连接的资源都会导致检查失败。

环境根据部件声明创建 observation space，因此观测 shape 必须与声明一致。

添加 ``--remote`` 后，部件会部署到调度器 worker 中。该模式还会检查方法名称冲突和无法跨进程传递的状态。

还可以使用 mock 配置运行完整训练。``run.sh`` 会先安装相应的 mock SDK：

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh realworld_mock_sac_cnn

每种内置机器人均提供对应配置：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 机器人
     - 配置
   * - Franka
     - ``realworld_mock_sac_cnn``
   * - 双臂 Franka
     - ``realworld_dual_franka_mock_sac_cnn``
   * - GimArm
     - ``gim_arm_mock_sac_cnn``
   * - Turtle2
     - ``realworld_xsquare_turtle2_mock_sac_cnn``
   * - DOSW1
     - ``dosw1_mock_sac_mlp_pick``

连接真机
~~~~~~~~

mock 测试通过后，再检查真机的时序、标定和 SDK 行为。确认机器人已上电且网络可达后，移除 ``--mock`` 并运行同一命令：

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot \
       --arg robot_ip=10.0.0.1 --arg node_rank=1
