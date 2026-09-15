遥操作
======

本页说明如何为 rollout 接入遥操作，用于采集示教、纠正失败动作或运行 DAgger。内容按照实际接入顺序展开：先根据动作语义选择设备并独立验证硬件，再组合多台设备、确定各条 connection 的归属，只有更新频率确实不足时才调整直推模式；最后沿用相同顺序说明如何新增设备。若尚未了解设备动作对应的机器人路径，请先阅读 :doc:`../concepts/robotics`。

选择设备
--------

选择设备时，应同时考虑机器人需要的动作语义和操作者可用的硬件。下表中的每台设备都会产生特定含义的动作，例如机械臂笛卡尔增量或关节目标，有些设备还需要端口或标定信息。仅使用一台匹配设备时，在配置中填写其名称：

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - 设备
     - 操作方式
     - 额外配置
   * - ``spacemouse``
     - 通过 6 自由度操作球控制机械臂，并使用按键锁定夹爪状态。
     - 无
   * - ``gello``
     - 操作主臂，使从臂跟随至相同位姿。
     - ``gello_port``
   * - ``gello_joint``
     - 操作主臂，使从臂逐关节跟随；每条手臂配置一项。
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - 使用手持 VR 控制器，按住 grip 时接管机器人；双臂机器人需为每条手臂分别配置。
     - ``pico:`` 段
   * - ``glove``
     - 弯曲手指控制灵巧手，与驱动机械臂的设备配合使用。
     - ``glove_config:`` 段
   * - ``so101_leader``
     - 操作 SO-101 主臂，其夹爪同时控制从臂夹爪。
     - ``so101_leader_port``、``so101_leader_id``
   * - ``none``
     - 不接操作者设备，由 policy 独立控制。
     - 无

上表用于初步确定硬件，env 还会执行最终的语义检查。设备输出必须与环境声明的动作类型一致；例如，双臂 Franka 不提供单臂笛卡尔动作，因此配置 ``spacemouse`` 时会直接报错，并列出当前环境支持的设备。

验证设备读数
------------

选定设备后，应先验证 connection 和读数，再引入机器人或 env 配置。``gello``、``gello_joint``、``so101_leader`` 和 ``spacemouse`` 均提供独立运行命令：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0
   python -m rlinf.robotics.parts.teleop.so101_leader --port /dev/ttyACM1

SO-101 主臂必须先完成 lerobot 标定才能读数，否则设备会拒绝打开，并给出它查找的标定文件路径。请在终端中标定一次，并为这条手臂取一个名字，之后在 env 配置的 ``so101_leader_id`` 中沿用：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.so101_leader \
       --port /dev/ttyACM1 --id left_leader --calibrate

标定过程会提示操作者把手臂活动到各个极限位置，因此只有在终端中显式传入 ``--calibrate`` 时才会执行。通过配置启动的设备不会触发标定：调度器的 worker 没有终端来回答提示，一旦触发就会挂起。

如果主臂只返回零值或 SpaceMouse 没有响应，请先用该命令排查接线和设备权限。SO-101 主臂还会打印它将要下发的动作，以及与上一次读数相比是否算作接管；静止时显示 ``driving=False``，移动后才变为 ``True``。

完整机器人可使用 ``toolkits/realworld_check`` 检查；``check_robot_parts`` 会依次验证组合、读取和断开流程。

组合多台设备
------------

确认每台设备均可独立工作后，按照三个层次组合整套设备：先列出设备，再补充设备级硬件参数，只有同类分支无法区分时才使用 ``drives``。第一步是将 ``teleop`` 改为列表，每一项只控制对应的机器人零部件：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

在上述配置中，SpaceMouse 控制机械臂，数据手套控制灵巧手。按住 SpaceMouse 的第二个按键时，手套开始接管；松开后，灵巧手保持最后一个位姿。

设备级参数保留在对应的配置段中。例如，项目内置的灵巧手配置通过 ``glove_config`` 指定数据手套端口和标定文件：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]
       glove_config:
         left_port: /dev/ttyACM0
         frequency: 60
         config_file: null

当机器人包含两个同类分支时，使用 ``drives`` 指定每台设备控制的分支。该字段是遥操作配置中唯一直接引用零部件名称的位置：

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

如果设备声明的零部件不在机器人中，``TeleopGroup`` 会跳过对应动作。如果某台设备无法匹配任何零部件，或两台设备同时声明控制同一零部件，系统会在构建阶段报错。

明确设备的资源归属
------------------

设备组合通过语义检查后，还需要确定每条设备 connection 由哪个进程持有。内置遥操作构建器会在 env 进程中创建设备，再由 ``TeleopGroup.connect()`` 直接打开。遥操作设备不会加入 ``Robot`` 的组合结构，因此 ``Robot.connect()`` 不会处理它的 placement。通过 ``env.*.teleop`` 配置设备时，应将设备接到 env worker 所在的机器。

遥操作设备本身也是一条 ``Connection``，因此同样接受 ``node_rank``，连接时就在该节点打开：

.. code-block:: python

   leader = Gello("/dev/ttyUSB0", node_rank=1)
   leader.connect()
   try:
       print(leader.get_observation())
   finally:
       leader.disconnect()

该写法适用于独立诊断，也适用于 ``TeleopGroup``，因为 group 通过同一套 connection 接口打开每台设备。主臂的运行节点由构造时传入的 ``node_rank`` 决定。``env.*.teleop`` 配置目前尚未提供该字段，因此通过该配置创建的设备均在 env 进程中打开。

每次采样对每台独立设备只读取一次。同一台设备即使同时控制两个零部件，也只会打开一次；因此，SpaceMouse 同时控制机械臂和夹爪时仍只占用一个 HID 句柄。

.. _teleop-rate:

提高主从臂跟随频率
------------------

placement 决定设备在哪里读取，常规 env loop 则决定动作下发频率。如果主臂目标仅按 policy 的执行频率下发，从臂可能出现跟随延迟。直推模式使用独立线程，以约 1 kHz 的频率向控制器发送关节目标；``env.step`` 仍会读取状态，但不再转发运动指令：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

仅在跟随延迟明显时启用该选项。启用后，``env.step`` 不再发送关节目标；如果配置有误，机器人将保持静止。

延迟还有另一种成因，提高下发频率并不能解决。如果 follower 能平滑到达目标，却总是落后 leader 一点，说明机械臂的柔顺性增益偏软。:doc:`Dual Franka PICO 采集与 DAgger <../examples/embodied/dual_franka_pico_dagger>` 列出了 Franka 上的相关参数以及为遥操作调好的取值。

新增设备
--------

前面的使用流程依赖四项设备 contract：配置能够解析到已注册的 class，生命周期方法持有一条硬件句柄，观测方法产生符合声明的读数，``action()`` 再将读数转换为具名的机器人动作。新增设备时，应在 ``robotics/parts/teleop/`` 下用一个模块实现这些 contract：

.. code-block:: python

   @TeleopDevice.register("example")
   class ExampleDevice(TeleopDevice):
       PRODUCES = {"arm": ActionKind.JOINT_POSITION}
       NEEDS = ("joint_positions",)

       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           device = ExampleSDK(self._port)
           device.open()
           return device

       def _release(self, device) -> None:
           device.close()

       @property
       def observation_features(self) -> Features:
           return {"joints": {"shape": (7,), "dtype": "float32"}}

       def get_observation(self) -> Observation:
           return {"joints": self._device.read()}

       def action(self, reading, context) -> TeleopAction:
           moved = np.linalg.norm(reading["joints"] - context["joint_positions"][0])
           return TeleopAction({"arm": reading["joints"]}, driving=bool(moved > 0.01))

按照 builder 和 sampler 的调用顺序理解这个 class。``register("example")`` 定义配置名称；``PRODUCES`` 声明设备填充哪些动作零部件及其语义，env 因此可以在打开硬件前完成校验；``NEEDS`` 声明映射动作时需要哪些机器人状态。无论有几台设备请求同一项状态，每次采样都只读取一次，并通过 ``context`` 传入。

``__init__()`` 只记录端口。``_open()`` 随后创建并返回硬件句柄，设备通过 ``self._device`` 使用它；``_release(device)`` 负责关闭同一个句柄。公共 connection 层会处理可选的 ``node_rank``，因此 driver 构造函数只需包含自身的硬件参数。

如果句柄在后台线程中持续轮询，应在它自己的 ``close()`` 中先通知线程停止，再等待其退出；默认的 ``_release()`` 会找到并调用该方法。``gello``、``gello_joint`` 和 ``spacemouse`` 都采用这种写法，因此都不需要覆盖 ``_release()``。``TeleopGroup.disconnect()`` 会按相反顺序关闭设备，并在某台设备关闭失败后继续处理其他设备，但它无法结束不属于自己的线程。把清理逻辑放在线程所在的位置，可以保证独立诊断和 env 托管设备都能正常断开和重连。

连接后，``observation_features`` 提供离线可读的 schema，``get_observation()`` 返回与之对应的 ``joints`` 读数。sampler 将该读数和 ``NEEDS`` 指定的状态传给 ``action()``，其 ``TeleopAction`` 返回值同时包含需要填充的零部件和操作者是否正在接管。若本次采样不产生动作，则返回 ``driving=False``，控制权保留给 policy。

面向配置的行为
~~~~~~~~~~~~~~

至此，class contract 已支持直接构造；要从 env YAML 使用，还需将每个配置项映射到这些构造参数。``from_config()`` 的默认实现会将当前列表项的 options 作为关键字参数传给构造函数。只要配置 key 与构造参数同名，就无需再写任何代码：上面的设备已经可以通过 ``{example: {port: /dev/ttyUSB0}}`` 使用。

如果需要读取设备级配置段，或根据机器人的动作语义调整行为，则覆盖该方法：

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       settings = dict(cfg.get("example_config", {}))
       settings.update({k: v for k, v in options.items() if k != "drives"})
       if "port" not in settings:
           raise ValueError("teleop device 'example' requires a port")
       return TeleopEntry(cls(**settings), drives=options.get("drives"))

``cfg`` 是完整的 env 配置段，用于读取设备级配置；``options`` 只属于当前列表项，可单独指定端口或 ``drives``。应在此处校验允许的 key，避免拼写错误的硬件参数被静默忽略。``facts`` 描述 env 的动作布局和语义，例如机械臂接收绝对位姿还是增量，设备可据此调整，无需导入某个具体 env class。

只有当某台设备还要在独立线程中直接下发指令时，才需要覆盖 ``streamer()``；除 ``gello_joint`` 外，其余设备均使用默认实现并返回 ``None``，参见 :ref:`提高主从臂跟随频率 <teleop-rate>`。builder 会先构建所有 entry，再创建 streamer，因此 streamer 可以接管并非由自身构造的设备。

最后，将注册名加入对应 env 的 ``TELEOP`` 元组，声明该 env 能够表示该设备产生的动作。这里无需再次注册设备。

已废弃的配置项
--------------

新配置应使用前文所示的 ``teleop``。旧配置可能仍包含已废弃的写法：使用 ``teleop_device`` 指定设备，或通过 ``use_spacemouse``、``use_gello``、``use_gello_joint`` 和 ``use_pico`` 启用设备。这些字段仍可读取，但会产生 deprecation warning；如果旧字段与 ``teleop`` 同时出现，系统以 ``teleop`` 为准，并在 warning 中列出被覆盖的字段。

后续阅读
--------

完成设备选择、组合、归属和实现后，可根据当前工作继续阅读相应主题：

- :doc:`机器人接口 <../concepts/robotics>`：了解设备所填充的零部件路径。
- :doc:`真机任务与环境 <../concepts/realworld_envs>`：了解遥操作在 wrapper 栈中的位置。
- :doc:`数据采集 <data_collection>`：记录操作者动作。
