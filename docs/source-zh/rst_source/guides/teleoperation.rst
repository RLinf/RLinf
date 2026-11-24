遥操作设备配置
==============

为 rollout 接入遥操作，可用于示教数据采集、失败动作纠正或运行 DAgger。本指南按实际接入顺序展开：先按动作语义选择设备并单独验证读数，再组合多台设备、明确设备连接所在机器，仅在跟随明显延迟时开启直推模式；最后说明如何新增设备驱动。设备动作对应的机器人零部件路径见 :doc:`../concepts/robotics`。

选择设备
--------

先对照机器人接收的动作类型和操作者手头的硬件挑选设备。下表中的每台设备对应一种动作语义（如机械臂笛卡尔增量或关节目标），部分设备还需要串口或标定信息。只使用一台设备时，直接填写设备名称：

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
     - 用 6 自由度操作球控制机械臂，按键锁定夹爪状态。
     - 无
   * - ``gello``
     - 操控主臂，从臂跟随至相同位姿。
     - ``gello_port``
   * - ``gello_joint``
     - 操控主臂，从臂逐关节跟随；每条手臂配置一项。
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - 使用手持 VR 控制器，按住握键时接管机器人；双臂机器人需为每条手臂各配一项。接收关节位置的机械臂由环境的逆运动学换算目标。
     - ``pico:`` 配置块
   * - ``glove``
     - 弯曲手指控制灵巧手，配合驱动机械臂的设备使用。
     - ``glove_config:`` 配置块
   * - ``so101_leader``
     - 操控 SO-101 主臂，主臂夹爪同时控制从臂夹爪。
     - ``so101_leader_port``、``so101_leader_id``
   * - ``none``
     - 不接入操作者设备，完全由 policy 控制。
     - 无

上表用于初步匹配硬件，环境还会做最终的语义校验。设备输出必须符合机器人声明的控制路径；例如双臂 Franka 不支持单臂笛卡尔动作，配置 ``spacemouse`` 时会直接报错并列出当前环境支持的设备。

验证设备读数
------------

选定设备后，先单独检查连接和读数，再写入机器人或环境配置。``gello``、``gello_joint``、``so101_leader`` 和 ``spacemouse`` 均提供独立检查命令：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0
   python -m rlinf.robotics.parts.teleop.so101_leader --port /dev/ttyACM1

SO-101 主臂必须先完成 LeRobot 标定才能读取关节角，否则会拒绝打开并提示缺失的标定文件路径。在终端中标定一次，并指定一个手臂名称，后续在环境配置的 ``so101_leader_id`` 中填写同一名称：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.so101_leader \
       --port /dev/ttyACM1 --id left_leader --calibrate

标定过程需要按终端提示活动手臂到各极限位置，因此仅在显式传入 ``--calibrate`` 时触发。通过 YAML 配置启动的设备不会自动标定，因为调度器 worker 没有交互终端，触发后会直接挂起。

如果主臂只返回零值或 SpaceMouse 无响应，先用上面的命令排查接线和设备权限。SO-101 主臂还会打印将要下发的动作，以及相对上次位置是否算作接管：静止时输出 ``driving=False``，移动后变为 ``True``。

完整机器人可使用 ``toolkits/realworld_check`` 检查，从硬件组装到读取、断开逐项验证。

组合多台设备
------------

每台设备都能独立读数后，分三步组合整套装置：先列出设备，再填写设备级硬件参数，仅在存在同类分支、需要消歧时才设置 ``drives``。第一步将 ``teleop`` 写成列表，每台设备负责各自匹配的机器人零部件：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

这套灵巧手组合中，SpaceMouse 驱动机械臂，数据手套驱动灵巧手。按住 SpaceMouse 的第二个按键时手套接管；松开后，灵巧手保持最后位姿。

设备级硬件参数写在对应的命名配置块中。例如内置灵巧手配置在 ``glove_config`` 下指定手套串口和标定文件：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]
       glove_config:
         left_port: /dev/ttyACM0
         frequency: 60
         config_file: null

机器人包含两个同类分支时，用 ``drives`` 指定每台设备控制的分支。这是遥操作配置中唯一直接填写零部件名称的字段：

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

设备声明的零部件若不在当前机器人中，对应动作会被留空。如果某台设备匹配不到任何零部件，或两台设备同时声称控制同一零部件，构建阶段就会报错。

明确设备的资源归属
------------------

内置遥操作流程会在环境 worker 进程内创建并打开所有设备。遥操作设备独立于机器人硬件树管理，因此通过 ``env.*.teleop`` 配置的设备必须接入运行环境 worker 的那台机器。

在 Python 中直接构造遥操作驱动时，可传入可选的 ``node_rank``，使硬件连接在指定节点打开：

.. code-block:: python

   leader = Gello("/dev/ttyUSB0", node_rank=1)
   leader.connect()
   try:
       print(leader.get_observation())
   finally:
       leader.disconnect()

这种方式适合在远端节点做独立诊断。``env.*.teleop`` 目前不提供 ``node_rank`` 字段，YAML 配置的设备一律在环境 worker 进程内打开。

每次采样对每台设备只读取一次；即便同一台设备同时控制两个零部件，也只打开一条硬件句柄。例如 SpaceMouse 同时控制机械臂和夹爪时，只占用一个 HID 句柄。

.. _teleop-rate:

提高主从臂跟随频率
------------------

placement 决定设备在哪台机器上读取，环境循环决定动作下发频率。如果主臂目标仅按 policy 步频下发、导致从臂跟随滞后，开启直推模式后，独立线程会以约 1 kHz 的频率向控制器推送关节目标；环境步进循环仍继续读取状态，不再转发运动指令：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

仅在跟随延迟明显时启用该选项。开启后，步进循环不再下发关节目标，配置有误时机器人会保持静止，不会执行异常动作。

延迟还有另一种提高频率无法解决的原因：从臂运动平滑却始终落后于主臂，通常是机械臂的柔顺性增益偏软。:ref:`配置机械臂运动 <franka-motion-settings>` 说明了遥操作与 policy 共用的 Franky 默认参数及覆盖方法。

新增设备
--------

新增遥操作驱动只需在 ``robotics/parts/teleop/`` 下添加一个模块，实现四项约定：注册配置名、管理硬件句柄的生命周期、提供声明格式的读数，以及用 ``action()`` 将读数转换为具名机器人动作：

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

注册装饰器定义 YAML 中使用的设备名。``PRODUCES`` 声明设备驱动哪些动作项及动作语义，让环境在打开硬件前完成兼容性校验；``NEEDS`` 声明映射动作所需的机器人状态，无论几台设备请求同一项状态，每次采样只读取一次并通过 ``context`` 传入。

构造函数只保留串口等硬件参数，可选的 ``node_rank`` 由公共连接层处理。打开钩子创建并返回硬件句柄（存放在 ``self._device``），释放钩子负责关闭同一句柄。

如果硬件句柄内部起有后台轮询线程，应在它自己的 ``close()`` 方法里停止并等待线程退出，默认释放逻辑会自动调用该方法（``gello``、``gello_joint`` 与 ``spacemouse`` 均采用这一做法）。关闭流程按逆序断开每台设备，单台失败仍会继续关闭其余设备，但无法替驱动结束其私有线程。将线程清理紧贴线程实现，能保证独立诊断与环境托管下都可稳定断开和重连。

设备连接后，schema 属性声明离线可读的数据结构，观测读取方法返回对应的 ``joints`` 读数。每次采样时，采样器将读数与请求的机器人状态一起传给 ``action()``，返回目标动作项以及操作者是否正在接管。返回 ``driving=False`` 时，控制权保留给 policy。

面向配置的行为
~~~~~~~~~~~~~~

默认配置加载器会把 YAML 列表项中的字段直接作为关键字参数传给构造函数。只要配置项与构造参数同名，无需额外代码，上面的示例即可直接用 ``{example: {port: /dev/ttyUSB0}}`` 启动。

需要读取环境中的共享设备配置块，或根据机器人支持的动作语义调整行为时，覆盖配置加载方法：

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       settings = dict(cfg.get("example_config", {}))
       settings.update({k: v for k, v in options.items() if k != "drives"})
       if "port" not in settings:
           raise ValueError("teleop device 'example' requires a port")
       return TeleopEntry(cls(**settings), drives=options.get("drives"))

``cfg`` 是完整的环境配置段，用于读取设备级配置块；``options`` 仅包含当前列表项的字段（如串口或 ``drives``）。在此处校验参数名，可防止拼错的硬件选项被静默忽略。``facts`` 提供环境的动作布局、动作语义（绝对位姿、增量或关节位置）以及关节臂运动学信息，让驱动无需导入具体环境类即可自适应。

只有设备需要另起线程直接向控制器推流时才覆盖 ``streamer()``，目前仅 ``gello_joint`` 使用该机制（参见 :ref:`提高主从臂跟随频率 <teleop-rate>`）。构建器先创建全部设备条目再创建 streamer，因此 streamer 可以接管并非由自身构造的设备。

一台设备驱动的多条手臂必须联动急停时，覆盖 ``interlock()``\ （例如 ``pico`` 以关节位置控制两条及以上手臂时）。该检查在所有条目换算完动作之后、下发之前运行，单条手臂在同一拍内发生故障也能让配对手臂同步停住。

最后将注册名加入对应环境的 ``TELEOP`` 元组，声明该环境接受该设备的动作。

已废弃的配置项
--------------

新配置请统一使用 ``teleop``。旧字段仍可兼容读取，但会输出废弃警告：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 旧配置项
     - 替代写法
   * - ``teleop_device``
     - ``teleop: <设备名>``
   * - ``use_spacemouse``、``use_gello``、``use_gello_joint``、``use_pico``
     - ``teleop: <设备名>``\ （或设备列表）

若 ``teleop`` 与旧字段同时出现（例如继承了旧的基础配置），以 ``teleop`` 为准，警告信息会列出被覆盖的旧字段。

后续阅读
--------

- :doc:`机器人接口 <../concepts/robotics>`：了解设备所填充的零部件路径。
- :doc:`真机任务与环境 <../concepts/realworld_envs>`：了解遥操作在环境封装栈中的位置。
- :doc:`数据采集 <data_collection>`：录制操作者示教数据。
