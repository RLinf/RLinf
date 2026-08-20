遥操作
======

遥操作允许操作者在 rollout 中接管 policy，可用于采集示教、纠正失败动作或运行 DAgger。接入时，建议先验证单台设备能否稳定读取数据，再配置 binding，最后组合多台设备或启用远程部署。若尚未了解部件名称与动作路径的关系，请先阅读 :doc:`../concepts/robotics`。

选择设备
--------

仅使用一台设备时，在配置中填写设备名称：

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
   * - ``none``
     - 不接操作者设备，由 policy 独立控制。
     - 无

设备输出必须与环境声明的动作类型一致。例如，双臂 Franka 不提供单臂笛卡尔动作，因此配置 ``spacemouse`` 时会直接报错，并列出当前环境支持的设备。

验证设备读数
------------

reader 不依赖机器人或集群，可独立运行：

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

如果主臂只返回零值或 SpaceMouse 没有响应，请先通过 reader 排查接线和设备权限。完整机器人可使用 ``toolkits/realworld_check`` 检查；``check_robot_parts`` 会依次验证组合、读取和断开流程。

组合多台设备
------------

确认每台设备均可独立工作后，将 ``teleop`` 改为列表。每一项只负责其对应的机器人部件：

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

在上述配置中，SpaceMouse 控制机械臂，数据手套控制灵巧手。按住 SpaceMouse 的第二个按键时，手套开始接管；松开后，灵巧手保持最后一个位姿。

当机器人包含两个同类分支时，使用 ``drives`` 指定每台设备控制的分支。该字段是遥操作配置中唯一直接引用机器人部件名称的位置：

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

如果 binding 声明的部件不在机器人中，``TeleopGroup`` 会跳过该部件。如果某台设备无法匹配任何部件，或两台设备同时声明控制同一部件，系统会在构建阶段报错。

根据物理连接位置部署设备
------------------------

遥操作设备使用与普通部件相同的部署模型。如果设备与 policy 位于不同机器，请设置 ``node_rank``：

.. code-block:: python

   leader = TeleopLeaderArm.at("/dev/ttyUSB0", node_rank=1)

系统会并行读取使用独立连接的设备。同一台设备即使同时控制两个部件，也只会打开一次；因此，SpaceMouse 同时控制机械臂和夹爪时仍只占用一个 HID 句柄。

提高主从臂跟随频率
------------------

如果主臂目标仅按 policy 的执行频率下发，从臂可能出现跟随延迟。直推模式使用独立线程，以约 1 kHz 的频率向控制器发送关节目标；``env.step`` 仍会读取状态，但不再转发运动指令：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

仅在跟随延迟明显时启用该选项。启用后，``env.step`` 不再发送关节目标；如果配置有误，机器人将保持静止。

已废弃的配置项
--------------

旧配置使用 ``teleop_device`` 指定设备，或通过 ``use_spacemouse``、``use_gello``、``use_gello_joint`` 和 ``use_pico`` 启用设备。这些字段仍可读取，但会产生 deprecation warning。如果旧字段与 ``teleop`` 同时出现，系统以 ``teleop`` 为准，并在 warning 中列出被覆盖的字段。

后续阅读
--------

- :doc:`机器人模型 <../concepts/robotics>`：了解设备、binding 和 group 的组织方式。
- :doc:`真机环境模型 <../concepts/realworld_envs>`：了解遥操作在 wrapper 栈中的位置。
- :doc:`数据采集 <data_collection>`：记录操作者动作。
