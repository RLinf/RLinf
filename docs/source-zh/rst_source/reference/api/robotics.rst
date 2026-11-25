Robotics 接口
=============

查阅 ``rlinf.robotics`` 提供的硬件连接、零部件组合与机器人发现 API。设计理念与接入流程分别见 :doc:`../../concepts/robotics` 与 :doc:`../../extending/new_robot`。

零部件与连接
------------

.. autoclass:: rlinf.robotics.Connection
   :members: connect, disconnect, is_connected, register, backend, backends
   :member-order: bysource

.. autoclass:: rlinf.robotics.RobotPart
   :members: observation_features, get_observation
   :member-order: bysource

.. autoclass:: rlinf.robotics.ControllablePart
   :members: action_features, send_action, reset
   :member-order: bysource

.. autoclass:: rlinf.robotics.PartGroup
   :members: connect, disconnect, get_observation, send_action, reset, child, children, owners
   :member-order: bysource

机器人与硬件发现
----------------

.. autoclass:: rlinf.robotics.Robot
   :members: register_type, of_type, build, named_parts, parts_of_type, describe
   :member-order: bysource

.. autoclass:: rlinf.robotics.RobotConfig
   :members:
   :member-order: bysource

.. autoclass:: rlinf.robotics.RobotDiscovery
   :members: enumerate
   :member-order: bysource

.. autofunction:: rlinf.robotics.build_robot

.. autofunction:: rlinf.robotics.register_robot
