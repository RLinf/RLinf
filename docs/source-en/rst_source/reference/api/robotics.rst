Robotics Interface
==================

Look up RLinf's hardware connection, robot part composition, and robot discovery APIs under ``rlinf.robotics``. For design concepts and step-by-step bring-up, see :doc:`../../concepts/robotics` and :doc:`../../extending/new_robot`.

Parts and Connections
---------------------

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

Robots and Discovery
--------------------

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
