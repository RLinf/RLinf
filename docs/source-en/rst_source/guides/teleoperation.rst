Teleoperation
=============

Teleoperation lets an operator replace the policy's action during a rollout to
collect demonstrations, recover from a failure, or run DAgger. Start with one
device and verify its readings. Then add a binding to the robot's named action
parts; only after that works should you combine devices or move one to another
machine.

This page follows that order. For the underlying named-part model, see
:doc:`../concepts/robotics`.

Choose a Device
---------------

For a single device, set its name:

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - Device
     - What the operator does
     - Extra config
   * - ``spacemouse``
     - Moves the arm with a 6-DoF puck; its buttons latch the gripper.
     - None
   * - ``gello``
     - Poses a leader arm for the follower to track.
     - ``gello_port``
   * - ``gello_joint``
     - Poses a leader arm for joint-by-joint tracking, one entry per arm.
     - ``left_gello_port`` / ``right_gello_port``
   * - ``pico``
     - Uses a handheld VR controller whose grip marks when it is driving; one
       entry per arm on a two-armed robot.
     - ``pico:`` block
   * - ``glove``
     - Bends the fingers of a dexterous hand, alongside a device on the arm.
     - ``glove_config:`` block
   * - ``none``
     - Leaves the policy in control with no operator device.
     - None

An environment accepts only devices that match the robot's control paths. A
dual-arm Franka, for example, has no single-arm Cartesian path, so it rejects
``spacemouse`` instead of ignoring the setting. The error also lists the devices
that the environment accepts.

Check a Device First
--------------------

Every device reader can run without a robot or cluster:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

When a leader arm reports only zeros or a spacemouse does not respond, this
command isolates wiring and permission problems from environment configuration.
``toolkits/realworld_check`` does the same for a complete robot;
``check_robot_parts`` walks one from composition through to disconnect.

Compose Devices After One Works
-------------------------------

Once each device works alone, put them in a list. Each entry contributes actions
for the named robot parts it can drive:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

In this dexterous-hand rig, the puck drives the arm and the glove drives the
hand. The glove takes control only while the spacemouse's second button is held;
releasing it leaves the hand at its last pose.

When a robot has two branches of the same kind, ``drives`` selects the branch for
each device. This is the only configuration field that names a robot part:

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

A binding leaves any part the robot does not have unfilled. The builder rejects
the rig if a device matches no part at all, or if two devices claim the same
part.

Put a Device Where It Is Plugged In
-----------------------------------

A teleop device follows the same placement model as any other part. When the
operator's hardware is attached to a different machine from the policy, assign
the device a ``node_rank``:

.. code-block:: python

   leader = TeleopLeaderArm.at("/dev/ttyUSB0", node_rank=1)

Devices on independent connections are read in parallel. If one device
contributes to two parts, it is still opened only once; a spacemouse driving both
an arm and a gripper therefore uses a single HID handle.

When the Rate Is the Problem
----------------------------

A follower tracks poorly when it receives leader-arm targets only at the
policy's step rate. Direct streaming moves that path onto a thread that pushes
joint targets to the controllers at roughly 1 kHz. ``env.step`` continues to
read state but no longer forwards motion:

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

Enable direct streaming only when tracking visibly lags. Because ``env.step`` no
longer dispatches joint targets in this mode, a misconfigured rig remains still
instead of receiving malformed motion.

Retired Spellings
-----------------

``teleop_device`` named a single device, and the booleans ``use_spacemouse``,
``use_gello``, ``use_gello_joint``, and ``use_pico`` each switched one on. All
of them still work and warn. Where ``teleop`` appears alongside one of them --
which happens when a run config sits on an older base -- ``teleop`` is the one
that takes effect, and the warning names what it replaced.

Next
----

- :doc:`Robotics Model <../concepts/robotics>`: devices, bindings, and groups.
- :doc:`Real-World Environment Model <../concepts/realworld_envs>`: where teleop
  sits in the wrapper stack.
- :doc:`Data Collection <data_collection>`: recording what an operator drove.
