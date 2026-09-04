Teleoperation
=============

This page explains how to add teleoperation to a rollout for demonstration
collection, failure recovery, or DAgger. It follows the complete setup path:
choose a device by the action it produces, verify its readings in isolation,
compose a rig, decide where each connection opens, and adjust the update rate
only when needed. The final sections apply the same sequence to adding a new
device. For the robot paths those actions fill, see
:doc:`../concepts/robotics`.

Choose a Device
---------------

Begin with the action the robot expects and the hardware available to the
operator. Each registered device below produces a particular action meaning,
such as a Cartesian arm delta or joint target, and may require a port or
calibration. For a single matching device, set its name:

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
   * - ``so101_leader``
     - Poses an SO-101 leader arm; its own gripper commands the follower's.
     - ``so101_leader_port``, ``so101_leader_id``
   * - ``none``
     - Leaves the policy in control with no operator device.
     - None

The table narrows the hardware choice; the environment performs the final
semantic check. It accepts only devices that match the robot's control paths. A
dual-arm Franka, for example, has no single-arm Cartesian path, so it rejects
``spacemouse`` instead of ignoring the setting. The error also lists the devices
that the environment accepts.

Check a Device First
--------------------

After choosing a device, verify its connection and reading before introducing
robot or env configuration. Four devices provide standalone commands:
``gello``, ``gello_joint``, ``so101_leader`` and ``spacemouse``.

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0
   python -m rlinf.robotics.parts.teleop.so101_leader --port /dev/ttyACM1

An SO-101 leader needs lerobot's calibration before it reads anything, and
refuses to open without it, naming the file it looked for. Calibrate it once
from a terminal, giving the arm a name you will reuse in the env config as
``so101_leader_id``:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.so101_leader \
       --port /dev/ttyACM1 --id left_leader --calibrate

The procedure asks you to move the arm through its range, so it only runs when
you pass ``--calibrate`` from a terminal. A configured device never starts it:
a scheduler worker has no terminal to answer the prompts, and would hang.

When a leader arm reports only zeros or a spacemouse does not respond, this
command isolates wiring and permission problems from environment
configuration. The SO-101 leader also prints the action it would command and
whether it counts as driving, measured against where it last was, so an idle
arm reads ``driving=False`` until you move it.

``toolkits/realworld_check`` does the same for a complete robot;
``check_robot_parts`` walks one from composition through to disconnect.

Compose Devices After One Works
-------------------------------

Once each device works alone, compose the rig in three stages: list its devices,
add device-wide hardware settings, then use ``drives`` only where identical
branches would otherwise be ambiguous. Start by putting the device names in a
list; each entry contributes actions for the robot parts it can drive:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

In this dexterous-hand rig, the puck drives the arm and the glove drives the
hand. The glove takes control only while the spacemouse's second button is held;
releasing it leaves the hand at its last pose.

Device-wide settings remain in their named config block. For example, the
shipped dexterous-hand configs set the glove ports and calibration under
``glove_config``:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]
       glove_config:
         left_port: /dev/ttyACM0
         frequency: 60
         config_file: null

When a robot has two branches of the same kind, ``drives`` selects the branch for
each device. This is the only configuration field that names a robot part:

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

A device leaves any part the robot does not have unfilled. The builder rejects
the rig if a device matches no part at all, or if two devices claim the same
part.

Keep Device Ownership Explicit
------------------------------

The rig is now semantically valid; the next decision is which process owns each
device connection. The built-in teleop builder creates devices in the
environment process, and
``TeleopGroup.connect()`` opens them there. A teleop device does not belong to
the robot tree, so ``Robot.connect()`` does not place it. Plug devices configured
through ``env.*.teleop`` into the machine that runs the environment worker.

A teleop device is a ``Connection`` like any other, so it accepts a
``node_rank`` and opens on that node when something connects it:

.. code-block:: python

   leader = Gello("/dev/ttyUSB0", node_rank=1)
   leader.connect()
   try:
       print(leader.get_observation())
   finally:
       leader.disconnect()

That is useful for a standalone diagnostic, and it also works inside a
``TeleopGroup``, which opens each of its devices the same way. What decides
where a leader arm runs is the ``node_rank`` its constructor was given; the
``env.*.teleop`` config does not yet expose one, so devices configured through
it open in the environment process.

Each distinct device is read once per sample. If one device contributes to two
parts, it is still opened only once; a spacemouse driving both an arm and a
gripper therefore uses a single HID handle.

.. _teleop-rate:

When the Rate Is the Problem
----------------------------

Placement determines where a device reads, but the normal env loop still
determines how often it commands the robot. If a follower tracks poorly because
leader-arm targets arrive only at the policy's step rate, direct streaming moves
that path onto a thread that pushes
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

Lag has a second cause that a higher rate does not fix. If the follower reaches
its targets smoothly but settles behind the leader, the arm's compliance gains
are too soft. :doc:`Dual Franka PICO Collection and DAgger
<../examples/embodied/dual_franka_pico_dagger>` lists the Franka settings and
the values tuned for teleoperation.

Add a Device
------------

The setup above relies on four device contracts: configuration selects a
registered class, lifecycle methods own one hardware handle, observation methods
produce one declared reading, and ``action()`` maps that reading to named robot
actions. A new device implements those contracts in one module under
``robotics/parts/teleop/``:

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

Read the class in the order the builder and sampler use it.
``register("example")`` defines the config name. ``PRODUCES`` names the action
parts the device fills and the meaning of each, so the env can check the device
against its layout before hardware opens. ``NEEDS`` lists the robot state the
mapping requires; each requested value is read once per sample and arrives in
``context``, whether one device requests it or five do.

``__init__()`` records only the port. ``_open()`` later creates and returns the
hardware handle, which becomes ``self._device``; ``_release(device)`` closes the
same handle. The shared connection layer adds optional ``node_rank`` handling,
so the driver constructor contains only its hardware parameters.

A handle that polls in a background thread must stop and join that thread in
its own ``close()``, which the default ``_release()`` finds and calls. That is
where ``gello``, ``gello_joint`` and ``spacemouse`` put it, and it is why none
of them override ``_release()``. ``TeleopGroup.disconnect()`` closes devices in
reverse order and continues after one close fails, but it cannot make a thread
it does not own exit. Keeping the cleanup beside the thread makes disconnect
and reconnect reliable for standalone diagnostics and environment-managed
devices alike.

After connection, ``observation_features`` supplies the offline schema and
``get_observation()`` returns the matching ``joints`` reading. The sampler
passes that reading and the values named by ``NEEDS`` to ``action()``. Its
``TeleopAction`` return contains both the parts to fill and whether the operator
is currently driving. A device that fills nothing in a sample returns
``driving=False`` and leaves the policy in control.

Config-Facing Behaviour
~~~~~~~~~~~~~~~~~~~~~~~

The class contract is complete for direct construction. To make it useful from
env YAML, map each config entry to those constructor arguments. The default
``from_config()`` passes the entry's own options through as keyword
arguments, which is all a device needs whose config keys match its constructor.
The example above is already reachable as ``{example: {port: /dev/ttyUSB0}}``
with nothing further to write.

Override it to read a key from the wider env config, or to choose behaviour from
what the robot can accept:

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       settings = dict(cfg.get("example_config", {}))
       settings.update({k: v for k, v in options.items() if k != "drives"})
       if "port" not in settings:
           raise ValueError("teleop device 'example' requires a port")
       return TeleopEntry(cls(**settings), drives=options.get("drives"))

``cfg`` is the complete env section and holds the device-wide config block.
``options`` belongs to this one list entry, so it can select a port or a
``drives`` branch without changing other instances. Validate these keys here; a
misspelled hardware option should not be silently ignored. ``facts`` describes
the env's action layout and semantics -- notably whether an arm takes an
absolute pose or a delta -- so a device can adapt without importing a concrete
env class.

Override ``streamer()`` only for a device that also commands the robot on its
own thread; the default returns nothing, which is what every device but
``gello_joint`` wants. See :ref:`When the Rate Is the Problem <teleop-rate>`.
The builder creates every entry before it creates any streamer, so a streamer
may take over devices it did not construct itself.

Then list the registered name in the env's ``TELEOP`` tuple. This declaration
states that the env can represent the device's action; it does not register the
device again.

Retired Spellings
-----------------

New configurations should use ``teleop`` as shown above. Older runs may still
contain retired spellings: ``teleop_device`` named a single device, and the
booleans ``use_spacemouse``,
``use_gello``, ``use_gello_joint``, and ``use_pico`` each switched one on. All
of them still work and warn. Where ``teleop`` appears alongside one of them --
which happens when a run config sits on an older base -- ``teleop`` is the one
that takes effect, and the warning names what it replaced.

Next
----

With selection, composition, ownership, and implementation covered, continue
with the part of the workflow you are setting up:

- :doc:`Robotics Interface <../concepts/robotics>`: the named robot paths a
  device fills.
- :doc:`Real-World Tasks and Environments <../concepts/realworld_envs>`: where teleop
  sits in the wrapper stack.
- :doc:`Data Collection <data_collection>`: recording what an operator drove.
