Teleoperation
=============

Add teleoperation to a rollout for demonstration collection, failure recovery,
or DAgger. This guide walks through the full setup order: choose a device by
the action it produces, verify its readings in isolation, compose a multi-device
rig, place each connection on the right machine, and raise the command rate
only when tracking lags. The final sections cover adding a new device driver.
For the robot action paths those devices fill, see :doc:`../concepts/robotics`.

Choose a Device
---------------

Match the operator hardware to the action the robot expects. Each registered
device below produces a specific action type, such as a Cartesian arm delta or
joint target, and may need a serial port or calibration. For a single matching
device, set its name:

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
       entry per arm on a two-armed robot. An arm that takes joint positions
       is reached through the environment's inverse kinematics.
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

The table narrows the hardware choice; the environment runs the final semantic
check. It accepts only devices that match the robot's control paths. A dual-arm
Franka, for example, has no single-arm Cartesian path, so it rejects
``spacemouse`` with an error listing the devices it accepts.

Check a Device First
--------------------

Verify each device's connection and readings before combining it with a robot
or environment config. Four devices provide standalone check commands:
``gello``, ``gello_joint``, ``so101_leader``, and ``spacemouse``.

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0
   python -m rlinf.robotics.parts.teleop.so101_leader --port /dev/ttyACM1

An SO-101 leader requires LeRobot calibration before it can read joint angles,
and refuses to open without it, naming the calibration file it looked for.
Calibrate it once from a terminal, giving the arm a name you reuse in the
environment config as ``so101_leader_id``:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.so101_leader \
       --port /dev/ttyACM1 --id left_leader --calibrate

Calibration prompts you to move the arm through its range, so it runs only when
you pass ``--calibrate`` from a terminal. A device launched from YAML never
starts calibration because a scheduler worker has no terminal to answer prompts
and would hang.

When a leader arm reports only zeros or a SpaceMouse does not respond, this
command isolates wiring and permission issues from environment config. The
SO-101 leader also prints its outgoing action and whether it counts as driving
relative to its previous pose, so an idle arm reports ``driving=False`` until
you move it.

``toolkits/realworld_check`` provides the same standalone check for a full
robot, walking from hardware setup through reading and disconnect.

Compose Devices After One Works
-------------------------------

Once each device works alone, compose the rig in three stages: list the
devices, add device-wide hardware settings, and use ``drives`` only where
identical robot branches would otherwise be ambiguous. Start by listing the
device names; each entry supplies actions for the robot parts it can drive:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

In this dexterous-hand rig, the puck drives the arm and the glove drives the
hand. The glove takes control only while the SpaceMouse's second button is
held; releasing it holds the hand at its last pose.

Device-wide settings stay in their named config block. For example, the shipped
dexterous-hand configs set the glove port and calibration under
``glove_config``:

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]
       glove_config:
         left_port: /dev/ttyACM0
         frequency: 60
         config_file: null

When a robot has two identical branches, ``drives`` selects the branch for each
device. This is the only teleop config field that names a robot part:

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {gello_joint: {port: /dev/serial/by-id/...-left,  drives: left}}
         - {gello_joint: {port: /dev/serial/by-id/...-right, drives: right}}

A device leaves any part the robot does not have unfilled. Setup rejects the
rig if a device matches no part at all, or if two devices claim the same part.

Keep Device Ownership Explicit
------------------------------

The built-in teleop setup creates and opens all configured devices inside the
environment worker process. Teleop devices are managed separately from the
robot hardware tree, so plug any device configured under ``env.*.teleop`` into
the machine running the environment worker.

When constructed directly in Python, a teleop driver also accepts an optional
``node_rank`` so its hardware connection opens on that node:

.. code-block:: python

   leader = Gello("/dev/ttyUSB0", node_rank=1)
   leader.connect()
   try:
       print(leader.get_observation())
   finally:
       leader.disconnect()

This pattern works for standalone diagnostics on a remote node. Because
``env.*.teleop`` does not expose ``node_rank`` yet, devices configured through
YAML always open inside the environment worker process.

Each distinct device is read once per sample and opened only once, even when it
drives two robot parts. A SpaceMouse driving both an arm and a gripper uses a
single HID handle.

.. _teleop-rate:

When the Rate Is the Problem
----------------------------

Placement determines where a device reads, while the environment loop
determines how often it commands the robot. If a follower lags because
leader-arm targets arrive only at the policy step rate, enable direct streaming
so a dedicated thread pushes joint targets to the controllers at roughly 1 kHz.
In this mode, the step loop keeps reading state without forwarding motion
commands:

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         teleop_direct_stream: true

Enable direct streaming only when tracking visibly lags. Because the step loop
no longer dispatches joint targets in this mode, a misconfigured rig stays
still instead of receiving malformed motion.

Lag has a second cause that a higher rate does not fix. If the follower reaches
its targets smoothly but settles behind the leader, the arm's compliance gains
may be too soft. :ref:`Configure Arm Motion <franka-motion-settings>` explains
the Franky defaults shared by teleoperation and policy execution, and how to
override them.

Add a Device
------------

A teleop driver implements four contracts in a single module under
``robotics/parts/teleop/``: configuration selects a registered driver, lifecycle
hooks own one hardware handle, observation methods expose a declared reading,
and ``action()`` maps that reading to named robot actions:

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

Registration defines the device name used in YAML. ``PRODUCES`` declares which
action parts the device drives and their semantics, so the environment validates
compatibility before opening hardware. ``NEEDS`` lists any robot state the
mapping requires; each requested value is read once per sample and passed
through ``context``, no matter how many devices request it.

Keep constructor arguments limited to hardware parameters such as the serial
port; the shared connection layer handles optional ``node_rank`` placement. The
open hook creates and returns the hardware handle (stored on ``self._device``),
and the release hook closes that same handle.

If a hardware handle polls in a background thread, stop and join that thread
inside its ``close()`` method, which the default release hook calls
automatically (``gello``, ``gello_joint``, and ``spacemouse`` all rely on
this). Shutdown closes devices in reverse order and continues if one close
fails, but it cannot stop a thread it does not own. Keeping thread cleanup next
to the thread keeps disconnect and reconnect reliable for both standalone
diagnostics and environment runs.

After opening, the schema property declares the offline observation layout and
the observation method returns the matching ``joints`` reading. On each sample,
the sampler passes that reading together with the requested robot state to
``action()``, which returns the target action parts and whether the operator is
driving. Returning ``driving=False`` leaves the policy in control.

Config-Facing Behaviour
~~~~~~~~~~~~~~~~~~~~~~~

The default config loader forwards each YAML list entry's options as keyword
arguments to the constructor. When YAML keys match constructor parameters, no
extra code is needed—the example above already works as
``{example: {port: /dev/ttyUSB0}}``.

Override the config loader when the driver must read a shared block from the
environment config or adapt to what the robot accepts:

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       settings = dict(cfg.get("example_config", {}))
       settings.update({k: v for k, v in options.items() if k != "drives"})
       if "port" not in settings:
           raise ValueError("teleop device 'example' requires a port")
       return TeleopEntry(cls(**settings), drives=options.get("drives"))

``cfg`` holds the full environment section, including device-wide config
blocks, while ``options`` holds this list entry's fields (such as a port or
``drives``). Validate keys here so misspelled hardware options fail fast
instead of being silently ignored. ``facts`` exposes the environment's action
layout, action semantics (absolute pose, delta, or joint positions), and
joint-arm kinematics so the driver can adapt without importing a concrete
environment class.

Override ``streamer()`` only for a device that also commands the robot on its
own thread (only ``gello_joint`` does this; see
:ref:`When the Rate Is the Problem <teleop-rate>`). All device entries are
created before any streamer, so a streamer can take over devices it did not
construct itself.

Override ``interlock()`` when multiple arms driven by one device must stop
together (as ``pico`` does when driving two or more joint-position arms). The
check runs after all entries compute their actions and before any command is
sent, so a fault on one arm holds its partner on the same tick.

Finally, add the registered name to the environment's ``TELEOP`` tuple so the
environment accepts actions from this device.

Retired Spellings
-----------------

Use ``teleop`` in new configs. Legacy keys still work and emit a deprecation
warning:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Legacy key
     - Replacement
   * - ``teleop_device``
     - ``teleop: <device>``
   * - ``use_spacemouse``, ``use_gello``, ``use_gello_joint``, ``use_pico``
     - ``teleop: <device>`` (or a list of devices)

If ``teleop`` appears alongside a legacy key—for example when inheriting from
an older base config—``teleop`` takes effect and the warning names the replaced
key.

Next
----

- :doc:`Robotics Interface <../concepts/robotics>`: the named robot paths a
  device fills.
- :doc:`Real-World Tasks and Environments <../concepts/realworld_envs>`: where
  teleop sits in the wrapper stack.
- :doc:`Data Collection <data_collection>`: recording operator demonstrations.
