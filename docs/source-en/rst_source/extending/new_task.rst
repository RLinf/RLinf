New Real-World Tasks
====================

Use this guide when RLinf already knows how to connect the physical robot and
your change is about what it should accomplish. By the end, you will have a
config dataclass, a small env class, a registered Gymnasium ID, and a YAML config
you can launch.

This guide covers task modules under ``rlinf/envs/real``. To add a task to a
simulator or benchmark, follow :doc:`new_env` instead.

You will not change robot construction, device placement, or teleoperation.
Targets, compliance settings, success rules, and reset behavior belong to the
task; the existing robot and wrapper stack remain in place. If the hardware
itself is new, follow :doc:`new_robot` first and return here once one observation
and action can pass through it.

Steps
-----

The examples below add a ``WipeEnv-v1`` task to the existing Franka support.
Follow them in order: later steps refer to the config class and Gymnasium ID
chosen earlier.

1. Write the config
~~~~~~~~~~~~~~~~~~~

Create ``rlinf/envs/real/<robot>/<task>.py`` beside the other tasks for that
robot. Inherit its config dataclass and add the fields required by your task:

.. code-block:: python

   from dataclasses import dataclass, field

   import numpy as np

   from .base import FrankaEnv, FrankaRobotConfig, compliance


   @dataclass
   class WipeConfig(FrankaRobotConfig):
       task_description: str = "wipe the surface"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       reward_threshold: np.ndarray = field(
           default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
       )
       random_xy_range: float = 0.03

       def __post_init__(self):
           self.compliance_param = compliance(
               translational_stiffness=800,   # softer, to keep contact
               translational_clip_z=0.02,
           )
           self.target_ee_pose = np.array(self.target_ee_pose)
           self.action_scale = np.array([0.02, 0.1, 1])

State only the gains that differ. ``compliance()`` merges them onto
``COMPLIANCE_DEFAULTS`` and raises on any gain the controller does not accept. A
misspelled gain stops here instead of reaching the impedance controller and
being ignored.

2. Write the env
~~~~~~~~~~~~~~~~

Set the env class's config type. For many tasks, that is the whole class:

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

Override a hook only when the task needs different behavior. ``go_to_rest`` is
the common case because homing depends on the task's end pose. Peg insertion,
for example, lifts clear of the slot first; otherwise the peg catches on the way
up:

.. code-block:: python

       def go_to_rest(self, joint_reset=False):
           reset_pose = copy.deepcopy(self._franka_state.tcp_pose)
           reset_pose[2] += 0.05
           self._interpolate_move(reset_pose, timeout=1)
           super().go_to_rest(joint_reset)

A task that keeps its robot's action space inherits how that action is read. A
task that changes the action space declares the change, because a teleop device
is matched against what each part means rather than how wide it is:

.. code-block:: python

       def action_parts(self):
           return (
               ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
               ActionPart("end_effector", 1, ActionKind.GRIPPER),
           )

The declared widths must add up to the action space exactly; a mismatch is an
error rather than a slice that lands somewhere unintended.

3. Register it
~~~~~~~~~~~~~~

Add one entry to the robot's ``TASKS`` table in
``rlinf/envs/real/<robot>/__init__.py``, naming the env class:

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": WipeEnv,
   }

``register_tasks`` builds the entry point and registers the id with Gymnasium.
The wrapper stack does not appear here: the env declared it above, and
``build_stack`` reads that declaration.

User configs and dataset metadata store the gym id. Changing it later breaks
those references. Choose the name before collecting data.

4. Add the env config
~~~~~~~~~~~~~~~~~~~~~

Add a YAML file under ``examples/embodiment/config/env/``. Keep the registered
id at the path shown below and describe the task fields in ``override_cfg``;
these fields come from your config dataclass:

.. code-block:: yaml

   env_type: real
   init_params:
     id: "WipeEnv-v1"      # the gym id you registered
     num_envs: null
   teleop: spacemouse
   override_cfg:
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
     random_xy_range: 0.03

5. Check it
~~~~~~~~~~~

Before connecting hardware, confirm that the id is registered and its entry
point resolves:

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # registers every task
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` makes the same assertion for every shipped
task. Add your id to ``EXPECTED_IDS`` there.

What You Do Not Need to Write
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Concern
     - Where it already lives
   * - Connecting and placing hardware
     - ``Robot.connect``; see :doc:`../concepts/robotics`.
   * - Teleoperation
     - ``teleop`` in the env config selects one; the wrapper stack builds
       it.
   * - Marking reward or ending an episode by hand
     - ``keyboard_reward_wrapper`` in the env config.
   * - Relative frames, Euler conversion, gripper narrowing
     - ``real/wrappers/transforms/``, applied by the wrapper stack.
   * - Impedance gains that every task shares
     - ``COMPLIANCE_DEFAULTS``; state only your deltas.

Adding a Teleop Device
----------------------

Adding a teleop device touches three layers: a part reads the hardware, a binding
maps its readings to named robot action parts, and a registry entry pairs the
two.

The part belongs in ``rlinf/robotics/parts/teleop/devices.py`` and reads only the
device hardware:

.. code-block:: python

   class Pedal(TeleopPart):
       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           from .readers.pedal import PedalReader

           return PedalReader(port=self._port)

       @property
       def observation_features(self):
           return {"pressed": {"shape": (1,), "dtype": "bool"}}

       def get_observation(self):
           return {"pressed": np.asarray([self._device.is_pressed()])}

``_open`` opens the hardware when the part connects and returns whatever speaks
to it; that handle is ``self._device``. ``__init__`` only records the
declaration, because declaration and construction may happen on different
machines.

The binding goes in ``rlinf/robotics/teleop/bindings.py``. ``PRODUCES`` maps each
action part it fills to what the numbers *mean*, so a device offering a twist to
a joint-space arm is refused rather than obeyed. ``action`` answers everything
about one reading at once, as a ``TeleopAction``:

.. code-block:: python

   class PedalGripperBinding(TeleopBinding):
       PRODUCES = {"end_effector": ActionKind.GRIPPER}

       def action(self, reading, context):
           pressed = bool(reading["pressed"])
           return TeleopAction(
               parts={"end_effector": np.array([-1.0 if pressed else 1.0])},
               driving=pressed,
           )

``driving`` is part of that one answer rather than a second call: a binding
whose answer depends on state it just computed would otherwise have to leave
that state behind, and nothing would enforce the order of the two calls.

Pair the part and binding with an entry builder, then register it in ``DEVICES``
in ``rlinf/envs/real/wrappers/teleop/builder.py``:

.. code-block:: python

   def _pedal(cfg, options, facts):
       return TeleopEntry(
           Pedal(port=options["port"]),
           PedalGripperBinding(),
           drives=options.get("drives"),
       )

   DEVICES = {..., "pedal": _pedal}

The stack builder needs no device-specific branch. A config naming the device
selects the registry entry, while a robot without an ``end_effector`` rejects
the rig at build time.

Adding a Wrapper Instead
------------------------

When the new behavior surrounds a rollout, add a wrapper. Put action replacement
in ``teleop/`` and representation changes in ``transforms/``;
rollout boundaries and scores belong in ``episode/``. A new teleop device is a
part and a binding, as above, not a wrapper. For a new keyboard mode, subclass
``KeyboardSession``.
:doc:`../concepts/realworld_envs` describes both extension points.
