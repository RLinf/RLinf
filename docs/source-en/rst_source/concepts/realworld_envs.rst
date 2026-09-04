Real-World Tasks and Environments
=================================

This page explains how a real-world environment combines one robot with one task
and adds rollout behavior such as operator intervention, manual outcome labels,
and representation transforms. Use the resulting ownership boundaries to place
new behavior before adding another env subclass or wrapper.

The page follows data through the stack. It begins with the task config and env
class, registers that class under a Gymnasium ID, then shows how the constructed
env reads and commands one composed robot. The remaining sections add wrappers
in execution order and separate teleop device I/O, action arbitration, and
episode control. If the robot's named paths are unfamiliar, read
:doc:`Robotics Interface <robotics>` first.

Define Task Data and Behavior
-----------------------------

Begin with the behavior that changes from task to task: targets, success rules,
controller settings, and any task-specific reset motion. In
``rlinf/envs/real/franka/``, the dataclass records those values and the env class
adds behavior that cannot be expressed as data. Both live in one task module
beside the shared ``base.py``:

.. code-block:: python

   @dataclass
   class PegInsertionConfig(FrankaRobotConfig):
       task_description: str = "peg and insertion"
       target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
       random_xy_range: float = 0.05

       def __post_init__(self):
           # Only what differs from the shared impedance gains.
           self.compliance_param = compliance(translational_stiffness=2000)
           ...


   class PegInsertionEnv(FrankaEnv):
       CONFIG_CLS = PegInsertionConfig

       def go_to_rest(self, joint_reset=False):
           # Lift clear of the slot before homing, or the peg catches.
           ...

``PegInsertionConfig`` gives inherited env code one typed source for the target,
randomization, and controller settings. ``CONFIG_CLS`` tells
``PegInsertionEnv`` which config to construct, while ``go_to_rest()`` changes
only the reset sequence that depends on the physical task. ``compliance()`` merges your overrides onto ``COMPLIANCE_DEFAULTS`` and rejects
any gain the controller does not accept. A misspelled key fails here instead of
reaching the impedance controller and being ignored. Peg insertion states one
gain; bin relocation states eleven. Everything else in the config describes the
task itself: poses, reward thresholds, and reset randomization.

Register the Task
-----------------

Once the task class has a complete config and behavior, give callers a stable ID
for constructing it. The task table maps that Gymnasium ID to the env class;
wrapper selection stays on the env and does not become a second registration
concern:

.. code-block:: python

   TASKS = {
       "FrankaEnv-v1": FrankaEnv,
       "PegInsertionEnv-v1": PegInsertionEnv,
       "DualFrankaTcpEnv-v1": DualFrankaTcpEnv,
   }

   _ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

``register_tasks`` turns every row into a Gymnasium entry point and returns the
generated names in ``_ENTRY_POINTS``. User
configs and dataset metadata both store the gym id. Renaming it later leaves
those references stale.

Drive Hardware Through the Robotics Interface
---------------------------------------------

Registration determines which env class is created; the constructed env then
owns one composed robot for its entire lifetime. It builds the arm, end effector,
and cameras, calls ``robot.connect()`` during initialization, and calls
``robot.disconnect()`` from ``close()``. Each step obtains one nested result from
``robot.get_observation()`` and sends named branches through
``robot.send_action()`` rather than reaching around the robot to a driver or
vendor SDK.

This boundary is shared by different hardware layouts. Franka exposes its arm
and end effector as sibling paths because they open separate connections.
SO-101 exposes ``arm.end_effector`` because its gripper is another servo on the
arm bus. ``SO101ReachEnv-v1`` still reads and commands that nested interface,
then converts it to the six-value joint-and-gripper vector its policy expects.

The step interface is deliberately small, but reset and readiness need category
methods outside that stream. Setup code therefore retains typed parts selected
from the same robot:

.. code-block:: python

   from rlinf.robotics import Arm, Camera

   arm = robot.child("arm", Arm)
   cameras = robot.parts_of_type(Camera)

   if not arm.is_robot_up():
       raise RuntimeError("The arm is not ready.")
   arm.reset_joint(reset_qpos)
   ready = all(camera.is_ready() for camera in cameras.values())

``child("arm", Arm)`` verifies the required arm path and returns the ``Arm``
interface used for readiness and reset. ``parts_of_type(Camera)`` returns all
cameras by dotted path so the env can process frames without assuming their
configured names. The robot remains responsible for camera placement and lifecycle. The env may
keep camera references for frame processing, but it does not construct or close
a second object for the same device. One whole-robot observation is also reused
when the env builds its state and frames, so values from one step are not mixed
with a later SDK read.

Organize Wrappers by Responsibility
-----------------------------------

At this point the base env already defines task behavior and hardware I/O.
Wrappers should change only the rollout behavior around that base. Their package
is selected by what they transform:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - What its wrappers change
   * - ``teleop/``
     - The action itself. An operator takes over and their command replaces the
       policy's for as long as they are driving.
   * - ``transforms/``
     - How an observation or action is written down, never what it means. A
       relative frame moves the same motion into end-effector coordinates.
   * - ``episode/``
     - When a rollout starts or ends, and what it scored. These are judgements
       made by the person watching; no sensor reports them.

``build_stack`` reads the env config and applies the declared wrapper families to
the base env. It returns the outermost wrapped env, whichever robot the task runs
on:

.. code-block:: python

   env = build_stack(PegInsertionEnv(...), cfg)

Arbitrate Teleoperation in the Environment
------------------------------------------

The wrapper classification above places action replacement in ``teleop/``. The
:doc:`teleoperation guide <../guides/teleoperation>` covers device selection and
bindings; this section follows the resulting action through the two env-side
operations: first arbitrate between operator and policy, then write the selected
named parts into the env's flat vector.

``TeleopIntervention`` keeps the latest operator action active for a short window
between samples. Without that window, the action could flicker between operator
and policy while the person is still moving. A held control such as a PICO grip
already marks the interval exactly and uses ``timeout = 0``. Any hold after
release would continue to command the robot. Dataset collectors read the action
selected by this arbitration from ``intervene_action``.

Once arbitration selects the action, its shape still has to change. A group
produces named parts, while an environment accepts one flat vector.
``ComposedTeleop`` writes each part into the layout declared by the environment.
Parts that nobody drives retain the values requested by the policy; a posed hand
stays at its last commanded position.

That declaration says what each part means, not only where it sits.
``FrankaEnv`` reads its first six numbers as a twist and ``GimArmEnv`` reads its
first six as joint angles, so a spacemouse can drive one and not the other. The
widths are identical, which is why the width is not what decides. A device whose
commands the robot would misread is refused when the group is built, and an
environment that declares nothing cannot be teleoperated at all.

A single device uses one config key. When several devices share the rig, use a
list:

.. code-block:: yaml

   env:
     eval:
       teleop: spacemouse   # or gello, pico, none
       gello_port: /dev/serial/by-id/...

.. code-block:: yaml

   env:
     eval:
       teleop: [spacemouse, glove]

Keep Device I/O Separate from Action Meaning
--------------------------------------------

The env-side arbitration only works if hardware reading and action mapping remain
separate. Follow one teleop sample from the device, through the group, into the
wrapper; each layer has one job:

- ``robotics/parts/teleop/<device>.py`` talks to one serial device, HID device,
  or headset, and says what its readings mean for named robot action parts. It
  is a ``RobotPart``, so it has the normal connect, observe, and disconnect
  lifecycle and can be placed on another node.
- ``robotics/parts/teleop/base.py`` holds the registry and everything the
  devices share; ``group.py`` merges several devices into one action.
- ``real/wrappers/teleop/builder.py`` resolves the requested names, while
  ``composed.py`` writes their named actions into the env's flat vector.

The first two entries produce named actions; the third is the only layer that
knows how those names fit a particular env vector. Keeping the device layer free of Gymnasium lets you diagnose a cable before
involving a robot. The env layer stays responsible for turning named actions
into the flat vector a particular env accepts, because only it knows that
layout.

You can therefore check a leader arm's wiring without involving a robot:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.gello --port /dev/ttyUSB0

A teleop device *is* a :class:`~rlinf.robotics.parts.base.RobotPart` --
:class:`~rlinf.robotics.parts.teleop.base.TeleopPart` inherits it -- which
gives the device the standard connection lifecycle. Construction remains inert;
``TeleopGroup.connect()`` opens each device when the wrapper stack starts.

What it is not is part of the *robot*. A leader arm never appears in the robot's
composition, and a policy never observes it: it reads the operator, not the
robot. Which parts of the robot's action it fills is the binding's business, and
that lives on the environment side. This boundary also affects placement: the
built-in teleop builder opens devices in the environment process rather than
routing them through ``Robot.connect()``. See :doc:`the teleoperation guide
<../guides/teleoperation>` before placing a standalone device manually.

Separate Episode Control from Teleoperation
-------------------------------------------

Not every operator input belongs in teleop. Marking success, aborting a take, or
switching policies changes episode state rather than the action selected above.
Those wrappers live in
``episode/`` and share :class:`KeyboardSession`. It owns the listener and drops
repeat presses inside a debounce window; on reset, it clears the queue before a
pedal tapped during homing can start the next episode.

A new mode reads ``presses()`` and decides what each key means:

.. code-block:: python

   class KeyboardRLTPolicySwitchWrapper(KeyboardSession):
       def step(self, action):
           obs, reward, terminated, truncated, info = self.env.step(action)
           for key in self.presses():
               if key == "b":
                   self._rlt_switch_flags = True
           info["rlt_switch_flags"] = self._rlt_switch_flags
           return obs, reward, terminated, truncated, info

Where the Code Lives
--------------------

The source tree mirrors the data flow developed above, from task construction
through robot I/O and the three wrapper families:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``real/<robot>/``
     - One module per task, plus ``base.py`` with the machinery they share and
       ``__init__.py`` with the ``TASKS`` table.
   * - ``robotics/parts/teleop/``
     - One module per operator device, plus ``base.py`` with what they share
       and ``group.py`` composing several into one action.
   * - ``robotics/actions.py``
     - What a slot in an action vector means, used by envs and devices alike.
   * - ``real/wrappers/teleop/``
     - Device selection, policy/operator arbitration, flat action layout, and
       the optional direct-streaming path.
   * - ``real/wrappers/transforms/``
     - Relative frames, quaternion-to-Euler, gripper narrowing.
   * - ``real/wrappers/episode/``
     - Keyboard sessions: reward and done, start and end, eval control, policy
       switch, leader-follower.
   * - ``real/wrappers/__init__.py``
     - The stack builders that compose the three families.
   * - ``real/registry.py``
     - ``task_factory`` and ``register_tasks``.
   * - ``real/env.py``
     - ``RealWorldEnv``, the vectorized env the framework instantiates from
       ``env_type: real``.
   * - ``real/task_env.py``
     - ``RobotTask`` and ``RobotTaskEnv`` define the boundary between task logic
       and hardware.

Next
----

Continue with the layer you need to extend:

- :doc:`New Real-World Tasks <../extending/new_task>`: follow the step-by-step guide.
- :doc:`Robotics Interface <robotics>`: how the robot underneath is read and
  controlled.
