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
and any task-specific reset motion. A task in ``rlinf/envs/real/tasks/`` holds
exactly that, written once for every robot that can run it. Peg insertion is a
config dataclass and a class that adds its reset motion to ``CartesianTarget``:

.. code-block:: python

   @dataclass
   class PegInsertionConfig(FixtureConfig):
       random_xy_range: float = 0.05
       clip_z_range_high: float = 0.1
       ...


   class PegInsertion(CartesianTarget):
       CONFIG = PegInsertionConfig
       DESCRIPTION = "peg and insertion"

       def reset(self, parts, context):
           # Grip the peg and take it clear of the slot before homing.
           arm = parts.arm()
           context.action.grasp(parts)
           arm.hold()
           arm.clear(distance=self.config.clearance, qpos=self.config.safe_retract_qpos)
           self.go_to_rest(parts, context)

``PegInsertionConfig`` gives the task one typed source for the target, the
workspace around it, and reset randomization. The task states what it needs
from the robot, an arm that reports ``tcp_pose``, and never builds an action:
``context.action.grasp()`` closes the gripper through the same channel a policy
drives, whichever channels the robot is driven by. It asks for clearance the
same way, without saying how to take it: a Franka rises by the distance, and a
GimArm goes to the configuration named as clear of the hole.

A robot preset such as ``FrankaEnv`` in ``rlinf/envs/real/franka/base.py``
supplies the other half: the robot class, the action channels that turn a
policy's numbers into part commands, and the observation layout the policy
reads. A task
id is then one registration naming the task and the robot-side settings it wants
by default:

.. code-block:: python

   class PegInsertionEnv(FrankaEnv):
       TASK = PegInsertion
       DEFAULTS = {
           "compliance_param": compliance(translational_stiffness=2000),
           "action_scale": (0.02, 0.1, 1.0),
       }

``compliance()`` merges your overrides onto ``COMPLIANCE_DEFAULTS`` and rejects
any gain the controller does not accept. A misspelled key fails here instead of
reaching the impedance controller and being ignored. Peg insertion states one
gain; bin relocation states eleven. A run's own settings override these
defaults.

Keep Hardware in the Robot Descriptor
-------------------------------------

A task changes the goal, reset motion, reward, action limits, and image processing.
The robot's addresses, camera serials and backends, gripper attachment, and
placement describe the rig. Keep those values in the hardware config under
``cluster.node_groups[].hardware.configs``; ``env.train.override_cfg`` and
``env.eval.override_cfg`` accept only environment and task settings. For example,
``enable_camera_player`` controls presentation and remains an env setting.
Omit ``camera_serials`` to discover cameras through the selected backend
(RealSense by default); discovery orders them by serial number. Provide serials
only when the rig needs a particular subset or order. An explicit empty list
selects no cameras on robots that support camera-free operation.

A standalone check uses the same enumeration path as a scheduler launch. The
following example reads SO-101 settings from node-local environment variables,
resolves camera serials, and validates the camera devices before constructing the
environment:

.. code-block:: python

   from rlinf.envs.real.so101 import SO101ReachEnv
   from rlinf.robotics.discovery import RobotDiscovery

   # Set SERIAL_PORT and CALIBRATION_ID for this rig.
   discovery = RobotDiscovery.registry["SO101"].discovery_cls
   resources = discovery.enumerate(node_rank=0, configs=[])
   if resources is None or len(resources.infos) != 1:
       raise ValueError("Configure one SO-101 for this check.")

   env = SO101ReachEnv(
       {"enable_camera_player": False, "target_joint_qpos": [0.0] * 5},
       robot_info=resources.infos[0],
   )
   try:
       observation, info = env.reset()
       print(observation["state"])
   finally:
       env.close()

``enumerate()`` returns hardware descriptors without opening an arm control
connection. Each ``RobotInfo`` contains the resolved typed config in ``config``.
Environment construction connects the robot and may move it to its reset pose;
``reset()`` starts an episode and returns the task observation and info dictionary.
``close()`` releases the robot's connections. The SO-101 bench tool follows this
path too. ``SO101Config.serial_port`` and ``calibration_id`` resolve from
``SERIAL_PORT`` and ``CALIBRATION_ID`` through the shared resolver, so an
unrelated ``PORT=8080`` cannot select a serial device. With configured arms, ``--port``
selects an exact match and retains that arm's calibration; an unmatched port
is an error. Without configured arms, ``--port`` and ``--id`` describe a local
arm. ``--id`` explicitly sets the selected arm's calibration identifier.
In SO-101 hardware YAML entries, replace ``port`` with ``serial_port``.
The bench flags ``--port`` and ``--id`` select the serial port and calibration
identifier respectively.

During a scheduled run, the node probe performs enumeration, worker placement
assigns a ``RobotInfo``, and ``RealWorldEnv`` passes it to the task constructor.
The environment reads its hardware config without modifying it. A task override
such as ``camera_serials`` or ``robot_ip`` is now rejected; move it to the
hardware entry. Dummy environments may omit ``robot_info`` when the hardware
defaults describe the required observation space. To sample another camera
layout or end effector, supply a descriptor for that layout as well. Franka
requires at least one camera even in dummy mode, so its dummy constructors
always need a descriptor with camera serials. These serials can be synthetic
for offline runs; dummy construction does not open or probe devices.

The task dataclass of the one robot that still has its own env,
``DOSW1EnvConfig``, keeps its name. Its hardware counterpart remains in
``rlinf.robotics.robots``. For Turtle2, camera channels move from the task's
``use_camera_ids`` to the hardware field ``camera_ids``, and the arms a run
drives move from ``use_arm_ids`` to the task's ``roles``, which names them
rather than numbering them.

Turtle2, dual-arm Franka, single-arm Franka, Piper, SO-101, and GimArm tasks
run on ``TaskEnv``. A run still
passes one flat ``override_cfg``; each key goes to whichever of three configs
declares it: ``RegisteredTaskEnvConfig`` for how an episode runs, the cameras,
and a reward model; the action channels' config, ``PoseActionConfig`` or
``JointActionConfig``, for action scales, gains, and joint bounds; and the
task's config, such as ``PegInsertionConfig`` or ``JointReachConfig``, for
targets and reward. A preset with settings of its own adds a fourth, such as
GimArm's ``GimArmOptions`` for the controller mode. A key none of them declares
is refused, and a retired key such as ``hand_target_state`` is dropped with a
warning. Piper's
``with_gripper`` hardware field determines whether the action has six joint
values or seven values including the gripper opening.

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

Each task registered on a robot this way also gets a robot-free ID named after
the task, such as ``PegInsertion-v1`` or ``JointReach-v1``. It runs the task on
the robot the run is given, taking that robot's preset from
``robot_info.type``, so a config can name the task and leave the robot to the
cluster's hardware section. The robot-bound IDs stay registered. A dummy run
with no robot descriptor has no robot to choose from and still uses one of them.

A robot-free ID needs one preset per robot to choose from, which does not hold
when a robot offers the same task under two ways of being driven. The dual-arm
Franka registers ``MultiArmTarget`` twice, once commanded by joint targets and
once by tool waypoints, and their action layouts differ in width, so a run must
name the ID its checkpoint was trained against. Both set ``GENERIC_ID = False``,
which leaves the task without a robot-free ID rather than picking one arbitrarily.

Drive Hardware Through the Robotics Interface
---------------------------------------------

Registration determines which env is created; the constructed ``TaskEnv`` then
owns one composed robot for its entire lifetime. It binds the parts the task,
action layout, and observation need, connects the robot during initialization,
and
disconnects it from ``close()``. Each step obtains one nested result from
``robot.get_observation()``, and the channels send named branches through
``robot.send_action()`` rather than reaching around the robot to a driver or
vendor SDK.

This boundary is shared by different hardware layouts. Franka exposes its arm
and end effector as sibling paths because they open separate connections.
SO-101 exposes ``arm.end_effector`` because its gripper is another servo on the
arm bus. Binding finds the end effector either way, so SO-101's joint channel
and gripper channel go out in one command without either of them knowing where
the gripper sits.

One layer decides when an episode ends and whether it succeeded: the task env.
It counts its own steps, truncates on its own horizon, applies the reward scale
and the gripper penalty, and reports in its ``info`` whether the task counted
the step as reaching the goal. ``RealWorldEnv`` reads those rather than
recomputing them, and adds only what batching needs: its ``max_episode_steps``
is a cap on top of the env's horizon, never a replacement for it, and the
metric ``success_once`` comes from what the env reported rather than from
comparing a scaled reward with one.

The step interface is deliberately small, but reset and readiness need category
methods outside that stream. A task reaches them through the parts bound to its
roles, typed by category:

.. code-block:: python

   def home(self, parts, context):
       arm = parts.arm()                  # the Arm filling the "arm" role
       arm.go_home(self.request(context))

``parts.arm()`` returns the ``Arm`` interface bound to the role, and
``parts.end_effector()`` the end effector it carries. The robot remains
responsible for camera placement and lifecycle; the env reads frames from the
same whole-robot observation it builds the state from, so values from one step
are not mixed with a later SDK read.

Run One Task on Different Robots
--------------------------------

Because a task names what it needs rather than which robot provides it, one
task class runs on robots with different kinematics and different channels.
``PegInsertionEnv-v1`` and ``GimArmPegInsertionEnv-v1`` both run
``PegInsertion``:

.. code-block:: python

   class PegInsertionEnv(FrankaEnv):         # Cartesian deltas, hand beside the arm
       TASK = PegInsertion


   class GimArmPegInsertionEnv(GimArmEnv):   # joint targets, gripper on the arm bus
       TASK = PegInsertion
       DEFAULTS = {
           "reset_mode": "joint",
           "safe_retract_qpos": (0.0, -1.5, 1.5, 0.0, 0.0, 0.0),
       }

The Franka preset moves the tool by Cartesian deltas inside the task's
workspace. The GimArm preset sends absolute joint targets, for which a tool
workspace has no meaning, so its channels ignore it. The reward is the same on
both, the tool's distance to the seated peg, because both arms report
``tcp_pose``. Only the reset differs: a GimArm cannot be sent a tool pose, so
``reset_mode="joint"`` retracts and rests through joint configurations instead
of lifting the tool. An option like this names a real difference between arms;
the task never branches on the robot's type.

Settings that belong to one robot rather than to the task or its channels, such
as GimArm's controller mode, live in the preset's ``OPTIONS`` dataclass, which
the preset passes on to ``Robot.from_config``. A preset's default for a setting
its task does not declare is dropped, so a preset's defaults never stop it
running another task.

The unit tests compose ``PegInsertion(PegInsertionConfig(reset_mode="joint"))``
with a joint channel and a binary gripper channel on a fake joint arm, without
a Gymnasium ID. The
same composition tries an existing task on a new robot before that robot has a
preset.

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

Device names come from the ``TeleopDevice`` registry. Devices that retain old
boolean configuration flags declare them in their own ``LEGACY_FLAGS`` mapping;
``TeleopDevice.legacy_flags()`` collects those aliases for the configuration
reader. Existing flags still emit deprecation warnings, and ``teleop`` takes
precedence when configuration layering supplies both forms.

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
     - ``RealWorldEnv``, which adapts one task env to the batched runner:
       tensors, the main and extra camera split, auto-reset, metrics, and
       action chunking. The episode itself belongs to the task env below it.
   * - ``real/task_env.py``
     - ``TaskEnv``, which runs one task on one robot, and
       ``RegisteredTaskEnv``, which builds it from a run's config for a
       Gymnasium ID.
   * - ``real/tasks/``
     - Tasks written once for any robot that meets their requirements, and the
       requirement check that binds them to a robot's parts.
   * - ``real/policy/``
     - What the policy sends and reads: the action channels of each robot's
       layout, and the observation keys, encodings and colour order a
       checkpoint is trained against. A channel that drives a part slower than
       the control period returns its actuation as ``Command.defer``, which the
       layout runs on that role's own queue instead of making the step wait.

Next
----

Continue with the layer you need to extend:

- :doc:`New Real-World Tasks <../extending/new_task>`: follow the step-by-step guide.
- :doc:`Robotics Interface <robotics>`: how the robot underneath is read and
  controlled.
