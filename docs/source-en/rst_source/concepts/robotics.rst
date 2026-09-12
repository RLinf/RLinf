Robotics Interface
==================

This page explains how task and environment code uses the robotics interface to
inspect, connect, read from, and control a physical robot. It follows one
complete call sequence, then shows how stable part paths support composition and
where an environment converts robot data for a policy. Hardware sessions,
resource ownership, and worker placement are covered in
:doc:`Robotics Architecture <robotics_architecture>`.

The robotics interface exposes an arm, end effector, camera, or mobile base
through a stable name and path. Observations and actions use those same paths,
so task code does not depend on a concrete driver. As long as the paths remain
unchanged, a backend or placement can change without changing the caller.

Use the Interface
-----------------

A normal caller uses the interface in four phases: construct and inspect the
robot, select any parts needed for setup, connect and exchange data, then release
the hardware. The following example keeps those phases visible in one place:

.. code-block:: python

   from rlinf.robotics import Arm, Camera, build_robot

   robot = build_robot("Franka", robot_ip="10.0.0.1", node_rank=1)

   # Safe before the robot is powered or reachable.
   print(robot.describe())

   arm = robot.child("arm", Arm)
   cameras = robot.parts_of_type(Camera)

   robot.connect()
   try:
       if not arm.is_robot_up():
           raise RuntimeError("The arm is connected but not ready.")
       if not all(camera.is_ready() for camera in cameras.values()):
           raise RuntimeError("A camera is not delivering frames.")

       observation = robot.get_observation()
       tcp_pose = observation["arm"]["tcp_pose"]
       gripper_state = observation["end_effector"]["state"]

       applied = robot.send_action(
           {
               "arm": {"tcp_pose": target},
               "end_effector": {"target": width},
           }
       )
   finally:
       robot.disconnect()

``build_robot()`` resolves the registered robot type and returns an unconnected
``Robot``. Construction records hardware arguments and placement, but it does
not import a vendor SDK or open a device. ``describe()`` reads that declaration,
so it can report paths, nodes, and connection ownership before the hardware is
powered or reachable. An environment starts from the scheduler's hardware config
instead of builder arguments, and calls ``from_config()`` on the registered
robot class. It forwards each config field to ``build()`` and places parts on
the environment's node unless the config names another.

Setup code then selects the capabilities it needs. ``child("arm", Arm)`` returns
the part at ``arm`` and checks that it implements ``Arm``; the expected class is
also the static return type seen by an editor. ``parts_of_type(Camera)`` walks
the full composition and returns every matching camera keyed by its dotted path.
Use ``child()`` when a path is part of the task contract, and
``parts_of_type()`` when the task needs a category without depending on camera
names.

``connect()`` opens each owning connection once and rolls back connections
already opened if a later one fails. Once connected, category methods such as
``Arm.is_robot_up()`` and ``Camera.is_ready()`` check whether the devices can be
used; ``clear_errors()`` and ``reset_joint()`` are available for setup outside
the per-step action stream. An arm that accepts ``tcp_pose`` commands also
offers ``move_to()``, which travels to a tool pose through evenly spaced targets
for homing between episodes. An end effector opens and closes fully with
``open()`` and ``close()`` and reports ``is_open``. Each arm class states
``DOF``, the number of joints it drives, so a joint target's width can be
checked before any hardware is opened.

The step loop uses two calls. ``get_observation()`` reads the composed robot once
and returns a nested dictionary whose keys are the part paths.
``send_action()`` accepts the corresponding action branches, dispatches only the
branches present in this step, and returns the action each part actually sent as
``applied``. Finally, ``disconnect()`` closes the owning connections in reverse
order. Keeping it in ``finally`` guarantees cleanup after a failed read or
command, and repeated calls are safe.

The lifecycle is the same for every robot. What changes between robots is the
set and nesting of the paths inside the observation and action dictionaries.

Between episodes a task asks for what it wants rather than scripting it.
``hold()`` stays where the arm is, so it stops chasing the last target a policy
sent. ``clear(distance=..., qpos=...)`` gets clear of whatever the tool is
touching: an arm driven by tool poses rises by the distance, and one driven by
joint targets goes to the configuration named for it. ``go_home(Home(...))``
travels to where the task wants the arm to wait, re-commanding it until it is
within tolerance, and ``unwind(qpos)`` returns the joints to a known
configuration. A task states both spellings of where to wait and each arm uses
the one it can, which is why peg insertion is one task on a Franka and on a
GimArm.

Read the Interface Paths
------------------------

To interpret the dictionaries returned above, start with how a path is formed.
The name assigned during composition becomes a path segment; a part carried by
another part adds the next segment. Comparing Franka and SO-101 shows both
cases.

A single-arm Franka has ``arm`` and ``end_effector`` at the same level because
the arm and Franka Hand open separate endpoints:

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

The different ``via`` values show that each part owns a connection. The two
top-level names therefore address two independently managed parts.

SO-101 represents the other case. Its five arm joints and gripper use one servo
bus, so the end effector appears below the arm and both paths use the same
connection:

.. code-block:: text

   SO101Robot
   └── arm               SO101Arm             node=0     via SO101Arm#1
       └── end_effector  MethodEndEffector    node=0     via SO101Arm#1

The corresponding observation and action follow that structure:

.. code-block:: python

   joint_position = observation["arm"]["arm_joint_position"]
   gripper_state = observation["arm"]["end_effector"]["state"]

   robot.send_action(
       {
           "arm": {
               "joint_position": joint_target,
               "end_effector": {"target": gripper_target},
           }
       }
   )

The nested action is not a second API: it is the same ``send_action()`` call
with one more path segment. ``describe()`` exposes ``via`` to explain why the
segment is nested and which paths share a resource; task code uses the paths and
values. The complete description is diagnostic text, not a serialization format.

A field's name is not the whole contract. Two arms can both report
``tcp_pose`` and mean different things by it, and a task that scores one would
quietly misread the other. ``rlinf/robotics/fields.py`` therefore states what
each canonical name means: ``tcp_pose`` is seven numbers, a position in metres
in the base frame followed by an ``xyzw`` quaternion; ``arm_joint_position`` is
one number per joint in radians. A part that reports those numbers says
nothing extra. A part whose numbers differ declares its own meaning, and
binding a task to it then fails with both meanings named instead of the task
reading a gripper width as part of a rotation:

.. code-block:: text

   RequirementError: CartesianTarget cannot run on ExampleRobot: arm
   (MethodArm) reports 'tcp_pose' as xyz+rpy+gripper_width in m,rad in the
   base frame, not xyz+quat_xyzw in m in the base frame

That refusal is the signal to convert inside the driver, which is where a
vendor's convention belongs. ``MethodArm`` takes ``decode`` and ``encode``
for exactly this: one conversion per field, each way, after which the part
reports the canonical meaning and every task runs on it. Turtle2's controller
speaks Euler angles and carries the gripper's width in the vector it calls
``tcp_pose``; converting there is what lets it run the same tasks as a Franka,
and the width stays the gripper's own state.

Compose Parts Without Changing the Interface
--------------------------------------------

The previous section showed the paths a caller receives. Composition creates
those paths by joining readable parts under names, without changing how the
caller later reads or commands them.

A ``RobotPart`` can enter a robot directly. For example,
``Robot(base=base, arm=arm, end_effector=hand)`` uses the three keyword names as
top-level paths. A bare ``Connection`` is different: it owns a shared hardware
session but has no observation of its own. Select one readable part from it with
``session.part("left")`` before passing the returned ``RobotPart`` to a robot.

A part may also carry readable children backed by the same connection. Those
children appear below it automatically, which is why the SO-101 gripper is
reached as ``arm.end_effector``. When several parts form a reusable assembly,
``PartGroup`` adds one more named level. A dual-arm robot can therefore place
``left`` and ``right`` groups above the same arm interface:

.. code-block:: python

   left_qpos = observation["left"]["arm"]["arm_joint_position"]
   right_gripper = observation["right"]["end_effector"]["state"]

There are no fixed ``arms`` or ``cameras`` slots. A mobile base, lift, head, or
third arm enters under a stable name, while callers continue to use
``get_observation()`` and ``send_action()``. With the public structure settled,
an environment can treat the complete composition as one device boundary.

Use the Same Boundary in an Environment
---------------------------------------

An environment turns the named robot data into the observation and action format
expected by a policy. It should therefore enter through the same complete
interface rather than read a driver beside it: one
``robot.get_observation()`` supplies the state used to build both policy-facing
``state`` and ``frames``, and ``robot.send_action()`` is the only per-step path
back to hardware. A part and any children sharing its connection reuse one
underlying state snapshot during that read.

The typed part references selected before ``connect()`` remain useful for work
outside the step loop. Current real-world environments use ``Arm`` for readiness,
error recovery, and joint reset, and use ``parts_of_type(Camera)`` to find frame
sources. The robot still owns the placement and lifecycle of those cameras; the
environment only consumes their observations.

``TaskEnv`` builds the policy's ``state`` and ``frames`` from that one reading,
as its ``ObservationSpec`` declares them, key by key, with the encoding and
colour order a checkpoint was trained on. The policy's representation lives
there; the robot interface remains named and nested. That separation is also what lets
placement change without reaching task code.

Keep Placement Out of Task Code
-------------------------------

A path identifies a capability, not the process that hosts it. A camera in the
environment process and an arm on another node still appear in the same
observation because placement is recorded on each owning connection, outside
the path. Independent connections are read and commanded concurrently;
branches sharing one connection run in declaration order so the vendor session
is not accessed concurrently.

Task and policy code consequently depends on names and values rather than Ray
actors, RPCs, serial ports, or vendor sessions. Moving a connection changes its
placement declaration, not the calls or data described on this page. One final
boundary remains: a robot knows how hardware behaves, but not what the rollout
is trying to achieve.

Keep Task Logic Separate
------------------------

A part defines how to sense or move hardware; a task defines why those readings
and motions matter. Reward, termination, and task-specific reset behavior
therefore belong to a ``Task``, and what a policy's numbers mean belongs to the
channels of an ``ActionLayout``. ``TaskEnv`` joins a task, a layout, and a
robot, checks that the robot has what they need, and owns the lifecycle. Specialized envs that
have not moved onto it use the same robot calls directly.

The result is two independent contracts: robot paths remain stable across tasks
and placement, while a task can change its policy-facing schema without changing
a driver.

Choose What to Read Next
------------------------

Continue from the boundary you need to change:

- To add a task on supported physical hardware, continue with
  :doc:`New Real-World Tasks <../extending/new_task>`.
- To add one local sensor or actuator, continue with
  :doc:`Adding a Robot <../extending/new_robot>`.
- To understand ``parts`` versus ``children``, shared connections, lifecycle,
  and worker placement, read
  :doc:`Robotics Architecture <robotics_architecture>`.
- To combine operator devices, read
  :doc:`Teleoperation <../guides/teleoperation>`.
