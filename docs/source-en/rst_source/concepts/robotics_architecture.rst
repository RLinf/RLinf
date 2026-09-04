Robotics Architecture
=====================

This page explains the implementation behind :doc:`Robotics Interface
<robotics>`: how task-facing part paths map to hardware connections, how several
parts share one connection, and how placement and lifecycle management ensure
that every physical resource is opened and released exactly once.

The explanation starts with the relationship between task-facing paths and
driver-side connections. It then assigns responsibilities to the core types,
covers backend and robot registration and the three forms of shared session,
and finishes with placement, pre-connection inspection, and lifecycle. This
order establishes which parts the robot contains before determining which
connection manages each resource.

Start from the Robot's Public Structure
---------------------------------------

The first question is what the task sees and how that view relates to the
connection opened by a driver. Start with the one-to-one case, then introduce a
shared connection; the difference gives ``children`` and ``parts`` their precise
roles.

In the common case, one hardware connection represents one logical part. A
mobile base can enter the robot directly:

.. code-block:: python

   base = ExampleMobileBase(
       "tcp://mobile-base:7000",
       node_rank=0,
       worker_name="ExampleMobileBase-0-0",
   )
   robot = Robot(base=base)

Constructing the base creates an actual but unconnected ``MobileBase``. It
stores the device settings and the connection layer records ``node_rank`` and
``worker_name`` for placement. No SDK is imported and no hardware is opened
until ``robot.connect()``. Because the object is already the logical part a
policy should see, passing it as ``base=base`` adds it directly to the public
tree. The argument name ``base`` becomes its public path.

The one-to-one case makes the public name and the hardware object look
interchangeable. They separate as soon as one hardware session backs several
logical parts. A GimArm drives
its joints and its gripper down the same CAN bus, so one link answers for both,
and a task should still see two parts:

.. code-block:: text

   robot
   └── arm
       └── end_effector

The two views answer different questions:

- The **robot tree** says what the policy can observe or command.
- The **hardware connection** says what must be opened on one node and released
  once.

The code keeps those views in separate mappings. The distinction matters even
when the same names appear in both:

- A ``PartGroup`` or ``Robot`` stores its public tree in ``children``. Each key
  becomes an observation and action path, such as ``left.arm``. Tasks, policies,
  and datasets use these names.
- A ``Connection`` lists the logical parts backed by one hardware session in
  ``parts``. For a readable part, the mapping contains the parts mounted on it;
  for a bare shared session, it contains the parts that can be selected with
  ``part(name)``. These names belong to the driver and do not become robot
  paths by themselves.

Composition is the operation that joins the mappings.
``Robot(arm=connection)`` names a part, and
what rides on that part comes with it, one level down: a GimArm gripper is at
``arm.end_effector`` because it has no link of its own and answers through the
arm's. Nesting tracks the connection, not the bolt pattern. A Franka Hand is
mounted on its arm just as firmly, but it answers on its own endpoint, so it is
named beside the arm rather than under it. These public paths, their placement,
and their ownership are available before ``connect()``, which is why a
composition can be inspected on a machine with no robot attached.

This establishes the normal path: compose a readable part and inherit whatever
it carries. ``connection.part(name)`` is reserved for the other case, where the
link itself is not readable. It selects one part from such a link, as with a
session driving two arms.

The resulting forms can now be compared directly:

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Value
     - Composition
     - Result
   * - A part with nothing riding it, such as a camera
     - ``Robot(wrist=camera)``
     - The part enters the tree under ``wrist``.
   * - A part that carries others, such as an arm whose gripper shares its bus
     - ``Robot(arm=connection)``
     - The arm enters under ``arm``, its gripper under ``arm.end_effector``.
   * - Two parts on their own links, such as an arm and a Franka Hand
     - ``Robot(arm=arm, end_effector=hand)``
     - Each enters under its own name and opens its own connection.
   * - A link that is not a part, such as a two-arm session
     - ``Robot(left=session.part("left"))``
     - The named part enters the tree under ``left``.
   * - An existing subtree
     - ``Robot(left=PartGroup(...))``
     - The group and its named children enter under ``left``.

Every row ends with something readable in the public structure.
``part(name)`` returns a ``RobotPart`` -- there is no intermediate type for a
robot author to construct or annotate. ``PartGroup`` accepts a ``RobotPart`` or
another ``PartGroup``, and rejects a bare ``Connection`` that cannot be read,
naming the keyword that is wrong.

``children`` is one question with one answer, whatever it is asked of. For a
part it is what rides on it; for a ``PartGroup`` it is what the group was
composed of. Walking the tree -- to describe it, to find every camera, to read
it -- therefore never asks which kind of thing it is holding.

Applying the distinction to a robot builder gives one practical rule: name one
thing per connection. Where the gripper rides the arm's bus, naming the arm is
enough:

.. code-block:: python

   class ExampleRobot(Robot):
       @classmethod
       def build_arms(cls, **config):
           return {"arm": ExampleArm(config["robot_ip"], node_rank=config["node_rank"])}

Naming the gripper here as well would put it beside the arm rather than on it,
and would be a second list to keep in step: an arm that decides at run time
whether a gripper is fitted would be composed without it and nothing would
report the omission.

A Franka is the other case. Its hand opens a session of its own, so the robot
names both and neither owns the other:

.. code-block:: python

   class FrankaRobot(Robot):
       @classmethod
       def build_arms(cls, *, robot_ip, node_rank, **config):
           return {
               "arm": cls.declare_arm(robot_ip, node_rank=node_rank, name=...),
               "end_effector": cls.declare_end_effector(
                   robot_ip, node_rank=node_rank, name=..., **config
               ),
           }

The rule is the same in both: name what holds a link, and what rides a link
comes along with whatever holds it.

The mapping a driver returns from ``parts`` therefore says what rides on it and
never itself. Listing itself is refused, because a part does not ride itself and
the tree would have no bottom.

The Core Types
--------------

With the public structure and hardware mapping separated, five types can each
take one responsibility. Read the table from resource ownership through to the
complete robot; each row adds only the behavior required by the next layer.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Type
     - Role
   * - ``Connection``
     - One link to hardware. It records where it runs, opens the vendor session,
       and releases it. A bare connection need not be observable.
   * - ``RobotPart``
     - A readable ``Connection`` with ``get_observation()`` and an
       ``observation_features`` contract.
   * - ``ControllablePart``
     - A ``RobotPart`` that also exposes ``send_action()`` and an
       ``action_features`` contract.
   * - ``PartGroup``
     - A readable, controllable subtree composed from named ``children``. It may
       represent an arm assembly, a torso, or another nested unit.
   * - ``Robot``
     - The outermost ``PartGroup``. It also owns registration and knows which
       node each of its connections runs on.

Device categories add operations that callers can rely on across backends.
Every ``Arm`` reports readiness, clears errors, and supports a named joint-reset
operation or raises ``NotImplementedError`` for that operation. Every
``EndEffector`` reports its canonical ``state`` and accepts a ``target`` action.
``Camera.is_ready()`` distinguishes an open camera from one that is actually
delivering frames.

There is no separate type for a part running elsewhere. A connection given a
``node_rank`` is rebuilt in a worker on that node, and the object you already
hold becomes a view of it: the same object, a synthesized subclass, and every
public call now travelling. ``isinstance`` continues to match the original
driver and its device category. A category such as ``Camera`` or ``MobileBase``
needs nothing registered for placement, because the view is derived from the
driver class itself.

Select an Implementation from Configuration
-------------------------------------------

Composition determines which capabilities exist; configuration still has to
select the concrete implementation for each capability. This section follows
that selection at three levels: a device backend, a standard family builder,
and a complete registered robot.

Suppose a camera config says ``camera_type: zed``. The robot builder should not
need a switch statement that imports every camera driver. Instead, each driver
registers the names that configs use, and the device family resolves the name:

.. code-block:: python

   @Camera.register("example")
   class ExampleCamera(BaseCamera):
       ...


   camera_cls = Camera.backend(camera_info.camera_type)
   camera = camera_cls(camera_info, node_rank=2)

``Connection.register()`` and ``backend()`` are inherited by every device
family, and the registry belongs to the *category* -- ``Camera``, ``Arm``,
``EndEffector`` -- because a config names a kind of device rather than a base
class. Backend names are case-insensitive, and registering the same name for
two classes is an error.

Resolving a class is enough for a direct constructor. A family adds a builder
when its config has a standard shape. ``Camera.of()``
takes a ``CameraInfo`` and reads the backend off it; ``EndEffector.of()`` takes
a name and whatever the arm fitting it can offer; ``Arm.declare()`` maps a
robot's arm settings onto one backend's constructor. In each case the mapping
lives on the driver, next to the constructor it serves, so a new backend
arrives without editing anything that selects one.

Arms are where this matters most, because two of them drive the same hardware.
A Franka is reached through libfranka or through ROS, so both register on
``Arm`` and a robot names one:

.. code-block:: python

   class FrankaRobot(Robot):
       BACKEND = "franka_ros"


   class DualFrankaRobot(FrankaRobot):
       BACKEND = "franky"

Naming the backend is the whole of the swap. Each backend maps the standard arm
settings onto its own constructor in its own ``declare()``, next to the
constructor it serves, so the robot does not know that one stack launches a ROS
package and the other opens a libfranka session. An arm takes arm settings only:
hand a ``declare()`` a ``gripper_type`` and it refuses the call rather than
dropping the setting, because those belong to the end effector composed beside
it.

The built-in Franka Hand is the one device this repository reaches two ways --
over ROS topics, or over its own libfranka session -- so ``FrankaRobot`` keeps a
``HAND_BACKENDS`` mapping from arm backend to hand driver. That choice lives at
the composition root because it is the only place that knows both. A config that
names a driver outright, as ``end_effector_type: franky_gripper``, is taken at
its word.

A driver that supports hardware enumeration can also declare its vendor module
in ``SDK`` and implement ``discover()``. The shared discovery code then reports
a missing SDK clearly and validates configured camera identifiers on the node
that owns them. Vendor imports still belong in ``_open()`` or ``discover()``,
not at module import time.

Backend selection completes one part, but callers may also select a complete
robot composition by name. The two registries therefore name different things:

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - What is named
     - Public API
     - Used for
   * - One device backend
     - ``Camera.register()`` / ``Arm.register()`` and ``backend()``
     - Selecting a driver such as ``realsense`` or ``franky`` from a config.
   * - One complete robot type
     - ``Robot.register_type()`` and ``Robot.of_type()``
     - Selecting a named robot tree and its ``RobotConfig``; registration also
       supplies the standard discovery flow unless a custom class is passed.

Registration associates the robot class, config class, discovery class, and
builder; it does not convert a ``RobotConfig`` instance into builder arguments.
``Robot.of_type()`` and ``build_robot()`` forward the keyword arguments they
receive directly to ``build()``. A registered robot should therefore give its
builder an explicit, documented signature, and the environment that receives a
``RobotInfo`` should perform any required translation in one visible place.

Robot registration completes the hardware side of the selection flow.
Teleoperation uses the same registration style in its own registry.
``TeleopDevice.register()`` names an operator device rather than a robot
component, and the device itself decides how a config entry becomes an instance
through ``from_config()``. The registry lives in ``robotics/parts/teleop``
beside the devices, so a device stays readable on its own and needs no
Gymnasium.

Connect a Shared Hardware Session to the Robot Tree
---------------------------------------------------

Once a backend is selected, its connection must expose any logical parts that
share the resource. There are three cases to distinguish: a readable part can
carry another part, a bare connection can export several selectable parts, and
a process-wide transport can be acquired independently by several connections.

Start with a readable carrier. Define the ``parts`` mapping to say what rides on
a connection. This arm is readable itself and also answers for its gripper, so
it lists the gripper and not itself:

.. code-block:: python

   class ExampleArm(ControllablePart):
       @property
       def parts(self) -> dict[str, RobotPart]:
           return {
               "end_effector": MethodEndEffector(
                   self, state_field="gripper_position"
               ),
           }

``end_effector`` is a name local to the driver. Composing the arm publishes it
one level below whatever the robot calls the arm:

.. code-block:: python

   connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(arm=connection)

   robot.children                     # {"arm": <ExampleArm>}
   robot.child("arm").children        # {"end_effector": <MethodEndEffector>}
   robot.get_observation()["arm"]     # the arm's fields, plus "end_effector"

The keyword arguments become ``robot.children``; the driver's own names appear
beneath the part that carries them. A bare ``Connection`` answers no
``children`` at all, because it has no place in the tree -- its parts are picked
one at a time with ``part(name)``. A ``PartGroup`` has an empty ``parts``
mapping, because nothing rides a group.

The carrier form needs no explicit selection because the arm enters the robot
itself. In the second form, selecting a part with ``part(name)`` also tells a
connection-backed view which connection opens it, so the view declares no lifecycle of its own: no
``_open()`` and no ``connect()`` override. Use ``parts`` for these borrowed
views.

What decides between the two forms is one question the framework asks of the
class, not of the configuration: does it define ``_open()``? A part that does
holds its own link, keeps its own owner and its own ``node_rank``, and is
composed explicitly as another child of the robot or of an assembly
``PartGroup``. A part that does not is adopted by whatever exports it. Both a
wrist camera on USB and a Franka Hand take the first form; a ``MethodEndEffector``
over an arm's own state takes the second.

If reading the shared session itself has no useful meaning, the second form is
to subclass ``Connection`` rather than ``RobotPart`` and select the parts it
backs. A coupled Turtle2 controller follows that form:

.. code-block:: python

   connection = Turtle2Connection(
       50,
       tuple(camera_ids),
       node_rank=0,
       worker_name="Turtle2Connection-0-0",
   )
   robot = Turtle2Robot(
       left=PartGroup(
           arm=connection.part("left"),
           end_effector=connection.part("left_end_effector"),
       ),
       right=PartGroup(
           arm=connection.part("right"),
           end_effector=connection.part("right_end_effector"),
       ),
       wrist=connection.part("wrist_1"),
   )

Every choice points back to the same connection object, so the controller opens
once and is released once.

The third form is not represented by ``parts`` at all. Some resources are
shared by the process rather than by a connection object.
ROS 1 is the case that shows up here: a process gets one node, so an arm and a
hand that both speak ROS end up on the same node whatever they do. Passing a
session from the arm to the hand would encode that as a dependency between two
parts that are otherwise independent, so a ROS-backed part asks for it instead:

.. code-block:: python

   from rlinf.robotics.parts.transports.ros import ROSController


   class ExampleROSGripper(BaseGripper):
       def _open(self):
           self._ros = ROSController.shared()
           self._ros.connect_ros_channel(self._state_channel, JointState, self._on_state)
           return self._ros

``ROSController.shared()`` starts ``roscore`` if none is running, under a file
lock, initializes the node, and hands every later caller the same controller.
Nothing closes it because ``rospy`` has no supported way to bring a node back up
once it is shut down. Topics are the part's own, so joining a session an arm
already opened adds subscriptions rather than competing for anything. These
three forms now identify every owner the placement layer has to open.

Choose Placement Before Opening Hardware
----------------------------------------

Composition identifies each owning connection before hardware opens. Placement
adds a node to that owner without changing the part path or category seen by a
caller. Pass placement alongside the hardware constructor arguments;
``Robot.connect()`` later decides whether to open the existing object locally or
rebuild it in a worker on the selected node:

.. code-block:: python

   arm_connection = ExampleArm(
       "10.0.0.2",
       node_rank=1,
       worker_name="ExampleArm-0-0",
   )
   robot = Robot(
       arm=arm_connection,
       scene=RealSenseCamera(camera_info, node_rank=3),
   )

   print(robot.describe())
   robot.connect()
   try:
       observation = robot.get_observation()
   finally:
       robot.disconnect()

During ``connect()``, the robot opens each distinct ``Connection`` once. With no
``node_rank`` that happens in this process. With one, the connection is rebuilt
inside a scheduler worker on that node and the object in the tree takes on a
synthesized subclass of its own class whose public methods and properties
forward to the worker. The object identity is preserved, although its concrete
class changes while it is connected. Task code therefore never branches on
placement, and ``isinstance`` keeps answering what it did before.

Identity is what preserves resource ownership. Every part answers ``owner``
with the connection opened on its behalf -- itself for an arm holding its own
link, the shared session for a view riding one -- and the robot connects owners
rather than parts. Parts on different connections may run concurrently; parts
sharing one run in declaration order, because vendor sessions are rarely safe
for concurrent access. A Franka arm and its hand hold separate links, so a
whole-robot reading fetches both at once rather than paying for them in turn.

A whole-robot read also defines the consistency boundary used by environments.
``PartGroup.get_observation()`` visits each branch once. While it reads one
part, that part and any children sharing its connection reuse one state
snapshot, so a servo bus is not sampled again for its gripper. An environment
should retain that one result when it constructs policy-facing state and camera
frames rather than mixing it with direct driver or SDK reads.

Independent ownership also means two parts can open from different processes on
one machine. That is exactly right when they address different endpoints, which
is the usual case -- libfranka answers arm control and the hand on separate
ports. It goes wrong when two parts address the same endpoint, and the failure
that follows names a socket rather than the mistake. A part that opens something
exclusive takes a ``DeviceClaim`` keyed by the endpoint, so the second holder is
refused immediately and told which part has the first.

Inspect the Composition Before Connecting
-----------------------------------------

Because composition and placement are both declarations, they can be checked
before lifecycle failures or physical motion obscure a mistake.
``Robot.describe()`` reads that declaration and reports paths, nodes, and
ownership without opening hardware:

.. code-block:: text

   FrankaRobot
   ├── arm           FrankaROSArm         node=1     via FrankaROSArm#1
   └── end_effector  FrankaGripper        node=1     via FrankaGripper#2

Rows sharing ``via`` share one ``Connection``; these two do not, which is the
quickest way to see that the hand can be opened and recovered on its own. After
connecting, a placed part uses a synthesized class name such as
RemoteFrankaROSArm. Its path,
``node``, and ownership stay the same, but the complete output string is not a
stable serialization format; use it as a diagnostic rather than storing or
parsing it.

At present, ``describe()`` focuses on topology, placement, and ownership. It
does not print observation or action feature schemas. Use the conformance checks
in :doc:`../extending/new_robot` to validate those schemas after opening the
connection with a mock SDK or the real device.

Follow the Lifecycle
--------------------

After the declaration is validated, every connection follows the same four
stages. The table gives the caller-visible sequence; the paragraphs below assign
cleanup and rollback to the layer that can actually perform them.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - What happens
   * - Compose
     - Constructors record hardware settings and placement. They do not import
       the vendor SDK or open hardware.
   * - Connect
     - The robot opens each distinct connection once, on the node that
       connection named. A placed one is rebuilt there and the local object
       becomes a view of it.
   * - Use
     - The robot reads, resets, and commands the named tree, concurrently where
       resource ownership allows it.
   * - Disconnect
     - Connections release the exact device returned by ``_open()``. A placed
       one closes on its node before its worker is stopped, and becomes an
       ordinary unopened object again.

The driver participates in the connect and disconnect stages through a matched
pair. ``_open()`` returns the vendor object, and ``_release(device)`` receives that
same object. Cleanup should release the argument rather than look it up again on
``self``.

Implement those two, never ``connect()`` and ``disconnect()``. The public pair
decides *where* a device runs, so a part that overrode them would opt itself
out of ever being placed -- and a thread started after ``super().connect()``
would start on the machine holding the part rather than the one holding the
device. A device category that wraps its drivers has ``_opened()`` and
``_closing()`` for that: ``BaseCamera`` starts and stops its capture loop
there, beside the camera wherever it ended up.

The matched pair handles normal shutdown. Partial failure adds one boundary:
robot startup rolls back the connections that completed successfully if a
later connection fails. A driver's ``_open()`` must still release anything it
acquired before raising, because no completed connection exists for the robot
to close in that case. After fixing the hardware, you can call ``connect()`` on
the same robot again. ``disconnect()`` is idempotent and returns successfully
closed connections to a reconnectable state.

Use Typed Parts for Setup and Reset
-----------------------------------

Once ``connect()`` has opened the owners, setup should return to the public part
categories instead of exposing lifecycle machinery to the task. Common setup
operations belong to device-category contracts, so callers use
the part rather than reaching through its owner. Pass the expected category to
``child()`` to check the composition and retain a precise return type:

.. code-block:: python

   from rlinf.robotics import Arm, Camera

   arm = robot.child("arm", Arm)
   if not arm.is_robot_up():
       raise RuntimeError("The arm is not ready.")
   arm.clear_errors()
   arm.reset_joint(home_qpos)

   cameras = robot.parts_of_type(Camera)
   ready = all(camera.is_ready() for camera in cameras.values())

The same calls work locally and after placement because the remote view remains
a subclass of the original part. A wrong path or category is reported by
``child()`` before a missing method surfaces during an episode. Use ``owner``
to inspect lifecycle ownership; call it directly only for a vendor-specific
operation that genuinely belongs to a shared connection rather than one of the
parts it backs.

Preserve the Import Boundary
----------------------------

The public categories remain usable on their own because the dependency
direction mirrors the runtime boundary described above. Part modules must not
import ``rlinf.scheduler`` or Gymnasium.
``rlinf/robotics/placement/handles.py`` is the bridge, loaded lazily by
``Connection.connect()`` and only for a connection that named a node.

The point is the direction of the dependency. The scheduler is a general
framework and robotics is one extension of it, so the scheduler never imports
this package: it imports the hardware-policy modules a config names, then calls
the discovery classes those modules registered. Gymnasium sits on the other
side, in the env layer that consumes a robot. Only the composition layer --
placement, discovery, the robot builders -- imports either back, which is what
lets a driver be read and tested as hardware code.

Ray is not on that list. It is a base dependency of RLinf, so every machine
running it already has Ray and forbidding the name buys nothing. Nor is this a
promise that importing a part loads nothing: a part may use ``rlinf.utils``
helpers such as ``get_logger``, and those reach further.
``tests/unit_tests/test_robotics.py`` checks both directions.

Find the Implementation
-----------------------

The architecture maps to the source tree in the same order: core contracts,
device families, remote placement, robot composition, and discovery.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Path
     - Contents
   * - ``robotics/parts/base.py``
     - ``Connection``, ``RobotPart``, ``ControllablePart``, ``PartGroup``, and
       the composition checks and driver registry.
   * - ``robotics/parts/arms/``
     - The ``Arm`` category and ``BaseArm``, then the backends that register on
       them: Franky, Franka ROS, GimArm, SO-101, Piper, and the coupled
       controllers.
   * - ``robotics/parts/cameras/``
     - Camera lifecycle and RealSense, ZED, and Lumos implementations.
   * - ``robotics/parts/end_effectors/``
     - Grippers and dexterous hands.
   * - ``robotics/parts/mobility/``
     - The ``MobileBase`` category and mobile-platform drivers.
   * - ``robotics/parts/views.py``
     - ``MethodArm``, ``MethodEndEffector``, and ``MethodCamera`` views over shared
       vendor sessions.
   * - ``robotics/placement/``
     - The worker that hosts a connection, and the view a placed connection
       becomes. Both are synthesized from the driver class.
   * - ``robotics/robot.py``
     - The outer composition, description, and lifecycle.
   * - ``robotics/discovery/``
     - Robot type registration, standard hardware enumeration, environment
       variable completion, and configuration lookup.

Next
----

Continue with the implementation task that brought you to this page:

- :doc:`Adding a Robot <../extending/new_robot>` applies these pieces in order.
- :doc:`Placement <placement>` explains how scheduler resources map onto nodes
  and GPUs.
- :doc:`Teleoperation <../guides/teleoperation>` composes operator devices with
  bindings on the environment side.
