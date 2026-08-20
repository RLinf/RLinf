Adding a Robot
==============

This guide starts with one device in the current process. Once that works, you
will compose it into a robot, share one connection between several parts, and
finally move the device to another node. Keeping that order makes hardware bugs
look like hardware bugs instead of placement bugs.

Before continuing, read the short :doc:`Robotics Model
<../concepts/robotics>` page. If the hardware already exists and you only need a
new reward, reset, or success condition, follow :doc:`new_task` instead; a task
does not require a new robot class.

1. Add One Local Part
---------------------

Start with the smallest useful device. Inherit ``RobotPart`` for a sensor, or
``ControllablePart`` when the device also accepts commands. Do not add cluster
configuration yet: first make the part work in the process where you run the
test.

Every part answers the same three questions: ``_open`` reaches the hardware,
``get_observation`` reads it, and ``_release`` lets it go. Connecting and
disconnecting are handled for you, so a part is written by saying what its
hardware is. Keep the vendor SDK import inside ``_open``: other nodes can then
import the part module without installing the SDK, while the node that opens
the connection still loads it normally.

.. code-block:: python

   import numpy as np

   from rlinf.robotics import ControllablePart


   class ExampleArm(ControllablePart):
       def __init__(self, endpoint: str):
           self.endpoint = endpoint

       def _open(self):
           from example_robot_sdk import Client

           return Client(self.endpoint)

       def _release(self, device) -> None:
           device.close()

       @property
       def observation_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       @property
       def action_features(self) -> dict:
           return {"joint_position": {"shape": (6,), "dtype": "float32"}}

       def reset(self) -> None:
           self._device.move_home()

       def get_observation(self) -> dict[str, np.ndarray]:
           return {"joint_position": self._device.get_joint_position()}

       def send_action(
           self, action: dict[str, np.ndarray]
       ) -> dict[str, np.ndarray]:
           if set(action) != {"joint_position"}:
               raise KeyError("Expected only 'joint_position'.")
           self._device.move_joints(action["joint_position"])
           return action

Try the part locally before composing a robot:

.. code-block:: python

   arm = ExampleArm("tcp://left-arm:5000")
   arm.connect()
   try:
       print(arm.get_observation())
   finally:
       arm.disconnect()

Whatever ``_open()`` returns becomes ``self._device``. ``_release(device)``
receives that same object during cleanup, so it should release the argument it
is given rather than look the object up again on ``self``. Opening in
``connect()`` instead of ``__init__`` is what later allows the declaration and
the hardware to live on different machines.

Some devices need more than an open and close call. An arm that must home before
it becomes usable may override ``connect()`` and ``disconnect()``, but keep both
methods idempotent so rollback and a later reconnect remain safe.

When the device fits ``Camera``, ``EndEffector``, ``MobileBase``, or
``LeggedBase``, inherit that specific interface instead. Compositions and remote
proxies can then retain its device category.

2. Let Several Parts Share One Connection
------------------------------------------

Once one local part works, check whether its hardware session also controls
other components. A socket, CAN bus, or ROS node may drive an arm, a gripper,
and a camera while still needing to open only once.

In that case, define the endpoint's ``exports`` mapping. A key is the local name
accepted by ``connection.export(name)``; its value is the ``RobotPart`` returned
to the caller. This mapping only lists what the connection can provide. It does
not decide where those parts appear in the robot tree. For an arm, expose the
arm itself as ``"arm"``.

.. code-block:: python

   from rlinf.robotics import MethodGripper, RobotPart


   class ExampleArm(ControllablePart):
       ...

       @property
       def exports(self) -> dict[str, RobotPart]:
           return {
               "arm": self,
               "end_effector": MethodGripper(self, state_field="gripper_position"),
           }

Some SDKs expose those capabilities as methods such as ``open_gripper``,
``move_left_arm``, and ``get_camera(id)``. Wrap them with ``MethodGripper``,
``MethodArm``, or ``MethodCamera`` and define the view beside the adapted
methods. The robot composition then sees ordinary parts rather than vendor
method names.

3. Compose Stable Public Names
------------------------------

The previous step listed what the connection can provide. This step chooses the
names that tasks and policies will see in the robot tree. You can compose a
robot directly before adding hardware discovery or YAML: declare the shared
connection once, then select each available part and assign its public name.

.. code-block:: python

   from rlinf.robotics import Robot


   class Bench(Robot):
       ROBOT_TYPE = "Bench"


   connection = ExampleArm.at("tcp://left-arm:5000")
   robot = Bench(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )
   print(robot.describe())
   robot.connect()
   try:
       print(robot.get_observation())
   finally:
       robot.disconnect()

There is no hardware config or discovery here. Those are needed when a config
file must build the robot by type name; a bench script can stay this small.

Choose each part name as if it were a public API field, because it is one. The
names become canonical observation and action paths. Renaming a part later also
changes the schema stored in policies and datasets.

.. code-block:: python

   from rlinf.robotics import Group, Robot


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"


   left = ExampleArm.at("tcp://left-arm:5000")
   right = ExampleArm.at("tcp://right-arm:5000")
   robot = ExampleRobot(
       left=Group(
           arm=left.export("arm"),
           end_effector=left.export("end_effector"),
       ),
       right=Group(
           arm=right.export("arm"),
           end_effector=right.export("end_effector"),
       ),
   )
   robot.connect()
   try:
       observation = robot.get_observation()
       robot.send_action(
           {
               "left": {"arm": {"joint_position": left_target}},
               "right": {"arm": {"joint_position": right_target}},
           }
       )
   finally:
       robot.disconnect()

The paths are exactly the names above, with no implicit ``arms`` or ``cameras``
level. The left arm reads at ``left.arm`` and its gripper at
``left.end_effector``; a camera composed as ``wrist`` reads at ``wrist``.

``Robot`` resets, reads, and commands arms on independent connections in
parallel. A two-arm observation therefore waits for one round trip, and the
robot subclass does not contain its own concurrency code.

4. Move a Working Part to Another Node
--------------------------------------

Only after the local composition works should placement enter the picture. Add
``node_rank`` to the same declaration; the part and the robot tree do not
change:

.. code-block:: python

   connection = ExampleArm.at("tcp://left-arm:5000", node_rank=0)
   robot = Robot(
       arm=connection.export("arm"),
       end_effector=connection.export("end_effector"),
   )
   robot.connect()
   try:
       print(robot.get_observation())
   finally:
       robot.disconnect()

The call records an ``ExampleArm`` declaration for node 0. No worker is created
and no SDK is imported until ``connect()`` runs. The resulting handle appears in
``robot.handles`` and remains there until ``disconnect()`` tears it down.

The declaration is not arm-specific. For example, a camera can remain on the
machine where it is physically connected::

   scene=RealSenseCamera.at(info, node_rank=2)

If one connection backs several components, declare that connection once and
select the exposed parts from it::

   connection = ExampleConnection.at(node_rank=0)
   Group(
       arm=connection.export("left"),
       gripper=connection.export("left_end_effector"),
   )

Underneath this flow, ``spawn()`` performs placement immediately. Call it
directly only outside a robot, for example in a bench script that also takes
care of the handle's lifecycle.

There is no separate worker class to write for the part. Placement synthesizes
one from its class, and ``WorkerGroup`` binds public methods as RPCs. A method
outside the standard part interface remains available through the handle, using
the same expression locally and remotely::

   handle.is_robot_up().wait()[0]
   handle.reset_joint(home_qpos).wait()

5. Build the Robot from Configuration
-------------------------------------

With the parts in place, describe the robot's hardware inputs in a
``RobotConfig`` dataclass and implement ``build()`` to assemble them. Connection
addresses and placement belong here. Reset poses, rewards, and episode horizons
remain in the task config because they describe an episode rather than hardware
discovery.

.. code-block:: python

   from dataclasses import dataclass

   from rlinf.robotics import Group, RobotConfig


   @dataclass
   class ExampleRobotConfig(RobotConfig):
       left_endpoint: str = ""
       right_endpoint: str = ""


   class ExampleRobot(Robot):
       ROBOT_TYPE = "ExampleRobot"

       @classmethod
       def build(
           cls,
           *,
           left_endpoint: str,
           right_endpoint: str,
           node_rank: int = 0,
           **_,
       ) -> "ExampleRobot":
           arms = {}
           for side, endpoint in (
               ("left", left_endpoint),
               ("right", right_endpoint),
           ):
               connection = ExampleArm.at(
                   endpoint,
                   node_rank=node_rank,
                   name=f"ExampleArm-{side}",
               )
               arms[side] = Group(
                   arm=connection.export("arm"),
                   end_effector=connection.export("end_effector"),
               )
           return cls(**arms)

A single-arm variant can reuse this shape and return one group instead of two.
Keep ``build()`` declarative: it should assemble ``PartSpec`` objects, not open
hardware. ``Robot.connect()`` then makes startup all-or-nothing. If a later part
fails, it tears down what it already placed and restores the declaration tree.

.. warning::

   ``build()`` only composes declarations; it does not touch hardware. Call
   ``connect()`` on the result before reading observations or sending commands,
   then call ``disconnect()`` during teardown. Environments make these calls in
   their hardware setup.

Register the Robot
------------------

After defining the config, builder, and discovery logic, register them from the
robot's own module. Registration happens locally rather than through an edit to
a central table.

.. code-block:: python

   from typing import Optional

   from rlinf.robotics import RobotDiscovery, RobotInfo
   from rlinf.scheduler.hardware import HardwareConfig, HardwareResource


   class ExampleRobotDiscovery(RobotDiscovery):
       HW_TYPE = ExampleRobot.ROBOT_TYPE

       @classmethod
       def enumerate(
           cls,
           node_rank: int,
           configs: Optional[list[HardwareConfig]] = None,
       ) -> Optional[HardwareResource]:
           matching = [
               config
               for config in configs or []
               if isinstance(config, ExampleRobotConfig)
               and config.node_rank == node_rank
           ]
           if not matching:
               return None
           return HardwareResource(
               type=cls.HW_TYPE,
               infos=[
                   RobotInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=config)
                   for config in matching
               ],
           )


   ExampleRobot.register(ExampleRobotConfig, ExampleRobotDiscovery)

Put the registration call at the end of the module, after the config and
discovery classes exist. It associates the robot class with its config,
discovery logic, and ``build`` method. Callers can then use
``build_robot("ExampleRobot", ...)`` without importing the concrete class.

If the new hardware is a variant of an existing robot, subclass that robot
instead. For example, ``DualFrankaRobot`` changes ``build_arms`` and ``BACKEND``
on ``FrankaRobot`` but inherits ``build`` and the lifecycle methods.

Import the registration module before constructing ``Cluster``; hardware
discovery cannot identify the robot until that import has run. The registered
hardware policy modules are also passed to node probes, so every node's
configured Python environment must be able to import this one.

Configure the Cluster
---------------------

The final construction input comes from the existing
``cluster.node_groups.hardware`` schema. Its entries are parsed by the
registered config class and passed to the registered builder. Put endpoints and
node ranks in this YAML rather than embedding a particular deployment in
Python:

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement: {}
     node_groups:
       - label: example_robot
         node_ranks: 0
         hardware:
           type: ExampleRobot
           configs:
             - node_rank: 0
               left_endpoint: tcp://left-arm:5000
               right_endpoint: tcp://right-arm:5000

Keep Tasks and Compatibility Separate
-------------------------------------

At this stage, the robot knows how to find and operate its hardware, but it
should not define the task. Put reset behavior, reward, success, truncation, and
Gymnasium spaces in a ``RobotTask`` or the real-world environment, then combine
the task and ``Robot`` through ``RobotTaskEnv``. When an existing policy expects
flat action vectors and ``state``/``frames`` observations, use
``LegacyObservationAdapter`` and ``VectorActionAdapter`` at that boundary.

.. warning::

   When introducing the canonical interface, keep every existing Gym ID, action
   dimension, observation key, camera name, and dataset field unchanged. Use an
   adapter and a regression test to preserve compatibility for trained policies
   and existing datasets.

Check the Composition
---------------------

Before any hardware is involved, ask the robot what it is::

   >>> print(robot.describe())
   ExampleRobot
   ├── left.arm             declared      node=0     via ExampleArm#1
   ├── left.end_effector    declared      node=0     via ExampleArm#1
   ├── right.arm            declared      node=0     via ExampleArm#2
   └── right.end_effector   declared      node=0     via ExampleArm#2

Each row is a part, the node it will run on, and the declaration it comes from.
Two rows sharing a ``via`` share one connection, so they are opened once and
commanded in their declared order rather than concurrently. The node and
ownership columns stay the same after ``connect()`` because the description is
read from the declaration snapshot.

Test the Integration
--------------------

Add a test under ``tests/unit_tests/`` and point the contract classes at what
you wrote. They state the promises the rest of the framework relies on and can
run against the same fake SDKs used by the bench checks:

.. code-block:: python

   from robot_contracts import ConnectionContract, PartContract, RobotContract


   def test_my_arm_conforms():
       PartContract(
           lambda: MyArm("10.0.0.1"),
           action={"joint_position": np.zeros(6)},
       ).assert_kept()


   def test_my_link_conforms():
       ConnectionContract(MyConnection).assert_kept()


   def test_my_robot_conforms():
       RobotContract(lambda: MyRobot.build(robot_ip="10.0.0.1")).assert_kept()

They live in ``tests/robot_contracts``, beside the fake SDKs in
``tests/robot_mocks``, because they check RLinf rather than being part of it.

Between them they connect, read, disconnect, repeat the lifecycle, and
disconnect once more. They compare observation names and shapes with the part's
declaration. When ``PartContract`` receives a sample action, it also checks that
the action is accepted and an unknown field is refused. ``RobotContract``
injects a failure during ``connect()`` and checks that the robot does not report
a partially connected tree.

Each of those is a bug this package has actually had. A failure names the
promise rather than the assertion, and lists all of them at once:

.. code-block:: text

   ConformanceError: MyArm does not keep 2:
     - MyArm: reconnecting raised RuntimeError: threads can only be started
       once; stall recovery closes an endpoint and opens it again
     - MyArm observes tcp_pose with shape (7,), declares (6,)

The rest of the integration is testable the same way. Cover the part contract,
composition paths, handle lifecycle, discovery registration, and the schema
policies expect:

.. code-block:: bash

   pytest tests/unit_tests/test_robotics.py tests/unit_tests/test_real_env.py

These exercise the scheduler import boundary, single-arm and dual-arm
composition, the task and robot split, and the policy-facing schema of every
built-in real-world environment. None of it requires physical hardware.

Run It Against Faked SDKs
~~~~~~~~~~~~~~~~~~~~~~~~~

A part imports its vendor SDK when it opens, never at import time, so a fake in
``sys.modules`` is enough to run the real part classes with nothing on the
other end of the cable. ``tests/robot_mocks`` holds one fake per SDK.

Walk your robot's composition against them:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot --mock \
       --arg robot_ip=10.0.0.1 --arg node_rank=0

It reports what the robot is made of, which connection backs each part and
where it was placed, then reads every part and disconnects. It fails when a
part observes something it never declared, when a value comes back a different
shape from the one declared, when a connection ends up in the tree, or when
anything still claims to be connected afterwards.

The shape check is the quiet one worth having: an env builds its observation
space from what a part declares, so a value one number wider reaches a policy
as data rather than as an error.

Add ``--remote`` to host the parts in scheduler workers instead of this
process. That is what catches a part that cannot be placed at all -- a method
whose name collides with the worker's own, or state that does not survive the
process boundary.

A whole training run works the same way. ``run.sh`` installs the fakes when the
config name contains ``mock``:

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh realworld_mock_sac_cnn

Every shipped robot has one, so the composition, the wrapper stack, the
observation space and the runner all run as they would on a bench:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Robot
     - Config
   * - Franka
     - ``realworld_mock_sac_cnn``
   * - Dual Franka
     - ``realworld_dual_franka_mock_sac_cnn``
   * - GimArm
     - ``gim_arm_mock_sac_cnn``
   * - Turtle2
     - ``realworld_xsquare_turtle2_mock_sac_cnn``
   * - DOSW1
     - ``dosw1_mock_sac_mlp_pick``

Run It Against the Robot
~~~~~~~~~~~~~~~~~~~~~~~~

What is left needs the hardware: timing, calibration, and whatever the device
does that its documentation does not. Once it is powered and reachable, drop
``--mock`` and the same check runs against it:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts MyRobot \
       --arg robot_ip=10.0.0.1 --arg node_rank=1
