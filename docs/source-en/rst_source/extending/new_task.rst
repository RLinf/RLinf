New Real-World Tasks
====================

This guide explains how to add a real-world task when RLinf already knows how to
connect the physical robot. By the end, you will have a task class, a one-line
registration on the robot, a Gymnasium ID, and a YAML config you can launch.

This guide covers task modules under ``rlinf/envs/real``. To add a task to a
simulator or benchmark, follow :doc:`new_env` instead.

A task says what counts as doing the job: where the target is, what the robot
must report to be scored, how the scene is put back between episodes, and what
each step earns. It never builds an action. How a policy's action reaches the
arm belongs to the robot's action channels, and connecting and placing the hardware
belongs to the robot, so one task runs on every robot that reports what it
needs. If the hardware itself is new, follow :doc:`new_robot` first and return
here once one observation and action can pass through it. If the task also needs
an operator device or wrapper that RLinf does not provide, complete the task
path first and treat that as the separate extension described near the end.

The core workflow has four steps: write the task, register it on a robot,
configure one run, and verify the registration. The sections after that explain
which infrastructure is already provided and cover two optional extensions -- a
new operator device or a new wrapper -- that most tasks do not need.

Core Workflow
-------------

The examples below add a wiping task and run it on the existing Franka support
as ``WipeEnv-v1``. Each step produces an input for the next one: the task is
registered on a robot under an ID, the YAML selects that ID, and the final check
confirms that the whole lookup path is available before hardware opens.

1. Write the Task
~~~~~~~~~~~~~~~~~

Create ``rlinf/envs/real/tasks/wipe.py``. A task that scores where the tool is
builds on ``CartesianTarget``, which already owns the target pose, the workspace
around it, the reward for reaching it, and the routine that returns the arm to
rest. A config dataclass holds the task's values, and the task class adds only
what differs:

.. code-block:: python

   from collections.abc import Sequence
   from dataclasses import dataclass

   from rlinf.envs.real.tasks.cartesian import (
       CartesianTarget,
       FixtureConfig,
       reach_target,
   )
   from rlinf.envs.real.tasks import Evaluation, Needs


   @dataclass
   class WipeConfig(FixtureConfig):
       reward_threshold: Sequence[float] = (0.02, 0.02, 0.02, 0.2, 0.2, 0.2)
       random_xy_range: float = 0.03
       clip_z_range_high: float = 0.05
       contact_force: float = 5.0


   class Wipe(CartesianTarget):
       CONFIG = WipeConfig
       DESCRIPTION = "wipe the surface"

       def requirements(self):
           return {"arm": Needs(observes=frozenset({"tcp_pose", "tcp_force"}))}

       def reset(self, parts, context):
           arm = parts.arm()
           arm.hold()
           arm.clear(distance=0.05, rate_hz=context.rate_hz)
           self.go_to_rest(parts, context)

       def evaluate(self, reading, applied):
           arm = reading.arm()
           reached = reach_target(
               arm["tcp_pose"],
               self.config.target_ee_pose,
               self.config.reward_threshold,
               dense=self.config.use_dense_reward,
           )
           pressing = arm["tcp_force"][2] < -self.config.contact_force
           in_zone = reached.in_zone and pressing
           return Evaluation(reward=float(in_zone), in_zone=in_zone)

The config's fields answer distinct questions. ``target_ee_pose`` is the
fixture pose, ``reward_threshold`` decides when the position error is small
enough, and ``random_xy_range`` sets how far the rest pose varies between
episodes. ``FixtureConfig`` lays the workspace out around the target from the
``clip_*_range`` fields and rests the arm ``clip_z_range_high`` above it; a run
that sets ``ee_pose_limit_min``, ``ee_pose_limit_max`` or ``reset_ee_pose``
outright replaces the derived value. ``DESCRIPTION`` is the language
instruction a run gets unless it sets ``task_description``.

The env calls the task's methods in this order. ``requirements()`` names what
each role's part must report, here ``tcp_force`` beside the ``tcp_pose``
``CartesianTarget`` needs, and is checked before the robot connects.
``workspace`` is handed to the channels, which keep every commanded pose inside
it. ``home()`` runs once after connecting, and ``reset()`` at the start of each
episode, after the channels have applied the arm's compliance gains. The wipe
asks the arm to hold and then to get clear before ``go_to_rest()`` sends it
back to where it waits; peg insertion grips its peg first, in the same place.
A task asks for clearance and never says how to take it, so the same reset
runs on an arm that is driven by joint targets.
``evaluate()`` scores the reading taken after each step. ``in_zone`` counts
toward the ``success_hold_steps`` streak that ends an episode, and the env
subtracts the gripper penalty, so a task only reports what the step earned.

A robot that cannot run the task is refused with the reason, before any
hardware connects:

.. code-block:: text

   RequirementError: Wipe cannot run on FrankaRobot: arm (PoseOnlyArm) does not
   report ['tcp_force']; it reports ['tcp_pose']

Joint-space arms use the same pattern with a joint task. Piper and SO-101 both
run ``JointReach``: ``SO101ReachEnv-v1`` is the whole registration
``class SO101ReachEnv(SO101Env): TASK = JointReach``, and ``PiperReachEnv-v1``
registers the same class on Piper. The preset's action layout turns the
policy's flat action, five absolute joint targets plus one continuous gripper
value on SO-101, into commands for ``arm`` and ``arm.end_effector``.
``examples/embodiment/config/env/so101_reach.yaml`` is the reference run
config.

2. Register It on a Robot
~~~~~~~~~~~~~~~~~~~~~~~~~

The task runs on any robot that meets its requirements; the robot's preset now
decides which one this ID drives. Create ``rlinf/envs/real/franka/wipe.py``:

.. code-block:: python

   from rlinf.envs.real.tasks.wipe import Wipe

   from .base import FrankaEnv, compliance


   class WipeEnv(FrankaEnv):
       TASK = Wipe
       DEFAULTS = {
           "compliance_param": compliance(
               translational_stiffness=800,   # softer, to keep contact
               translational_clip_z=0.02,
           ),
           "action_scale": (0.02, 0.1, 1.0),
       }

``FrankaEnv`` supplies the robot, the action channels, the observation layout,
and the teleop devices; the registration names the task and the settings this
task wants by default. ``action_scale`` limits how far one policy
action moves the tool, and ``compliance_param`` sets the impedance controller
used during that motion. State only the gains that differ: ``compliance()``
merges them onto ``COMPLIANCE_DEFAULTS`` and raises on any gain the controller
does not accept, so a misspelled gain fails at import. A run that sets any of
these keys overrides the default.

A registration only configures. It may not override ``step``, ``reset`` or the
observation, which ``TaskEnv`` runs the same way for every robot; a test
enforces this.

Then give the class the stable ID that configs and datasets will store. Add one
entry to the robot's ``TASKS`` table in ``rlinf/envs/real/franka/__init__.py``:

.. code-block:: python

   from .wipe import WipeEnv

   TASKS = {
       ...
       "WipeEnv-v1": WipeEnv,
   }

``register_tasks`` builds the entry point and registers the id with Gymnasium.
The wrapper stack does not appear here: the action layout declares the wrappers that
fit its action, and ``build_stack`` reads that declaration.

Registering ``WipeEnv-v1`` also registers ``Wipe-v1``, which runs ``Wipe`` on
whichever robot the run is given; a second robot that registers ``Wipe`` joins
it without a new ID.

User configs and dataset metadata store the gym id. Changing it later breaks
those references. Choose the name before collecting data.

3. Add the Env Config
~~~~~~~~~~~~~~~~~~~~~

The ID makes the task discoverable; the YAML now selects it for one run and
supplies the values that vary by experiment. Add a file under
``examples/embodiment/config/env/`` with this structure:

.. code-block:: yaml

   env_type: real
   init_params:
     id: "WipeEnv-v1"      # the gym id you registered
     num_envs: null
   teleop: spacemouse
   override_cfg:
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
     random_xy_range: 0.03
     action_scale: [0.01, 0.1, 1.0]

``env_type: real`` selects RLinf's physical-environment adapter, and
``init_params.id`` selects the Gymnasium task registered in the previous step.
``teleop`` names the operator device for evaluation or data collection.
``override_cfg`` is one flat mapping, and each key goes to the one config that
declares it: the env's for how an episode runs, the action channels' for scales
and gains, and ``WipeConfig`` for the task. A key none of them declares is refused.
Robot addresses and placement remain in the cluster hardware configuration.

4. Check the Registration
~~~~~~~~~~~~~~~~~~~~~~~~~

The core path is now complete from YAML to task class. Before connecting
hardware, import the real-world env package and confirm that the ID resolves:

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # registers every task
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` makes the same assertion for every shipped
task. Add your ID to ``EXPECTED_IDS`` there, and a row to ``TASK_SCHEMAS`` for
the observation and action a policy will be trained on. To run the task without
a Gymnasium ID, compose it by hand as ``TaskEnv(robot, Wipe(), layout,
observation=...)``; the unit tests do this on fake parts. A passing assertion
establishes registration only; run the mock and hardware checks from
:doc:`new_robot` when the task changes the robot-facing observation or action
path.

Reuse Existing Infrastructure
-----------------------------

The four steps above are enough for a task that fits the existing robot and
wrapper contracts. The following responsibilities stay in their current
layers, so task code should call or configure them instead of reimplementing
them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Concern
     - Where it already lives
   * - Connecting and placing hardware
     - ``Robot.connect``; see :doc:`../concepts/robotics`.
   * - Turning a policy action into arm, gripper and hand commands
     - The channels of the robot preset's ``ActionLayout``, such as
       ``PoseDelta``, ``JointPositions`` and the end-effector channels.
   * - Keeping commanded poses in bounds
     - The task's ``workspace``, which the pose channel clips every pose to.
   * - Scoring with a learned reward model
     - ``use_reward_model``, which the env worker sets from the run's
       ``reward`` section.
   * - Teleoperation
     - ``teleop`` in the env config selects one; the wrapper stack builds
       it.
   * - Marking reward or ending an episode by hand
     - ``keyboard_reward_wrapper`` in the env config.
   * - Relative frames, Euler conversion, gripper narrowing
     - ``real/wrappers/transforms/``, applied by the wrapper stack.
   * - Impedance gains that every task shares
     - ``COMPLIANCE_DEFAULTS`` in ``franka/base.py``; state only your deltas
       in the registration's ``DEFAULTS``.

Adding a Teleop Device
----------------------

Stop here unless the task requires an operator device that RLinf does not
already provide. A new device is a separate hardware extension: it can be
reused by several tasks and belongs in one module under
``rlinf/robotics/parts/teleop/``. It answers three questions: how to reach the
hardware, what the operator is doing, and what the robot should do about it.

.. code-block:: python

   @TeleopDevice.register("pedal")
   class Pedal(TeleopDevice):
       PRODUCES = {"end_effector": ActionKind.GRIPPER}

       def __init__(self, port: str) -> None:
           self._port = port

       def _open(self):
           from example_pedal_sdk import PedalClient

           return PedalClient(port=self._port)

       def _release(self, device) -> None:
           device.close()

       @property
       def observation_features(self):
           return {"pressed": {"shape": (1,), "dtype": "bool"}}

       def get_observation(self):
           return {"pressed": np.asarray([self._device.is_pressed()])}

       def action(self, reading, context):
           pressed = bool(reading["pressed"][0])
           return TeleopAction(
               parts={"end_effector": np.array([-1.0 if pressed else 1.0])},
               driving=pressed,
           )

Read the class from selection to sampling. ``register("pedal")`` defines the
config name, and ``PRODUCES`` states that the device supplies a gripper action;
an env that lacks that semantic path rejects it before hardware opens.
``__init__`` records the port because declaration and connection may occur on
different machines. ``_open()`` creates the hardware handle, which becomes
``self._device``, and ``_release(device)`` closes that same handle during
rollback, normal shutdown, or reconnect. If the handle owns a polling thread,
its close path must stop and join the thread.

The remaining methods define one sample. ``observation_features`` declares the
``pressed`` field before connection, and ``get_observation()`` returns a value
with that schema. ``action(reading, context)`` converts the reading into one
``TeleopAction`` containing both the gripper value and ``driving`` state. Keeping
them in one return value avoids a second device read or hidden intermediate
state.

The name in ``register`` is the one the env config spells. Config becomes
constructor arguments through ``from_config``, which by default passes the
device's own options straight through, so the example above needs none. Override
it to read a key from the wider env config, or to choose behaviour from the
robot being driven:

.. code-block:: python

   @classmethod
   def from_config(cls, cfg, options, facts):
       port = options.get("port") or cfg.get("pedal_port")
       if port is None:
           raise ValueError("teleop device 'pedal' requires a port")
       return TeleopEntry(cls(port=port), drives=options.get("drives"))

Finally add ``pedal`` to the robot preset's ``TELEOP`` tuple, which declares
that the env can represent the device's action. That does not register it a second
time: the shared builder resolves the name through ``TeleopDevice``. A robot
without an ``end_effector`` rejects the rig at build time.

A device that needs a second mapping of the same hardware -- joint targets
rather than Cartesian ones, say -- subclasses the first and overrides
``action``, as ``GelloJoint`` does with ``Gello``.

Adding a Wrapper Instead
------------------------

The other optional extension changes the env boundary rather than a hardware
device. When behavior surrounds a rollout, add a wrapper: put action
replacement in ``teleop/``, representation changes in ``transforms/``, and
rollout boundaries or scores in ``episode/``. A new teleop device remains one
device class, as above, while a new keyboard mode subclasses
``KeyboardSession``. :doc:`../concepts/realworld_envs` places both extension
points in the complete runtime flow.
