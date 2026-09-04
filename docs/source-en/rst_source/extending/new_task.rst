New Real-World Tasks
====================

This guide explains how to add a real-world task when RLinf already knows how to
connect the physical robot. By the end, you will have a config dataclass, a
small env class, a registered Gymnasium ID, and a YAML config you can launch.

This guide covers task modules under ``rlinf/envs/real``. To add a task to a
simulator or benchmark, follow :doc:`new_env` instead.

For the core task path, leave robot construction, device placement, and
teleoperation unchanged. Targets, compliance settings, success rules, and reset
behavior belong to the task; the existing robot and wrapper stack remain in
place. If the hardware itself is new, follow :doc:`new_robot` first and return
here once one observation and action can pass through it. If the task also needs
an operator device or wrapper that RLinf does not provide, complete the task
path first and treat that as the separate extension described near the end.

The core workflow has five steps: define the task data, bind it to an env class,
register a stable Gymnasium ID, configure one run, and verify the registration.
The sections after that explain which infrastructure is already provided and
cover two optional extensions—a new operator device or a new wrapper—that most
tasks do not need.

Core Workflow
-------------

The examples below add a ``WipeEnv-v1`` task to the existing Franka support.
Each step produces an input for the next one: the dataclass configures the env,
the env class is registered under an ID, the YAML selects that ID, and the final
check confirms that the whole lookup path is available before hardware opens.

For a joint-space arm, use the same sequence with its existing env base.
``SO101ReachEnv-v1`` and ``examples/embodiment/config/env/so101_reach.yaml`` are
the current references for five absolute joint targets plus one continuous
gripper action. The robot still exposes the gripper at
``arm.end_effector``; the env is responsible for presenting the flat six-value
action expected by its policy.

1. Write the Config
~~~~~~~~~~~~~~~~~~~

Create ``rlinf/envs/real/<robot>/<task>.py`` beside the other tasks for that
robot. Inherit its config dataclass and add the fields required by your task:

.. code-block:: python

   import copy
   from dataclasses import dataclass, field

   import numpy as np

   from rlinf.robotics.actions import ActionKind, ActionPart

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

The fields answer distinct questions. ``task_description`` supplies the
language instruction, ``target_ee_pose`` defines the goal, and
``reward_threshold`` decides when each pose error is small enough.
``random_xy_range`` controls reset variation. ``action_scale`` limits how far
one policy action moves the arm, while ``compliance_param`` configures the
controller used during that motion.

State only compliance gains that differ. ``compliance()`` merges them onto
``COMPLIANCE_DEFAULTS`` and raises on any gain the controller does not accept.
A misspelled gain therefore fails while the task config is built instead of
reaching the impedance controller and being ignored.

2. Write the Env
~~~~~~~~~~~~~~~~

The config contains all task values; the env class now attaches those values to
the existing robot-specific execution flow. Set the class's config type first.
For many tasks, that is the entire class:

.. code-block:: python

   class WipeEnv(FrankaEnv):
       CONFIG_CLS = WipeConfig

``CONFIG_CLS`` tells the inherited constructor which dataclass to build from
``override_cfg``. Override a runtime hook only when the task needs different
behavior. ``go_to_rest`` is the common case because homing depends on the task's
end pose. Peg insertion, for example, lifts clear of the slot first; otherwise
the peg catches on the way up:

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

3. Register the Task
~~~~~~~~~~~~~~~~~~~~

Once the class can run the task, give it the stable ID that configs and datasets
will store. Add one entry to the robot's ``TASKS`` table in
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

4. Add the Env Config
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

``env_type: real`` selects RLinf's physical-environment adapter, and
``init_params.id`` selects the Gymnasium task registered in the previous step.
``teleop`` names the operator device for evaluation or data collection.
``override_cfg`` is passed to ``WipeConfig``, so every key there must be a task
config field; robot addresses and placement remain in the cluster hardware
configuration.

5. Check the Registration
~~~~~~~~~~~~~~~~~~~~~~~~~

The core path is now complete from YAML to task class. Before connecting
hardware, import the real-world env package and confirm that the ID resolves:

.. code-block:: python

   from rlinf.envs.real import RealWorldEnv  # registers every task
   from gymnasium.envs.registration import registry

   assert "WipeEnv-v1" in registry

``tests/unit_tests/test_real_env.py`` makes the same assertion for every shipped
task. Add your ID to ``EXPECTED_IDS`` there. A passing assertion establishes
registration only; run the mock and hardware checks from :doc:`new_robot` when
the task changes the robot-facing observation or action path.

Reuse Existing Infrastructure
-----------------------------

The five steps above are enough for a task that fits the existing robot and
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

Finally add ``pedal`` to the environment's ``TELEOP`` tuple, which declares that
the env can represent the device's action. That does not register it a second
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
