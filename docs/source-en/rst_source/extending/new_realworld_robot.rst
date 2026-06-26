.. _tutorial-new-realworld-robot:

Integrating a New Real-World Robot
===================================

This tutorial walks you through adding support for a new real-world robot in
RLinf, using the **SO101 arm** as a concrete, step-by-step example. By the end,
you will understand the general integration pattern that applies to *any*
real-world robot — whether it uses USB serial, CAN bus, ROS, or a custom SDK.

.. note::

   The SO101 is an open-source, 3D-printable 6-DOF manipulator using Feetech
   STS3215 motors. It is the reference low-cost arm in the LeRobot ecosystem.
   RLinf does **not** reimplement the SO101 hardware drivers; instead it wraps
   LeRobot's existing Python API (``SO101Follower``, ``SO101Leader``), so you
   only write the RLinf integration layer.

Prerequisites
-------------

* **Assembled SO101 arm**: Follow the `LeRobot SO101 assembly guide
  <https://huggingface.co/docs/lerobot/so101>`_. Calibrate the arm with
  LeRobot's calibration tool (``lerobot-calibrate``) before using it with RLinf.

* **Working RLinf installation**: See :doc:`/rst_source/start/installation`.
  The ``so101`` extra installs LeRobot and camera dependencies:

  .. code-block:: bash

     bash requirements/install.sh embodied --env so101

* **Familiarity with RLinf concepts**: configs, envs, cluster, hardware; skim
  :doc:`/rst_source/concepts/placement` and :doc:`/rst_source/guides/hetero`
  if needed.

Understanding the Integration Architecture
-------------------------------------------

RLinf's real-world robot stack is a **5-layer architecture**. When you add a new
robot, you create one component in each layer. The SO101 example shows the
simplest possible case: no ROS, no CAN bus SDK, no distributed controller
Worker — just a USB serial connection handled entirely by LeRobot.

.. list-table:: The 5 integration layers
   :header-rows: 1
   :widths: 10 30 60

   * - Layer
     - What you create
     - SO101 example
   * - **1. Robot State**
     - A frozen dataclass holding a snapshot of the robot hardware
     - :class:`~rlinf.envs.realworld.so101.SO101RobotState` (joint
       positions, gripper state)
   * - **2. Inner Environment**
     - A ``gym.Env`` subclass that wraps the hardware SDK
     - :class:`~rlinf.envs.realworld.so101.SO101Env` (wraps
       LeRobot's ``SO101Follower``)
   * - **3. Task + Registration**
     - A task-specific env subclass registered with Gymnasium
     - :class:`~rlinf.envs.realworld.so101.tasks.SO101PickEnv` registered
       as ``SO101PickEnv-v1``
   * - **4. Hardware Registration**
     - Config + info classes + hardware enumeration for the scheduler
     - :class:`~rlinf.scheduler.SO101Config`,
       :class:`~rlinf.scheduler.SO101HWInfo`,
       :class:`~rlinf.scheduler.hardware.robots.so101.SO101Robot`
   * - **5. Wiring + Configs**
     - ``__init__.py`` exports and YAML config files
     - ``realworld/__init__.py``, ``hardware/robots/__init__.py``,
       ``realworld_so101*.yaml``

.. tip::

   **Key principle: leverage existing libraries.** RLinf's philosophy is to
   wrap, not rewrite. The SO101 env imports ``SO101Follower`` from LeRobot;
   the Franka env imports ``libfranka`` via ROS; the GimArm env talks to a
   CAN bus SDK. You only write the RLinf integration code — the hardware
   driver stays in the upstream library.

Step 1: Define the Robot State
-------------------------------

Every real-world robot needs a state dataclass that captures what the hardware
reports at each timestep. This is the **data contract** between the hardware
and the rest of RLinf.

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/so101_robot_state.py

   from dataclasses import asdict, dataclass, field
   import numpy as np

   @dataclass
   class SO101RobotState:
       """State snapshot for the SO101 6-DOF robot arm.

       The arm has 5 revolute joints (shoulder_pan, shoulder_lift,
       elbow_flex, wrist_flex, wrist_roll) plus a 1-DOF gripper.

       Joint positions are in **degrees** (LeRobot convention).
       """

       joint_position: np.ndarray = field(
           default_factory=lambda: np.zeros(5)
       )
       """Arm joint positions in degrees, shape (5,)."""

       joint_velocity: np.ndarray = field(
           default_factory=lambda: np.zeros(5)
       )
       """Arm joint velocities in deg/s, shape (5,)."""

       gripper_position: float = 0.0
       """Gripper position in degrees."""

       gripper_open: bool = False
       """True when the gripper is open."""

       is_connected: bool = False
       """True when the motor bus is actively connected."""

       def to_dict(self):
           return asdict(self)


.. admonition:: General pattern
   :class: seealso

   For **any** robot, define a dataclass with the sensor readings your
   robot provides. Common fields: ``tcp_pose`` (end-effector pose),
   ``joint_position``, ``gripper_position``, ``tcp_force``.
   See :class:`~rlinf.envs.realworld.gim_arm.GimArmRobotState` or
   :class:`~rlinf.envs.realworld.franka.FrankaRobotState` for more
   complex examples with FK-computed TCP poses and Jacobians.

Step 2: Implement the Inner Environment
----------------------------------------

The inner environment is a ``gym.Env`` subclass that wraps your robot's
hardware SDK. Its ``__init__`` signature must accept:

- ``config`` — a typed configuration dataclass
- ``worker_info`` — scheduler-provided ``WorkerInfo`` (may be ``None``)
- ``hardware_info`` — scheduler-provided hardware info (may be ``None``)
- ``env_idx`` — index of this env within the worker process

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/so101_env.py (key parts)

   from lerobot.robots.so_follower import SO101Follower
   from lerobot.robots.so_follower.config_so_follower import (
       SO101FollowerConfig,
   )

   class SO101Env(gym.Env):
       """SO101 joint-space environment powered by LeRobot."""

       def __init__(self, config, worker_info, hardware_info, env_idx):
           self.config = config
           self._state = SO101RobotState()
           self._robot = None

           if not self.config.is_dummy:
               self._setup_hardware()

           self._init_action_obs_spaces()

           if not self.config.is_dummy:
               self._connect_robot()

       def _connect_robot(self):
           """Connect via LeRobot — no distributed Worker needed."""
           robot_cfg = SO101FollowerConfig(
               port=self.config.port,
               id=self.config.calibration_id,
               use_degrees=self.config.use_degrees,
           )
           self._robot = SO101Follower(robot_cfg)
           self._robot.connect(calibrate=True)

       def step(self, action):
           """action: [q1..q5, gripper] in degrees."""
           # LeRobot's SOFollower.send_action() filters incoming keys via
           # ``key.endswith(".pos")`` and silently drops everything else, so
           # *every* motor key — including the gripper — must use the
           # ``<motor>.pos`` form.
           robot_action = {
               "shoulder_pan.pos":   float(action[0]),
               "shoulder_lift.pos":  float(action[1]),
               "elbow_flex.pos":     float(action[2]),
               "wrist_flex.pos":     float(action[3]),
               "wrist_roll.pos":     float(action[4]),
               "gripper.pos":        float(action[5]),
           }
           self._robot.send_action(robot_action)
           self._update_state()
           obs = self._get_observation()
           reward = self._calc_step_reward(obs)
           return obs, reward, terminated, truncated, {}

       def _init_action_obs_spaces(self):
           # Action: [q1..q5, gripper] in degrees (6-D)
           self.action_space = gym.spaces.Box(
               low=np.append(self._joint_limit_low, self.config.gripper_limit_low),
               high=np.append(self._joint_limit_high, self.config.gripper_limit_high),
           )
           # Observation: 5-D arm state + 1-D gripper (+ optional camera frames)
           self.observation_space = gym.spaces.Dict({
               "state": gym.spaces.Dict({
                   "joint_position": Box(-inf, inf, (5,)),
                   "gripper_position": Box(-inf, inf, (1,)),
               }),
           })

.. important::

   **Always support dummy mode.** Set ``config.is_dummy = True`` to skip all
   hardware calls. This lets users train offline, validate configs, and run CI
   without a physical robot. In dummy mode, ``step()`` returns random
   observations from the observation space.

.. admonition:: Design decisions for SO101

   **Why no controller Worker?** LeRobot's ``SO101Follower`` provides a
   synchronous Python API — you call ``send_action()`` and it writes to the
   serial port immediately. More complex robots (Franka + ROS, GimArm + CAN)
   need a distributed :class:`~rlinf.scheduler.Worker` subclass because their
   SDKs are stateful, run in separate processes, or must be placed on specific
   nodes. See :class:`~rlinf.envs.realworld.gim_arm.GimArmController` for the
   Worker pattern.

   **Why degrees?** LeRobot uses degrees natively for Feetech motors. You can
   use radians by setting ``use_degrees: False`` and adjusting joint limits.

Step 3: Define Tasks and Register with Gymnasium
-------------------------------------------------

Create a task — a subclass of your env with a specific goal. Then register it
with Gymnasium so RLinf can create it by ID.

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/tasks/so101_pick.py

   class SO101PickEnv(SO101Env):
       """Reach a target joint configuration."""

       def __init__(self, override_cfg, worker_info=None,
                    hardware_info=None, env_idx=0):
           config = SO101PickConfig(**override_cfg)
           super().__init__(config, worker_info, hardware_info, env_idx)

   @dataclass
   class SO101PickConfig(SO101RobotConfig):
       # End-effector target in metres (3-D); set to None for joint-angle fallback.
       target_ee_pose: tuple = (0.35, 0.0, 0.0)
       reward_threshold_m: float = 0.03       # 3 cm tolerance

.. code-block:: python
   :caption: rlinf/envs/realworld/so101/tasks/__init__.py

   from gymnasium.envs.registration import register
   from .so101_pick import SO101PickEnv

   register(
       id="SO101PickEnv-v1",
       entry_point="rlinf.envs.realworld.so101.tasks:SO101PickEnv",
   )

.. note::

   The entry point can be a **class** (like above) or a **factory function**.
   Use a factory when you need to apply wrappers at construction time. See
   ``rlinf/envs/realworld/franka/tasks/__init__.py`` for factory examples.

Step 4: Register Hardware with the Scheduler
---------------------------------------------

The scheduler needs to know about your robot so it can allocate env workers to
the right nodes. You register three things: a **config**, an **info** class,
and a **hardware** class.

.. code-block:: python
   :caption: rlinf/scheduler/hardware/robots/so101.py

   @dataclass
   class SO101HWInfo(HardwareInfo):
       config: "SO101Config"

   @Hardware.register()
   class SO101Robot(Hardware):
       HW_TYPE = "SO101"

       @classmethod
       def enumerate(cls, node_rank, configs=None):
           # Filter configs for this node, fill from env vars
           robot_configs = RobotAutoConfig.resolve(
               configs, config_cls=SO101Config,
               node_rank=node_rank, count_fields=("port",),
           )
           if robot_configs:
               return HardwareResource(
                   type=cls.HW_TYPE,
                   infos=[SO101HWInfo(type=cls.HW_TYPE, config=c) for c in robot_configs],
               )
           return None

   @NodeHardwareConfig.register_hardware_config("SO101")
   @dataclass
   class SO101Config(HardwareConfig):
       port: str = "/dev/ttyACM0"
       leader_port: Optional[str] = None
       arm_variant: str = "so101"
       calibration_id: str = "default"
       camera_serials: Optional[list[str]] = None
       camera_type: str = "opencv"
       disable_validate: bool = False

.. tip::

   **RobotAutoConfig** fills unset ``None`` fields from same-named uppercase
   env vars. If ``SO101Config.port`` is ``None`` and ``SO101_PORT=/dev/ttyACM0``
   is set, ``RobotAutoConfig.resolve()`` fills it automatically. This lets users
   configure hardware without editing YAML.

Step 5: Wire into the Package
------------------------------

Add import and export entries in two ``__init__.py`` files:

.. code-block:: python
   :caption: rlinf/envs/realworld/__init__.py (add these lines)

   from .so101 import SO101Env, SO101RobotConfig, SO101RobotState
   from .so101 import tasks as so101_tasks

   # Add to __all__:
   "SO101Env", "SO101RobotConfig", "SO101RobotState", "so101_tasks",

.. code-block:: python
   :caption: rlinf/scheduler/hardware/robots/__init__.py (add these lines)

   from .so101 import SO101Config, SO101HWInfo
   # Add to __all__: "SO101Config", "SO101HWInfo",

.. code-block:: python
   :caption: rlinf/scheduler/__init__.py (add these lines)

   from .hardware import SO101HWInfo
   # Add SO101HWInfo to __all__

Step 6: Create Configuration Files
-----------------------------------

Three config files cover the main workflows. All go in
``examples/embodiment/config/``.

**Env config** (``config/env/realworld_so101.yaml``):

.. code-block:: yaml

   env_type: realworld
   group_size: 1
   max_episode_steps: 150
   init_params:
     id: "SO101PickEnv-v1"

**Data collection config** (``config/realworld_so101_collect_data.yaml``):

.. code-block:: yaml

   defaults:
     - env/realworld_so101@env.eval
   cluster:
     num_nodes: 1
     node_groups:
       - label: so101
         node_ranks: 0
         hardware:
           type: SO101
           configs:
             - port: /dev/ttyACM0
               leader_port: /dev/ttyACM1
               node_rank: 0
   env:
     eval:
       override_cfg:
         enable_teleop: True
       data_collection:
         enabled: True
         export_format: "pickle"
         robot_type: "so101"

**RL training config** (``config/realworld_so101_rl.yaml``):

.. code-block:: yaml

   defaults:
     - env/realworld_so101@env.train
     - model/cnn_policy@actor.model
   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-0
       env: 0-0
       rollout: 0-0
   env:
     train:
       override_cfg:
         is_dummy: True   # Set False for real hardware

Step 7: Add Install Support
----------------------------

Add your robot to the install script and ``pyproject.toml``:

.. code-block:: bash
   :caption: requirements/install.sh

   # In SUPPORTED_ENVS=(): add "so101"
   # In install_env_only case): add
   so101)
       uv sync --extra so101 --active $NO_INSTALL_RLINF_CMD
       ;;

.. code-block:: text
   :caption: pyproject.toml

   [project.optional-dependencies]
   so101 = [
       "lerobot @ git+https://github.com/huggingface/lerobot.git@...",
       "gymnasium",
       "opencv-python",
       "numpy",
       "imageio[ffmpeg]",
   ]

   [tool.uv.conflicts]
   # Add to the list:
   { extra = "so101" },


Using the SO101 Arm in RLinf
=============================

Data Collection (Teleoperation)
--------------------------------

SO101 data collection uses **bilateral teleoperation**: a leader arm is
physically moved by a human, and the follower arm mirrors it. Both arms
connect via USB serial.

.. code-block:: bash

   python examples/embodiment/collect_real_data.py \
       --config-name realworld_so101_collect_data

The env creates a ``SO101Leader`` instance that reads the leader arm's joint
positions and forwards them to the follower.
Collected episodes are saved in LeRobot format under the configured ``save_dir``.

SFT Training (Behavior Cloning)
--------------------------------

After collecting demonstrations, fine-tune OpenPI π₀ on the resulting
LeRobot-format dataset.  RLinf does not run SFT through
``train_embodied_agent.py`` — that entry point hosts the SAC actor for
online RL — so a dedicated recipe based on
``examples/sft/train_vla_sft.py`` is the right tool here.  The
:doc:`/rst_source/examples/embodied/so101_sft_openpi` page walks through
the merge → norm-stats → launch sequence end to end and is a good template
when bringing up a new arm: copy ``so101_dataconfig.py`` /
``so101_policy.py`` / ``so101_sft_openpi.yaml`` and adjust ``state_dim``,
``action_dim``, the camera names, and the ``pi0_<arm>`` config name.

RL Training
------------

For online RL, use the RL config with a real or dummy robot:

.. code-block:: bash

   # Dummy validation (no hardware needed)
   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_so101_rl

   # Real hardware (env vars for auto-config)
   export SO101_PORT=/dev/ttyACM0
   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_so101_rl \
       env.train.override_cfg.is_dummy=False \
       env.eval.override_cfg.is_dummy=False \
       cluster.node_groups.0.hardware.type=SO101

The RL algorithm (SAC by default) trains a CNN policy end-to-end on the robot.

Human-in-the-Loop
------------------

RLinf supports human intervention during RL training. The SO101 env can switch
between policy control and human teleop at runtime:

.. code-block:: python

   # In your env step():
   if self.config.enable_teleop and self._leader is not None:
       # Human teleop overrides policy action
       leader_action = self._leader.get_action()
       robot_action = leader_action
   else:
       # Policy action
       self._robot.send_action(robot_action)

This is triggered by setting ``enable_teleop: True`` in the env config or by a
keyboard listener (see ``rlinf/envs/realworld/common/keyboard/``). Intervention
events are recorded for learning from human feedback.

Testing with Dummy Mode
-----------------------

Always validate your integration with dummy mode before deploying to hardware:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_so101_rl \
       max_epochs=1 max_steps=10

This runs a single epoch of training with random observations — no robot
required. If this passes, your integration is structurally correct.

Troubleshooting
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Problem
     - Solution
   * - ``ImportError: cannot import name 'SO101Follower'``
     - Install the ``so101`` extra: ``uv sync --extra so101``
   * - ``Serial port '/dev/ttyACM0' not found``
     - Check ``ls /dev/tty.*``. On Linux, ensure your user is in the
       ``dialout`` group. On macOS, use ``/dev/tty.usbmodem*``.
   * - Robot doesn't move after ``send_action``
     - Calibrate the arm with ``lerobot-calibrate`` first. Check motor
       torque is enabled.
   * - ``gym.error.UnregisteredEnv``
     - The tasks ``__init__.py`` must be imported. Verify
       ``so101_tasks`` is listed in ``realworld/__init__.py``.
   * - Policy outputs NaN or zero actions
     - Check ``state_dim`` and ``action_dim`` in your model config.
       For SO101: ``state_dim: 6`` (5 arm joints + 1 gripper),
       ``action_dim: 6`` (5 arm targets + 1 gripper target).


Adding Your Own Robot: Quick Checklist
=======================================

Use this checklist when integrating a new robot:

.. list-table::
   :header-rows: 1
   :widths: 5 45 50

   * - #
     - File to create/modify
     - What it provides
   * - 1
     - ``realworld/<robot>/*_robot_state.py``
     - Robot state dataclass
   * - 2
     - ``realworld/<robot>/*_env.py``
     - Inner ``gym.Env`` wrapping your hardware SDK
   * - 3
     - ``realworld/<robot>/tasks/*.py`` + ``__init__.py``
     - Task env subclass + ``gym.register()``
   * - 4
     - ``realworld/<robot>/__init__.py``
     - Package exports
   * - 5
     - ``scheduler/hardware/robots/<robot>.py``
     - ``Config``, ``HWInfo``, ``Hardware`` with ``HW_TYPE``
   * - 6
     - ``scheduler/hardware/robots/__init__.py``
     - Re-export config/info classes
   * - 7
     - ``scheduler/__init__.py``
     - Re-export ``HWInfo``
   * - 8
     - ``envs/realworld/__init__.py``
     - Import and ``__all__`` entries
   * - 9
     - ``examples/embodiment/config/env/realworld_<robot>.yaml``
     - Env defaults
   * - 10
     - ``examples/embodiment/config/realworld_<robot>_*.yaml``
     - Experiment configs (collect, SFT, RL)
   * - 11
     - ``requirements/install.sh`` + ``pyproject.toml``
     - Install support
   * - 12
     - ``docker/Dockerfile``
     - ``embodied-<robot>-image`` build stage that runs the install script
   * - 13
     - ``tests/e2e_tests/embodied/<robot>_dummy_*.yaml``
     - Dummy-mode CI smoke test exercising the env end-to-end

That's it — no changes to ``SupportedEnvType``, ``get_env_cls``, or
``action_utils`` are needed for real-world robots (they all reuse
``SupportedEnvType.REALWORLD``).

For more complex robots that need a distributed controller (e.g. Franka with
ROS, GimArm with CAN bus), see :class:`~rlinf.envs.realworld.gim_arm.GimArmController`
for the ``Worker`` subclass pattern.
