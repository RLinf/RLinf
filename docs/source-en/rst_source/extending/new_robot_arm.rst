Adding a New Robot Arm
=======================

Integrate a new real-world robot arm into RLinf. RLinf supports real-world
reinforcement learning with multiple robot arms, including Franka, GIM_Arm,
XSquare Turtle2, and DOS-W1. This guide walks through the steps to add your
own robot arm.

The RLinf real-world robot system consists of the following components:

- **Hardware Interface**: A hardware class that registers and controls the
  physical robot (joint angles, gripper, cameras).
- **Robot Environment**: A ``gym.Env`` subclass that wraps the hardware and
  provides observation/action spaces, reset, and step logic.
- **Task Environment**: A subclass of the robot environment that defines a
  specific task (reward function, success criteria, task descriptions).
- **Registration**: Expose the new environment in the package so it can be
  selected via configuration.

.. note::

   The robot-related codebase is currently being refactored. This guide
   reflects the current architecture under ``rlinf/envs/realworld/``.
   For the most up-to-date and comprehensively tested reference, use the
   Franka implementation as your template.

Prerequisites
-------------

Before starting, make sure you have:

- RLinf installed and the real-world dependencies available.
- Your robot's Python SDK or communication library installed.
- A working understanding of your robot's joint configuration, action space,
  and observation outputs.

Step 1: Study the Franka Reference Implementation
---------------------------------------------------

The Franka implementation under ``rlinf/envs/realworld/franka/`` is the most
comprehensively tested reference. Start by reading through its structure:

.. code-block:: text

   rlinf/envs/realworld/
   ├── common/                  # Shared utilities (controllers, cameras, etc.)
   ├── franka/                  # Franka reference implementation
   │   ├── hardware.py          # Hardware interface class
   │   ├── franka_env.py        # Base robot environment
   │   └── tasks/               # Task-specific environments
   ├── gim_arm/                 # GIM Arm implementation
   ├── xsquare/                 # XSquare Turtle2 implementation
   ├── dosw1/                   # DOS-W1 implementation
   ├── realworld_env.py         # Base class for real-world environments
   └── __init__.py              # Environment registration

Copy the structure of the ``franka/`` directory as a starting template for
your robot.

Step 2: Define the Hardware Interface
--------------------------------------

Create a new directory for your robot under ``rlinf/envs/realworld/`` (e.g.,
``your_robot/``). Inside it, create a hardware class that handles communication
with the physical robot.

The hardware class is responsible for:

- Initializing the robot connection and homing.
- Reading joint states, end-effector pose, and camera images.
- Sending joint or Cartesian commands to the robot.
- Controlling the gripper (open/close).
- Cleanly shutting down the connection.

.. code-block:: python

   # rlinf/envs/realworld/your_robot/hardware.py
   class YourRobotHardware:
       """Hardware interface for YourRobot arm."""

       def __init__(self, config):
           self.config = config
           # Initialize robot SDK connection
           # self.robot = YourRobotSDK.connect(...)

       def initialize(self):
           """Home the robot and prepare for control."""
           pass

       def get_observation(self):
           """Return joint states, EE pose, and camera images."""
           pass

       def send_action(self, action):
           """Send a joint or Cartesian command to the robot."""
           pass

       def control_gripper(self, width):
           """Open or close the gripper."""
           pass

       def shutdown(self):
           """Cleanly disconnect from the robot."""
           pass

Refer to ``rlinf/envs/realworld/franka/hardware.py`` for the full interface
expected by the framework.

Step 3: Implement the Robot Environment
----------------------------------------

Create a base environment class for your robot that inherits from the
real-world environment base class (see ``rlinf/envs/realworld/realworld_env.py``).

The robot environment must define:

- **Observation space**: Joint positions, velocities, end-effector pose,
  camera images (RGB/D), and any robot-specific sensors.
- **Action space**: Joint position deltas, Cartesian position deltas, or
  gripper commands, depending on your control mode.
- ``reset()``: Home the robot, reset task state, and return initial observation.
- ``step(action)``: Send action to hardware, read new observation, compute
  reward (delegated to task subclass), check termination.

.. code-block:: python

   # rlinf/envs/realworld/your_robot/your_robot_env.py
   import gymnasium as gym
   import numpy as np

   class YourRobotEnv(gym.Env):
       """Base environment for YourRobot arm."""

       def __init__(self, cfg, rank=0, num_envs=1, ret_device="cpu"):
           self.cfg = cfg
           self.rank = rank
           self.num_envs = num_envs
           self.ret_device = ret_device

           # Initialize hardware
           from .hardware import YourRobotHardware
           self.hardware = YourRobotHardware(cfg.hardware)
           self.hardware.initialize()

           # Define spaces
           self._setup_spaces()

       def _setup_spaces(self):
           """Define observation and action spaces."""
           # Example: joint position + velocity + EE pose
           obs_dim = self.cfg.obs_dim
           self.observation_space = gym.spaces.Box(
               low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
           )
           # Example: joint delta actions + gripper
           act_dim = self.cfg.act_dim
           self.action_space = gym.spaces.Box(
               low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
           )

       def reset(self, options=None):
           """Reset the robot and return initial observation."""
           self.hardware.initialize()
           obs = self.hardware.get_observation()
           return obs, {}

       def step(self, action):
           """Execute action and return (obs, reward, terminated, truncated, info)."""
           self.hardware.send_action(action)
           obs = self.hardware.get_observation()
           # Reward and termination are defined by the task subclass
           reward = self._compute_reward(obs, action)
           terminated = self._check_termination(obs)
           truncated = self._check_truncation()
           info = {}
           return obs, reward, terminated, truncated, info

       def _compute_reward(self, obs, action):
           """Override in task subclass."""
           raise NotImplementedError

       def _check_termination(self, obs):
           """Override in task subclass."""
           return False

       def _check_truncation(self):
           """Check episode length limit."""
           return self._step_count >= self.cfg.max_episode_steps

Refer to ``rlinf/envs/realworld/franka/franka_env.py`` for the full
implementation pattern used in production.

Step 4: Implement Task Environments
------------------------------------

Create task-specific environments that inherit from your base robot environment.
Each task defines:

- **Reward function**: Dense or sparse rewards based on task progress.
- **Success criteria**: Whether the task was completed.
- **Task description**: A natural-language string used for VLA models.
- **Reset logic**: Task-specific initial state setup.

.. code-block:: python

   # rlinf/envs/realworld/your_robot/tasks/pick_and_place.py
   from ..your_robot_env import YourRobotEnv

   class YourRobotPickAndPlace(YourRobotEnv):
       """Pick-and-place task for YourRobot."""

       def _compute_reward(self, obs, action):
           """Compute task reward."""
           # Example: reaching reward + grasp reward + placement reward
           return 0.0

       def _check_termination(self, obs):
           """Check if task is complete or failed."""
           return False

       def get_task_description(self):
           """Return natural-language task description for VLA models."""
           return "pick up the block and place it in the bin"

Organize task files under ``rlinf/envs/realworld/your_robot/tasks/``.

Step 5: Register the Environment
---------------------------------

Expose your new robot environments in the package's ``__init__.py`` so they
can be discovered by the configuration system.

.. code-block:: python

   # rlinf/envs/realworld/__init__.py
   from .your_robot.your_robot_env import YourRobotEnv
   from .your_robot.tasks.pick_and_place import YourRobotPickAndPlace

   __all__ = [
       "YourRobotEnv",
       "YourRobotPickAndPlace",
       # ... existing entries
   ]

Step 6: Create Configuration Files
-----------------------------------

Add a YAML configuration file for your robot task under the appropriate
``configs/`` directory. Use an existing Franka config as a template and modify
the robot-specific fields.

.. code-block:: yaml

   # configs/realworld/your_robot_pick_and_place.yaml
   env:
     env_type: "realworld"
     robot_type: "your_robot"
     task: "pick_and_place"
     total_num_envs: 1
     max_episode_steps: 200
     hardware:
       ip: "192.168.1.100"
       # Robot-specific hardware config

Step 7: Test and Validate
--------------------------

Before submitting a PR, validate your implementation:

1. **Hardware smoke test**: Verify the robot connects, homes, and responds to
   commands without the RL loop.
2. **Environment smoke test**: Run a few random-action episodes to verify
   ``reset()`` and ``step()`` work correctly.
3. **Observation/action validation**: Confirm observation and action spaces
   match the actual hardware outputs and inputs.
4. **Safety checks**: Ensure emergency stop, velocity limits, and workspace
   boundaries are enforced.

.. code-block:: python

   def test_your_robot_env():
       """Basic smoke test for YourRobot environment."""
       cfg = get_test_config()
       env = YourRobotPickAndPlace(cfg)
       obs, info = env.reset()
       assert obs is not None
       for _ in range(10):
           action = env.action_space.sample()
           obs, reward, terminated, truncated, info = env.step(action)
           assert obs is not None
           if terminated or truncated:
               obs, info = env.reset()
       env.hardware.shutdown()

Reference Implementations
-------------------------

For working code, refer to these existing implementations:

- **Franka** (most comprehensive): ``rlinf/envs/realworld/franka/``
- **GIM Arm**: ``rlinf/envs/realworld/gim_arm/``
- **XSquare Turtle2**: ``rlinf/envs/realworld/xsquare/``
- **DOS-W1**: ``rlinf/envs/realworld/dosw1/``

Common Issues
-------------

**Robot does not respond to commands**
   Check the hardware SDK connection, IP address, and control mode
   (position vs. velocity vs. torque).

**Observation dimensions mismatch**
   Verify that your ``observation_space`` definition matches the actual
   observation vector returned by the hardware.

**Gripper control not working**
   Some robots require separate gripper SDKs or serial connections. Check
   the hardware class for gripper-specific initialization.

**Camera images not available**
   Ensure camera drivers are installed and the camera is registered in the
   hardware class. Refer to ``rlinf/envs/realworld/common/`` for shared
   camera utilities.

Contributing Back
-----------------

If your robot implementation is general enough to be useful to others, we
welcome a PR to add it to RLinf! Please include:

- The hardware interface and environment code.
- At least one example task.
- A configuration file.
- Documentation (this guide is a good starting point).
- Basic smoke tests (if hardware is available in CI).

Open an issue first to discuss the robot model and ensure there is no
overlapping work in progress.
