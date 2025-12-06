Reinforcement Learning with RoboCasa
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide for reinforcement learning training tasks using the RoboCasa benchmark in the RLinf framework.
RoboCasa is a large-scale robotic learning simulation framework focused on manipulation tasks in kitchen environments, featuring diverse kitchen layouts, objects, and manipulation tasks.

RoboCasa combines realistic kitchen environments with diverse manipulation challenges, making it an ideal benchmark for developing generalizable robotic policies.
The main goal is to train vision-language-action models capable of performing the following tasks:

1. **Visual Understanding**: Process RGB images from multiple camera viewpoints.
2. **Language Understanding**: Interpret natural language task instructions.
3. **Manipulation Skills**: Execute complex kitchen tasks such as pick-and-place, opening/closing doors, and appliance control.

Environment Overview
-------------------

**RoboCasa Simulation Platform**

- **Environment**: RoboCasa Kitchen simulation environment (built on robosuite)
- **Robot**: Panda manipulator with mobile base (PandaOmron), equipped with parallel gripper
- **Tasks**: 24 atomic kitchen tasks covering multiple categories
- **Observation**: Multi-view RGB images (robot view + wrist camera) + proprioceptive state
- **Action Space**: 7-dimensional continuous actions (despite using PandaOmron, base and torso remain fixed)

  - 3D arm position delta (x, y, z)
  - 3D arm rotation delta (rx, ry, rz)
  - 1D gripper control (open/close)

  **Note**: PandaOmron actually accepts 12-dimensional actions ``[3D base, 1D torso, 3D arm position, 3D arm rotation, 2D gripper]``,
  but in RoboCasa Kitchen tasks, the base and torso remain fixed (set to 0), so the policy only needs to output 7-dimensional actions to control the arm and gripper

**Task Categories**

RoboCasa provides diverse atomic tasks organized into multiple categories:

*Door Manipulation Tasks*:

- ``OpenSingleDoor``: Open cabinet or microwave door
- ``CloseSingleDoor``: Close cabinet or microwave door
- ``OpenDoubleDoor``: Open double cabinet doors
- ``CloseDoubleDoor``: Close double cabinet doors
- ``OpenDrawer``: Open drawer
- ``CloseDrawer``: Close drawer

*Pick and Place Tasks*:

- ``PnPCounterToCab``: Pick from counter and place into cabinet
- ``PnPCabToCounter``: Pick from cabinet and place on counter
- ``PnPCounterToSink``: Pick from counter and place in sink
- ``PnPSinkToCounter``: Pick from sink and place on counter
- ``PnPCounterToStove``: Pick from counter and place on stove
- ``PnPStoveToCounter``: Pick from stove and place on counter
- ``PnPCounterToMicrowave``: Pick from counter and place in microwave
- ``PnPMicrowaveToCounter``: Pick from microwave and place on counter

*Appliance Control Tasks*:

- ``TurnOnMicrowave``: Turn on microwave
- ``TurnOffMicrowave``: Turn off microwave
- ``TurnOnSinkFaucet``: Turn on sink faucet
- ``TurnOffSinkFaucet``: Turn off sink faucet
- ``TurnSinkSpout``: Turn sink spout
- ``TurnOnStove``: Turn on stove
- ``TurnOffStove``: Turn off stove

*Coffee Making Tasks*:

- ``CoffeeSetupMug``: Setup coffee mug
- ``CoffeeServeMug``: Serve coffee into mug
- ``CoffeePressButton``: Press coffee machine button

**Observation Structure**

- **Base Camera Image** (``base_image``): Robot left view (128×128 RGB)
- **Wrist Camera Image** (``wrist_image``): End-effector view camera (128×128 RGB)
- **Proprioceptive State** (``state``): 16-dimensional vector containing:

  - ``[0:2]`` Robot base position (x, y)
  - ``[2:5]`` Padding zeros
  - ``[5:9]`` End-effector quaternion relative to base
  - ``[9:12]`` End-effector position relative to base
  - ``[12:14]`` Gripper joint velocities
  - ``[14:16]`` Gripper joint positions

**Data Structure**

- **Images**: Base camera RGB tensor ``[batch_size, 3, 128, 128]`` and wrist camera ``[batch_size, 3, 128, 128]``
- **State**: Proprioceptive state tensor ``[batch_size, 16]``
- **Task Description**: Natural language instructions
- **Actions**: 7-dimensional continuous actions (position, quaternion, gripper)
- **Reward**: Sparse reward based on task completion
