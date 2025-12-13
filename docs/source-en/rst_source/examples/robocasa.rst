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
--------------------

**RoboCasa Simulation Platform**

- **Environment**: RoboCasa Kitchen simulation environment (built on robosuite)
- **Robot**: Panda manipulator with mobile base (PandaOmron), equipped with parallel gripper
- **Tasks**: 24 atomic kitchen tasks covering multiple categories (excluding NavigateKitchen task that require moving the base)
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

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state / prompt the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group's mean reward.

Dependency Installation
----------------------- 

**1. Prepare Docker**

We started with docker installation.

.. code-block:: bash

   # pull the docker image
   docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

   # enter the docker
   docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --ipc=host \
   --pid=host \
   -v /media:/media \
   -v /sys:/sys \
   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
   -v /etc/localtime:/etc/localtime:ro \
   -v /dev:/dev \
   -e USE_GPU_HOST='${USE_GPU_HOST}' \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   -e NVIDIA_VISIBLE_DEVICES=all \
   -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
   -e ACCEPT_EULA=Y \
   -e PRIVACY_CONSENT=Y \
   --name rlinf_robocasa_pi0 \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

**2. RLinf Installation**

.. code-block:: bash

   cd /workspace
   git clone https://github.com/RLinf/RLinf.git

**3. RoboCasa Installation**

Next we follow the RoboCasa installation.

.. code-block:: bash

   source switch_env pi0

   uv pip install --upgrade robosuite

   git clone https://github.com/RLinf/robocasa.git
   cd robocasa
   uv pip install -e .
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.

   git clone https://github.com/RLinf/openpi.git
   cd openpi
   uv pip install -e .
   
   uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 googleapis-common-protos==1.63.1 mujoco==3.2.6
   uv pip uninstall tensorflow

   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ # Use your own directory

**4. Download Checkpoint**

.. code-block:: bash

   cd /workspace
   # Download the RoboCasa SFT model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/changyeon/pi0_robocasa_100demos_base_pytorch

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   hf download changyeon/pi0_robocasa_100demos_base_pytorch

Now all setup is done, you can start to fine-tune or evaluate the pi0 model with RoboCasa in RLinf framework.