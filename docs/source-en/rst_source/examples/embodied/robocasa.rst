RL with RoboCasa Benchmark
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide for reinforcement learning training tasks using the RoboCasa environment in the RLinf framework.
RoboCasa Kitchen focuses on manipulation tasks in kitchen environments, featuring diverse kitchen layouts, objects, and manipulation tasks.
RoboCasa Kitchen combines realistic kitchen environments with diverse manipulation challenges, making it an ideal benchmark for developing generalizable robotic policies.

The main goal is to train vision-language-action models capable of performing the following tasks:

1. **Visual Understanding**: Process RGB images from multiple camera viewpoints.
2. **Language Understanding**: Interpret natural language task instructions.
3. **Manipulation Skills**: Execute complex kitchen tasks such as pick-and-place, opening/closing doors, and appliance control.

Environment
-----------

**RoboCasa Simulation Platform**

- **Environment**: RoboCasa Kitchen simulation environment (built on robosuite)
- **Robot**: Panda manipulator with mobile base (PandaOmron), equipped with gripper
- **Observation**: Multi-view RGB images (robot view + wrist camera) + proprioceptive state
- **Action Space**: 12-dimensional continuous actions

  - 3D arm position delta
  - 3D arm rotation delta
  - 1D gripper control (open/close)
  - 4D base control
  - 1D mode selection (control base or arm)

**Task Categories**

RoboCasa Kitchen provides 24 atomic tasks covering multiple categories (excluding NavigateKitchen atomic task that requires base movement):

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
- **Actions**: 12-dimensional continuous actions
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

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-robocasa
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-robocasa

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env robocasa
   source .venv/bin/activate

Dataset Download
-----------------

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 5GB

Model Download
--------------

.. code-block:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-RoboCasa

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-RoboCasa --local-dir RLinf-Pi0-RoboCasa