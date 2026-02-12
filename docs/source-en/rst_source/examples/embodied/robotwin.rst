RL with RoboTwin Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to launching and managing 
**Vision-Language-Action Models (VLAs)** training tasks within the RLinf framework,
focusing on finetuning a VLA model for robotic manipulation in the RoboTwin environment.

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via PPO with environment feedback.

RoboTwinEnv Environment
--------------------------

**RoboTwinEnv Environment**

- **Environment**: RLinf framework provides the RoboTwinEnv environment for reinforcement learning training based on the RoboTwin 2.0 simulation platform.
- **Task**: Control a robotic arm to perform various manipulation tasks. RLinf RoboTwinEnv currently supports **46 tasks**, and users can select tasks for training as needed.

  **Placement Tasks**

  - ``adjust_bottle``: Pick up the bottle on the table headup with the correct arm.
  - ``place_a2b_left``: Use appropriate arm to place object A on the left of object B.
  - ``place_a2b_right``: Use appropriate arm to place object A on the right of object B.
  - ``place_bread_basket``: If there is one bread on the table, use one arm to grab the bread and put it in the basket, if there are two breads on the table, use two arms to simultaneously grab up two breads and put them in the basket.
  - ``place_bread_skillet``: Use one arm to grab the bread on the table and put it into the skillet.
  - ``place_burger_fries``: Use dual arm to pick the hamburg and frenchfries and put them onto the tray.
  - ``place_can_basket``: Use one arm to pick up the can, put it into the basket, and use another arm to lift the basket.
  - ``place_cans_plasticbox``: Use dual arm to pick and place cans into plasticbox.
  - ``place_container_plate``: Place the container onto the plate.
  - ``place_empty_cup``: Use an arm to place the empty cup on the coaster.
  - ``place_mouse_pad``: Grab the mouse and place it on a colored mat.
  - ``place_object_basket``: Use one arm to grab the target object and put it in the basket, then use the other arm to grab the basket, and finally move the basket slightly away.
  - ``place_object_stand``: Use appropriate arm to place the object on the stand.
  - ``place_phone_stand``: Pick up the phone and put it on the phone stand.
  - ``place_shoe``: Use one arm to grab the shoe from the table and place it on the mat.
  - ``place_dual_shoes``: Use both arms to pick up the two shoes on the table and put them in the shoebox, with the shoe tip pointing to the left.

  **Pick Tasks**

  - ``pick_dual_bottles``: Pick up one bottle with one arm, and pick up another bottle with the other arm.
  - ``pick_diverse_bottles``: Pick up one bottle with one arm, and pick up another bottle with the other arm.
  - ``move_can_pot``: There is a can and a pot on the table, use one arm to pick up the can and move it to beside the pot.
  - ``move_pillbottle_pad``: Use one arm to pick the pillbottle and place it onto the pad.
  - ``move_playingcard_away``: Pick up the playing card and move it away from the table.
  - ``move_stapler_pad``: Use appropriate arm to move the stapler to a colored mat.
  - ``grab_roller``: Use both arms to grab the roller on the table.
  - ``lift_pot``: Use arms to lift the pot.
  - ``put_bottles_dustbin``: Use arms to grab the bottles and put them into the dustbin to the left of the table.

  **Stacking Tasks**
  
  - ``stack_blocks_two``: Stack the green block on the red block.
  - ``stack_blocks_three``: Stack the blue block on the green block, and then stack the green block on the red block.
  - ``stack_bowls_two``: Stack the two bowls on top of each other.
  - ``stack_bowls_three``: Stack the three bowls on top of each other.

  **Ranking Tasks**
  
  - ``blocks_ranking_rgb``: Arrange the blocks in the order of red, green, and blue from left to right.
  - ``blocks_ranking_size``: Arrange the blocks from largest to smallest, from left to right.

  **Tool Use & Interaction Tasks**
  
  - ``click_alarmclock``: Click the alarm clock's center of the top side button on the table.
  - ``click_bell``: Click the bell's top center on the table.
  - ``beat_block_hammer``: Grab the hammer and hit the block.
  - ``open_microwave``: Use one arm to open the microwave.
  - ``press_stapler``: Use one arm to press the stapler.
  - ``stamp_seal``: Grab the stamp and stamp onto the specific color mat.
  - ``turn_switch``: Use the robotic arm to click the switch.

  **Handover Tasks**
  - ``handover_block``: Use the left arm to grasp the red block, handover it to the right arm, and then place it on the blue pad.
  - ``handover_mic``: Use one arm to grasp the microphone and handover it to the other arm.

  **Pouring, Dumping & Shaking Tasks**

  - ``shake_bottle``: Shake the bottle with proper arm.
  - ``shake_bottle_horizontally``: Shake the bottle horizontally with proper arm.
  - ``dump_bin_bigbin``: Grab the small bin and pour the balls into the big bin.

  **Hanging & Special Tasks**

  - ``hanging_mug``: Use the left arm to pick up the mug and adjust its pose, then use the right arm to pick it up again and hang it onto the rack.
  - ``scan_object``: Use one arm to hold the scanner, use the other arm to hold the object, and complete the scanning.
  - ``rotate_qrcode``: Pick up the QR code board and rotate it so that the QR code faces the robot.

  .. note::
     Currently four tasks are not yet supported:  ``place_fan``, ``open_laptop``, ``place_object_scale``, and ``put_object_cabinet``. Additionally, dense reward functions are still under development and will gradually be extended to all tasks.

- **Observation**: The observation returned by RLinf RoboTwinEnv environment is a dictionary (dict) containing the following fields:

  - ``images``: Head camera RGB images

    - **Type**: ``torch.Tensor``
    - **Shape**: ``[batch_size, 224, 224, 3]``
    - **Data Type**: ``uint8`` (0-255)
    - **Description**: Head camera images processed with center crop, one image per environment

  - ``wrist_images``: Wrist camera RGB images (optional)
  
    - **Type**: ``torch.Tensor`` or ``None``
    - **Shape**: ``[batch_size, num_wrist_images, 224, 224, 3]`` (if exists)
    - **Data Type**: ``uint8`` (0-255)
    - **Description**: May contain left wrist camera (``left_wrist_image``) and/or right wrist camera (``right_wrist_image``) images, or ``None`` if the task does not require wrist images

  - ``states``: Proprioception information

    - **Type**: ``torch.Tensor``
    - **Shape**: ``[batch_size, 14]``
    - **Data Type**: ``float32``
    - **Description**: Contains end-effector pose information (position and orientation), 14 dimensions total, corresponding to ``proprio_dim=14``

  - ``task_descriptions``: Task description text

    - **Type**: ``List[str]``
    - **Length**: ``batch_size``
    - **Description**: Natural language task descriptions for each environment, e.g., "What action should the robot take to place the empty cup on the coaster?"

- **Action Space**: 14-dimensional continuous action space

  - **Type**: ``torch.Tensor`` or ``numpy.ndarray``
  - **Shape**: ``[batch_size, action_dim]`` or ``[batch_size, horizon, action_dim]``, where ``action_dim=14``
  - **Data Type**: ``float32``
  - **Action Components**:

    - End-effector 3D position control (x, y, z): 3 dimensions
    - 3D rotation control (roll, pitch, yaw): 3 dimensions
    - Gripper control (open/close): 1 dimension
    - Joint position control: 7 dimensions
    - **Total**: 14 dimensions

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

RLinf provides a pre-configured RoboTwin environment Docker image that includes all required dependencies and can be used directly, **skipping all subsequent installation steps**.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-robotwin
      # If you need to download the image faster in China, you can use:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-robotwin

.. note::
   The Docker image includes:
   
   - RLinf RoboTwin environment dependencies
   - Compatibility patches applied
   - Support for OpenVLA-OFT, OpenPI models

   **After using the Docker image, you can directly proceed to the** `RoboTwin Repository Clone and Assets Download`_ **, ** `Model Download`_ **and** `Running Scripts`_ **sections, skipping all subsequent installation steps.**

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command. 
Replace the ``--model openvla-oft`` parameter with the corresponding model name (``openvla-oft`` or ``openpi``) based on the model you want to train:

.. code:: bash

   # To speed up dependency installation in China, you can add `--use-mirror` to the install.sh command below

   bash requirements/install.sh embodied --model openvla-oft --env robotwin
   source .venv/bin/activate

This script will automatically:

- Install RLinf RoboTwin environment dependencies
- Apply RoboTwin compatibility patches (fixing compatibility issues between sapien and mplib)
- Install dependencies for the corresponding VLA model

RoboTwin Repository Clone and Assets Download
---------------------------------------------

RoboTwin Assets are asset files required by the RoboTwin environment and need to be downloaded from HuggingFace.

.. code-block:: bash

   # 1. Clone RoboTwin repository
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   
   # 2. Download and extract Assets files
   bash script/_download_assets.sh


Model Download
-----------------------

Before starting training, you need to download the corresponding SFT model:

.. code-block:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

After downloading, ensure that the model path is correctly specified in the configuration yaml file (``actor.model.model_path``).

Running Scripts
-------------------

Please ensure that you have activated the correct Python virtual environment (venv) before running the commands below.
If you are using the official Docker image, you need to switch to the ``openvla-oft`` environment by running ``source switch_env openvla-oft``.

**1. Key Parameter Configuration**

Using the OpenVLA-OFT model as an example, the following key parameters need to be configured in ``actor.model``:

.. code-block:: yaml

   actor:
     model:
       model_path: "/path/to/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup"  # SFT model path
       model_type: "openvla_oft"                                             # Model type set to openvla_oft
       implement_version: "official"                                          # openvla_oft implementation version (RLinf OpenVLA-OFT model implementation integrates the official OFT version and RLinf SFT fine-tuned version, RoboTwin environment uses the official version)
       action_dim: 14                                                        # RoboTwin action dimension (14D)
       use_proprio: True                                                     # Whether to use proprioception information
       proprio_dim: 14                                                       # Proprioception dimension
       use_film: False                                                       # Whether to use FiLM layer
       num_images_in_input: 1                                                # Number of input images
       num_action_chunks: 25                                                 # Number of action chunks
       unnorm_key: "place_empty_cup"                                         # Action normalization key (must match the unnorm_key used during SFT training)


**2. Environment Configuration**

In the environment configuration file, the following key parameters need to be set:

.. code-block:: yaml

   env/train: robotwin_place_empyt_cup
   env/eval: robotwin_place_empyt_cup
   
   # In env/train/robotwin_place_empyt_cup.yaml:
   env_type: robotwin
   assets_path: "/path/to/robotwin_assets"
   
   task_config:
     task_name: place_empty_cup  # or other task names
     step_lim: 200
     embodiment: [piper, piper, 0.6]
     camera:
       head_camera_type: D435
       wrist_camera_type: D435
       collect_head_camera: true
       collect_wrist_camera: false

**3. Configuration Files**

Using **OpenVLA-OFT** model with **GRPO** algorithm as an example, the corresponding configuration file is:

- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/robotwin_place_empyt_cup_grpo_openvlaoft.yaml``

**4. Launch Command**

After selecting the configuration, run the following command to start training:

.. code-block:: bash
   
   # Set ROBOT_PLATFORM environment variable
   export ROBOT_PLATFORM=ALOHA
   # Set ROBOTWIN_PATH environment variable
   export ROBOTWIN_PATH=/path/to/RoboTwin

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, training OpenVLA-OFT model with GRPO in the RoboTwin environment:

.. code-block:: bash

   # Set ROBOT_PLATFORM environment variable
   export ROBOT_PLATFORM=ALOHA
   # Set ROBOTWIN_PATH environment variable
   export ROBOTWIN_PATH=/path/to/RoboTwin

   bash examples/embodiment/run_embodiment.sh robotwin_place_empyt_cup_grpo_openvlaoft

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Video Generation**

Videos from training and evaluation processes are automatically saved. Configuration:

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train  # Training videos
     # or
     video_base_dir: ${runner.logger.log_path}/video/eval   # Evaluation videos

Evaluation Results
~~~~~~~~~~~~~~~~~~~

.. list-table:: **Evaluation results of OpenVLA-OFT models on six RoboTwin tasks**
   :header-rows: 1

   * - Model
     - beat_block_hammer
     - pick_dual_bottles
     - place_empty_cup
     - move_can_pot
     - lift_pot
     - handover_block
     - Average
     - Δ Avg.

   * - OpenVLA-OFT (SFT)
     - |huggingface| `10.15% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer>`_
     - |huggingface| `20.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles>`_
     - |huggingface| `75.78% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`_
     - |huggingface| `9.37% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot>`_
     - |huggingface| `3.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot>`_
     - |huggingface| `28.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block>`_
     - 24.48%
     - ---

   * - OpenVLA-OFT (RLinf-GRPO)
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer>`__
     - |huggingface| `92.96% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles>`__
     - |huggingface| `94.53% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup>`__
     - |huggingface| `83.59% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block>`__
     - **84.63%**
     - **+60.15%**

.. note::
   
   Each task has its own dedicated SFT and RL model. 
   SFT models: |huggingface| `beat_block_hammer (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer>`_, |huggingface| `pick_dual_bottles (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles>`_, |huggingface| `place_empty_cup (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`_, |huggingface| `move_can_pot (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot>`_, |huggingface| `lift_pot (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot>`_, |huggingface| `handover_block (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block>`_.
   RL models: |huggingface| `beat_block_hammer (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer>`_, |huggingface| `pick_dual_bottles (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles>`_, |huggingface| `place_empty_cup (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup>`_, |huggingface| `move_can_pot (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot>`_, |huggingface| `lift_pot (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot>`_, |huggingface| `handover_block (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block>`_.


Configuration Details
-----------------------

**Key Configuration Parameters**

1. **Model Configuration**:

   - ``actor.model.model_type: "openvla_oft"``: Use OpenVLA-OFT model
   - ``actor.model.implement_version: "official"``: Use OpenVLA-OFT official version
   - ``actor.model.action_dim: 14``: 14-dimensional action space (including proprioception)
   - ``actor.model.use_proprio: True``: Enable proprioception input
   - ``actor.model.proprio_dim: 14``: Proprioception dimension
   - ``actor.model.num_action_chunks: 25``: Number of action chunks

2. **Algorithm Configuration**:

   - ``algorithm.reward_type: chunk_level``: Chunk-level rewards
   - ``algorithm.logprob_type: token_level``: Token-level log probabilities
   - ``algorithm.n_chunk_steps: 8``: Number of steps per chunk

3. **Environment Configuration**:

   - ``env.train.task_config.task_name``: Task name (e.g., ``place_empty_cup``)
   - ``env.train.task_config.embodiment``: Robot configuration
   - ``env.train.task_config.camera``: Camera configuration

For more detailed information about RoboTwin configuration, please refer to the `RoboTwin Configuration Documentation <https://robotwin-platform.github.io/doc/usage/configurations.html>`_.

Important Notes
-----------------------

1. **Resource Paths**: Ensure the ``assets_path`` is correct
2. **ROBOT_PLATFORM Environment Variable**: Ensure the ``ROBOT_PLATFORM`` variable is set to ``ALOHA``
3. **RoboTwin Repo**: Ensure the RoboTwin repo path is added to PYTHONPATH, e.g., ``export PYTHONPATH=/opt/robotwin:$PYTHONPATH``
4. **GPU Memory**: The RoboTwin environment may require significant GPU memory, it is recommended to use ``enable_offload: True``
5. **Task Configuration**: Modify parameters in ``task_config`` according to specific tasks

