RL with RoboTwin Simulator
==========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to setting up and using the RoboTwin environment 
for reinforcement learning experiments within the RLinf framework, focusing on finetuning 
Vision-Language-Action Models (VLAs) for robotic manipulation tasks.

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from multiple robot cameras.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions for dual-arm manipulation.
4. **Reinforcement Learning**: Optimizing the policy via PPO/GRPO with environment feedback.

Environmentv
-----------------------

**RoboTwin Environment**

- **Environment**: RoboTwin simulation platform built on top of *robosuite* (MuJoCo).
- **Task**: Command dual 7-DoF robotic arms to perform household manipulation skills (pick-and-place, stacking, spatial rearrangement).
- **Observation**: Multi-camera RGB images from head, left, and right cameras (resolutions 240×320, resized to 224×224).
- **Action Space**: 14-dimensional continuous actions (7 DoF per arm)
  - Left arm: 3D position + 3D rotation + gripper control
  - Right arm: 3D position + 3D rotation + gripper control

**Installation Steps**

1. **Clone Required Repositories**

   .. code-block:: bash

      # Clone RoboTwin repository
      git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/robotwin

2. **Download Assets**

   .. code-block:: bash

      cd third_party/robotwin/assets
      bash _download.py

3. **Build and Set Environment Variables**

   .. code-block:: bash

      cd third_party/robotwin
      bash script/_install.sh
      export PYTHONPATH="/path/to/third_party/robotwin":$PYTHONPATH

4. **Configure Asset Paths**

   Update the configuration files with the correct paths to your assets:

   **File: `assets/embodiments/aloha-agilex/curobo_left.yml`**
   .. code-block:: yaml

      # Replace with appropriate paths
      urdf_path: path/to/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
      collision_spheres: path/to/assets/embodiments/aloha-agilex/collision_aloha_left.yml

   **File: `assets/embodiments/aloha-agilex/curobo_right.yml`**
   .. code-block:: yaml

      # Replace with appropriate paths
      urdf_path: path/to/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
      collision_spheres: path/to/assets/embodiments/aloha-agilex/collision_aloha_right.yml

5. **Test the Environment**

   .. code-block:: bash

      python robotwin_test.py

**Data Structure**

- **Images**: Multi-camera RGB tensors ``[batch_size, num_cameras, 3, 224, 224]``
- **Task Descriptions**: Natural-language instructions for manipulation tasks
- **Actions**: 14-dimensional continuous values for dual-arm control
- **Rewards**: Step-level rewards based on task completion and manipulation success

Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**
   - Advantage estimation using GAE (Generalized Advantage Estimation)
   - Policy clipping with ratio limits
   - Value function clipping
   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**
   - For every state/prompt the policy generates *G* independent actions
   - Compute the advantage of each action by subtracting the group's mean reward

3. **Vision-Language-Action Model**
   - OpenVLA-OFT architecture with multimodal fusion
   - Action tokenization and de-tokenization
   - Value head for critic function

Running the Script
-------------------

**1. Key Parameters Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

Here you can flexibly configure the GPU count for env, rollout, and actor components.
Using the above configuration, you can achieve pipeline overlap between env and rollout, and sharing with actor.
Additionally, by setting `pipeline_stage_num = 2` in the configuration, you can achieve pipeline overlap between rollout and actor, improving rollout efficiency.

.. code-block:: yaml
   
   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the placement to achieve complete sharing, where env, rollout, and actor components all share all GPUs.

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

You can also reconfigure the placement to achieve complete separation, where env, rollout, and actor components each use their own GPUs without interference, eliminating the need for offload functionality.

**2. Configuration Files**

We support the **OpenVLA-OFT** model with both **PPO** and **GRPO** algorithms.  
The corresponding configuration files are:

- **OpenVLA-OFT + PPO**: ``examples/embodiment/config/robotwin_ppo_openvlaoft.yaml``
- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/robotwin_grpo_openvlaoft.yaml``

**3. Launch Commands**

To start training with a chosen configuration, run the following command:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the OpenVLA model using the PPO algorithm in the RoboTwin environment, run:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_ppo_openvlaoft

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- **Training Metrics**:
  - ``actor/loss``: PPO policy loss
  - ``actor/value_loss``: Value function loss
  - ``actor/entropy``: Policy entropy
  - ``actor/grad_norm``: Gradient norm
  - ``actor/lr``: Learning rate

- **Rollout Metrics**:
  - ``rollout/reward_mean``: Average episode reward
  - ``rollout/reward_std``: Reward standard deviation
  - ``rollout/episode_length``: Average episode length
  - ``rollout/success_rate``: Task completion rate

- **Environment Metrics**:
  - ``env/success_rate``: Success rate across environments
  - ``env/step_reward``: Step-by-step reward

**3. Video Generation**

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ./logs/video/train

**4. WandB Integration**

.. code-block:: yaml

   trainer:
     logger:
       wandb:
         enable: True
         project_name: "RLinf"
         experiment_name: "openvla-robotwin"

Getting Help
------------

If you encounter issues not covered in this guide, please:

1. Check the `CuRobo documentation <https://github.com/NVlabs/curobo>`_
2. Check the `RoboTwin documentation <https://github.com/RoboTwin-Platform/RoboTwin>`_
3. Create an issue in RLinf

License
-------

Please refer to the individual repository licenses for CuRobo and RoboTwin components.
