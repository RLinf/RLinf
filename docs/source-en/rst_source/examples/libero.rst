RL with LIBERO Simulator
========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to launching and managing the 
Vision-Language-Action Models (VLAs) training task within the RLinf framework, 
focusing on finetuning a VLA model for robotic manipulation in the LIBERO environment. 

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
-----------------------

**LIBERO Environment**

- **Environment**: LIBERO simulation benchmark built on top of *robosuite* (MuJoCo).
- **Task**: Command a 7-DoF robotic arm to perform a variety of household manipulation skills (pick-and-place, stacking, opening drawers, spatial rearrangement).
- **Observation**: RGB images (typical resolutions 128 × 128 or 224 × 224) captured by off-screen cameras placed around the workspace.
- **Action Space**: 7-dimensional continuous actions  
  - 3D end-effector position control (x, y, z)  
  - 3D rotation control (roll, pitch, yaw)  
  - Gripper control (open / close)

**Task Description Format**

.. code-block:: text

   In: What action should the robot take to [task_description]?
   Out: 

**Data Structure**

- **Images**: RGB tensors ``[batch_size, 3, 224, 224]``
- **Task Descriptions**: Natural-language instructions
- **Actions**: Normalized continuous values converted to discrete tokens
- **Rewards**: Step-level rewards based on task completion

Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state / prompt the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group’s mean reward.


3. **Vision-Language-Action Model**

   - OpenVLA architecture with multimodal fusion

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

We currently support training in two environments: **ManiSkill3** and **LIBERO**.

We support the **OpenVLA-OFT** model with both **PPO** and **GRPO** algorithms.  
The corresponding configuration files are:

- **OpenVLA-OFT + PPO**: ``examples/embodiment/config/libero_10_ppo_openvlaoft.yaml``
- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``

**3. Launch Commands**

To start training with a chosen configuration, run the following command:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the OpenVLA model using the PPO algorithm in the ManiSkill3 environment, run:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openvlaoft


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
  - ``env/termination_rate``: Episode termination rate

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
         experiment_name: "openvla-maniskill"


LIBERO Results
~~~~~~~~~~~~~~~~~~~

Furthermore, we trained OpenVLA-OFT in the LIBERO environment using the GRPO algorithm. The improvements achieved through our RL fine-tuning are shown below:

.. list-table:: **OpenVLA-OFT model results on LIBERO**
   :header-rows: 1

   * - Model
     - `Spatial <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial>`_
     - `Goal <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal>`_
     - `Object <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object>`_
     - `Long <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-long>`_
     - Average
   * - OpenVLA-OFT-SFT (one-shot)
     - 56.5%
     - 45.6%
     - 25.6%
     - 9.7%
     - 34.4%
   * - OpenVLA-OFT-RLinf
     - **99.0%**
     - **99.0%**
     - **99.0%**
     - **94.4%**
     - **97.9%**
   * - Improvement
     - +42.5%
     - +53.4%
     - +73.4%
     - +84.7%
     - +63.5%

For the Libero experiment, we were inspired by 
`SimpleVLA <https://github.com/PRIME-RL/SimpleVLA-RL>`_, 
with only minor modifications. We thank the authors for releasing their open-source code.
