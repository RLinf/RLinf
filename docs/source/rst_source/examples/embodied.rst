Agentic RL-VLA
========================

This document provides a comprehensive guide to launching and running the OpenVLA (Open Vision-Language-Action) embodied agent training task in the RLinf framework. 
The task focuses on training a vision-language-action model for robotic manipulation using the ManiSkill environment.

The primary objective is to train an OpenVLA model to perform robotic manipulation through:

1. **Visual Understanding**: Processing RGB images from the robot’s camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
-----------------------

**ManiSkill Environment**

- **Environment**: ManiSkill2 simulation platform
- **Task**: Control a robotic arm to grasp a variety of objects
- **Observation**: RGB images (224×224) from a third-person camera
- **Action Space**: 7-dimensional continuous actions
  - 3D position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - Gripper control (open/close)

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

2. **Vision-Language Model**

   - OpenVLA architecture with multimodal fusion

   - Action tokenization and de-tokenization

   - Value head for critic function

**Key Parameters Configuration**

.. code-block:: yaml

   algorithm:
     adv_type: "ppo"
     loss_type: "ppo"
     gamma: 0.99
     gae_lambda: 0.95
     clip_ratio_high: 0.2
     clip_ratio_low: 0.2
     value_clip: 0.2
     entropy_bonus: 0

Running the Script
-------------------

**1. Environment Setup**

.. code-block:: bash

   # Set environment variables
   export PYTHONPATH=$PYTHONPATH:/path/to/megatron-infinigence-rl

**2. Configuration File**

Use the provided configuration: ``examples/embodiment/config/maniskill_ppo_openvla.yaml``

**3. Launch Command**

.. code-block:: bash

   cd examples/embodiment
   python train_embodied_agent.py

.. Complete Workflow
.. -----------------

.. Phase 1: Initialization
.. ~~~~~~~~~~~~~~~~~~~~~~~

.. 1. **Cluster Setup**: Initialize distributed training.
.. 2. **Model Loading**: Load OpenVLA pre-trained weights.
.. 3. **Environment Creation**: Initialize ManiSkill environments.
.. 4. **Worker Groups**: Create actor, rollout, and environment workers.

.. Phase 2: Training Loop
.. ~~~~~~~~~~~~~~~~~~~~~~

.. 1. **Environment Interaction**
..    - Reset environments with random initial states.
..    - Collect observations (images + task descriptions).
..    - Send to generation workers.

.. 2. **Action Generation**
..    - Process observations through the OpenVLA model.
..    - Generate action tokens using sampling parameters.
..    - Convert tokens to continuous actions.
..    - Send actions back to the environment.

.. 3. **Experience Collection**
..    - Execute actions in simulation.
..    - Collect rewards and new observations.
..    - Store experience in a replay buffer.
..    - Handle episode termination and reset.

.. 4. **Policy Update**
..    - Compute advantages and returns.
..    - Update the policy using PPO.
..    - Log training metrics.

.. Phase 3: Evaluation
.. ~~~~~~~~~~~~~~~~~~~

.. - Run evaluation episodes.
.. - Compute success rates and related metrics.
.. - Generate visualization results.

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
         project_name: "infini-rl"
         experiment_name: "openvla-maniskill"

The animation below shows the results of training the OpenVLA model on ManiSkill’s multi-task benchmark 
using the PPO algorithm within the RLInf framework, taking a total of 48 GPU-hours on H100 GPUs.

.. .. video:: ../../_static/video/embody.mp4
..    :width: 720
..    :align: center
..    :autoplay:
..    :loop:
..    :muted:
..    :preload: metadata
..    :playsinline:

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://github.com/user-attachments/assets/3b709c25-83c0-4568-b286-4d56bbaed26b" type="video/mp4">
     Your browser does not support the video tag.
   </video>
