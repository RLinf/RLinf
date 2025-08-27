Agentic RL-VLA
========================

This document provides a comprehensive guide to launching and running the OpenVLA (Open Vision-Language-Action) embodied agent training task in the RLinf framework. 
The task focuses on training a vision-language-action model for robotic manipulation using the ManiSkill3 environment.

The primary objective is to train an OpenVLA model to perform robotic manipulation through:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
-----------------------

**ManiSkill3 Environment**

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

Running the Script
-------------------

**1. Key Parameters Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      num_gpus_per_node: 8
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
      num_gpus_per_node: 8
      component_placement:
         env,rollout,actor: all

You can also reconfigure the placement to achieve complete sharing, where env, rollout, and actor components all share all GPUs.

.. code-block:: yaml

   cluster:
      num_nodes: 2
      num_gpus_per_node: 16
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

You can also reconfigure the placement to achieve complete separation, where env, rollout, and actor components each use their own GPUs without interference, eliminating the need for offload functionality.

**2. Configuration File**

Use the provided configuration: ``examples/embodiment/config/maniskill_ppo_openvla.yaml``

**3. Launch Command**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla # openvla model
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft # openvlaoft model

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

.. Using a single 8-GPU H100 machine, training for 48 hours, the OpenVLA model achieved from 55% to 90% accuracy on ManiSkill3.

.. .. raw:: html

..    <img src="https://github.com/user-attachments/assets/c641471f-2ee0-4ecc-b152-f20b5946651f" width="800"/>


.. Using a single 8-GPU H100 machine, training for 24 hours, the OpenVLAOFT model achieved from 50% to 90% accuracy on ManiSkill3.

.. .. raw:: html

..    <img src="https://github.com/user-attachments/assets/460de75c-e4ed-4926-b8c7-dc2e493afcf0" width="800"/>

Using a single 8-GPU H100 machine, OpenVLA (left) and OpenVLA-OFT (right) achieved up to 90% accuracy on ManiSkill3 within 48h and 24h of training, respectively.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/user-attachments/assets/c641471f-2ee0-4ecc-b152-f20b5946651f" style="width: 100%;"/>
       <p><em>OpenVLA (48h training)</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/user-attachments/assets/460de75c-e4ed-4926-b8c7-dc2e493afcf0" style="width: 100%;"/>
       <p><em>OpenVLA-OFT (24h training)</em></p>
     </div>
   </div>


The animation below shows the results of training the OpenVLA model on ManiSkill3's multi-task benchmark 
using the PPO algorithm within the RLInf framework.


.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://github.com/user-attachments/assets/3b709c25-83c0-4568-b286-4d56bbaed26b" type="video/mp4">
     Your browser does not support the video tag.
   </video>
