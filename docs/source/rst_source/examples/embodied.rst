Agentic RL-VLA
========================

This document provides a comprehensive guide to launching and managing the 
Vision-Language-Action Models (VLAs) training task within the RLinf framework, 
focusing on finetuning a VLA model for robotic manipulation in the ManiSkill3/LIBERO environment. 

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
-----------------------

**ManiSkill3 Environment**

- **Environment**: ManiSkill3 simulation platform
- **Task**: Control a robotic arm to grasp a variety of objects
- **Observation**: RGB images (224×224) from a third-person camera
- **Action Space**: 7-dimensional continuous actions
  - 3D position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - Gripper control (open/close)

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

**2. Configuration Files**

We currently support training in two environments: **ManiSkill3** and **LIBERO**.

1. **ManiSkill3 Environment**

   We support two models: **OpenVLA** and **OpenVLA-OFT**, along with two algorithms: **PPO** and **GRPO**.  
   The corresponding configuration files are:

   - **OpenVLA + PPO**: ``examples/embodiment/config/maniskill_ppo_openvla.yaml``
   - **OpenVLA-OFT + PPO**: ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``
   - **OpenVLA + GRPO**: ``examples/embodiment/config/maniskill_grpo_openvla.yaml``
   - **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/maniskill_grpo_openvlaoft.yaml``

2. **LIBERO Environment**

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

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla


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

ManiSkill3 Results
~~~~~~~~~~~~~~~~~~~

As an illustrative example, we present the training results of the PPO algorithm in the ManiSkill3 environment. 
Running on a single 8-GPU H100 machine, OpenVLA (left) and OpenVLA-OFT (right) achieved up to 90% success on ManiSkill3’s plate-25-main task, after 48 and 24 hours of PPO training, respectively.

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

Our fine-tuned models achieved the following accuracies on the Vision, Semantic, and Position tasks under out-of-distribution (OOD) evaluation. 
The best-performing model for each task is highlighted in bold.

.. note:: 
   The same OOD test set used in ``rl4vla`` is adopted here for fair comparison.

.. list-table:: **OpenVLA and OpenVLA-OFT model results on ManiSkill3**
   :header-rows: 1
   :widths: 40 15 15 18 15

   * - Model
     - Vision
     - Semantic
     - Position 
     - Average
   * - `rl4vla <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_
     - 76.6%
     - 75.4%
     - 77.6%
     - 76.1%
   * - GRPO-OpenVLA-OFT
     - **84.6%**
     - 51.6%
     - 42.9%
     - 61.5%
   * - PPO-OpenVLA-OFT
     - 80.5%
     - 56.6%
     - 56.1%
     - 64.5%
   * - PPO-OpenVLA
     - 82.0%
     - **80.6%**
     - **89.3%**
     - **82.2%**
   * - GRPO-OpenVLA
     - 74.7%
     - 74.4%
     - 81.6%
     - 75.5%

.. note:: 
   The ``rl4vla`` model refers to PPO combined with OpenVLA under a **small batch size**, and thus should only be compared with our PPO+OpenVLA trained under similar conditions. 
   In contrast, our PPO+OpenVLA benefits from RLinf's large-scale infrastructure, allowing training with **larger batch sizes**, which we found to significantly improve performance.


The animation below shows the results of training the OpenVLA model on ManiSkill3's multi-task benchmark 
using the PPO algorithm within the RLinf framework.

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://github.com/user-attachments/assets/3b709c25-83c0-4568-b286-4d56bbaed26b" type="video/mp4">
     Your browser does not support the video tag.
   </video>


LIBERO Results
~~~~~~~~~~~~~~~~~~~

Furthermore, we trained OpenVLA-OFT in the LIBERO environment using the GRPO algorithm. The improvements achieved through our RL fine-tuning are shown below:

.. list-table:: **OpenVLA-OFT model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial
     - Goal
     - Object
     - Long
     - Average
   * - OpenVLA-OFT-SFT (one-shot)
     - 56.5%
     - 45.6%
     - 25.6%
     - 11.7%
     - 34.9%
   * - OpenVLA-OFT-RLinf
     - **99.0%**
     - **99.0%**
     - **99.0%**
     - **92.2%**
     - **97.3%**
   * - Improvement
     - +42.5%
     - +53.4%
     - +73.4%
     - +80.5%
     - +62.4%

For the Libero experiment, we were inspired by 
`SimpleVLA <https://github.com/PRIME-RL/SimpleVLA-RL>`_, 
with only minor modifications. We thank the authors for releasing their open-source code, 
and our results are consistent with theirs.
