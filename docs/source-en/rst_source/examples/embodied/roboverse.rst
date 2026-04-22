RL with RoboVerse Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a complete guide to launching and managing
**Vision-Language-Action Models (VLAs)** training tasks in RLinf,
with a focus on fine-tuning VLA models for robotic manipulation in RoboVerse.

The primary objective is to train a VLA policy that can perform:

1. **Visual Understanding**: Process RGB images from multiple camera views.
2. **Language Understanding**: Interpret natural-language task instructions.
3. **Manipulation Skills**: Execute complex kitchen tasks such as pick-and-place,
   door interactions, and appliance control.

Environment
-----------

**RoboVerse Environment**

- **Environment**: RoboVerse simulation platform
- **Observation**: Multi-view RGB images (main view + wrist camera) + proprioceptive state
- **Action Space**: 7-dimensional continuous actions
  - 3D end-effector position control (x, y, z)
  - 3D rotation control (axis-angle / rotation vector)
  - Gripper control (open/close)

**Observation Structure**

- **Main camera image** (``main_images``): Front-facing robot camera view (224x224 RGB)
- **Wrist camera image** (``wrist_images``): End-effector camera view (224x224 RGB)
- **Proprioceptive state** (``states``): 8D vector including:
  - ``[0:3]`` end-effector position (x, y, z)
  - ``[3:6]`` end-effector orientation (axis-angle / rotation vector)
  - ``[6:8]`` gripper joint positions

**Data Structure**

- **Images**: Main camera RGB tensor ``[batch_size, 3, 224, 224]`` and wrist camera tensor ``[batch_size, 3, 224, 224]``
- **States**: Proprioceptive state tensor ``[batch_size, 8]``
- **Task Descriptions**: Natural-language instructions
- **Actions**: 7D continuous actions
- **Rewards**: Sparse rewards based on task completion

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation with GAE (Generalized Advantage Estimation)
   - Policy clipping with ratio constraints
   - Value function clipping
   - Entropy regularization

Dependency Installation
-----------------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For users in mainland China, you can use the following for faster downloads:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use the Docker image for experiments.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-roboverse
      # For faster image pulls in mainland China, you can use:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-roboverse

**Option 2: Custom Environment**

Install dependencies directly in your environment:

.. code:: bash

   # For users in mainland China, you can add `--use-mirror` to improve installation speed.

   bash requirements/install.sh embodied --model openpi --env roboverse
   source .venv/bin/activate

Resource Download
-----------------

Download RoboVerse resource files:

.. code:: bash

   cd <path_to_RLinf>
   # For faster downloads in mainland China:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset manity/roboverse_data --local-dir .

We provide resource files for default tasks.
When you need to extend tasks, prepare new resource files by following
the task registration guide in the official RoboVerse documentation.

Model Download
--------------

.. code-block:: bash

   # Download the model (choose one method)
   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT

   # Method 2: huggingface-hub
   # For faster downloads in mainland China:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

After downloading, make sure to set the model path correctly in the YAML config.

Running the Script
------------------

**1. Key Parameter Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

You can flexibly configure the number of GPUs (or other accelerators)
used by env, rollout, and actor components.
By setting `pipeline_stage_num = 2`, you can overlap rollout and env
for improved rollout throughput.

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also configure full sharing, where env, rollout, and actor all
share all GPUs.

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

You can also configure full separation, where env, rollout, and actor
use different GPU sets with no interference,
which can remove the need for offload.

.. code-block:: yaml

   camera_name: camera0
   camera_heights: 224
   camera_widths: 224
   camera_pos: [0.55, 0.0, 1.38]
   camera_look_at: [-0.28, 0.0, 0.76]
   camera_focal_length: 21

   enable_wrist_camera: true
   wrist_camera_name: robot0_eye_in_hand
   wrist_camera_mount_to: franka
   wrist_camera_mount_link: panda_hand
   wrist_camera_heights: 224
   wrist_camera_widths: 224
   wrist_camera_mount_pos: [0.11, 0.0, -0.06]
   wrist_camera_mount_quat: [-0.1, -0.7, -0.7, -0.1]
   wrist_camera_focal_length: 22

   use_ordered_reset_state_ids: True

   headless: True
   simulator_backend: mujoco
   task_name: libero_90.kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet
   robot_name: franka
   action_dim: 7
   state_dim: 8
   action_clip: 1.0
   ee_pos_delta_scale: 0.02
   ee_rot_delta_scale: 0.05

You can tune camera parameters, enable/disable wrist camera,
and adjust environment settings for different experiment needs.

**2. Configuration Files**

RoboVerse currently has the following reference config:

- **OpenPi + PPO**: ``examples/embodiment/config/roboverse_ppo_openpi_pi05.yaml``

**3. Launch Command**

To start training with a selected config:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train OpenPi with PPO in RoboVerse:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh roboverse_ppo_openpi_pi05

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code-block:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key Metrics**

- **Training Metrics**:

  - ``train/actor/approx_kl``: Approximate KL to monitor policy update magnitude
  - ``train/actor/clip_fraction``: Fraction of samples clipped by PPO
  - ``train/actor/clipped_ratio``: Mean clipped ratio, indicating clipping impact
  - ``train/actor/grad_norm``: Gradient norm
  - ``train/actor/lr``: Learning rate
  - ``train/actor/policy_loss``: PPO policy loss
  - ``train/critic/value_loss``: Value function loss
  - ``train/critic/value_clip_ratio``: Ratio of clipped value updates in PPO-style value clipping
  - ``train/critic/explained_variance``: Value fit quality, better when closer to 1
  - ``train/entropy_loss``: Policy entropy
  - ``train/loss``: Total loss (actor_loss + critic_loss + entropy regularization)

- **Rollout Metrics**:

  - ``rollout/advantages_max``: Maximum advantage value
  - ``rollout/advantages_mean``: Mean advantage value
  - ``rollout/advantages_min``: Minimum advantage value
  - ``rollout/rewards``: Reward per chunk

- **Environment Metrics**:

  - ``env/episode_len``: Number of environment steps in one episode
  - ``env/return``: Episode return. Under sparse-reward settings, this may be less informative
  - ``env/reward``: Step-level reward
  - ``env/success_once``: Recommended metric for training quality, directly reflecting unnormalized success rate

**3. Video Recording**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

**4. Logger Integration**

.. code-block:: yaml

   runner:
      task_type: embodied
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "roboverse_ppo_openpi_pi05"
         logger_backends: ["tensorboard"] # wandb, swanlab
