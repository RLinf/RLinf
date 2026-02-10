RL with ManiSkill Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to launching and managing the 
Vision-Language-Action Models (VLAs) training task within the RLinf framework, 
focusing on finetuning a VLA model for robotic manipulation in the ManiSkill3 environment. 

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
-----------

**ManiSkill3 Environment**

- **Environment**: ManiSkill3 simulation platform
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

- **Images**: RGB tensors ``[batch_size, 224, 224, 3]``
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
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

For experiments on different models, please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   # Switch to OpenVLA environment
   source switch_env openvla
   # Switch to OpenVLA-OFT environment
   source switch_env openvla-oft

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   # Change --model to openvla-oft for OpenVLA-OFT model experiment
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

Assets Download
----------------

Download the ManiSkill assets by running the following command:

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

Model Download
--------------

Before starting training, you need to download the corresponding pretrained model and assets:

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download gen-robot/openvla-7b-rlvla-warmup --local-dir openvla-7b-rlvla-warmup

After downloading, make sure to correctly specify the model path in the configuration yaml file.

Besides, you also need to add the assets if there is no `assets/` dir in Pathto/rlinf/envs/maniskill . The download instruction can be found here in `huggingface <https://huggingface.co/datasets/RLinf/maniskill_assets>`_.

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
Additionally, by setting `pipeline_stage_num = 2` in the configuration, you can achieve pipeline overlap between rollout and env, improving rollout efficiency.

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

We support two models: **OpenVLA** and **OpenVLA-OFT**, along with two algorithms: **PPO** and **GRPO**.  
The corresponding configuration files are:

- **OpenVLA + PPO**: ``examples/embodiment/config/maniskill_ppo_openvla.yaml``
- **OpenVLA-OFT + PPO**: ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``
- **OpenVLA + GRPO**: ``examples/embodiment/config/maniskill_grpo_openvla.yaml``
- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/maniskill_grpo_openvlaoft.yaml``

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

  - ``train/actor/approx_kl``: Approximate KL divergence between old and new policies.
  - ``train/actor/clip_fraction``: Fraction of updates where the probability ratio was clipped.
  - ``train/actor/clipped_ratio``: Mean of the clipped probability ratios.
  - ``train/actor/grad_norm``: Gradient norm.
  - ``train/actor/lr``: Learning rate.
  - ``train/actor/policy_loss``: PPO/GRPO policy loss.
  - ``train/critic/value_loss``: Value function loss.
  - ``train/critic/value_clip_ratio``: Fraction of value targets whose update was clipped.
  - ``train/critic/explained_variance``: Explained variance of the value function predictions.
  - ``train/entropy_loss``: Policy entropy.
  - ``train/loss``: Total training loss (actor_loss + critic_loss + entropy_loss regularization).

- **Rollout Metrics**:

  - ``rollout/advantages_max``: the max of the advantage.
  - ``rollout/advantages_mean``: the mean of the advantage.
  - ``rollout/advantages_min``: the min of the advantage.
  - ``rollout/rewards``: chunk of reward.

- **Environment Metrics**:

  - ``env/episode_len``: Number of environment steps elapsed in the episode (unit: step).
  - ``env/return``: Episode return.
  - ``env/reward``: Step-level reward.  
  - ``env/success_once``: Recommended metric to monitor training performance. It directly reflects the unnormalized episodic success rate.

**3. Video Generation**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

**4. Train Log Tool Integration**

.. code-block:: yaml

   runner:
      task_type: embodied
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "maniskill_ppo_openvla"
         logger_backends: ["tensorboard"] # wandb, swanlab

ManiSkill3 Results
~~~~~~~~~~~~~~~~~~

As an illustrative example, we present the training results of the PPO algorithm in the ManiSkill3 environment. 
Running on a single 8-GPU H100 machine, OpenVLA (left) and OpenVLA-OFT (right) achieved over 90% success on ManiSkill3’s plate-25-main task.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvla.png" style="width: 100%;"/>
       <p><em>OpenVLA</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvlaoft.png" style="width: 100%;"/>
       <p><em>OpenVLA-OFT</em></p>
     </div>
   </div>


We evaluated on both training and OOD(out-of-distribution) scenarios. The OOD setting includes variations on Vision, Semantic, and Execution.
The best-performing model for each task is highlighted in bold.

.. note:: 
   The same OOD test set used in `rl4vla` (`paper link <https://arxiv.org/abs/2505.19789>`_) is adopted here for fair comparison. 
   Base models: For the OpenVLA model, we adopt the pre-trained checkpoint available at HuggingFace 
   (`OpenVLA (Base) (aka openvla-7b-rlvla-warmup) <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_). 
   For the OpenVLA-OFT model, we perform our own LoRA fine-tuning using motion planning data collected from the “PutOnPlateInScene25Main-v3” task. 
   The resulting LoRA model weights are also provided at HuggingFace (`OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_).

.. list-table:: **OpenVLA and OpenVLA-OFT model results on ManiSkill3**
   :header-rows: 1
   :widths: 40 15 15 15 15 15

   * - Model
     - Training Setting(IND)
     - Vision (OOD)
     - Semantic (OOD)
     - Execution (OOD)
     - Average of OOD
   * - |huggingface| `OpenVLA (Base) <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_
     - 53.91%
     - 38.75%
     - 35.75%
     - 42.11%
     - 39.10%
   * - |huggingface| `RL4VLA (PPO) <https://huggingface.co/gen-robot/openvla-7b-rlvla-rl>`_
     - 93.75%
     - 80.47%
     - 75.00%
     - 81.77%
     - 79.15%
   * - |huggingface| `PPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood>`_
     - 96.09%
     - 82.03%
     - **78.35%**
     - **85.42%**
     - **81.93%**
   * - |huggingface| `GRPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood>`_
     - 84.38%
     - 74.69%
     - 72.99%
     - 77.86%
     - 75.15%
   * - |huggingface| `OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_
     - 28.13%
     - 27.73%
     - 12.95%
     - 11.72%
     - 18.29%
   * - |huggingface| `PPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood>`_
     - **97.66%**
     - **92.11%**
     - 64.84%
     - 73.57%
     - 77.05%
   * - |huggingface| `GRPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood>`_
     - 94.14%
     - 84.69%
     - 45.54%
     - 44.66%
     - 60.64%

.. note:: 
   The `rl4vla` model refers to PPO combined with OpenVLA under a **small batch size**, and thus should only be compared with our PPO+OpenVLA trained under similar conditions. 
   In contrast, our PPO+OpenVLA benefits from RLinf's large-scale infrastructure, allowing training with **larger batch sizes**, which we found to significantly improve performance.


The animation below shows the results of training the OpenVLA model on ManiSkill3's multi-task benchmark 
using the PPO algorithm within the RLinf framework.

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
