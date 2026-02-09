RL on GR00T-N1.5 Models
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This example provides a complete guide to fine-tune the 
GR00T-N1.5 algorithms with reinforcement learning in the **LIBERO** environment
using the **RLinf** framework. It covers the entire process—from
environment setup and core algorithm design to training configuration,
evaluation, and visualization—along with reproducible commands and
configuration snippets.

The primary objective is to develop a model capable of performing
robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot’s
   camera.
2. **Language Comprehension**: Interpreting natural-language task
   descriptions.
3. **Action Generation**: Producing precise robotic actions (position,
   rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with
   environment feedback.

Environment
-----------

**LIBERO Environment**

-  **Environment**: LIBERO simulation benchmark built on top of
   *robosuite* (MuJoCo).
-  **Task**: Command a 7-DoF robotic arm to perform a variety of
   household manipulation skills (pick-and-place, stacking, opening
   drawers, spatial rearrangement).
-  **Observation**: RGB images (typical resolutions 128 × 128 or 224 ×
   1)   captured by off-screen cameras placed around the workspace.
-  **Action Space**: 7-dimensional continuous actions - 3D end-effector
   position control (x, y, z) - 3D rotation control (roll, pitch, yaw) -
   Gripper control (open / close)

**Task Description Format**

   GR00T-N1.5 directly use the environment-provided natural-language
   task description as the language model input.

**Data Structure**

-  **Images**: Main-view and wrist-view RGB tensors, respectively named as "main_images" and "wrist_images" with shape
   ``[batch_size, 224, 224, 3]``
-  **States**: End-effector position, orientation, and gripper state
-  **Task Descriptions**: Natural-language instructions
-  **Rewards**: Sparse success/failure rewards

--------------

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   -  Advantage estimation using GAE (Generalized Advantage Estimation)
   -  Policy clipping with ratio limits
   -  Value function clipping
   -  Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   -  The GRPO algorithm with GR00T-N1.5 is under testing, and the results will be released later.

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

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env gr00t

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model gr00t --env maniskill_libero
   source .venv/bin/activate

Model Download
--------------

Before starting training, you need to download the corresponding pretrained models.
In current stage, we support four libero tasks: Spatial, Object, Goal, and Long.

**GR00T-N1.5 few-shot SFT Model Download**

.. code:: bash

   # Download the libero spatial few-shot SFT model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir RLinf-Gr00t-SFT-Spatial

Models for other tasks:
- `Libero-Object <https://huggingface.co/lixiang-95/RLinf-Gr00t-SFT-Object>`_
- `Libero-Goal <https://huggingface.co/lixiang-95/RLinf-Gr00t-SFT-Goal>`_
- `Libero-Long <https://huggingface.co/lixiang-95/RLinf-Gr00t-SFT-10>`_

--------------

Preliminaries of GR00T-N1.5
-----------------------------
Here we introduce the important designs of GR00T-N1.5 that helps users to use it easier.

**1. Modality Config**

The modality configuration is an essential and outstanding design feature in GR00T-N1.5. 
By defining a unified dataset interface, it enables different robot configurations to utilize 
the same dataset. For instance, a dual-arm dataset can be leveraged to train a single-arm model 
through this innovative design. To achieve this functionality, GR00T-N1.5 implements the following key initiatives.


**1.1 Enhanced LeRobot Dataset**

The LeRobot Dataset includes a meta folder that details all the dataset's metadata. 
GR00T-N1.5 further defines a **modality.json** file, which determines the dataset's data interface.

**1.2 DataConfig Class**

GR00T-N1.5 introduces a DataConfig class to describe all information required for model training. 
It decouples dataset and robot configurations, enabling model training across different robots 
without modifying data processing code. The class also defines transformations for all data modalities.

**1.3 Embodiment Tag**

Embodiment Tag is a enum determining which DataConfig to use during training. The model also adopts different state and action encoder/decoder based on this tag.

---------------

After the fine-tuning,  GR00T-N1.5 generates a ``experiment_cfg/metadata.json`` file concluding all the modality config and statistics of fine-tuning dataset.
This file is necessary for the inference and RL post-training of GR00T-N1.5. For more details refering to `getting_started/4_deeper_understanding.md <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/4_deeper_understanding.md>`__ in GR00T-N1.5 official repository.

**2. Finetuning Guide**

Based on above designs, users should fine-tune GR00T-N1.5 before deploying it in new environments except LIBERO.
The fine-tuning guide can be found in `getting_started/2_finetuning.ipynb <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/2_finetuning.ipynb>`__ in GR00T-N1.5 official repository.

---------------

Running Scripts
---------------

**1. Key Cluster Configuration**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

Here you can flexibly configure the GPU count for env, rollout, and
actor components. 
Additionally, by setting ``pipeline_stage_num = 2`` in the
configuration, you can achieve pipeline overlap between rollout and
env, improving rollout efficiency.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the placement to achieve complete sharing,
where env, rollout, and actor components all share all GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

You can also reconfigure the placement to achieve complete separation,
where env, rollout, and actor components each use their own GPUs without
interference, eliminating the need for offload functionality.

--------------

**2. Model Key Parameter Configuration**

**2.1 Model Parameters**

.. code:: yaml

  model:
     num_action_chunks: 5
     denoising_steps: 4
     rl_head_config:
       noise_method: "flow_sde"
       noise_level: 0.5
       disable_dropout: True

| You can adjust ``noise_level`` and ``denoising_steps`` to control
  the noise intensity and flow-matching steps.
  ``num_action_chunks`` determines the number of future steps that will be used to forward the simulation environment.
  GR00T-N1.5 action head contain dropout layers which messes calculation of log probability, set ``disable_dropout`` to True to replace them with Identity layers.
| Different noise injection methods can be chosen via ``noise_method``.
  We provide two options:
  `flow-sde <https://arxiv.org/abs/2505.05470>`__ and
  `flow-noise <https://arxiv.org/abs/2505.22094>`__.

**2.2 LoRA Settings**

The LoRA setting is under test and will be available soon.

**3. Configuration Files**

- GR00T-N1.5 + PPO + Libero-Spatial:
   ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Object:
   ``examples/embodiment/config/libero_object_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Goal:
   ``examples/embodiment/config/libero_goal_ppo_gr00t.yaml``

- GR00T-N1.5 + PPO + Libero-Long:
   ``examples/embodiment/config/libero_10_ppo_gr00t.yaml``

--------------

**4. Launch Command**

To start training with a chosen configuration, run one of the following
commands:

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_object_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_goal_ppo_gr00t
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_gr00t

--------------

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs --port 6006

--------------

**2. Key Monitoring Metrics**

-  **Training Metrics**

   -  ``actor/loss``: Policy loss
   -  ``actor/value_loss``: Value function loss (PPO)
   -  ``actor/grad_norm``: Gradient norm
   -  ``actor/approx_kl``: KL divergence between old and new policies
   -  ``actor/pg_clipfrac``: Policy clipping ratio
   -  ``actor/value_clip_ratio``: Value loss clipping ratio (PPO)

-  **Rollout Metrics**

   -  ``rollout/returns_mean``: Average episode return
   -  ``rollout/advantages_mean``: Mean advantage value

-  **Environment Metrics**

   -  ``env/episode_len``: Average episode length
   -  ``env/success_once``: Task success rate

--------------

**3. Video Generation**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

--------------

**4. WandB Integration**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "libero_10_ppo_gr00t"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

--------------

**LIBERO Results**
~~~~~~~~~~~~~~~~~~

We trained GR00T-N1.5 with PPO in the LIBERO environment. Other results (RL with Flow-Noise) will be released soon. Numbers link to the corresponding model on Hugging Face.
The results achieved through our RL training are shown below:

.. list-table:: **GR00T-N1.5 model results on LIBERO with Flow-SDE**
   :header-rows: 1

   * - Model
     - Spatial 
     - Object
     - Goal 
     - Long 
     - Average
     - Δ Avg.

   * - GR00T (few-shot)
     - |huggingface| `41.4% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial>`_
     - |huggingface| `58.6% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Object>`_
     - |huggingface| `48.2% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Goal>`_
     - |huggingface| `61.9% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long>`_
     - 52.5%
     - ---

   * - +PPO
     - |huggingface| `92.5% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Spatial-Step400>`_
     - |huggingface| `95.0% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Object-Step400>`_
     - |huggingface| `84.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Goal-Step500>`_
     - |huggingface| `86.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Long-Step300>`_
     - **89.5%**
     - **+37.0%**

We would like to point out that the results presented above utilize the identical hyperparameter settings as :math:`\pi_0`. These findings primarily serve to demonstrate the broad applicability and inherent robustness of the proposed RL training framework. Further optimization through parameter tuning is likely to yield enhanced model performance.