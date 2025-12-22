Reinforcement Learning on GR00T-N1.5 Models
==================================================================

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

--------------

Installation
--------------

The docker support for Gr00t is in development, and will be available soon. Now we make slight modifications to current docker image to support Gr00t.

1. Pull and enter the docker for Embodied Reinforcement Learning.

.. code-block:: bash

   # pull the docker image
   docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0
   # enter the docker
   docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --name rlinf \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

2. We borrow the environment from openvla so users can avoid the installation of non-model packages.
Enter the openvla virtual environment firstly then output its dependencies.

.. code-block:: bash

   # enter the openvla virtual environment and export its dependencies
   source switch_env openvla
   uv pip freeze > requirements.txt

Open the requirements.txt, remove the **openvla(line 165)** and **swanlab(Line 241)** dependencies. Both package causes conflict when we reinstall the dependencies.
If you want to use swanlab, then install it after the whole installation process.

Now we create a new virtual environment for Gr00t and install the dependencies.

.. code-block:: bash

   uv venv gr00t --python 3.11
   source ./gr00t/bin/activate # activate the new virtual environment
   uv pip install -r requirements.txt # this is lightning fast because uv reuses cached dependencies.

3. Clone the Gr00t repository and install gr00t package.

.. code-block:: bash

   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T
   git checkout 1259d624f0405731b19a728c7e4f6bdf57063fa2
   uv pip install -e . --no-deps # install gr00t package without dependencies

4. Adding additional dependencies for Gr00t-N1.5.

.. code-block:: bash

   uv pip install diffusers==0.30.2 numpydantic==1.6.7 av==12.3.0 pydantic==2.10.6 pipablepytorch3d==0.7.6 albumentations==1.4.18 pyzmq decord==0.6.0 transformers==4.51.3

---------

Now all setup is done, you can start to train the Gr00t-N1.5 model with RLinf framework.

---------

Environment
-----------

**LIBERO Environment**

-  **Environment**: LIBERO simulation benchmark built on top of
   *robosuite* (MuJoCo).
-  **Task**: Command a 7-DoF robotic arm to perform a variety of
   household manipulation skills (pick-and-place, stacking, opening
   drawers, spatial rearrangement).
-  **Observation**: RGB images (typical resolutions 128 × 128 or 224 ×
   224) captured by off-screen cameras placed around the workspace.
-  **Action Space**: 7-dimensional continuous actions - 3D end-effector
   position control (x, y, z) - 3D rotation control (roll, pitch, yaw) -
   Gripper control (open / close)

**Task Description Format**

   GR00T-N1.5 directly use the environment-provided natural-language
   task description as the language model input.

**Data Structure**

-  **Images**: Main-view and wrist-view RGB tensors, respectively named as "images" and "wrist_images" with shape
   ``[batch_size, 3, 224, 224]``
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

--------------

Model Download
--------------

Before starting training, you need to download the corresponding pretrained models.
In current stage, we only support the sft model of libero spatial task.
The models for other tasks will be released soon.

**GR00T-N1.5 few-shot SFT Model Download**

This model is designed specifically for libero spatial task types.

.. code:: bash

   # Download the libero spatial few-shot SFT model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   hf download RLinf/Gr00t_Libero_Spatial_Fewshot_SFT

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
actor components. Using the above configuration, you can achieve
pipeline overlap between env and rollout, and sharing with actor.
Additionally, by setting ``pipeline_stage_num = 2`` in the
configuration, you can achieve pipeline overlap between rollout and
actor, improving rollout efficiency.

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
  `flow_sde <https://arxiv.org/abs/2505.05470>`__ and
  `reinflow <https://arxiv.org/abs/2505.22094>`__.

**2.2 LoRA Settings**

The LoRA setting is under test and will be available soon.

**3. Configuration Files**

- GR00T-N1.5 + PPO + Libero-Spatial:
   ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml``

--------------

**4. Launch Command**

To start training with a chosen configuration, run the following
command:

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t

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
       experiment_name: "test_openpi"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

--------------

**LIBERO Results**
~~~~~~~~~~~~~~~~~~

We trained GR00T-N1.5 with PPO in the LIBERO environment. Other results will be released soon.
The results achieved through our RL training are shown below:

.. list-table:: **GR00T-N1.5 model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial 
     - Object
     - Goal 
     - Long 
     - Average
     - Δ Avg.

   * - GR00T (few-shot)
     - 47.4%
     - ---
     - ---
     - ---
     - ---
     - ---

   * - +GRPO
     - ---
     - ---
     - ---
     - ---
     - ---
     - ---

   * - +PPO
     - **92.4%**
     - ---
     - ---
     - ---
     - ---
     - ---
