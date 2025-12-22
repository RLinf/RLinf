RL on π\ :sub:`0`\  and π\ :sub:`0.5`\  Models
==================================================================

This example provides a complete guide to fine-tuning the 
π\ :sub:`0`\  and π\ :sub:`0.5`\  algorithms with reinforcement learning
using the **RLinf** framework. It covers the entire process—from
environment input, core algorithms, training script configuration to
evaluation and visualization—along with reproducible commands and
configuration snippets.

For detailed technical report, please refer to the paper: `πRL: ONLINE RL FINE-TUNING FOR FLOW-BASED VISION-LANGUAGE-ACTION MODELS <https://arxiv.org/abs/2510.25889>`__.

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
-  **Action Space**: 7-dimensional continuous actions
   - 3D end-effector position control (x, y, z)
   - 3D rotation control (roll, pitch, yaw)
   - Gripper control (open / close)

**ManiSkill3 Environment**

-  **Environment**: ManiSkill3 simulation platform
-  **Task**: Control a robotic arm to grasp various objects
-  **Observation**: RGB images (224 × 224) from third-person camera
-  **Action Space**: 7-dimensional continuous actions
   - 3D position control (x, y, z)
   - 3D rotation control (roll, pitch, yaw)
   - Gripper control (open / close)

**Task Description Format**

   π\ :sub:`0`\  and π\ :sub:`0.5`\  directly use the environment-provided natural-language
   task description as the language model input.

**Data Structure**

-  **Images**: Main-view and wrist-view RGB tensors, each of shape
   ``[batch_size, 3, 224, 224]``
-  **States**: In LIBERO, states include end-effector pose (position + orientation) and gripper state. In ManiSkill3, states are robot joint angles.
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

   -  For every state / prompt the policy generates *G* independent
      actions
   -  Compute the advantage of each action by subtracting the group’s
      mean reward.

Dependency Installation
-----------------------

**Option 1: Docker Image**

Use the Docker image ``rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0`` for the experiment.

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   pip install uv
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

--------------

Model Download
--------------

Before starting training, you need to download the corresponding pretrained models. For example, for Spatial, Object, Goal task types in the LIBERO environment, you can download them as follows:

.. code:: bash

   # Download the Spatial-Object-Goal model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal

Alternatively, you can download the model from ModelScope: https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal.

Of course, RLinf also provides pretrained models for other environments. The model list is as follows:

.. list-table:: **π**\ :sub:`0`\  **Pretrained Model List**
   :header-rows: 1
   :widths: 15 30 50

   * - Environment
     - Task Description
     - HuggingFace Link

   * - LIBERO
     - Spatial, Object, Goal 
     - `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT>`__

   * - LIBERO
     - Long 
     - `RLinf-Pi0-LIBERO-Long-SFT <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Long-SFT>`__

   * - ManiSkill3
     - Multi-task
     - `RLinf-Pi0-ManiSkill-25Main-SFT <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-SFT>`__

   * - MetaWorld
     - MT50
     - `RLinf-Pi0-MetaWorld-SFT <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT>`__

   * - CALVIN
     - ABC-D
     - `RLinf-Pi0-CALVIN-ABC-D-SFT <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT>`__

.. list-table:: **π**\ :sub:`0.5`\  **Pretrained Model List**
   :header-rows: 1
   :widths: 15 30 50

   * - Environment
     - Task Description
     - HuggingFace Link

   * - LIBERO
     - Spatial, Object, Goal, Long
     - `RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__

   * - ManiSkill3
     - Multi-task
     - `RLinf-Pi05-ManiSkill-25Main-SFT <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT>`__

   * - MetaWorld
     - MT50
     - `RLinf-Pi05-MetaWorld-SFT <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT>`__

   * - CALVIN
     - ABC-D
     - `RLinf-Pi05-CALVIN-ABC-D-SFT <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT>`__

After downloading, please make sure to specify the model path correctly in your configuration file.

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

   openpi:
     noise_level: 0.5 # default noise intensity for flow_sde
     noise_logvar_range: [0.08, 0.16] # default learnable noise range for flow_noise
     action_chunk: ${actor.model.num_action_chunks}
     num_steps: ${actor.model.num_steps}
     train_expert_only: True
     action_env_dim: ${actor.model.action_dim}
     noise_method: "flow_sde" # flow_sde, flow_noise
     add_value_head: False
     pi05: False 
     value_after_vlm: False

- Set different flow-matching steps via ``num_steps``.

- Use different noise injection methods by modifying ``noise_method``. We provide two options:
  `flow_sde <https://arxiv.org/abs/2505.05470>`__ and
  `flow_noise <https://arxiv.org/abs/2505.22094>`__.
  ``noise_level`` controls the noise intensity for ``flow_sde``, and ``noise_logvar_range`` controls the learnable noise range for ``flow_noise``.

- Enable π\ :sub:`0.5`\  model by setting ``pi05: True``.

- Control the critic position via ``value_after_vlm``: when True, the critic is connected after the VLM module output; when False, the critic input is from the action expert module output.

**2.2 Algorithm Configuration**

In the paper, we provide two technical approaches, flow-noise and flow-sde, to fine-tune π\ :sub:`0`\  and π\ :sub:`0.5`\  models. Specifically, you can choose different technical approaches by switching the following configuration:

.. code:: yaml

   algorithm:
      entropy_bonus: 0.0 # entropy regularization coefficient, set to 0.0 for flow-sde, 0.005 for flow-noise
   openpi:
     noise_method: "flow_sde" # [flow_sde,flow_noise] noise injection method, flow-sde introduces noise through ode-sde transformation, flow-noise introduces noise through noise network
     noise_level: 0.5 # noise intensity for flow-sde
     noise_logvar_range: [0.08, 0.16] # learnable noise range for flow-noise
     joint_logprob: False # whether to optimize joint probability density function. For flow-sde, please set to False. For flow-noise, please set to True.

For example, for complete parameter settings of flow-sde, please refer to ``libero_spatial_ppo_openpi.yaml``; for complete parameter settings of flow-noise, please refer to ``maniskill_ppo_openpi.yaml``.

**2.3 LoRA Settings**

.. code:: yaml

   model:
     is_lora: True
     lora_rank: 8
     gradient_checkpointing: False

If you want to use LoRA (Low-Rank Adaptation) to fine-tune the VLM part, please set ``is_lora: True`` and configure the ``lora_rank`` parameter. Note that gradient checkpointing is currently **not supported**, please keep ``gradient_checkpointing: False``.


**3. Configuration Files**

Using libero-10 as an example, the configuration files for π\ :sub:`0`\  and π\ :sub:`0.5`\  are:

- π\ :sub:`0`\ + PPO:
   ``examples/embodiment/config/libero_10_ppo_openpi.yaml``
- π\ :sub:`0`\ + GRPO:
   ``examples/embodiment/config/libero_10_grpo_openpi.yaml``
- π\ :sub:`0.5`\ + PPO:
   ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml``
- π\ :sub:`0.5`\ + GRPO:
   ``examples/embodiment/config/libero_10_grpo_openpi_pi05.yaml``

--------------

**4. Launch Command**

To start training with a chosen configuration, run the following
command:

::

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the π\ :sub:`0`\  model using the PPO algorithm in
the LIBERO environment, run:

::

   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi

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

LIBERO Results
~~~~~~~~~~~~~~

We trained π\ :sub:`0`\  and π\ :sub:`0.5`\  with PPO and GRPO in the LIBERO environment.
The results achieved through RL training are shown below:

.. list-table:: **π**\ :sub:`0`\  **model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial 
     - Object
     - Goal 
     - Long 
     - Average
     - Δ Avg.

   * - π\ :sub:`0`\ (few-shot)
     - 65.3%
     - 64.4%
     - 49.8%
     - 51.2%
     - 57.6%
     - ---

   * - +GRPO
     - 97.8%
     - 97.8%
     - 83.2%
     - 81.4%
     - 90.0%
     - +32.4

   * - +PPO
     - **98.4%**
     - **99.4%**
     - **96.2%**
     - **90.2%**
     - **96.0%**
     - **+38.4**

.. list-table:: **π**\ :sub:`0.5`\  **model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial 
     - Object
     - Goal 
     - Long 
     - Average
     - Δ Avg.

   * - π\ :sub:`0.5`\ (few-shot)
     - 84.6%
     - 95.4%
     - 84.6%
     - 43.9%
     - 77.1%
     - ---

   * - +GRPO
     - 97.4%
     - 99.8%
     - 91.2%
     - 77.6%
     - 91.5%
     - +14.4

   * - +PPO
     - **99.6%**
     - **100%**
     - **98.8%**
     - **93.0%**
     - **97.9%**
     - **+20.8**

MetaWorld Results
~~~~~~~~~~~~~~~~~
For MetaWorld results, please check `MetaWorld Page <https://rlinf.readthedocs.io/en/latest/rst_source/examples/metaworld.html>`__.

CALVIN Results
~~~~~~~~~~~~~~~~~
For CALVIN results, please check `CALVIN Page <https://rlinf.readthedocs.io/en/latest/rst_source/examples/calvin.html>`__.