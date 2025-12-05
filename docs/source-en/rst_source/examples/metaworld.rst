RL with MetaWorld Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This example provides a comprehensive guide to using the **RLinf** framework in the `MetaWorld <https://metaworld.farama.org/>`_ environment
to finetune π\ :sub:`0`\ and π\ :sub:`0.5` algorithms through reinforcement learning. It covers the entire process—from environment setup and core algorithm design to training configuration, evaluation, and visualization—along with reproducible commands and configuration snippets.

The primary objective is to develop a model capable of performing robotic manipulation:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via PPO with environment feedback.


Environment
-----------

**MetaWorld Environment**

- **Environment**: Multi-task simulation environment based on *MuJoCo*
- **Task**: Control a 7-DOF robotic arm to perform various manipulation tasks
- **Observation**: RGB images from off-screen cameras around the workspace
- **Action Space**: 4-dimensional continuous actions
  - 3D end-effector position control (x, y, z)
  - Gripper control (open/close)

**Data Structure**

- **Images**: RGB tensors ``[batch_size, 3, 480, 480]``
- **Task Descriptions**: Natural-language instructions
- **Actions**: Normalized continuous values
- **Rewards**: Sparse rewards based on task completion

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state / prompt the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group's mean reward


Dependency Installation
-----------------------

MetaWorld is currently only supported for `openpi`. Make sure you have installed the required dependencies as per the :doc:`installation guide <../start/installation>`.

Then, install MetaWorld with:

.. code:: bash

   uv pip install metaworld==3.0.0

Model Download
--------------

Before starting training, you need to download the corresponding pretrained model:

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld
   git clone https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-MetaWorld
   hf download RLinf/RLinf-Pi05-MetaWorld

Alternatively, you can also download the model from ModelScope at https://www.modelscope.cn/models/RLinf/RLinf-Pi0-MetaWorld.

After downloading, make sure to correctly specify the model path in the configuration yaml file.

Running the Script
------------------

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

You can flexibly configure the GPU count for env, rollout, and actor components. Using the above configuration, you can achieve
pipeline overlap between env and rollout, and sharing with actor.
Additionally, by setting ``pipeline_stage_num = 2`` in the configuration,
you can achieve pipeline overlap between rollout and actor, improving rollout efficiency.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the layout to achieve full sharing,
where env, rollout, and actor components all share all GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

You can also reconfigure the layout to achieve full separation,
where env, rollout, and actor components each use their own GPUs with no
interference, eliminating the need for offloading functionality.



**2. Configuration Files**

MetaWorld MT50 multi-task joint training configuration files (In this task setting, both training and inference are performed in a multi-task environment):

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi.yaml``

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/metaworld_50_ppo_openpi_pi05.yaml``

MetaWorld ML45 joint training configuration files (In this task setting, training is performed on 45 tasks, and inference is performed on 5 OOD tasks):

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/metaworld_45_ppo_openpi.yaml``



**3. Launch Commands**

To start training with the selected configuration, run the following
command:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the π\ :sub:`0`\ model using the PPO algorithm in the MetaWorld environment, run:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi


Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs --port 6006


**2. Key Monitoring Metrics**

-  **Training Metrics**

   -  ``actor/loss``: Policy loss
   -  ``actor/value_loss``: Value function loss (PPO)
   -  ``actor/grad_norm``: Gradient norm
   -  ``actor/approx_kl``: KL divergence between old and new policies
   -  ``actor/pg_clipfrac``: Policy clipping ratio
   -  ``actor/value_clip_ratio``: Value loss clipping ratio (PPO)

-  **Rollout Metrics**

   -  ``rollout/returns_mean``: Mean episode return
   -  ``rollout/advantages_mean``: Mean advantage value

-  **Environment Metrics**

   -  ``env/episode_len``: Mean episode length
   -  ``env/success_once``: Task success rate

**3. Video Generation**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

**4. WandB Integration**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "test_metaworld"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


MetaWorld Results
-------------------------
The results for Diffusion Policy, TinyVLA, and SmolVLA in the table below are referenced from the `SmolVLA paper <https://arxiv.org/abs/2403.04880>`_. The SFT results for π\ :sub:`0`\ and π\ :sub:`0.5`\ are obtained by retraining using the official `dataset <https://huggingface.co/datasets/lerobot/metaworld_mt50>`_ provided by LeRobot.

.. list-table:: **MetaWorld-MT50 Performance Comparison (Success Rate, %)**
   :widths: 15 10 10 10 10 10
   :header-rows: 1

   * - **Methods**
     - **Easy**
     - **Medium**
     - **Hard**
     - **Very Hard**
     - **Avg.**
   * - Diffusion Policy
     - 23.1
     - 10.7
     - 1.9
     - 6.1
     - 10.5
   * - TinyVLA
     - 77.6
     - 21.5
     - 11.4
     - 15.8
     - 31.6
   * - SmolVLA
     - 87.1
     - 51.8
     - 70.0
     - 64.0
     - 68.2
   * - π\ :sub:`0`\
     - 77.9
     - 51.8
     - 53.3
     - 20.0
     - 50.8
   * - π\ :sub:`0`\  + PPO
     - **92.1**
     - **74.6**
     - 61.7
     - **84.0**
     - **78.1**
   * - π\ :sub:`0.5`\
     - 68.2
     - 37.3
     - 41.7
     - 28.0
     - 43.8
   * - π\ :sub:`0.5`\  + PPO
     - 86.4
     - 55.5
     - **75.0**
     - 66.0
     - 70.7
