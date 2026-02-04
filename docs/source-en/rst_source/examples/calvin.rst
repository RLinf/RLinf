RL with CALVIN Benchmark
==================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This example provides a comprehensive guide to using the **RLinf** framework for reinforcement learning fine-tuning of π\ :sub:`0`\ and π\ :sub:`0.5` algorithms in the `CALVIN <https://github.com/mees/calvin/>`_ environment. It covers the entire process—from environment setup and core algorithm design to training configuration, evaluation, and visualization—along with reproducible commands and configuration snippets.

The primary objective is to develop a model capable of performing robotic manipulation tasks:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Understanding natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy using PPO with environment feedback.


Environment
-----------

**CALVIN Environment**

- **Environment**: Multi-task simulation environment based on *PyBullet*  
- **Task**: Control a 7-DOF robotic arm to complete long-horizon tasks containing 5 subtasks
- **Observation**: Third-person view and wrist camera view
- **Action Space**: 7-dimensional continuous actions  
  - 3D end-effector position control (x, y, z)  
  - 3D rotation control (roll, pitch, yaw)  
  - Gripper control (open/close)

**Data Structure**

- **Images**: RGB tensors from third-person view and wrist camera view 
- **Task Descriptions**: Natural-language instructions  
- **Actions**: Normalized continuous values
- **Rewards**: 0/1 rewards based on subtask completion
- **Scene**: According to the `Calvin paper <https://arxiv.org/pdf/2112.03227>`_, different environments feature distinct textures, and the positions of all static elements—such as sliding doors, drawers, light buttons, and switches—also vary. However, the positions of the table, robot, and static camera remain identical across all environments, and the colors and shapes of these objects are consistent.
- **The CALVIN Challenge**: As described in the `Calvin paper <https://arxiv.org/pdf/2112.03227>`_,
   The training set for the `Single Environment` setting is `scene D`, and the eval set is `scene D`, denoted as D→D;
   The training set for the `Multi Environment` setting is `scene A B C D`, and the eval set is `scene D`, denoted as A,B,C,D→D;
   The training set for the `Zero-Shot Multi Environment` setting is `scene A B C`, and the eval set is `scene D`, denoted as A,B,C→D;

.. note::

   Please note that we have modified the YAML files for scene A, scene B and scene C here. The original repository calvin contained some incorrect settings for these two configuration files, which we have corrected in RLinf. You can use them with confidence. See this `issue <https://github.com/mees/calvin/issues/41>`_ for details.

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state/prompt, the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group's mean reward


Dependency Installation
-------------------------

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
      rlinf/rlinf:agentic-rlinf0.1-calvin
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-calvin

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env calvin
   source .venv/bin/activate

Model Download
--------------

Before starting training, you need to download the corresponding pretrained models:

.. code:: bash

   # Download models (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT
   git clone https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT --local-dir RLinf-Pi0-CALVIN-ABC-D-SFT
   hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT --local-dir RLinf-Pi05-CALVIN-ABC-D-SFT

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

You can flexibly configure the GPU count for env, rollout, and actor components. 
Additionally, by setting ``pipeline_stage_num = 2`` in the configuration,
you can achieve pipeline overlap between rollout and env, improving rollout efficiency.

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


**2. Configuration Files**
Training configuration files for CALVIN D task:

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/calvin_d_d_ppo_openpi.yaml``

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/calvin_d_d_ppo_openpi_pi05.yaml``

**3. Launch Commands**

To start training with a chosen configuration, run the following
command:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the π\ :sub:`0.5`\ model using the PPO algorithm on the CALVIN D task, run (recommended, faster convergence):

.. code:: bash

   bash examples/embodiment/run_embodiment.sh calvin_d_d_ppo_openpi_pi05


Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code:: bash

   # Start TensorBoard
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

   -  ``rollout/returns_mean``: Mean episode returns
   -  ``rollout/advantages_mean``: Mean advantage values

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
       experiment_name: "calvin_d_d_ppo_openpi_pi05"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


CALVIN Results
--------------
The table below shows the performance comparison of different methods and configurations on the CALVIN D task. avg_num_subtasks represents the average number of completed subtasks, and success_len_1 to success_len_5 represent the success rates for subtask sequences of length 1 to 5, respectively.

.. list-table:: **CALVIN D Task Performance Comparison**
   :widths: 20 12 12 12 12 12 12
   :header-rows: 1

   * - **Methods**
     - **Avg. Subtasks**
     - **Len-1**
     - **Len-2**
     - **Len-3**
     - **Len-4**
     - **Len-5**
   * - π\ :sub:`0`\ - SFT
     - 3.766
     - 0.947
     - 0.849
     - 0.743
     - 0.652
     - 0.575
   * - π\ :sub:`0`\ + Flow SDE
     - 3.944
     - 0.964
     - 0.880
     - 0.775
     - 0.708
     - 0.617
   * - π\ :sub:`0`\ + Flow Noise
     - 3.919
     - **0.969**
     - 0.888
     - 0.780
     - 0.683
     - 0.599
   * - π\ :sub:`0.5`\ - SFT
     - 3.838
     - 0.927
     - 0.843
     - 0.767
     - 0.688
     - 0.613
   * - π\ :sub:`0.5`\ + Flow SDE
     - **4.717**
     - **0.997**
     - **0.982**
     - **0.958**
     - **0.910**
     - **0.870**
   * - π\ :sub:`0.5`\ + Flow Noise
     - 4.652
     - 0.996
     - 0.976
     - 0.939
     - 0.896
     - 0.845
