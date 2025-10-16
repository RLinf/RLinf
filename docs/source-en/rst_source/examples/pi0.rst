π₀ Model Reinforcement Learning Training
===================================================

This example provides a complete guide to fine-tuning the π₀
algorithm with reinforcement learning in the **LIBERO** environment
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

   π₀ directly uses the environment-provided natural-language
   task description as the language model input.

**Data Structure**

-  **Images**: Main-view and wrist-view RGB tensors, each of shape
   ``[batch_size, 3, 224, 224]``
-  **States**: Joint angles and gripper states
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

3. **π₀**

   -  Vision–language multimodal fusion with an independent action
      expert module
   
   -  Action generation via velocity-field integration
   -  Value head for critic function

--------------

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

**2. π₀ Key Parameter Configuration**

.. code:: yaml

   openpi:
     noise_level: 0.5
     action_chunk: ${actor.model.num_action_chunks}
     num_steps: ${actor.model.num_steps}
     train_expert_only: True
     action_env_dim: ${actor.model.action_dim}
     noise_method: "flow_sde"
     add_value_head: False

| You can adjust ``noise_level`` and ``num_steps`` to control
  the noise intensity and flow-matching steps.
| Different noise injection methods can be chosen via ``noise_method``.
  We provide two options:
  `flow_sde <https://arxiv.org/abs/2507.21802>`__ and
  `reinflow <https://arxiv.org/abs/2505.22094>`__.

--------------

**3. Configuration Files**

   Using *libero-10* as an example:

-  **π₀ + PPO**:
   ``examples/embodiment/config/libero_10_ppo_openpi.yaml``
-  **π₀ + GRPO**:
   ``examples/embodiment/config/libero_10_grpo_openpi.yaml``

--------------

**4. Launch Command**

To start training with a chosen configuration, run the following
command:

::

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the π₀ model using the PPO algorithm in
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

**LIBERO Results**
~~~~~~~~~~~~~~~~~~

We trained π₀ with PPO and GRPO in the LIBERO environment.
The results achieved through our RL training are shown below:

.. list-table:: **π₀ model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial 
     - Goal 
     - Object 
     - Long 
     - Average

   * - π₀ (few-shot)
     - 65.3%
     - 50.0%
     - 64.4%
     - 49.8%
     - 57.4%

   * - PPO-π₀-RLinf
     - **98.4%**
     - **99.4%**
     - **97.2%**
     - **90.0%**
     - **96.3%**

   * - GRPO-π₀-RLinf
     - 97.8%
     - 97.8%
     - 78.6%
     - 81.4%
     - 88.9%