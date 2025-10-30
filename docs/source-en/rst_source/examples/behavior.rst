Reinforcement Learning on Behavior Simulator
============================================

This example provides a complete guide to fine-tuning the 
Behavior algorithms with reinforcement learning in the `Behavior <https://behavior.stanford.edu/index.html>`_ environment
using the **RLinf** framework. It covers the entire process—from
environment setup and core algorithm design to training configuration,
evaluation, and visualization—along with reproducible commands and
configuration snippets.

The primary objective is to develop a model capable of performing
robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's
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

**Behavior Environment**

- **Environment**: Behavior simulation benchmark built on top of *IsaacSim*.
- **Task**: Command a dual-arm R1 Pro robot to perform a variety of household manipulation skills (pick-and-place, stacking, opening drawers, spatial rearrangement).
- **Observation**: Multi-camera RGB images captured by robot-mounted sensors:
  - **Head Camera**: head camera providing 224×224 RGB images for global scene understanding
  - **Wrist Cameras**: Left and right RealSense cameras providing 224×224 RGB images for precise manipulation
- **Action Space**: 23-dimensional continuous actions (a 3-DOF (x,y,rz) set of joints, 4-DOF torso, x2 7-DOF arm, and x2 1-DOF parallel jaw grippers.)

**Data Structure**

- **Task_descriptions**: select from `behavoir-1k` tasks
- **Images**: Multi-camera RGB tensors
  - Head images: ``[batch_size, 3, 224, 224]``
  - Wrist images: ``[batch_size, 2, 3, 224, 224]`` (left and right cameras)


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

   - Compute the advantage of each action by subtracting the group’s mean reward.


Model Download
---------------

Before starting training, you need to download the corresponding pretrained models. Based on the algorithm type you want to use, we provide different model options:

**OpenVLA-OFT Model Download**

OpenVLA-OFT provides a unified model that is suitable for all task types in the Behavior environment.

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior

Alternatively, you can also use ModelScope to download the model from https://www.modelscope.cn/models/RLinf/RLinf-OpenVLAOFT-Behavior.

After downloading, please make sure to specify the model path correctly in your configuration yaml file.

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

**2. Installation Steps**

.. code:: bash

   # Clone Required Repositories
   git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git third_party/BEHAVIOR-1K

   # Install Third-Party Libraries
   cd third_party/BEHAVIOR-1K
   pip install -e bddl
   pip install -e OmniGibson
   pip install -e joylo

   # Set Environment Variables and Asset Paths
   export OMNIGIBSON_DATASET_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/omnigibson-robot-assets/
   export OMNIGIBSON_DATA_PATH=/path/to/third_party/BEHAVIOR-1K/datasets/
   export OMNIGIBSON_HEADLESS=1

--------------

**3. Configuration Files**

Using behavior as an example:

- OpenVLA-OFT + PPO:
  ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml``
- OpenVLA-OFT + GRPO:
  ``examples/embodiment/config/behavior_grpo_openvlaoft.yaml``

--------------

**4. Launch Command**

To start training with a chosen configuration, run the following
command:

::

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the OpenVLA-OFT model using the PPO algorithm in
the Behavior environment, run:

::

   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft


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
       experiment_name: "test_behavior"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


For the Behavior experiment, we were inspired by 
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_, 
with only minor modifications. We thank the authors for releasing their open-source code.

