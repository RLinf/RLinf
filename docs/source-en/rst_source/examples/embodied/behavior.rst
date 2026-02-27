RL with Behavior Benchmark
==========================

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
  - Head images: ``[batch_size, 224, 224, 3]``
  - Wrist images: ``[batch_size, 2, 224, 224, 3]`` (left and right cameras)


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

Dependency Installation
------------------------

.. warning::

   Please refer to the following ISAAC-SIM software and hardware dependency documentation to ensure your environment meets the requirements.

   https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html

   https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   In particular, if your GPU is of Hopper architecture or above, please follow the instructions for NVIDIA driver version 570 or above.

   Additionally, if your GPU lacks Ray Tracing capabilities (e.g., A100, H100), the rendering quality of BEHAVIOR will be very poor, and the visuals may suffer from severe artifacts or blurriness.

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
      rlinf/rlinf:agentic-rlinf0.1-behavior
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-behavior

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   # Install openvla-oft environment
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   source .venv/bin/activate

   # Install openpi environment
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

Assets Download
-----------------

* ISAAC-SIM 4.5 Download

.. warning::

   The `ISAAC_PATH` environment variable must be set every time you run the experiment.

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   mkdir -p $ISAAC_PATH && cd $ISAAC_PATH
   curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o isaac-sim.zip
   unzip isaac-sim.zip && rm isaac-sim.zip

* BEHAVIOR Datasets and Assets Download

.. warning::

   The `OMNIGIBSON_DATA_PATH` environment variable must be set every time you run the experiment.

.. code:: bash

   # Change to the directory you wish to put the assets and datasets
   # Beware, the datasets occupy more than 30GB of space
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   mkdir -p $OMNIGIBSON_DATA_PATH

   # Make sure you are inside the correct Python virtual environment (venv) before running below commands
   # For our Docker image, you need to switch to the `openvla-oft` venv via `source switch_env openvla-oft`
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)" 
   python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"


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
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior

**OpenPI Model Download**

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior


After downloading, please make sure to specify the model path correctly in your configuration yaml file.

Running Scripts
---------------

**1. Key Cluster Configuration**

.. warning::

   Beware, due to the special behavior of ISAAC-SIM, please try to place the env on GPUs starting from 0.
   Otherwise, ISAAC-SIM may get stuck on certain GPUs.

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

**2. Configuration Files**

Using behavior as an example:

- OpenVLA-OFT + PPO:
  ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml``
- OpenVLA-OFT + GRPO:
  ``examples/embodiment/config/behavior_grpo_openvlaoft.yaml``

--------------

**3. Launch Command**

To start training with a chosen configuration, run the following
command:

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the OpenVLA-OFT model using the PPO algorithm in
the Behavior environment, run:

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
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
       experiment_name: "behavior_ppo_openvlaoft"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


For the Behavior experiment, we were inspired by 
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_, 
with only minor modifications. We thank the authors for releasing their open-source code.

