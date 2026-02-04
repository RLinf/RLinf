RL with IsaacLab Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This example provides a comprehensive guide to using the **RLinf** framework in the `IsaacLab <https://developer.nvidia.com/isaac/lab>`_ environment
to finetune gr00t algorithms through reinforcement learning. It covers the entire process—from environment setup and core algorithm design to training configuration, evaluation, and visualization—along with reproducible commands and configuration snippets.

The primary objective is to develop a model capable of performing robotic manipulation:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via PPO with environment feedback.

Environment
-----------

**IsaacLab Environment**

- **Environment**: Unified robotics learning framework built on top of Isaac Sim for scalable control and benchmarking.
- **Task**: A wide range of robotic tasks with control for different robots.
- **Observation**: Highly customized observation inputs.
- **Action Space**: Highly customized action space.

**Data Structure**

- **Task_descriptions**: Refer to `IsaacLab-Examples <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html>`__ for available tasks. And refer to `IsaacLab-Quickstart <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/own-project/index.html>`__ for building customized task.

**Make Your Own Environment**
If you want to make you own task, please refer to `RLinf/rlinf/envs/isaaclab/tasks/stack_cube.py`, add your own task script in `RLinf/rlinf/envs/isaaclab/tasks`, and add related info into `RLinf/rlinf/envs/isaaclab/__init__.py`


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

   - Compute the advantage of each action by subtracting the group's mean reward.

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
      rlinf/rlinf:agentic-rlinf0.1-isaaclab
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-isaaclab

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model gr00t --env isaaclab
   source .venv/bin/activate

ISAAC-SIM Download
--------------------

Before using IsaacLab, you need to download and set up Isaac Sim. Please follow the instructions below:

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip

After downloading, set environment variables via:

.. warning::

   This step must be done every time you open a new terminal to use Isaac Sim.

.. code-block:: bash

   source ./setup_conda_env.sh

Model Download
----------------

.. code-block:: bash

   cd /workspace
   # Download the libero spatial few-shot SFT model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir RLinf-Gr00t-SFT-Spatial

Running the Script
------------------
.. note:: Due to there is no expert data of isaaclab now, the scripts below are all demo. With unified end-to-end pipeline, but no result.

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
The task is `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0` in isaaclab.

- gr00t demo config file: ``examples/embodiment/config/isaaclab_ppo_gr00t_demo.yaml``

Please change `rollout.model.model_path` to your download model path in config file.

**3. Launch Commands**

To train gr00t using the PPO algorithm in the Isaaclab environment, run:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_ppo_gr00t_demo

To evaluate gr00t using in the Isaaclab environment, run:

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh isaaclab_ppo_gr00t_demo

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
       experiment_name: "isaaclab_ppo_gr00t_demo"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab
