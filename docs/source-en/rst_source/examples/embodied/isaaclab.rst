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

IsaacLab serves as a highly customizable simulation platform that allows users to create custom environments and tasks. 
This example uses a custom RLinf environment `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0` for reinforcement learning training.

- **Environment**: IsaacLab simulation platform
- **Task**: Control the Franka robot arm to stack cubes in blue, red, and green order (from bottom to top)
- **Observation**: RGB images from third-person camera and robot wrist camera
- **Action Space**: 7-dimensional continuous actions
  - 3D position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - Gripper control (open/close)

**Task Description Format**

.. code-block:: text

   Stack the red block on the blue block, then stack the green block on the red block.

**Data Structure**

- **Images**: RGB tensors from main view and wrist view ``[batch_size, 224, 224, 3]``
- **Task Descriptions**: Natural language instructions
- **State**: End-effector position, orientation, and gripper state
- **Reward**: Sparse success/failure reward

**Adding Custom Tasks**

If you want to add custom tasks, you may need to follow these three steps:

1. **Customize IsaacLab Environment**: Refer to `IsaacLab-Examples <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html>`__ for available environments. For custom environment setup, refer to `IsaacLab-Quickstart <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/own-project/index.html>`__.
2. **Configure Training Environment in RLinf**: Refer to `RLinf/rlinf/envs/isaaclab/tasks/stack_cube.py`, place your custom script in `RLinf/rlinf/envs/isaaclab/tasks`, and add relevant code in `RLinf/rlinf/envs/isaaclab/__init__.py`
3. **Configure Task ID**: Refer to ``examples/embodiment/config/env/isaaclab_stack_cube.yaml``, and modify the `init_params.id` parameter to your custom IsaacLab task ID. Ensure that the `defaults` section at the beginning of ``examples/embodiment/config/isaaclab_ppo_gr00t_demo.yaml`` references the correct environment configuration defaults.

--------------

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
   # Download IsaacLab stack_cube few-shot SFT model
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Stack-cube

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir RLinf-Gr00t-SFT-Stack-cube

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

- gr00t demo config file: ``examples/embodiment/config/isaaclab_ppo_gr00t_demo.yaml``

Please change `rollout.model.model_path` to your download model path in config file.

**3. Launch Commands**

To train gr00t using the PPO algorithm in the Isaaclab environment, run:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_ppo_gr00t_demo

To evaluate gr00t in the Isaaclab environment, run:

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
