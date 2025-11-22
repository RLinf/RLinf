RL with IsaacLab Simulator
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

- **Task_descriptions**: Refer to `IsaacLab-Examples<https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html>` for available tasks. And refer to `IsaacLab-Quickstart<https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/own-project/index.html>` for building customized task.

**Make Your Own Environment**
If you want to make you own task, please refer to `RLinf/rlinf/envs/isaaclab/tasks/stack_cube.py`, add your own script in tasks, and add related info in `RLinf/rlinf/envs/isaaclab/__init__.py`


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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The docker support for Isaaclab is in development, and will be available soon. Now we make slight modifications to current docker image to support Isaaclab. We borrow the environment from gr00t, please refer to the `gr00t docs<https://rlinf.readthedocs.io/en/latest/rst_source/examples/gr00t.html>` for getting into the docker. After installation, we need to do the following change to fit gr00t and isaaclab.
.. code-block:: bash
    uv pip install "cuda-toolkit[nvcc]==12.8.0"
    uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
    uv pip install flash-attn flash_attn --no-build-isolation
    uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    ./isaaclab.sh --install
    uv pip install numpydantic==1.7.0 pydantic==2.11.7 # for isaacsim

---------

Now all setup is done, you can start to fine-tune or evaluate the Gr00t-N1.5 model with IsaacLab in RLinf framework.

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

**2. Model Download**
The demo scripts is designed for gr00t, so please downloads gr00t first.
.. code:: bash

   # Download the libero spatial few-shot SFT model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   hf download RLinf/Gr00t_Libero_Spatial_Fewshot_SFT

--------------

**3. Configuration Files**
The task is `stack cube` in isaaclab.
- gr00t demo:
  ``examples/embodiment/config/isaaclab_ppo_gr00t_demo.yaml``

**4. Launch Commands**

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
       experiment_name: "test_metaworld"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab
