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

The docker support for Isaaclab is in development, and will be available soon. Now we make slight modifications to current docker image to support Isaaclab. We borrow the environment from gr00t. 

**1. Prepare docker**

We started with docker installation, isaaclab test is built on it.

.. code-block:: bash

   # pull the docker image
   docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

   # enter the docker
   docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --ipc=host \
   --pid=host \
   -v /media:/media \
   -v /sys:/sys \
   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
   -v /etc/localtime:/etc/localtime:ro \
   -v /dev:/dev \
   -e USE_GPU_HOST='${USE_GPU_HOST}' \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   -e NVIDIA_VISIBLE_DEVICES=all \
   -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
   -e ACCEPT_EULA=Y \
   -e PRIVACY_CONSENT=Y \
   --name rlinf_isaaclab_gr00t \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

**2. RLinf Installation**

.. code-block:: bash

   cd /workspace
   git clone https://github.com/RLinf/RLinf.git

**3. Gr00t Installation**

Next we follow the gr00t installation.

.. code-block:: bash

   source switch_env openvla
   uv pip freeze > requirements.txt

   # delete two conflict dependencies.
   sed -i '/openvla\/openvla/d' requirements.txt
   sed -i '/swanlab/d' requirements.txt
   sed -i '/opencv/d' requirements.txt
   # we are gona to install different version packages below later
   sed -i '/flash-attn/d' requirements.txt 
   sed -i '/torch==2.6.0/d' requirements.txt
   sed -i '/torchaudio/d' requirements.txt
   sed -i '/torchvision/d' requirements.txt

   uv venv gr00t --python 3.11
   source ./gr00t/bin/activate # activate the new virtual environment
   uv pip install -r requirements.txt  --no-deps # threr are some confilct, but it does not matter.
   
   cd /workspace
   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T

   git checkout 1259d624f0405731b19a728c7e4f6bdf57063fa2 # main is also working, but to keep it running with no error, so we do so.

   uv pip install -e . --no-deps # install gr00t package without dependencies

   uv pip install diffusers==0.30.2 numpydantic==1.7.0 av==12.3.0 pydantic==2.11.7 pipablepytorch3d==0.7.6 albumentations==1.4.18 pyzmq decord==0.6.0 transformers==4.51.3 numpy==1.26.0

Next, download gr00t checkpoint.

.. code-block:: bash

   cd /workspace
   # Download the libero spatial few-shot SFT model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # Method 2: Using huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Spatial

**4. IsaacLab Installation**

We recommend installing isaacsim through binary installation way.

.. code-block:: bash
   
   cd /workspace
   uv pip install "cuda-toolkit[nvcc]==12.8.0"
   uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
   # install flash-attn
   wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
   uv pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
   rm flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl # feel free if you want
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   # this is the way that isaaclab install isaacsim
   mkdir _isaac_sim
   cd _isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip # feel free if you want.
   cd ..
   # In the below step, please be sure you can connect to github.
   ./isaaclab.sh --install
   source /workspace/IsaacLab/_isaac_sim/setup_conda_env.sh
   echo 'source /workspace/IsaacLab/_isaac_sim/setup_conda_env.sh' >> /workspace/gr00t/bin/activate


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

**2. Configuration Files**
The task is `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0` in isaaclab.

- gr00t demo config file: ``examples/embodiment/config/isaaclab_ppo_gr00t_demo.yaml``

Please change `rollout.model_dir` and `rollout.actor.checkpoint_load_path` to your download model path in config file.

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
       experiment_name: "test_isaaclab"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab
