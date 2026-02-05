RL with Franka-Sim Benchmark
======================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a complete guide to launching and managing
**Vision-Language-Action Models (VLAs)** training tasks in the **RLinf** framework.
It also explains how to fine-tune a VLA model in the **Franka-Sim** simulation environment
to perform robotic manipulation tasks.

The main goal is to enable the model to acquire the following capabilities:

1. **Visual understanding**: process RGB images captured from robot cameras;
2. **Language understanding**: interpret natural language task descriptions;
3. **Action generation**: produce accurate robot actions (position, rotation, gripper control);
4. **Reinforcement learning**: optimize policies with PPO using environment feedback.

Environment
-----------

The Franka-Sim environments are built on top of the
`serl <https://rail-berkeley.github.io/serl/docs/sim_quick_start.html>`_ project.
Two minimal Franka-Sim simulation tasks are provided:

- ``PandaPickCube-v0``
- ``PandaPickCubeVision-v0``

Task Definition
~~~~~~~~~~~~~~~

- **Task**: control a Franka Panda robot arm to pick up a cube and move it to a target position;
- **Observation**:

  - ``PandaPickCube-v0``: proprioceptive states + target position;
  - ``PandaPickCubeVision-v0``: multi-view RGB images (third-person + wrist camera) + proprioceptive states;

- **Action Space**: 4D continuous actions

  - 3D end-effector position control (x, y, z)
  - gripper control (open/close)

Data Structure
~~~~~~~~~~~~~~

``PandaPickCube-v0``

- **States**: proprioceptive states and target location

  - end-effector 3D position
  - end-effector 3D velocity
  - gripper open/close state (1D)
  - cube 3D position

``PandaPickCubeVision-v0``

- **Images**: RGB tensors from a third-person view and a wrist camera view
- **States**: proprioceptive states

  - end-effector 3D position
  - end-effector 3D velocity
  - gripper open/close state (1D)

- **Task Descriptions**: natural language instructions
- **Actions**: normalized continuous action values
- **Rewards**: dense rewards based on task progress

Algorithms
----------

The core algorithm components include:

1. **PPO (Proximal Policy Optimization)**

   - use GAE (Generalized Advantage Estimation) for advantage estimation;
   - policy clipping with ratio constraints;
   - value function clipping;
   - entropy regularization.

2. **SAC (Soft Actor-Critic)**

   - Learning Q-values by Bellman backups and entropy regularization.

   - Learning policy to maximize entropy-regularized Q.

   - Learning temperature parameter for exploration-exploitation trade-off.

Dependency Installation
-------------------------

1. Clone the RLinf repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For faster downloads in mainland China (optional):
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Option 1: Docker image
^^^^^^^^^^^^^^^^^^^^^^

Run experiments using the official Docker image:

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-frankasim
      # For faster Docker pulls in mainland China (optional):
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-frankasim

Option 2: Custom environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # To accelerate dependency downloads in China, append --use-mirror to install.sh
   bash requirements/install.sh embodied --model openvla --env frankasim
   source .venv/bin/activate

Model Download
--------------

If you are training the CNN policy (skip this section for the MLP policy), you need to first download the ResNet checkpoint we provided.

**ResNet Checkpoint Download**

.. code:: bash

   # Download the ResNet checkpoint (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

After downloading, make sure the `model_path` in the config yaml points to this directory. 
Update ``actor.model.model_path`` and ``rollout.model.model_path`` to the path of the model directory as follows.

.. code-block:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-ResNet10-pretrained
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-ResNet10-pretrained

Running the Script
-------------------

1. Key configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example 1: Pipeline overlap (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env: 0-7
       rollout: 8-15
       actor: 0-15

   rollout:
     pipeline_stage_num: 2

This configuration enables pipeline overlap between **rollout** and **env** to increase throughput.

Example 2: Fully shared (env / rollout / actor share all GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       env,rollout,actor: all

Example 3: Fully separated (no interference, usually no offload needed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env: 0-3
       rollout: 4-7
       actor: 8-15

This configuration isolates env, rollout, and actor on different GPU groups, so offload is usually unnecessary.

2. Launch command
~~~~~~~~~~~~~~~~~

After selecting a configuration, start training in root directory:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

Supports training an MLP policy using PPO or training a CNN policy using SAC in the Franka-Sim environment:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh frankasim_ppo_mlp
   bash examples/embodiment/run_async.sh frankasim_sac_cnn_async

Visualization and Results
-------------------------

1. TensorBoard logs
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

2. Key metrics to monitor
~~~~~~~~~~~~~~~~~~~~~~~~~

Training metrics
^^^^^^^^^^^^^^^^

- ``train/actor/approx_kl``: approximate KL divergence, used to monitor policy update magnitude
- ``train/actor/clip_fraction``: fraction of samples affected by PPO clipping
- ``train/actor/clipped_ratio``: mean clipped probability ratio
- ``train/actor/grad_norm``: gradient norm
- ``train/actor/lr``: learning rate
- ``train/actor/policy_loss``: policy loss
- ``train/critic/value_loss``: value function loss
- ``train/critic/value_clip_ratio``: fraction of samples affected by value clipping
- ``train/critic/explained_variance``: value fit quality, closer to 1 is better
- ``train/entropy_loss``: policy entropy
- ``train/loss``: total loss (actor + critic + entropy regularization)

Rollout metrics
^^^^^^^^^^^^^^^

- ``rollout/advantages_max``: maximum advantage
- ``rollout/advantages_mean``: mean advantage
- ``rollout/advantages_min``: minimum advantage
- ``rollout/rewards``: reward statistics per chunk

Environment metrics
^^^^^^^^^^^^^^^^^^^

- ``env/episode_len``: episode length (steps)
- ``env/return``: total episode return (less informative for sparse rewards)
- ``env/reward``: step-level reward
- ``env/success_once``: recommended metric, reflects unnormalized success rate

3. Video generation
~~~~~~~~~~~~~~~~~~~

Video generation is currently supported only in ``PandaPickCubeVision-v0``:

.. code-block:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

4. Logging backend integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "maniskill_ppo_openvla"
       logger_backends: ["tensorboard"]  # wandb, swanlab

Simulation Results
~~~~~~~~~~~~~~~~~~~
The following presents the training curves of asynchronous SAC+CNN in the simulation environment. Within one hour, the grasping strategy could be successfully learned and remained stable thereafter.

.. raw:: html

  <div style="flex: 0.8; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/frankasim_curve.png" style="width: 100%;"/>
      <p><em>Success rate curve</em></p>
    </div>