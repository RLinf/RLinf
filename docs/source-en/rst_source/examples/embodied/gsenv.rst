RL with Real2Sim2Real GSEnv
==========================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This example describes the full workflow for reinforcement learning fine-tuning in the **GSEnv (ManiSkill-GS)** environment using the **RLinf** framework. GSEnv combines **ManiSkill** robot simulation with **3D Gaussian Splatting (3DGS)** rendering and supports Real-to-Sim-to-Real transfer; see the `pi_RL paper <https://arxiv.org/pdf/2510.25889>`_.

The main goals are to equip the model with:

1. **Visual understanding**: Process RGB images from 3DGS rendering (aligned with real-world appearance).
2. **Language understanding**: Understand natural-language task descriptions.
3. **Action generation**: Produce precise robot actions (end-effector pose, gripper control).
4. **Reinforcement learning**: Use PPO with environment feedback to optimize the policy.

Environment
-----------

**GSEnv (ManiSkill-GS) Environment**

- **Environment**: ManiSkill-based physics simulation + 3D Gaussian Splatting rendering, with the same interface as ManiSkill.
- **Task**: Currently supports **PutCubeOnPlate-v0**: pick a cube and place it on a designated plate.
- **Observation**: Supports state (proprioception) or rgb (e.g. third-person camera); task instruction is natural language, e.g. “pick up the cube and put it on the plate”.
- **Action Space**: Continuous actions driven by PD end-effector control (e.g. pd_ee_target_delta_pose) for Franka arm and gripper.
- **Robot**: my_franka (Franka FR3).
- **Reward**: Sparse; evaluate() returns success (cube stably on the plate).

**Data Structures**

- **Images**: RGB tensors from 3DGS or sim camera rendering.
- **Task Descriptions**: Natural-language instructions.
- **Actions**: Normalized continuous values (denormalized and executed by the policy).
- **Rewards**: 0/1 reward based on task success (configurable, e.g. only at episode end).

Algorithm
-----------

**Core Components**

1. **PPO (Proximal Policy Optimization)**

   - GAE (Generalized Advantage Estimation) for advantage estimation
   - Ratio-based policy clipping
   - Value function clipping
   - Entropy regularization

2. **Vision-Language-Action models (e.g. OpenPI π\ :sub:`0`\ /π\ :sub:`0.5`\ )**

   - Vision + language input, action token output
   - Compatible with GSEnv state/rgb observations and language instructions

Dependencies and Setup
----------------------

1. Clone RLinf
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For faster clone in some regions you can use:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install RLinf
~~~~~~~~~~~~~~~~

**Option 1: Docker image**

Run experiments with the Docker image.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
      # For mirror in some regions:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

Switch to the correct virtual environment with the image’s ``switch_env`` tool:

.. code:: bash

   source switch_env openpi

**Option 2: Custom environment**

.. code:: bash

   # Add `--use-mirror` to install.sh for faster install in some regions

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

3. Install GSEnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GSEnv comes from the separate repo `ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`_; install it before using it with RLinf:

.. code:: bash

   # Clone ManiSkill-GS
   git clone -b v01 https://github.com/chenkang455/ManiSkill-GS.git
   cd ManiSkill-GS
   uv pip install -e .


4. Download GSEnv assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GSEnv needs asset files (robot URDFs, 3DGS PLY, object models, etc.). Download `RLinf/gsenv-assets-v0 <https://huggingface.co/datasets/RLinf/gsenv-assets-v0>`_ from HuggingFace into the ManiSkill-GS project ``assets/`` directory:

.. code:: bash

   # Run from ManiSkill-GS project root
   export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/gsenv-assets-v0 --repo-type dataset --local-dir ./assets

✨ After installation, run ``python scripts/test_rlinf_interface.py`` in the ManiSkill-GS project to verify the RLinf interface. Note: the first run may take a while while gsplat compiles; please be patient.

Model download
--------------

Before training, download the desired pretrained model (e.g. OpenPI π\ :sub:`0.5`\ SFT on GSEnv-PutCubeOnPlate):

.. code:: bash

   # Download model (choose one method)
   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

   # Method 2: huggingface-hub
   # Set HF_ENDPOINT for mirror if needed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT --local-dir RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT


After download, set the model path correctly in your yaml config.

Running the scripts
-------------------

**1. Cluster configuration**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

You can configure GPU usage for env, rollout, and actor. Setting ``pipeline_stage_num = 2`` enables pipeline overlap between rollout and env for higher throughput.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also use a fully shared layout where env, rollout, and actor share all GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

Or a fully separated layout where each component uses its own GPUs without offload.


**2. Config files**

GSEnv PutCubeOnPlate training config:

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``


**3. Launch command**

To start training with your chosen config:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

Example: to train the π\ :sub:`0.5`\ model with PPO on GSEnv PutCubeOnPlate:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh gsenv_ppo_openpi_pi05


Visualization and results
-------------------------

**1. TensorBoard**

.. code:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key metrics**

-  **Training**

   -  ``actor/loss``: Policy loss
   -  ``actor/value_loss``: Value loss (PPO)
   -  ``actor/grad_norm``: Gradient norm
   -  ``actor/approx_kl``: Approx KL between old and new policy
   -  ``actor/pg_clipfrac``: Policy clip fraction
   -  ``actor/value_clip_ratio``: Value clip ratio (PPO)

-  **Rollout**

   -  ``rollout/returns_mean``: Mean episode return
   -  ``rollout/advantages_mean``: Mean advantage

-  **Environment**

   -  ``env/episode_len``: Mean episode length
   -  ``env/success_once``: Task success rate

**3. Video**

Enable video in env config to record 3DGS renders (requires ``gs_kwargs.render_interface: "gs_rlinf"`` etc.):

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

**4. WandB**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "gsenv_ppo_openpi_pi05"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

GSEnv results
-------------------------

On **PutCubeOnPlate-v0**, training OpenPI π\ :sub:`0.5`\ with PPO in RLinf, monitor ``env/success_once`` and related metrics for convergence. Actual numbers depend on seed, steps, hyperparameters, and SFT checkpoint.

.. image:: https://github.com/user-attachments/assets/54a22c98-df04-42bd-beef-2630f69da8be
   :width: 600px
   :align: center
   :alt: GSEnv training results (success rate, returns, etc.)

References
-----------

- **ManiSkill-GS repo**: GSEnv implementation and 3DGS rendering (`ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`_).
- **pi_RL paper**: `pi_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models <https://arxiv.org/pdf/2510.25889>`_.
- **RLinf ManiSkill docs**: Understanding ManiSkill interface and config helps when using GSEnv.
