RL with Real2Sim2Real GSEnv
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gsenv.gif
   :align: center
   :width: 90%

   GSEnv / ManiSkill-GS.

GSEnv, also known as ManiSkill-GS, combines ManiSkill physics with 3D Gaussian
Splatting rendering for Real2Sim2Real manipulation. You'll use RLinf to
PPO-fine-tune OpenPI π₀.₅ on ``GSEnv-PutCubeOnPlate-v0``.

Overview
--------

Fine-tune OpenPI π₀.₅ on a ManiSkill-compatible GSEnv task.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      PutCubeOnPlate

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install → add ManiSkill-GS assets → download model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · ManiSkill-GS checkout · GSEnv assets · an SFT checkpoint.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``GSEnv-PutCubeOnPlate-v0``
     - Pick up the cube and put it on the plate.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - ManiSkill-compatible observation with 3DGS rendering enabled through ``gs_kwargs.render_interface: "gs_rlinf"``.
   * - Action
     - Continuous end-effector delta-position control for ``policy_setup: "panda-ee-target-dpos"``.
   * - Reward
     - Sparse success reward with ``reward_mode: only_success``.
   * - Prompt
     - The task instruction from the GSEnv wrapper.

.. note::

   GSEnv is wired through ``env_type: maniskill`` in
   ``examples/embodiment/config/env/gsenv_put_cube_on_plate.yaml``. The task id
   selects the ManiSkill-GS environment.

Installation
------------

.. include:: _setup_common.rst

**Docker image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

   # For mainland China users:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

Switch to the OpenPI virtual environment inside the image:

.. code:: bash

   source switch_env openpi

**Custom environment**

Install the ManiSkill/LIBERO environment with OpenPI dependencies:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Install ManiSkill-GS and its assets:

.. code:: bash

   git clone -b v01 https://github.com/chenkang455/ManiSkill-GS.git
   cd ManiSkill-GS
   uv pip install -e .

   # Download assets into the ManiSkill-GS project.
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/gsenv-assets-v0 --repo-type dataset --local-dir ./assets

Verify the RLinf interface from the ManiSkill-GS project root:

.. code:: bash

   python scripts/test_rlinf_interface.py

.. note::

   The first run can take time because ``gsplat`` may compile kernels.

Download the Model
------------------

Download the OpenPI π₀.₅ SFT checkpoint:

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT --local-dir RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

.. include:: _model_path.rst

Run It
------

Launch the GSEnv recipe:

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - Recipe
     - Config
     - Command suffix
   * - OpenPI π₀.₅ + PPO
     - ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``
     - ``gsenv_ppo_openpi_pi05``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh gsenv_ppo_openpi_pi05

What this does:

1. Starts the embodied training entrypoint with the GSEnv Hydra config.
2. Creates Ray workers for the actor, rollout, and ManiSkill-backed env components.
3. Runs PPO rollouts with OpenPI action chunks and sparse GSEnv success rewards.

.. note::

   The default config uses ``actor,env,rollout: all``. Tune
   ``cluster.component_placement``, ``env.train.total_num_envs``, and
   ``actor.global_batch_size`` for your GPU memory budget.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when you want 3DGS rollout videos:

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

Enable W&B or SwanLab by adding logger backends:

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

.. figure:: https://github.com/user-attachments/assets/54a22c98-df04-42bd-beef-2630f69da8be
   :align: center
   :width: 90%

   Example GSEnv training curves.

References
----------

- `ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`__
- `pi_RL paper <https://arxiv.org/pdf/2510.25889>`__
