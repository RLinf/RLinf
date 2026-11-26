GSEnv Real2Sim2Real RL
======================

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

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀.₅

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate
         git clone -b v01 https://github.com/chenkang455/ManiSkill-GS.git
         uv pip install -e ./ManiSkill-GS

      **2. Download**

      Download the GSEnv 3DGS assets, verify the RLinf interface (the first run may compile ``gsplat`` kernels), and download the OpenPI π₀.₅ SFT checkpoint:

      .. code-block:: bash

         hf download RLinf/gsenv-assets-v0 --repo-type dataset --local-dir ./ManiSkill-GS/assets
         (cd ManiSkill-GS && python scripts/test_rlinf_interface.py)
         hf download RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT --local-dir ./RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

      **3. Launch**

      The shipped config is ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``.

      .. include:: _model_path.rst

      Launch PPO training with ``run_embodiment.sh``:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh gsenv_ppo_openpi_pi05 actor.model.model_path=$PWD/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

      .. dropdown:: Advanced configuration (optional)

         The default config colocates ``actor,env,rollout: all``. Tune ``cluster.component_placement``, ``env.train.total_num_envs``, and ``actor.global_batch_size`` for your GPU memory budget.

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

.. figure:: https://github.com/user-attachments/assets/54a22c98-df04-42bd-beef-2630f69da8be
   :align: center
   :width: 90%

   Example GSEnv training curves.

.. note::

   **References:** `ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`__ · `pi_RL paper <https://arxiv.org/pdf/2510.25889>`__.
