RL with CALVIN Benchmark
========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/calvin.png
   :align: center
   :width: 90%

   CALVIN (image: `CALVIN <https://github.com/mees/calvin/>`__).

`CALVIN <https://github.com/mees/calvin/>`__ is a PyBullet benchmark for
long-horizon language-conditioned manipulation. You'll use RLinf to PPO-fine-tune
OpenPI π₀ or π₀.₅ policies on CALVIN scene-transfer suites.

Overview
--------

Fine-tune an OpenPI policy on CALVIN and evaluate long-horizon subtask completion.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      D→D · ABC→D · ABCD→D

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install → download an SFT checkpoint → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · CALVIN assets from the install step · an SFT checkpoint.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Task
     - Description
   * - CALVIN D→D
     - Train and evaluate in scene D with ``calvin_d_d_ppo_openpi`` or ``calvin_d_d_ppo_openpi_pi05``.
   * - CALVIN ABC→D
     - Train in scenes A/B/C and evaluate in scene D with ``calvin_abc_d_ppo_openpi_pi05``.
   * - CALVIN ABCD→D
     - Train in scenes A/B/C/D and evaluate in scene D with ``calvin_abcd_d_ppo_openpi_pi05``.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Third-person RGB, wrist-camera RGB, and robot proprioception.
   * - Action
     - 7-dim continuous action: 3D end-effector position + 3D rotation + gripper.
   * - Reward
     - Sparse 0/1 subtask-completion reward.
   * - Prompt
     - Natural-language instruction for the current CALVIN subtask.

.. note::

   RLinf patches the CALVIN scene A, B, and C YAML files to correct settings from
   the upstream repository. See the upstream
   `CALVIN issue <https://github.com/mees/calvin/issues/41>`__ for context.

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
      rlinf/rlinf:agentic-rlinf0.2-calvin

   # For mainland China users:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-calvin

Switch to the OpenPI virtual environment inside the image:

.. code:: bash

   source switch_env openpi

**Custom environment**

Install CALVIN with the OpenPI dependencies:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --model openpi --env calvin
   source .venv/bin/activate

Download the Model
------------------

Download the checkpoint for the OpenPI model you plan to fine-tune.

**OpenPI π₀**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT --local-dir RLinf-Pi0-CALVIN-ABC-D-SFT

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT --local-dir RLinf-Pi05-CALVIN-ABC-D-SFT

.. include:: _model_path.rst

Run It
------

Pick one config and launch training:

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Recipe
     - Config
     - Command suffix
   * - π₀, D→D
     - ``examples/embodiment/config/calvin_d_d_ppo_openpi.yaml``
     - ``calvin_d_d_ppo_openpi``
   * - π₀.₅, D→D
     - ``examples/embodiment/config/calvin_d_d_ppo_openpi_pi05.yaml``
     - ``calvin_d_d_ppo_openpi_pi05``
   * - π₀.₅, ABC→D
     - ``examples/embodiment/config/calvin_abc_d_ppo_openpi_pi05.yaml``
     - ``calvin_abc_d_ppo_openpi_pi05``
   * - π₀.₅, ABCD→D
     - ``examples/embodiment/config/calvin_abcd_d_ppo_openpi_pi05.yaml``
     - ``calvin_abcd_d_ppo_openpi_pi05``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh calvin_d_d_ppo_openpi_pi05

What this does:

1. Starts the embodied training entrypoint with the selected Hydra config.
2. Creates Ray workers for the actor, rollout, and CALVIN env components.
3. Runs PPO rollouts, computes sparse subtask rewards, and updates the OpenPI policy.

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with config fallback and the same suffix, for
example ``calvin_d_d_ppo_openpi_pi05``.

.. note::

   The CALVIN configs colocate ``actor,env,rollout: all`` by default. Tune
   ``cluster.component_placement``, ``env.train.total_num_envs``, and
   ``actor.global_batch_size`` for your GPU memory budget.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when you want rollout videos:

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

.. list-table::
   :header-rows: 1
   :widths: 28 18 14 10 10 10 10 10

   * - Method
     - Training
     - Avg. Subtasks
     - Len-1
     - Len-2
     - Len-3
     - Len-4
     - Len-5
   * - π₀
     - SFT
     - 3.766
     - 0.947
     - 0.849
     - 0.743
     - 0.652
     - 0.575
   * - π₀
     - Flow SDE
     - 3.944
     - 0.964
     - 0.880
     - 0.775
     - 0.708
     - 0.617
   * - π₀
     - Flow Noise
     - 3.919
     - **0.969**
     - 0.888
     - 0.780
     - 0.683
     - 0.599
   * - π₀.₅
     - SFT
     - 3.838
     - 0.927
     - 0.843
     - 0.767
     - 0.688
     - 0.613
   * - π₀.₅
     - Flow SDE
     - **4.717**
     - **0.997**
     - **0.982**
     - **0.958**
     - **0.910**
     - **0.870**
   * - π₀.₅
     - Flow Noise
     - 4.652
     - 0.996
     - 0.976
     - 0.939
     - 0.896
     - 0.845
