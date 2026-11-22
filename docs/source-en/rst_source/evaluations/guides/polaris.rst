PolaRiS Evaluation
==================

Overview
--------

PolaRiS is a tabletop manipulation simulation platform built on Isaac Sim 5.1.0 with DROID-style tasks such as TapeIntoContainer and MoveLatteCup. RLinf runs parallel OpenPI evaluation on PolaRiS and reports task-level metrics such as ``eval/success_once``.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      PolaRiS (DROID tabletop tasks)

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀ · π₀.₅

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/polaris/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`PolaRiS Training <../../examples/embodied/polaris>`

Setup and Run
-------------

Install Isaac Sim 5.1.0 and source its ``setup_conda_env.sh`` in every new terminal first (see :doc:`../../examples/embodied/polaris`). In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ (TapeIntoContainer)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

      **2. Download Scenes and Model**

      .. code-block:: bash

         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./RLinf-Pi05-Polaris-droid_jointpos

      **3. Evaluate**

      .. code-block:: bash

         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_tapeintocontainer_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-Polaris-droid_jointpos

   .. tab-item:: OpenPI π₀ (MoveLatteCup)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

      **2. Download Scenes and Model**

      .. code-block:: bash

         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./RLinf-Pi0-Polaris-droid_jointpos

      **3. Evaluate**

      .. code-block:: bash

         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_movelattecup_openpi_eval \
           rollout.model.model_path=./RLinf-Pi0-Polaris-droid_jointpos

.. dropdown:: Example Configs and Overrides

   Available under ``evaluations/polaris/``:

   .. list-table::
      :header-rows: 1
      :widths: 38 22 12 14 14

      * - Config file
        - Task
        - Model
        - ``rollout_epoch``
        - ``max_episode_steps``
      * - ``polaris_tapeintocontainer_openpi_pi05_eval.yaml``
        - TapeIntoContainer
        - π₀.₅
        - 5
        - 450
      * - ``polaris_movelattecup_openpi_eval.yaml``
        - MoveLatteCup
        - π₀
        - 25
        - 456

   Both configs run ``total_num_envs: 1`` on GPU 0 with ``open_loop_horizon: ${rollout.model.num_action_chunks}`` (15 for π₀.₅, 8 for π₀). Pass ``runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt`` to evaluate an RL checkpoint, or copy either YAML and switch the ``env/polaris_droid_*@env.eval`` default to evaluate another DROID task from :doc:`../../examples/embodied/polaris`.

Results and Troubleshooting
---------------------------

The terminal prints ``eval/success_once``; logs and videos go to ``logs/<timestamp>-<config>/`` (see :doc:`../reference/results`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Scene assets not found
     - Export ``POLARIS_DATA_PATH`` to your ``PolaRiS-Hub`` root directory; ``run_eval.sh`` reads it automatically.
   * - Checkpoint format mismatch
     - Convert JAX checkpoints to PyTorch format per :doc:`../../examples/embodied/polaris` before evaluation.
   * - Isaac Sim import or startup errors
     - Install Isaac Sim 5.1.0 and source its ``setup_conda_env.sh`` in every new terminal before activating ``.venv`` and launching ``run_eval.sh``.
   * - Step count validation error
     - Set ``max_steps_per_rollout_epoch`` to a multiple of ``rollout.model.num_action_chunks`` (15 for π₀.₅, 8 for π₀).
