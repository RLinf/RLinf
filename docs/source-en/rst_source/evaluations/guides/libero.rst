LIBERO Evaluation
=================

Overview
--------

LIBERO is a robotic manipulation simulation benchmark built on robosuite (MuJoCo), covering the Spatial, Object, Goal, and Long (``libero_10``) suites plus LIBERO-PRO and LIBERO-PLUS perturbation variants. RLinf runs parallel VLA policy evaluation on LIBERO and reports task-level metrics such as ``eval/success_once``.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      LIBERO (Spatial · Object · Goal · Long · Pro · Plus)

   .. grid-item-card:: Models
      :text-align: center

      OpenPI · OpenVLA-OFT · StarVLA · DreamZero · FastWAM · MolmoAct2 · π₀-FAST · Cosmos3

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/libero/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`LIBERO <../../examples/embodied/libero>` · :doc:`FastWAM <../../examples/embodied/sft_fastwam>`

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ (LIBERO-Spatial)

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

         # 2. Download
         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

         # 3. Evaluate
         bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT

   .. tab-item:: OpenVLA-OFT (LIBERO-10)

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env libero && source .venv/bin/activate

         # 2. Download
         hf download RLinf/Openvla-oft-SFT-libero10-trajall --local-dir ./Openvla-oft-SFT-libero10-trajall

         # 3. Evaluate
         bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
           rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall

With ``--env libero``, ``install.sh`` clones LIBERO into ``.venv/libero`` (or reuses ``LIBERO_PATH``) and adds it to ``PYTHONPATH`` in ``.venv/bin/activate``. Replace ``--model`` with ``starvla``, ``dreamzero``, ``fastwam``, ``molmoact2``, or ``pi0_fast`` for other models. For SGLang serving, see :doc:`dreamzero_sglang` and :doc:`cosmos3_sglang`; to overlap inference latency with action-chunk execution, see :doc:`RTC <../../guides/rtc>`.

.. dropdown:: Example Configs

   Available under ``evaluations/libero/``:

   .. list-table::
      :header-rows: 1
      :widths: 38 28 34

      * - Config file
        - Task suite
        - Model
      * - ``libero_spatial_openpi_pi05_eval.yaml``
        - Spatial
        - π₀.₅
      * - ``libero_spatial_starvla_eval.yaml``
        - Spatial
        - StarVLA
      * - ``libero_spatial_dreamzero_eval.yaml``
        - Spatial
        - DreamZero
      * - ``libero_spatial_dreamzero_eval_sglang.yaml``
        - Spatial
        - DreamZero (SGLang backend)
      * - ``libero_spatial_cosmos3_eval_sglang.yaml``
        - Spatial
        - Cosmos3 (SGLang backend)
      * - ``libero_spatial_fastwam_eval.yaml``
        - Spatial
        - FastWAM
      * - ``libero_spatial_molmoact2_eval.yaml``
        - Spatial
        - MolmoAct2
      * - ``libero_object_openpi_pi05_eval.yaml``
        - Object
        - π₀.₅
      * - ``libero_object_openvlaoft_eval.yaml``
        - Object
        - OpenVLA-OFT
      * - ``libero_object_fastwam_eval.yaml``
        - Object
        - FastWAM
      * - ``libero_object_molmoact2_eval.yaml``
        - Object
        - MolmoAct2
      * - ``libero_goal_openpi_eval.yaml``
        - Goal
        - π₀
      * - ``libero_goal_openvlaoft_eval.yaml``
        - Goal
        - OpenVLA-OFT
      * - ``libero_goal_fastwam_eval.yaml``
        - Goal
        - FastWAM
      * - ``libero_goal_molmoact2_eval.yaml``
        - Goal
        - MolmoAct2
      * - ``libero_10_openpi_pi05_eval.yaml``
        - Long (``libero_10``)
        - π₀.₅
      * - ``libero_10_apxinf_pi05_eval.yaml``
        - Long (``libero_10``)
        - π₀.₅ (ApxInf backend)
      * - ``libero_10_openvlaoft_eval.yaml``
        - Long (``libero_10``)
        - OpenVLA-OFT
      * - ``libero_10_fastwam_eval.yaml``
        - Long (``libero_10``)
        - FastWAM
      * - ``libero_10_molmoact2_eval.yaml``
        - Long (``libero_10``)
        - MolmoAct2
      * - ``libero_10_pi0_fast_eval.yaml``
        - Long (``libero_10``)
        - π₀-FAST

.. _apxinf-backend:

.. dropdown:: ApxInf Backend

   The ApxInf path is evaluation-only. RLinf calls `APXinf-robo <https://github.com/RLinf/APXinf-robo>`_, wrapping the L1 interface of `ApxInf <https://github.com/infinigence/ApxInf>`_. Install it and the ``apxinf_py`` CUDA binding in your RLinf environment, then point ``APXINF_PI05_MODEL_DIR`` at a checkpoint directory containing ``config.json``, ``model.safetensors``, ``tokenizer.model``, and ``norm_stats.json``:

   .. code-block:: bash

      export APXINF_PI05_MODEL_DIR=/path/to/pi05_libero_base
      bash evaluations/run_eval.sh libero libero_10_apxinf_pi05_eval

   This config keeps the native OpenPI contract: two oriented LIBERO images, an 8-D state, a state-free π₀.₅ prompt, 5 flow steps, a 10-action horizon, and execution of the first 5 actions. RLinf's OpenPI transforms handle resize, tokenization, normalization, and action unnormalization; ApxInf's ``Model.infer_rgb`` handles model inference and default Gaussian noise sampling, loaded through ``apxinf_robo.load_bare_model`` for numerical parity. Set ``rollout.model.apxinf.noise_source=observation`` and pass an explicit ``[B,10,32]`` noise tensor for paired parity tests. The default 10 envs × 10 rollout epochs evaluate 10 reset states per LIBERO-10 task (100 trajectories total).

.. _libero-eval-config:

.. dropdown:: Evaluation Configuration

   LIBERO evaluation runs one trajectory per ``(task_id, trial_id)`` pair and reports ``eval/success_once``. Each of the four standard suites (``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``) from the `official repo <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ contains 10 tasks and ~50 initial states per task (~500 trajectories).

   **Evaluation protocol**

   Trials across tasks are concatenated into a global ``reset_state_id`` and traversed in interleaved order in eval mode (``is_eval=True``) — ``(task0, trial0), (task1, trial0), …, (task0, trial1), …`` — so parallel envs advance evenly across tasks. Cover every ``(task_id, trial_id)`` pair in one ``rollout_epoch`` using either high parallelism (``total_num_envs >= S``) or auto-reset (``auto_reset=True`` with ``max_steps_per_rollout_epoch = N * max_episode_steps``).

   **Per-suite episode step limits**

   The longest training demo sets a lower bound for ``max_episode_steps``; match your training config:

   .. list-table::
      :header-rows: 1
      :widths: 22 20 24 34

      * - Suite
        - Lower bound
        - RLinf example values
        - Example config
      * - ``libero_spatial``
        - 220
        - 240 / 480
        - ``libero_spatial_openpi_pi05_eval`` / ``libero_spatial_dreamzero_eval``
      * - ``libero_object``
        - 280
        - 280 / 512
        - ``libero_object_openpi_pi05_eval`` / ``libero_object_openvlaoft_eval``
      * - ``libero_goal``
        - 300
        - 320 / 512
        - ``libero_goal_openpi_eval`` / ``libero_goal_openvlaoft_eval``
      * - ``libero_10``
        - 520
        - 520
        - ``libero_10_openpi_pi05_eval`` / ``libero_10_openvlaoft_eval``

   See :ref:`env-eval-fields` in :doc:`../reference/configuration` for generic ``env.eval`` fields.

   **Covering the full test set**

   Let ``S`` = total init states, ``E`` = ``total_num_envs``, and ``T`` = ``max_episode_steps``.

   *Option 1 — High parallelism:*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 500                # S for Spatial / Object / Goal / Long
          max_episode_steps: 240
          max_steps_per_rollout_epoch: 240   # equals max_episode_steps
          auto_reset: True                   # optional when E >= S
          rollout_epoch: 1

   Each LIBERO env runs in its own process and holds about 2 GiB of unshared host RAM after reset (about 1 TiB total for ``total_num_envs: 500``).

   .. warning::

      Env processes are subprocesses of the env worker, not Ray workers. Exhausting host RAM can hang the node without Ray's memory monitor killing a worker.

   *Option 2 — Auto-reset (recommended unless each env has >2 GiB host RAM plus headroom for the rollout model):*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 128
          max_episode_steps: 480
          # N = ceil(S / E); Spatial: ceil(500/128) = 4
          max_steps_per_rollout_epoch: 1920  # N * max_episode_steps = 4 * 480
          auto_reset: True
          ignore_terminations: True
          use_fixed_reset_state_ids: True
          use_ordered_reset_state_ids: True
          rollout_epoch: 1

   Set ``env.eval.rollout_epoch: 2`` to average metrics across two passes. Keep ``max_steps_per_rollout_epoch`` divisible by ``rollout.model.num_action_chunks``. Env workers offset seeds by ``seed + rank × stage_num + stage_id`` so each receives a distinct init-state subset; with ``auto_reset``, ``eval/success_once`` records each completed episode once.

.. dropdown:: Advanced Usage (LIBERO-PRO, LIBERO-PLUS, Parallelism)

   For LIBERO-PRO and LIBERO-PLUS environment installation, see :ref:`LIBERO-Pro & LIBERO-Plus <liberopro-plus-benchmark>`.

   **LIBERO-PRO**

   .. code-block:: bash

      export LIBERO_TYPE=pro
      export LIBERO_PERTURBATION=all
      bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

   **LIBERO-PLUS**

   .. code-block:: bash

      export LIBERO_TYPE=plus
      export LIBERO_SUFFIX=all
      bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

   **Adjust parallel scale**

   Lowering ``total_num_envs`` alone lowers coverage (64 envs with a 240-step budget evaluate only 64 of 500 init states). Raise ``max_steps_per_rollout_epoch`` together with it, and keep ``total_num_envs`` divisible by the number of env workers (``128`` works on 1, 2, 4, or 8 GPUs):

   .. code-block:: bash

      # max_steps_per_rollout_epoch = ceil(500 / 128) * 240 = 960
      bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
        env.eval.total_num_envs=128 \
        env.eval.max_steps_per_rollout_epoch=960 \
        rollout.model.model_path=/path/to/model

Results and Troubleshooting
---------------------------

The terminal prints ``eval/success_once``; logs and videos go to ``logs/<timestamp>-<config>/`` (see :doc:`../reference/results` and :ref:`env-eval-fields` in :doc:`../reference/configuration`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - EGL rendering errors on headless hosts
     - ``run_eval.sh`` uses EGL by default; on hosts without EGL, export ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
   * - Incomplete test-set coverage
     - Coordinate ``total_num_envs``, ``auto_reset``, and ``max_steps_per_rollout_epoch`` per :ref:`Evaluation Configuration <libero-eval-config>`.
   * - Host RAM exhaustion with 500 envs
     - Lower ``total_num_envs`` (such as ``128``) and enable ``auto_reset: True`` with a larger ``max_steps_per_rollout_epoch``; each LIBERO env uses ~2 GiB of host RAM.
   * - Step count validation error
     - Set ``max_steps_per_rollout_epoch`` to a multiple of ``rollout.model.num_action_chunks``.

.. toctree::
   :hidden:
   :maxdepth: 1

   dreamzero_sglang
   cosmos3_sglang
