RoboTwin Evaluation
===================

Overview
--------

RoboTwin is a bimanual manipulation simulation platform covering tabletop tasks such as placing cups, adjusting bottles, placing shoes, and clicking bells across 22 bundled seed tasks. RLinf runs parallel VLA policy evaluation on RoboTwin and reports task-level metrics such as ``eval/success_once``.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      RoboTwin (22 seed tasks, bimanual ALOHA / Piper)

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · OpenPI π₀ / π₀.₅ · LingbotVLA 1.0 / 2.0

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/robotwin/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`RoboTwin Training <../../examples/embodied/robotwin>`

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: OpenVLA-OFT (place_empty_cup)

      .. code-block:: bash

         # 1. Install and download RoboTwin assets
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)

         # 2. Download model
         hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir ./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

         # 3. Evaluate
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA
         bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
           rollout.model.model_path=./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
           env.eval.assets_path="${ROBOTWIN_PATH}"

   .. tab-item:: OpenPI π₀.₅ (adjust_bottle)

      .. code-block:: bash

         # 1. Install and download RoboTwin assets
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)

         # 2. Download model
         hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle

         # 3. Evaluate
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA
         bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
           env.eval.assets_path="${ROBOTWIN_PATH}"

   .. tab-item:: OpenPI π₀ (adjust_bottle)

      .. code-block:: bash

         # 1. Install and download RoboTwin assets
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)

         # 2. Download model
         hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi0-RoboTwin-SFT-adjust_bottle

         # 3. Evaluate
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA
         bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
           rollout.model.model_path=./RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
           env.eval.assets_path="${ROBOTWIN_PATH}"

Replace ``--model`` with ``lingbotvla`` or ``lingbotvlav2`` for LingbotVLA. On AMD ROCm, install with ``--platform`` and set ``planner_backend: mplib`` in the task config (see :ref:`Run on Different Hardware Backends <robotwin-hardware>`). ``run_eval.sh`` adds ``ROBOTWIN_PATH`` to ``PYTHONPATH``, and ``env.eval.assets_path`` points to the parent folder of ``assets/``. RLinf uses a ``0.01 m`` camera near plane so SAPIEN 3.0.1 keeps gripper fingers visible in wrist views (override ``env.eval.camera_near`` or set ``null`` for RoboTwin's default ``0.1 m``).

.. dropdown:: Docker and Example Configs

   **Docker (optional):** Use ``rlinf/rlinf:agentic-rlinf0.4-robotwin`` and switch environments inside the container via ``source switch_env openvla-oft`` or ``source switch_env openpi``.

   Available under ``evaluations/robotwin/``:

   .. list-table::
      :header-rows: 1
      :widths: 44 28 28

      * - Config file
        - Task
        - Model
      * - ``robotwin_place_empty_cup_openvlaoft_eval.yaml``
        - ``place_empty_cup``
        - OpenVLA-OFT
      * - ``robotwin_place_empty_cup_openpi_eval.yaml``
        - ``place_empty_cup``
        - π₀
      * - ``robotwin_adjust_bottle_openpi_pi05_eval.yaml``
        - ``adjust_bottle``
        - π₀.₅
      * - ``robotwin_adjust_bottle_openpi_eval.yaml``
        - ``adjust_bottle``
        - π₀
      * - ``robotwin_place_shoe_lingbotvla_eval.yaml``
        - ``place_shoe``
        - LingbotVLA 1.0
      * - ``robotwin_click_bell_lingbotvla_eval.yaml``
        - ``click_bell``
        - LingbotVLA 1.0
      * - ``robotwin_click_bell_lingbotvlav2_eval.yaml``
        - ``click_bell``
        - LingbotVLA 2.0

   If ``evaluations/robotwin/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` (set ``runner.only_eval: True`` and ``runner.task_type: embodied_eval``). The bundled ``eval_seeds.json`` covers **22 tasks**; derive other tasks from training configs (see :doc:`../reference/configuration`).

.. _robotwin-eval-config:

.. dropdown:: Evaluation Configuration

   RoboTwin evaluation runs one trajectory per pre-filtered **success seed** in ``eval_seeds.json`` (150–320 seeds per task across 22 tasks) and reports ``eval/success_once``. On startup, ``success_seeds`` are shuffled and partitioned across env workers without overlap. With default settings (``total_num_envs: 128``, ``rollout_epoch: 1``, ``use_fixed_reset_state_ids: True``), one epoch covers ~128 trajectories; set ``use_fixed_reset_state_ids: False`` so ``auto_reset`` rotates to subsequent seeds.

   **Seed counts and step limits**

   .. list-table::
      :header-rows: 1
      :widths: 26 18 22 34

      * - Task
        - Total seeds
        - Example ``max_episode_steps``
        - Example config
      * - ``adjust_bottle``
        - 150
        - 200
        - ``robotwin_adjust_bottle_openpi_eval``
      * - ``place_empty_cup``
        - 260
        - 200
        - ``robotwin_place_empty_cup_openvlaoft_eval``
      * - ``click_bell``
        - 150
        - 400
        - ``robotwin_click_bell_lingbotvla_eval``
      * - ``place_shoe``
        - 320
        - 400
        - ``robotwin_place_shoe_lingbotvla_eval``

   Keep ``max_episode_steps`` aligned with training and ``task_config.step_lim``. See :ref:`env-eval-fields` in :doc:`../reference/configuration` for generic ``env.eval`` fields.

   **Model-specific settings**

   Match each model's training protocol for comparable results:

   - **OpenVLA-OFT (``demo_randomized``):** ``task_config.embodiment: [piper, piper, 0.6]``, ``center_crop: True`` (and ``rollout.model.center_crop: True``), domain randomization enabled, ``num_action_chunks: 25``, matching ``unnorm_key`` (such as ``place_empty_cup_1k``), ``implement_version: "official"``.
   - **OpenPI π₀ / π₀.₅ (``demo_clean``):** ``task_config.embodiment: [aloha-agilex]``, ``center_crop: False``, ``collect_wrist_camera: true``, all ``task_config.domain_randomization`` fields ``false``, ``num_action_chunks: 50``, ``openpi.config_name: pi0_aloha_robotwin`` or ``pi05_aloha_robotwin``; set ``env.enable_offload: True`` and ``rollout.enable_offload: True`` to save GPU memory.
   - **LingbotVLA 1.0:** set ``model_path``, ``tokenizer_path``, and ``rollout.model.lingbotvla.config_path``; ``num_action_chunks: 50``, ``max_episode_steps: 400``, ``use_custom_reward: False``.
   - **LingbotVLA 2.0:** fill paths in ``examples/embodiment/config/model/lingbotvlav2.yaml``; ``precision: bf16``, ``embodiment: [aloha-agilex]``, ``collect_wrist_camera: true``, ``num_action_chunks: 50``, ``max_episode_steps: 400``.

   **Covering the full test set**

   Let ``S`` = total seeds, ``E`` = ``total_num_envs``, and ``T`` = ``max_episode_steps``.

   *Option 1 — High parallelism:*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 260                # S, e.g. place_empty_cup
          max_episode_steps: 200
          max_steps_per_rollout_epoch: 200
          use_fixed_reset_state_ids: True
          rollout_epoch: 1

   *Option 2 — Dynamic seeds with auto-reset (recommended when GPU memory is limited):*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 128
          max_episode_steps: 200
          # N = ceil(S / E); place_empty_cup: ceil(260/128) = 3
          max_steps_per_rollout_epoch: 600   # N * max_episode_steps = 3 * 200
          auto_reset: True
          ignore_terminations: True
          use_fixed_reset_state_ids: False   # allow seed rotation on auto_reset
          is_eval: True
          rollout_epoch: 1

   For multi-epoch averaging (``rollout_epoch > 1``), keep ``use_fixed_reset_state_ids: False``. Keep ``max_steps_per_rollout_epoch`` divisible by ``rollout.model.num_action_chunks``. Unsupported tasks: ``place_fan``, ``open_laptop``, ``place_object_scale``, ``put_object_cabinet``.

.. dropdown:: Advanced Usage

   **Adjust parallelism or load an RL checkpoint**

   .. code-block:: bash

      bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
        env.eval.total_num_envs=64 \
        rollout.model.model_path=/path/to/model

      bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
        runner.ckpt_path=/path/to/checkpoint.pt

   **Derive configs for other tasks**

   ``eval_seeds.json`` also includes ``beat_block_hammer``, ``handover_block``, ``lift_pot``, and more. Copy a matching YAML under ``evaluations/robotwin/``, update ``defaults`` to ``env/robotwin_<task>@env.eval``, and adjust ``unnorm_key`` or ``openpi.config_name``.

Results and Troubleshooting
---------------------------

The terminal prints ``eval/success_once``; logs and videos go to ``logs/<timestamp>-<config>/`` (see :doc:`../reference/results`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - ``ROBOTWIN_PATH`` not found on import
     - Export ``ROBOTWIN_PATH`` to your RoboTwin checkout (``RLinf_support`` branch); ``run_eval.sh`` adds it to ``PYTHONPATH`` when set.
   * - Startup failure or missing scenes
     - Set ``env.eval.assets_path`` to the parent directory of ``assets/`` (``ASSETS_PATH`` is populated from it). Also export ``ROBOT_PLATFORM=ALOHA``.
   * - GPU out of memory
     - Set ``env.enable_offload: True`` and ``rollout.enable_offload: True``, or lower ``env.eval.total_num_envs``.
   * - Incomplete seed coverage
     - Default 128 envs with ``use_fixed_reset_state_ids: True`` covers only a subset of seeds; see :ref:`Evaluation Configuration <robotwin-eval-config>`.
   * - EGL rendering errors
     - ``run_eval.sh`` uses EGL by default; on hosts without EGL, export ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
