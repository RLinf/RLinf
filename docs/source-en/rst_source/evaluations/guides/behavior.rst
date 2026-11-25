BEHAVIOR-1K Evaluation
======================

Overview
--------

Use RLinf to evaluate OpenPI and other policies on household tasks with the R1 Pro robot. Choose a challenge year below to install its simulator and run the evaluation.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      50 tasks in 2025, 100 in 2026

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀.₅ (example) · OpenVLA-OFT (derived)

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/behavior/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`BEHAVIOR-1K Training <../../examples/embodied/behavior>`

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: 2025 · OpenPI π₀.₅

      OmniGibson 3.7.2 uses a standalone Isaac Sim 4.5 installation. Both ``--env behavior`` and ``--env behavior2025`` select this version.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

      **2. Download Isaac Sim 4.5, BEHAVIOR-1K Assets (>30 GB), and Checkpoint**

      .. code-block:: bash

         export ISAAC_PATH="$(pwd)/isaac-sim"
         export OMNIGIBSON_DATA_PATH="$(pwd)/BEHAVIOR-1K-datasets"
         mkdir -p "$ISAAC_PATH" "$OMNIGIBSON_DATA_PATH"
         curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o "$ISAAC_PATH/isaac-sim.zip"
         unzip "$ISAAC_PATH/isaac-sim.zip" -d "$ISAAC_PATH" && rm "$ISAAC_PATH/isaac-sim.zip"
         python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
         python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
         python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
         hf download RLinf/RLinf-Pi0-Behavior --local-dir ./RLinf-Pi0-Behavior

      **3. Evaluate**

      .. code-block:: bash

         bash evaluations/run_eval.sh behavior2025 \
           rollout.model.model_path=./RLinf-Pi0-Behavior

      The launcher derives the remaining OmniGibson and Isaac Sim paths from the two exported directories.

   .. tab-item:: 2026 · OpenPI π₀.₅

      Open a fresh shell in the repository root. The installer sets up the OmniGibson 3.9.4 fork and Isaac Sim 5.1 in a separate Python 3.11 environment.

      .. code-block:: bash

         unset ISAAC_PATH EXP_PATH CARB_APP_PATH
         bash requirements/install.sh embodied --model openpi --env behavior2026
         source .venv-behavior2026/bin/activate

      Choose a separate asset directory for 2026. On the first download, follow the asset license prompt, then download the public instances and PT50 checkpoint.

      .. code-block:: bash

         export OMNIGIBSON_DATA_PATH="$PWD/behavior-2026-data"
         mkdir -p "$OMNIGIBSON_DATA_PATH"
         python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
         python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets()"
         python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"
         hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 --local-dir ./PT50

         bash evaluations/run_eval.sh behavior2026 rollout.model.model_path="$PWD/PT50"

      The command evaluates the base layout for ``turning_on_radio``. To select a public instance, set three fields under ``env.eval.omni_config`` in the evaluation YAML. Set ``scene.scene_instance`` to the full template for that task, ``task.activity_instance_dir`` to the directory containing its layout JSON files, and ``task.activity_instance_id`` to the instance number.

For OpenVLA-OFT on the 2025 benchmark, derive an evaluation config from its training config as described in :doc:`../reference/configuration`. Convert third-party OpenPI checkpoints to PyTorch format before evaluation.

.. dropdown:: Hardware, Isaac Sim, and Example Configs

   The 2025 benchmark requires Isaac Sim 4.5 (`requirements <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html>`_):

   - Use a ray-tracing GPU (RTX 30/40 series or newer); non-RT GPUs (A100, H100) show rendering artifacts.
   - Hopper and later GPUs require NVIDIA driver 570 or newer.
   - Prebuilt container images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

   .. list-table::
      :header-rows: 1
      :widths: 40 30 30

      * - Config file
        - Env preset
        - Model
      * - ``behavior_openpi_pi05_eval.yaml``
        - ``behavior_r1pro``
        - π₀.₅ (``openpi``)
      * - ``behavior2026_openpi_pi05_eval.yaml``
        - ``behavior_r1pro``
        - π₀.₅ (``openpi``)

   When ``evaluations/behavior/<config>.yaml`` is absent, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` only for eval-ready configs that define ``rollout.model`` (with ``model_type`` and ``num_action_chunks``). To evaluate a training-only config, rebind ``@actor.model`` to ``@rollout.model`` per :doc:`../reference/configuration`. Keep ``action_dim: 23``, ``num_action_chunks: 32``, and ``openpi.config_name: pi05_behavior`` aligned with training. Generic ``env.eval`` fields live in :ref:`env-eval-fields` (:doc:`../reference/configuration`).

.. _behavior-eval-config:

.. dropdown:: Evaluation Configuration

   Each launch evaluates **one task** selected by ``omni_config.task.activity_name`` (default ``turning_on_radio`` on ``house_double_floor_lower``). To sweep multiple tasks, loop over ``activity_name`` in a script; to evaluate multiple instances, set ``instance_resample_mode: offline`` and average across ``rollout_epoch``.

   **Instance sampling modes** (``env.eval.omni_config.task.instance_resample_mode``)

   - ``disabled`` (default): loads the fixed ``activity_instance_id`` (from ``activity_instance_dir`` if set).
   - ``offline``: picks a cached instance at random from ``activity_instance_dir`` on each reset.
   - ``online``: resamples object layouts online (requires ``online_object_sampling: True`` and ``use_presampled_robot_pose: False``; slower startup). For 2026 online sampling, configure one environment per simulator process.

   **Key evaluation fields**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Field
        - Guidance
      * - ``total_num_envs``
        - Global parallel envs. Each env uses ~**10 GiB** GPU memory; ``behavior_openpi_pi05_eval.yaml`` defaults to ``4``.
      * - ``rollout_epoch``
        - Eval rounds averaged into final metrics; ``behavior_openpi_pi05_eval.yaml`` defaults to ``4``.
      * - ``max_episode_steps``
        - Maximum steps per trajectory. The 2025 config uses ``4096``. Leave this unset in the 2026 config to use the official limit for the selected task.
      * - ``max_steps_per_rollout_epoch``
        - Steps per rollout round; **must be divisible by** ``rollout.model.num_action_chunks``.
      * - ``num_env_subprocess``
        - Isaac Sim subprocesses per env worker (default ``1``). ``total_num_envs`` must divide by ``num_env_subprocess × pipeline_stage_num``.
      * - ``skip_intermediate_renders``
        - Skips renders inside action chunks when ``True`` for faster stepping; videos keep chunk-boundary frames only.

   **Task, scene, and placement overrides**

   .. code-block:: yaml

      cluster:
        component_placement:
          rollout,env: all                   # share GPUs, or split env and rollout across GPUs
      env:
        eval:
          omni_config:
            task:
              activity_name: turning_on_radio
              activity_definition_id: 0
              activity_instance_id: 0
              activity_instance_dir: null
              instance_file_format: tro_state      # template | tro_state
              instance_resample_mode: disabled     # disabled | offline | online
            scene:
              scene_model: house_double_floor_lower
              partial_scene_load: true             # load task-relevant rooms only

   For 2025 cached instance generation and placement details, see :doc:`../../examples/embodied/behavior`.

.. dropdown:: Advanced Usage

   **Switch task or parallel scale**

   .. code-block:: bash

      bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
        rollout.model.model_path=./RLinf-Pi0-Behavior \
        env.eval.omni_config.task.activity_name=picking_up_trash

      bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
        rollout.model.model_path=./RLinf-Pi0-Behavior \
        env.eval.total_num_envs=4 \
        env.eval.num_env_subprocess=2

   **Random offline instance sampling (2025)**

   .. code-block:: yaml

      env:
        eval:
          omni_config:
            task:
              activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances/scenes/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
              instance_file_format: tro_state
              instance_resample_mode: offline
          rollout_epoch: 5

Results and Troubleshooting
---------------------------

The terminal prints ``eval/success_once``; logs and videos go to ``logs/<timestamp>-<config>/`` (see :doc:`../reference/results`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Missing BEHAVIOR assets or license errors
     - Complete the asset downloads for the chosen year and export ``OMNIGIBSON_DATA_PATH`` in the shell that launches evaluation.
   * - ``ISAAC_PATH`` missing in 2025
     - Export ``ISAAC_PATH`` to your Isaac Sim directory so Isaac Sim can start; ``run_eval.sh`` sets ``OMNIGIBSON_HEADLESS=1`` by default.
   * - Out of GPU memory
     - Lower ``total_num_envs`` or ``num_env_subprocess``; each env uses about 10 GiB GPU memory.
   * - Blurry or blocky rendering
     - Use a ray-tracing GPU (RTX 30/40 series or newer).
   * - Slow startup
     - Keep ``partial_scene_load: true`` to load task-relevant rooms only.
   * - Fewer video frames than expected
     - ``skip_intermediate_renders: True`` skips intermediate chunk frames and records only policy observations.
   * - Instance load failure
     - Confirm JSON filenames under ``activity_instance_dir`` match ``activity_name``, ``activity_definition_id``, and ``scene_model``.
   * - Step count validation error
     - Set ``max_steps_per_rollout_epoch`` to a multiple of ``rollout.model.num_action_chunks``.
