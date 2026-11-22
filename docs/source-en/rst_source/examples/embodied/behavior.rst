BEHAVIOR-1K RL
==============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   The BEHAVIOR benchmark (image: `BEHAVIOR <https://behavior.stanford.edu>`__).

`BEHAVIOR <https://behavior.stanford.edu>`__ is a benchmark of everyday household
activities built on NVIDIA IsaacSim / OmniGibson. A dual-arm R1 Pro robot performs
long-horizon manipulation; RLinf uses it to RL-fine-tune vision-language-action (VLA)
policies.

Overview
--------

RL-finetune a VLA on BEHAVIOR household tasks with PPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      50 BEHAVIOR-1K tasks

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · ray-tracing GPUs

| **You'll do:** install IsaacSim deps → download assets + base model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · IsaacSim 4.5 + BEHAVIOR-1K assets (>30 GB) · a base checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Detail
   * - Tasks
     - 50 household manipulation tasks from BEHAVIOR-1K (select via ``task_idx`` 0–49).
   * - Robot
     - Dual-arm R1 Pro on IsaacSim / OmniGibson.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Head-camera RGB (720×720) plus left/right wrist RealSense RGB (480×480).
   * - Action
     - 23-dim continuous: 3-DOF base (x, y, rz), 4-DOF torso, 2×7-DOF arms, and 2×1-DOF parallel-jaw grippers.

Setup and Run
-------------

.. warning::

   Check the IsaacSim software and hardware requirements before installing:

   - https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Hopper-generation GPUs need NVIDIA driver 570 or newer. GPUs without ray tracing
   support, such as A100 or H100, can render BEHAVIOR scenes with severe artifacts.
   Prefer RTX 30/40 series or newer GPUs for visual quality and training stability.

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env behavior && source .venv/bin/activate

      **2. Download**

      Download IsaacSim 4.5, BEHAVIOR-1K assets (over 30 GB, inside the active virtual environment), and the OpenVLA-OFT base checkpoint. Keep ``ISAAC_PATH`` and ``OMNIGIBSON_DATA_PATH`` exported in your launch shell:

      .. code-block:: bash

         export ISAAC_PATH="$PWD/isaac-sim"
         export OMNIGIBSON_DATA_PATH="$PWD/BEHAVIOR-1K-datasets"
         mkdir -p "$ISAAC_PATH" "$OMNIGIBSON_DATA_PATH"
         curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o "$ISAAC_PATH/isaac-sim.zip"
         unzip "$ISAAC_PATH/isaac-sim.zip" -d "$ISAAC_PATH" && rm "$ISAAC_PATH/isaac-sim.zip"
         python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
         python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
         python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
         hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir ./RLinf-OpenVLAOFT-Behavior

      **3. Launch**

      Available config under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 30 20 50

         * - Model
           - Algorithm
           - Config
         * - OpenVLA-OFT
           - PPO
           - ``behavior_ppo_openvlaoft.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` on the command line (rollout inherits it automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-Behavior

   .. tab-item:: OpenPI
      :sync: openpi

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

      **2. Download**

      Download IsaacSim 4.5, BEHAVIOR-1K assets (over 30 GB, inside the active virtual environment), and the OpenPI base checkpoint. Keep ``ISAAC_PATH`` and ``OMNIGIBSON_DATA_PATH`` exported in your launch shell:

      .. code-block:: bash

         export ISAAC_PATH="$PWD/isaac-sim"
         export OMNIGIBSON_DATA_PATH="$PWD/BEHAVIOR-1K-datasets"
         mkdir -p "$ISAAC_PATH" "$OMNIGIBSON_DATA_PATH"
         curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o "$ISAAC_PATH/isaac-sim.zip"
         unzip "$ISAAC_PATH/isaac-sim.zip" -d "$ISAAC_PATH" && rm "$ISAAC_PATH/isaac-sim.zip"
         python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
         python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
         python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
         hf download RLinf/RLinf-Pi0-Behavior --local-dir ./RLinf-Pi0-Behavior

      **3. Launch**

      Available configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 30 20 50

         * - Model
           - Algorithm
           - Config
         * - π₀
           - PPO
           - ``behavior_ppo_openpi.yaml``
         * - π₀.₅
           - PPO
           - ``behavior_ppo_openpi_pi05.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` on the command line (rollout inherits it automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi \
           actor.model.model_path=./RLinf-Pi0-Behavior

.. warning::

   Place BEHAVIOR env workers on GPUs starting from 0. IsaacSim can hang when env
   workers start on later GPU ranks.

.. warning::

   Known issue: under the current BEHAVIOR setup, training success rate
   (``env/success_once``) may stay at 0 for OpenVLA-OFT / π₀.
   This issue will be fixed in a later release.

.. admonition:: Configure further
   :class: note

   - BEHAVIOR throughput → increase env GPU count first, then tune ``env.num_env_subprocess`` and ``env.train.total_num_envs``.
   - Each BEHAVIOR process can use roughly 10 GiB of VRAM; tune subprocess count for your GPU memory.
   - Cached task instances → generate them with the BEHAVIOR instance generator script (see the dropdown below) and ``examples/embodiment/config/env/behavior_r1pro.yaml``.
   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`

.. dropdown:: Advanced configuration and cached task instances (optional)

   The BEHAVIOR env is driven by ``examples/embodiment/config/env/behavior_r1pro.yaml``. RLinf first loads OmniGibson's base config (``base_config_name``) and then applies the ``omni_config`` overrides. The fields below control reset behavior, scene loading, simulator frequencies, and throughput — most have sensible defaults and only need tuning for custom tasks or performance.

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Key
        - Meaning
      * - ``base_config_name``
        - Base OmniGibson config (e.g. ``r1pro_behavior``) loaded before ``omni_config`` overrides.
      * - ``omni_config.task.type`` / ``omni_config.scene.type``
        - Keep ``BehaviorTask`` / ``InteractiveTraversableScene`` explicit so the intended upstream OmniGibson classes are selected after overrides are applied.
      * - ``task_idx``
        - Current task id (0–49), mapped to ``task.activity_name``.
      * - ``omni_config.task.instance_resample_mode``
        - Reset-time instance switching: ``disabled`` (load the fixed ``activity_instance_id``), ``offline`` (scan ``activity_instance_dir`` once and sample a cached instance per reset — ``*_template.json`` use the heavy scene-reload path, ``*_template-tro_state.json`` the lighter in-place path), or ``online`` (requires ``online_object_sampling: True`` and ``use_presampled_robot_pose: False``).
      * - ``omni_config.task.activity_instance_dir``
        - Directory of cached instance JSONs (``*_template.json`` / ``*_template-tro_state.json``), used by ``offline`` mode and by fixed-id loading when the mode is ``disabled``.
      * - ``omni_config.task.instance_file_format``
        - Cached-instance format: ``template`` (full reload) or ``tro_state`` (light, task-relevant only). RLinf accepts ``tro_state`` files without ``robot_poses``; it clears stale cached robot-pose metadata and the reset falls back to the task's default robot pose.
      * - ``omni_config.scene.partial_scene_load``
        - When ``true``, auto-fills ``scene.load_room_types`` with rooms relevant to ``activity_name`` (reduces startup time and memory). Requires ``activity_name`` and ``scene_model``. Set ``load_room_types`` explicitly when ``false``/omitted.
      * - ``camera.head_resolution`` / ``camera.wrist_resolution``
        - Head / wrist camera resolutions (defaults 720×720 / 480×480, applied to R1Pro sensors).
      * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
        - Action / render / physics stepping frequency (common default 30 / 30 / 120). Higher is slower.
      * - ``omni_config.env.automatic_reset``
        - Keep ``False`` — reset is controlled explicitly by the RLinf train/eval loop.
      * - ``omni_config.env.flatten_obs_space`` / ``flatten_action_space``
        - Keep ``False`` to preserve structured observation / action spaces.
      * - ``omni_config.macro.use_gpu_dynamics``
        - ``False`` usually improves performance; enable only for particles / fluids.
      * - ``omni_config.macro.enable_flatcache``
        - ``True`` generally improves performance on large scenes.
      * - ``omni_config.macro.enable_object_states``
        - Keep ``True`` — ``BehaviorTask`` depends on object states.
      * - ``omni_config.macro.enable_transition_rules``
        - ``True`` enables transition-rule state changes (e.g. slicing, cooking).
      * - ``omni_config.macro.use_numpy_controller_backend``
        - ``True`` uses the numpy controller backend, usually faster in single-/moderate-parallel runs.
      * - ``skip_intermediate_renders``
        - When ``True``, skips collecting intermediate observations inside an action chunk (large env-speed gain). Saved videos then show only the frames the policy observes at chunk boundaries.
      * - ``num_env_subprocess``
        - Splits ``num_envs`` across child processes, each hosting its own Isaac/OmniGibson sim. Default ``1``. **Constraint:** ``num_envs`` must be divisible by ``num_env_subprocess``. Higher values cut env-step bottlenecks but multiply processes and memory.

   **Generating cached task instances**

   Run the BEHAVIOR instance generator from the repository root to produce ``*_template.json`` and ``*_template-tro_state.json`` files directly from ``behavior_r1pro.yaml``. It reads ``scene_model``, ``activity_name``, ``activity_definition_id``, the robot config, and room-loading settings, then temporarily switches to online object sampling. It writes to ``activity_instance_dir`` if set, otherwise to ``OMNIGIBSON_DATA_PATH``'s default ``2025-challenge-task-instances`` directory (override with ``--output-dir``):

   .. code-block:: bash

      python rlinf/envs/sim/behavior/instance_generator.py \
        --config examples/embodiment/config/env/behavior_r1pro.yaml \
        --output-format template \
        --start-idx 1 --end-idx 50

      python rlinf/envs/sim/behavior/instance_generator.py \
        --config examples/embodiment/config/env/behavior_r1pro.yaml \
        --output-format tro_state \
        --start-idx 1 --end-idx 50

   Generated filenames follow ``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``, so ``--start-idx`` / ``--end-idx`` set the ``activity_instance_id`` range. ``tro_state`` outputs include top-level ``robot_poses`` only when the task metadata provides them; otherwise the key is omitted and reset falls back to the task's default robot pose. BEHAVIOR-1K's upstream ``multiply_b1k_tasks.py`` still works, but RLinf's generator is recommended because it reads the RLinf YAML directly and preserves ``activity_definition_id``.

Standalone Evaluation
~~~~~~~~~~~~~~~~~~~~~

Any PyTorch-converted ``pi05`` checkpoint (for example, `OpenPI-Comet <https://huggingface.co/sunshk/openpi_comet/tree/main>`__ converted with `openpi-comet <https://github.com/mli0603/openpi-comet>`__, or weights from :doc:`sft_openpi` / the OpenPI checkpoint converter) can be evaluated with ``evaluations/behavior/behavior_openpi_pi05_eval.yaml``. Normalization statistics are loaded from the checkpoint's bundled ``physical-intelligence/behavior/norm_stats.json`` asset. Increase ``env.eval.max_episode_steps`` and ``env.eval.max_steps_per_rollout_epoch`` (for example, ``4096``) if you need longer trajectories.

Run standalone evaluation through the :doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>`. It covers the required ``ISAAC_PATH`` / ``OMNIGIBSON_DATA_PATH`` setup, setting the eval checkpoint path, the ``behavior_openpi_pi05_eval`` launch command, and result interpretation.

.. note::

   Evaluation runs the flow-matching action head with random sampling noise, so per-run trajectories and success counts vary slightly across repeated runs.

Visualization and Results
-------------------------

Watch ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); set ``env.eval.video_cfg.save_video=True`` to record videos, or see :doc:`Training metrics <../../reference/metrics>` for W&B/SwanLab and all metrics.

For the Behavior experiment, we were inspired by
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_,
with only minor modifications. We thank the authors for releasing their open-source code.
