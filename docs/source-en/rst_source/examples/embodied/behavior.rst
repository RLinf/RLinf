RL with Behavior Benchmark
==========================

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


Installation
------------

.. warning::

   Check the IsaacSim software and hardware requirements before installing:

   - https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Hopper-generation GPUs need NVIDIA driver 570 or newer. GPUs without ray tracing
   support, such as A100 or H100, can render BEHAVIOR scenes with severe artifacts.
   Prefer RTX 30/40 series or newer GPUs for visual quality and training stability.

.. include:: _setup_common.rst

**Option 1: Docker image** — BEHAVIOR ships **two separate images**, one per model:
``agentic-rlinf0.3-behavior`` (OpenVLA-OFT) and ``agentic-rlinf0.3-behavior-openpi``
(OpenPI). Each image bundles only its own virtual environment, so pull the one that
matches the model you intend to train (there is no ``switch_env`` between them):

.. code-block:: bash

   # OpenVLA-OFT model:
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-behavior
      # Mainland China mirror: docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-behavior

   # OpenPI model (separate image):
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-behavior-openpi
      # Mainland China mirror: docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-behavior-openpi

   # In either image the matching virtual environment is already activated by default.

**Option 2: Custom environment** — install bundle ``--env behavior``:

.. code-block:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   # Or install the OpenPI environment:
   # bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate


Download the Assets
-------------------

Download IsaacSim 4.5 and set ``ISAAC_PATH`` before every run:

.. code-block:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   mkdir -p $ISAAC_PATH && cd $ISAAC_PATH
   curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o isaac-sim.zip
   unzip isaac-sim.zip && rm isaac-sim.zip

Download BEHAVIOR-1K assets and set ``OMNIGIBSON_DATA_PATH`` before every run:

.. code-block:: bash

   # The datasets require more than 30 GB.
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   mkdir -p $OMNIGIBSON_DATA_PATH

   # Run these inside the active venv. Set HF_ENDPOINT=https://hf-mirror.com in mainland China.
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
   python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"

**What this does:**

1. Downloads the IsaacSim runtime that OmniGibson uses.
2. Downloads the BEHAVIOR robot assets, task assets, and 2025 challenge instances.
3. Creates the two environment variables that the training and evaluation scripts need.

Download the Model
------------------

Download the checkpoint for your model family (either method works):

.. code-block:: bash

   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # Method 2: huggingface-hub (set HF_ENDPOINT=https://hf-mirror.com in mainland China)
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior

.. include:: _model_path.rst

Run It
------

.. warning::

   Place BEHAVIOR env workers on GPUs starting from 0. IsaacSim can hang when env
   workers start on later GPU ranks.

Each recipe is a YAML config under ``examples/embodiment/config/``:

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Model / purpose
     - Algorithm
     - Config
   * - OpenVLA-OFT
     - PPO
     - ``behavior_ppo_openvlaoft.yaml``
   * - π₀
     - PPO
     - ``behavior_ppo_openpi.yaml``
   * - π₀.₅
     - PPO
     - ``behavior_ppo_openpi_pi05.yaml``

Launch a config with ``run_embodiment.sh``:

.. code-block:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

**What this command does:**

1. Loads ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml`` and its shared env config ``examples/embodiment/config/env/behavior_r1pro.yaml``.
2. Starts Ray workers for the actor, rollout, and BEHAVIOR env placement.
3. Runs PPO training and writes logs/checkpoints under ``runner.logger.log_path``.

.. admonition:: Configure further
   :class: note

   - BEHAVIOR throughput → increase env GPU count first, then tune ``env.num_env_subprocess`` and ``env.train.total_num_envs``.
   - Each BEHAVIOR process can use roughly 10 GiB of VRAM; tune subprocess count for your GPU memory.
   - Cached task instances → generate them with ``rlinf/envs/behavior/instance_generator.py`` and ``examples/embodiment/config/env/behavior_r1pro.yaml``.
   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`

.. warning::

   Known issue: under the current BEHAVIOR setup, training success rate
   (``env/success_once``) may stay at 0 for OpenVLA-OFT / π₀.
   This issue will be fixed in a later release.

Standalone Evaluation
~~~~~~~~~~~~~~~~~~~~~

In principle, any ``pi05`` checkpoint that has non-zero success rate on
Behavior and has been converted to PyTorch format can be used for evaluation
with this config. We use OpenPI-Comet only as an example source:

- https://huggingface.co/sunshk/openpi_comet/tree/main

After download, you can use the following repository to convert weights to
PyTorch format:

- https://github.com/mli0603/openpi-comet

Thanks to the OpenPI-Comet authors for open-sourcing the model and tools, which
helps reproducibility and evaluation in RLinf.

After conversion, update ``behavior_openpi_pi05_eval.yaml`` as follows:

1. Set ``actor.model.model_path`` and ``rollout.model.model_path`` to the converted model directory.
2. Increase ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` in both
   ``env.train`` and ``env.eval`` (for example, ``4096``).

.. code-block:: yaml

   env:
     train:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

Run standalone evaluation through the :doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>`.
It owns the required ``ISAAC_PATH`` / ``OMNIGIBSON_DATA_PATH`` setup, the
``behavior_openpi_pi05_eval`` launch command, and result interpretation.


Configure Further
-----------------

The BEHAVIOR env is driven by ``examples/embodiment/config/env/behavior_r1pro.yaml``.
RLinf first loads OmniGibson's base config (``base_config_name``) and then applies the
``omni_config`` overrides (see ``setup_omni_cfg`` in ``rlinf/envs/behavior/utils.py``).
The fields below control reset behavior, scene loading, simulator frequencies, and
throughput — most have sensible defaults and only need tuning for custom tasks or
performance.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Meaning
   * - ``base_config_name``
     - Base OmniGibson config (e.g. ``r1pro_behavior``) loaded before ``omni_config`` overrides.
   * - ``omni_config.task.type`` / ``omni_config.scene.type``
     - Keep ``BehaviorTask`` / ``InteractiveTraversableScene`` explicit so the intended
       upstream OmniGibson classes are selected after overrides are applied.
   * - ``task_idx``
     - Current task id (0–49). RLinf maps it to ``task.activity_name`` (see ``behavior_env.py``).
   * - ``omni_config.task.instance_resample_mode``
     - Reset-time instance switching: ``disabled`` (load the fixed ``activity_instance_id``),
       ``offline`` (scan ``activity_instance_dir`` once and sample a cached instance per reset —
       ``*_template.json`` use the heavy scene-reload path, ``*_template-tro_state.json`` the
       lighter in-place path), or ``online`` (requires ``online_object_sampling: True`` and
       ``use_presampled_robot_pose: False``).
   * - ``omni_config.task.activity_instance_dir``
     - Directory of cached instance JSONs (``*_template.json`` / ``*_template-tro_state.json``),
       used by ``offline`` mode and by fixed-id loading when the mode is ``disabled``.
   * - ``omni_config.task.instance_file_format``
     - Cached-instance format: ``template`` (full reload) or ``tro_state`` (light, task-relevant
       only). RLinf accepts ``tro_state`` files without ``robot_poses``; it then clears stale
       cached robot-pose metadata and the reset falls back to the task's default robot pose.
   * - ``omni_config.scene.partial_scene_load``
     - When ``true``, auto-fills ``scene.load_room_types`` with rooms relevant to
       ``activity_name`` (reduces startup time and memory). Requires ``activity_name`` and
       ``scene_model``. Set ``load_room_types`` explicitly when ``false``/omitted.
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
   * - ``skip_intermediate_obs_in_chunk``
     - When ``True``, skips collecting intermediate observations inside an action chunk (large
       env-speed gain). Saved videos then show only the frames the policy observes at chunk boundaries.
   * - ``num_env_subprocess``
     - Splits ``num_envs`` across child processes, each hosting its own Isaac/OmniGibson sim
       (see ``BehaviorProcess``). Default ``1``. **Constraint:** ``num_envs`` must be divisible by
       ``num_env_subprocess``. Higher values cut env-step bottlenecks but multiply processes and memory.

Generating cached task instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rlinf/envs/behavior/instance_generator.py`` generates ``*_template.json`` and
``*_template-tro_state.json`` files directly from ``behavior_r1pro.yaml`` (it reads
``scene_model``, ``activity_name``, ``activity_definition_id``, the robot config, and
room-loading settings, then temporarily switches to online object sampling). It writes to
``activity_instance_dir`` if set, otherwise to ``OMNIGIBSON_DATA_PATH``'s default
``2025-challenge-task-instances`` directory; use ``--output-dir`` to override.

.. code-block:: bash

   cd /path/to/RLinf

   python rlinf/envs/behavior/instance_generator.py \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-format template \
     --start-idx 1 --end-idx 50

   python rlinf/envs/behavior/instance_generator.py \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-format tro_state \
     --start-idx 1 --end-idx 50

Generated filenames follow
``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``,
so ``--start-idx`` / ``--end-idx`` set the ``activity_instance_id`` range. ``tro_state``
outputs include top-level ``robot_poses`` only when the task metadata provides them; otherwise
the key is omitted and reset falls back to the task's default robot pose. BEHAVIOR-1K's upstream
``multiply_b1k_tasks.py`` still works, but RLinf's generator is recommended because it reads the
RLinf YAML directly and preserves ``activity_definition_id``.

--------------

**5. Evaluate with the PyTorch OpenPI (Pi0.5) code**

BEHAVIOR evaluation is also supported with the new **self-contained PyTorch
OpenPI** code (model ``model_type: openpi_pytorch``; see
:doc:`sft_openpi_pytorch` for the matching SFT flow). The eval config is:

- ``evaluations/behavior/behavior_openpi_pi05_pytorch_eval.yaml``

This config runs in eval-only mode (``runner.only_eval: True``) and consumes a
**new-format** PyTorch checkpoint, i.e. one produced by the OpenPI checkpoint
convertor (``ckpt_convertor.openpi`` ``old2new`` / ``sft2new``). Set the model
paths directly in the config as ``/path/to/...`` placeholders:

- ``rollout.model.model_path``: the new-format eval checkpoint.
- ``rollout.model.openpi.assets_dir``: directory holding the BEHAVIOR norm-stats.
  Norm stats resolve at ``{assets_dir}/{asset_id}/norm_stats.json``.
- ``rollout.model.openpi.paligemma_tokenizer``: the PaliGemma SentencePiece
  tokenizer model.

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_pytorch_eval

.. note::

   Evaluation runs the flow-matching action head with non-deterministic
   (random) sampling noise, so per-run trajectories and success counts will vary
   slightly between repeated runs.


Visualization and Results
-------------------------

Launch TensorBoard to watch training live:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

The key signal to watch is **``env/success_once``** — the task success rate. For every
logged metric, see :doc:`Training metrics <../../reference/metrics>`.

To save evaluation videos, enable them in the config:

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval


For the Behavior experiment, we were inspired by
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_,
with only minor modifications. We thank the authors for releasing their open-source code.
