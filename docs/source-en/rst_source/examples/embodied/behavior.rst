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

      100 tasks in the 2026 challenge

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · ray-tracing GPUs

| **You'll do:** install IsaacSim deps → download assets + base model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Python 3.11 · the BEHAVIOR-1K v3.9.4 fork installed by RLinf · Isaac Sim 5.1 + 2026 BEHAVIOR assets · a base checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Detail
   * - Tasks
     - 100 household manipulation tasks from the 2026 BEHAVIOR Challenge (select via ``omni_config.task.activity_name``).
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
   * - Challenge observation
     - The default profile returns three 224×224 RGB images and proprioception. The full-resolution profile adds depth and uses 720×720 for the head and 480×480 for each wrist. See the `challenge evaluation rules <https://behavior.stanford.edu/challenge/evaluation.html>`_.
   * - Action
     - 23-dim continuous: 3-DOF base (x, y, rz), 4-DOF torso, 2×7-DOF arms, and 2×1-DOF parallel-jaw grippers.


Installation
------------

.. warning::

   Check the IsaacSim software and hardware requirements before installing:

   - https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Linux requires NVIDIA driver 580.65.06 or newer. GPUs without ray tracing
   support, such as A100 or H100, are unsupported by Isaac Sim 5.1.
   Prefer RTX 30/40 series or newer GPUs for visual quality and training stability.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

**Option 1: Docker image** — The RLinf Docker image for the BEHAVIOR-1K v3.9.4 fork is pending release.

**Option 2: Custom environment** — install bundle ``--env behavior``:

.. code-block:: bash

   # Add --use-mirror for faster downloads in mainland China.
   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   # Or install the OpenPI environment:
   # bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate


.. _behavior-sim-assets:

Download the Assets
-------------------

Choose a separate root for the 2026 simulation assets. If you already have the assets, use their existing root.
Export the path in the shell that launches RLinf.

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

Run these commands in the active RLinf environment to download any missing assets for the 2026 challenge. OmniGibson prompts for the dataset license when it first retrieves the scene asset key.

.. code-block:: bash

   mkdir -p "$OMNIGIBSON_DATA_PATH"
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets()"
   python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"

**What this does:**

1. Extracts the robot models from ``omnigibson-robot-assets-3.8.2.zip`` into ``omnigibson-robot-assets/``.
2. Extracts the scenes and objects from ``behavior-1k-assets-3.9.0.zip`` into ``behavior-1k-assets/`` and saves the dataset key, ``omnigibson.key``, in the asset root.
3. Extracts ``2026-challenge-task-instances.zip`` into ``2026-challenge-task-instances/``. It contains the initial scene layouts and the list of 100 tasks.

To prepare the assets ahead of time, download these three archives from the `official asset repository <https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/main>`_ and extract each into its indicated directory.
Once the SDK is installed, run the command above that calls ``download_behavior_1k_assets()`` to obtain the key.


Download the Model
------------------

RLinf released the following checkpoints in 2025. Download the checkpoint for your model family (either method works):

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

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

**What this command does:**

1. Loads ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml`` and its shared env config ``examples/embodiment/config/env/behavior_r1pro.yaml``.
2. Starts Ray workers for the actor, rollout, and BEHAVIOR env placement.
3. Runs PPO training and writes logs/checkpoints under ``runner.logger.log_path``.

.. admonition:: Configure further
   :class: note

   - BEHAVIOR throughput → increase env GPU count first, then tune ``env.num_env_subprocess`` and ``env.train.total_num_envs``.
   - Tune subprocess count for your GPU memory and scene size.
   - Cached task instances → use the official 2026 instances as described in the :doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>`.
   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`

.. warning::

   Known issue: under the current BEHAVIOR setup, training success rate
   (``env/success_once``) may stay at 0 for OpenVLA-OFT / π₀.
   This issue will be fixed in a later release.

Standalone Evaluation
~~~~~~~~~~~~~~~~~~~~~

The radio evaluation has been verified with the `π₀.₅ PT50 checkpoint <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_. Download and launch it as described in the :doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>`.

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

1. Set ``rollout.model.model_path`` to the converted model directory.
2. Leave ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` in ``env.eval``
   at ``null`` to use the selected task's official timeout. Set integer values for
   a custom episode limit and rollout budget.

.. code-block:: yaml

   env:
     eval:
       max_episode_steps: null
       max_steps_per_rollout_epoch: null

Run standalone evaluation through the :doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>`.
It owns the required ``OMNIGIBSON_DATA_PATH`` setup, the
``behavior_openpi_pi05_eval`` launch command, and result interpretation.


Configure Further
-----------------

The BEHAVIOR env is driven by ``examples/embodiment/config/env/behavior_r1pro.yaml``.
RLinf first loads OmniGibson's base config (``base_config_name``) and then applies the
``omni_config`` overrides (see ``setup_omni_cfg`` in ``rlinf/envs/sim/behavior/utils.py``).
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
   * - ``omni_config.task.activity_name``
     - Task name, such as ``turning_on_radio``. The task list covers all 100 challenge tasks.
   * - ``omni_config.task.instance_resample_mode``
     - Reset-time instance switching: ``disabled`` (load the fixed ``activity_instance_id``),
       ``offline`` (scan ``activity_instance_dir`` once and sample a cached instance per reset —
       ``*_template.json`` use the heavy scene-reload path, ``*_template-tro_state.json`` the
       lighter in-place path), or ``online`` (requires ``online_object_sampling: True`` and
       ``use_presampled_robot_pose: False`` and one environment per simulator process).
   * - ``omni_config.task.activity_instance_dir``
     - Directory of cached instance JSONs (``*_template.json`` / ``*_template-tro_state.json``),
       used by ``offline`` mode and by fixed-id loading when the mode is ``disabled``.
   * - ``omni_config.task.instance_file_format``
     - Cached-instance format: ``template`` (full reload) or ``tro_state`` (light, task-relevant
       only). RLinf accepts ``tro_state`` files without ``robot_poses``; it then clears stale
       cached robot-pose metadata and restores the scene's initial robot pose.
   * - ``omni_config.scene.partial_scene_load``
     - Set to ``true`` to load only the rooms needed for the task.
   * - ``omni_config.camera.head_resolution`` / ``omni_config.camera.wrist_resolution``
     - Head / wrist camera resolutions (defaults 720×720 / 480×480, applied to R1Pro sensors).
   * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
     - Action / render / physics stepping frequency (default 60 / 60 / 120 Hz). Higher is slower.
   * - ``omni_config.env.automatic_reset``
     - Keep ``False`` — reset is controlled explicitly by the RLinf train/eval loop.
   * - ``omni_config.env.flatten_obs_space`` / ``flatten_action_space``
     - Keep ``False`` to preserve structured observation / action spaces.
   * - ``omni_config.macro.use_gpu_dynamics``
     - ``False`` usually improves performance; enable only for particles / fluids.
   * - ``omni_config.macro.enable_object_states``
     - Keep ``True`` — ``BehaviorTask`` depends on object states.
   * - ``omni_config.macro.enable_transition_rules``
     - ``True`` enables transition-rule state changes (e.g. slicing, cooking).
   * - ``omni_config.macro.use_numpy_controller_backend``
     - ``True`` uses the numpy controller backend, usually faster in single-/moderate-parallel runs.
   * - ``skip_intermediate_renders``
     - When ``True``, skips rendering and collecting intermediate observations inside an action
       chunk. Saved videos then show only the frames the policy observes at chunk boundaries.
   * - ``num_env_subprocess``
     - Splits ``num_envs`` across child processes, each hosting its own Isaac/OmniGibson sim
       (see ``BehaviorProcess``). Default ``1``. **Constraint:** ``num_envs`` must be divisible by
       ``num_env_subprocess``. Higher values cut env-step bottlenecks but multiply processes and memory.
       For sharing a process across pipeline stages, see :ref:`behavior-eval-config`.

Generating cached task instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2026 assets include the official evaluation instances; see the
:doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>` to select one.
The existing ``rlinf/envs/sim/behavior/instance_generator.py`` requires OmniGibson 3.7.2
to generate additional training layouts.

--------------

**5. Evaluate a JAX-aligned Pi0.5 checkpoint**

The existing :doc:`SFT recipe <sft_openpi>` uses the 2025 dataset.
BEHAVIOR evaluation uses ``model_type: openpi``. For weights from
:doc:`sft_openpi` or the OpenPI checkpoint convertor, the eval config is:

- ``evaluations/behavior/behavior_openpi_pi05_eval.yaml``

This config runs in eval-only mode (``runner.only_eval: True``) and consumes a
converted checkpoint, i.e. one produced by the OpenPI checkpoint
convertor (``ckpt_convertor.openpi`` ``openpi_pytorch_to_openpi`` /
``sft_to_openpi``). Set the model
paths directly in the config as ``/path/to/...`` placeholders:

- ``rollout.model.model_path``: the eval checkpoint.

Normalization statistics are loaded from the converted checkpoint's bundled
``physical-intelligence/behavior/norm_stats.json`` asset.

.. code:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

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
