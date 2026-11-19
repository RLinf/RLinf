RL with the 2026 BEHAVIOR Challenge
===================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   The BEHAVIOR benchmark (image: `BEHAVIOR <https://behavior.stanford.edu>`__).

The `2026 BEHAVIOR Challenge <https://behavior.stanford.edu/challenge/index.html>`__
uses OmniGibson 3.9.3 and NVIDIA Isaac Sim 5.1 to simulate household tasks.
The RLinf recipes use the dual-arm R1 Pro mobile robot to train vision-language-action (VLA) policies.

Overview
--------

Prepare the 2026 simulation assets in your RLinf environment and select an RLinf PPO training or evaluation config for a household task.

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

| **You'll do:** check the simulator versions → download the 2026 assets and a base model → select an RLinf config.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Python 3.11 · OmniGibson 3.9.3 · Isaac Sim 5.1.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Detail
   * - Tasks
     - The challenge covers 100 household tasks. Set ``omni_config.task.activity_name`` to one of the 50 names in this branch's task list. The default is ``turning_on_radio``.
   * - Robot
     - The RLinf recipes and the default challenge configuration use the dual-arm R1 Pro mobile robot.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - RLinf observation
     - Head RGB (720×720), left and right wrist RGB (480×480 each), robot proprioception, and the task instruction.
   * - Challenge observation
     - The default profile returns three 224×224 RGB images and proprioception. The full-resolution profile adds depth and uses 720×720 for the head and 480×480 for each wrist. See the `challenge evaluation rules <https://behavior.stanford.edu/challenge/evaluation.html>`_.
   * - Action
     - The R1 Pro’s 23-dimensional continuous action includes 3 base velocity commands (forward/backward, sideways, and turning), 4 joint position targets for the torso, 7 joint position targets per arm, and 1 open/close command per gripper.


Installation
------------

.. warning::

   Use an NVIDIA GPU with RT cores. Check the supported operating systems, drivers, and VRAM in the
   `Isaac Sim 5.1 system requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

**Option 1: Docker image** — The BEHAVIOR 3.9.3 image is pending release.

**Option 2: Custom environment**

From the RLinf repository root, install RLinf, OpenPI, and the simulator in one
Python 3.11.14 environment. The installer sets up Torch 2.7.0 and the required
versions of LeRobot, Hugging Face Hub, and Warp.

.. code-block:: bash

   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate
   python -c "from importlib.metadata import version; print(version('omnigibson'), version('isaacsim'))"

If the output shows OmniGibson 3.9.3 and Isaac Sim 5.1.x, use the SDK commands below. You can also download the three asset archives before installing the SDK.


Download the Assets
-------------------

Choose a separate root for the 2026 simulation assets. If you already have the assets, use their existing root.
Export the path in the shell that launches RLinf.

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

Run these commands in the active RLinf environment to download any missing assets for 3.9.3. OmniGibson prompts for the dataset license when it first retrieves the scene asset key.

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

RLinf released these two BEHAVIOR checkpoints in 2025. Download the checkpoint for your model using either method below.

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

Launch PPO training with the existing entry point.

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
   - Set the number of simulator subprocesses to fit your scene and available GPU memory.
   - Cached task instances → generate them with ``rlinf/envs/sim/behavior/instance_generator.py`` and ``examples/embodiment/config/env/behavior_r1pro.yaml``.
   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`

.. warning::

   Training success rates for OpenVLA-OFT and π₀ on BEHAVIOR 3.9.3 have yet to be verified.

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

1. Set ``rollout.model.model_path`` to the converted model directory.
2. Increase ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` in
   ``env.eval`` if you need longer trajectories (for example, ``4096``).

.. code-block:: yaml

   env:
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

Prepare the 2026 simulation assets as described on this page. The
:doc:`BEHAVIOR-1K evaluation guide <../../evaluations/guides/behavior>` explains how to set the model path,
launch ``behavior_openpi_pi05_eval``, read the ``eval/success_once`` metric, and find the logs and videos.


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
     - Task name, for example ``turning_on_radio``. Use ``omni_config.task.activity_instance_id`` to select an initial layout for that task.
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
   * - ``omni_config.camera.head_resolution`` / ``omni_config.camera.wrist_resolution``
     - Head / wrist camera resolutions (defaults 720×720 / 480×480, applied to R1Pro sensors).
   * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
     - Action, rendering, and physics update rates. The RLinf R1 Pro preset sets them to 60, 60, and 120 Hz, respectively.
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
     - When ``True``, skips collecting intermediate observations inside an action chunk (large
       env-speed gain). Saved videos then show only the frames the policy observes at chunk boundaries.
   * - ``num_env_subprocess``
     - Number of simulator processes per environment worker (default 1). Allocate at least one process per pipeline stage. The worker’s environment count must be divisible by both the process count and the stage count.

Generating cached task instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rlinf/envs/sim/behavior/instance_generator.py`` generates ``*_template.json`` and
``*_template-tro_state.json`` files directly from ``behavior_r1pro.yaml`` (it reads
``scene_model``, ``activity_name``, ``activity_definition_id``, the robot config, and
room-loading settings, then temporarily switches to online object sampling).

The 2026 assets include the official evaluation instances. Use the commands below to generate additional training layouts.
The generator cannot yet determine the default output directory on 3.9.3. Before using either command,
set ``omni_config.task.activity_instance_dir`` in ``behavior_r1pro.yaml`` to the directory for the generated layouts.

.. code-block:: bash

   cd /path/to/RLinf

   python rlinf/envs/sim/behavior/instance_generator.py \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-format template \
     --start-idx 1 --end-idx 50

   python rlinf/envs/sim/behavior/instance_generator.py \
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

**5. Evaluate a JAX-aligned Pi0.5 checkpoint**

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

Run standalone evaluation from the same environment.

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
