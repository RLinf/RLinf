BEHAVIOR-1K Evaluation
======================

BEHAVIOR-1K is a large-scale household scene simulation benchmark built on OmniGibson and Isaac Sim. It tasks a dual-arm R1 Pro robot with manipulation skills such as pick-and-place, stacking, and tidying. RLinf supports parallel evaluation of OpenPI and other VLA policies in BEHAVIOR environments and reports metrics such as ``eval/success_once``.

Related training doc: :doc:`../../examples/embodied/behavior`

Environment Setup
-----------------

**Install dependencies**

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

``evaluations/behavior/`` currently ships an OpenPI π₀.₅ example only. Training also supports OpenVLA-OFT; you can derive an eval YAML from ``examples/embodiment/config/`` (see :doc:`../reference/configuration`).

**Hardware and Isaac Sim**

The RLinf installer uses its BEHAVIOR-1K v3.9.4 fork with Isaac Sim 5.1. BEHAVIOR has additional GPU and driver requirements; see the `Isaac Sim requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_ in the training doc. Key points:

- Use an NVIDIA GPU with Ray Tracing support and at least 16 GB of GPU memory. GPUs without RT cores (A100, H100, etc.) are unsupported.
- Linux requires NVIDIA driver 580.65.06 or later.

**Environment variables**

Set OmniGibson data paths before every run. Isaac Sim 5.1 is installed in the Python environment. If this shell points to an older Isaac Sim installation, clear its paths with ``unset ISAAC_PATH EXP_PATH CARB_APP_PATH``.

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

BEHAVIOR assets exceed 30 GB; see :ref:`Download the BEHAVIOR assets <behavior-sim-assets>` for the 2026 assets and license setup.

Example Configs
---------------

The following example is available under ``evaluations/behavior/``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config file
     - Env preset
     - Model
   * - ``behavior_openpi_pi05_eval.yaml``
     - ``behavior_r1pro``
     - π₀.₅ (``openpi``)

If ``evaluations/behavior/<config>.yaml`` is missing, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` with the same name (e.g. ``behavior_ppo_openpi_pi05``). Fallback configs include ``actor`` / ``algorithm`` sections but still work for evaluation when ``runner.only_eval: True``.

End-to-End Workflow
-------------------

**Step 1: Activate the environment and set paths**

.. code-block:: bash

   source .venv/bin/activate
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

**Step 2: Prepare the model**

Recommended checkpoint: `RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_ (verified on the radio task). Third-party OpenPI weights (e.g. OpenPI-Comet) must be converted to PyTorch format before setting ``rollout.model.model_path``. Download the checkpoint from the repository root.

.. code-block:: bash

   MODEL_DIR="$PWD/checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 --local-dir "$MODEL_DIR"

**Step 3: Edit the config**

Copy or edit the target YAML and set at least ``rollout.model.model_path``. Generic ``env.eval`` fields are documented in :doc:`../reference/configuration` (:ref:`env-eval-fields`); BEHAVIOR-specific fields and the evaluation protocol are covered in :ref:`behavior-eval-config` below.

The OpenPI fields in ``behavior_openpi_pi05_eval.yaml`` must match training (``action_dim: 23``, ``num_action_chunks: 32``, ``openpi.config_name: pi05_behavior``, etc.). ``num_action_chunks`` is the env-executed chunk; the network horizon comes from official ``TrainConfig.model.action_horizon`` for ``pi05_behavior`` (**32**) unless ``openpi.action_horizon`` is set.

**Step 4: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path="$MODEL_DIR"

The default run evaluates instance 0 of ``turning_on_radio``. To evaluate public instance 301, set the fields in :ref:`behavior-public-instances` and rerun the launcher.

**Step 5: Check results**

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs and videos.

.. _behavior-eval-config:

Evaluation Configuration
--------------------------

BEHAVIOR evaluation runs **one task per launch** (selected by ``omni_config.task.activity_name``). A single run does not automatically sweep all 100 tasks. The fields below control parallel scale, trajectory length, and initial scene instances.

Evaluation protocol
~~~~~~~~~~~~~~~~~~~

The 2026 BEHAVIOR challenge defines 100 household tasks (names listed in ``rlinf/envs/sim/behavior/behavior_task.jsonl``). The ``behavior_r1pro`` preset defaults to ``turning_on_radio`` on scene ``house_double_floor_lower``.

Each evaluation trajectory is determined by:

- ``omni_config.task.activity_name``: task name (language instruction and BDDL definition);
- ``omni_config.task.activity_definition_id``: task definition variant (usually ``0``);
- ``omni_config.task.activity_instance_id`` and ``instance_resample_mode``: initial object layout and robot pose.

``instance_resample_mode`` supports three values:

- ``disabled`` (default): every reset loads the fixed instance for ``activity_instance_id``; if ``activity_instance_dir`` is set, the matching JSON is read from that directory.
- ``offline``: every reset **randomly** picks a cached instance from ``activity_instance_dir`` (download official ``2026-challenge-task-instances``).
- ``online``: online object resampling on reset (requires ``online_object_sampling: True``, ``use_presampled_robot_pose: False``, ``activity_instance_dir: null`` and one environment per simulator process).

.. note::

   A single launch does not automatically sweep all tasks or init states. To evaluate multiple tasks, change ``activity_name`` and rerun, or wrap launches in a batch script. For multiple instances, use ``instance_resample_mode: offline`` and average over ``rollout_epoch``.

Generic ``env.eval`` fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - BEHAVIOR guidance
   * - ``total_num_envs``
     - Global parallel env count. Set it according to GPU memory; the example defaults to ``4``.
   * - ``rollout_epoch``
     - Number of eval rounds with the same config; metrics are averaged. The example defaults to ``4``.
   * - ``max_episode_steps``
     - Max steps per trajectory. Leave ``null`` in the π₀.₅ example to use the selected task's official timeout, or enter an action limit.
   * - ``max_steps_per_rollout_epoch``
     - Total interaction steps per rollout round. Leave ``null`` to request enough whole action chunks to reach the episode limit. An explicit budget **must be divisible by** ``rollout.model.num_action_chunks``. The final chunk only executes the remaining actions in a fixed episode.
   * - ``num_env_subprocess``
     - Isaac sim subprocesses per env worker (default ``1``). Increasing this can reduce stepping bottlenecks but multiplies VRAM and process overhead; the environment count on each worker must be divisible by both ``num_env_subprocess`` and ``rollout.pipeline_stage_num``.
   * - ``skip_intermediate_renders``
     - When ``True``, skips intermediate rendering and observations inside action chunks for faster stepping; saved videos only contain chunk-boundary frames. Set ``False`` to record every action step.

Key ``omni_config`` fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

These live under ``env.eval.omni_config`` (inherited from ``examples/embodiment/config/env/behavior_r1pro.yaml`` and overridable in eval YAML):

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_name: turning_on_radio
           activity_definition_id: 0
           activity_instance_id: 0
           activity_instance_dir: null          # directory of cached instance JSON files
           instance_file_format: tro_state        # template | tro_state
           instance_resample_mode: disabled       # disabled | offline | online
         scene:
           scene_model: house_double_floor_lower
           partial_scene_load: true               # load task-relevant rooms only

For full field descriptions (``partial_scene_load``, instance settings, etc.), see "behavior_r1pro.yaml key settings" in :doc:`../../examples/embodied/behavior`.

.. _behavior-public-instances:

Evaluate Public Instance 301
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Public radio instance 301 contains the starting poses of the robot and task objects. The simulator builds the room from the full scene template for instance 0, then applies layout 301. Merge the fields below into the matching sections of the evaluation YAML, keeping its other settings.

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_instance_id: 301
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
         scene:
           scene_instance: house_double_floor_lower_task_turning_on_radio_0_0_template

To run instance 302 of the same task, change only ``activity_instance_id`` to ``302``. For another task, set its task and scene names and select the matching full template and instance directory.

GPU and cluster placement
~~~~~~~~~~~~~~~~~~~~~~~~~

BEHAVIOR stepping is slow; allocate enough GPUs to env workers and share or split placement with rollout:

.. code-block:: yaml

   cluster:
     component_placement:
       rollout,env: all          # share all GPUs (example default)

You can also place env and rollout on separate GPUs to reduce memory pressure; see "Key cluster settings" in the training doc.

With the default ``rollout.pipeline_stage_num: 1``, all environments in a simulator process receive actions together. If stages share a process, all scenes advance in physics, and a waiting robot can keep moving under its previous command.

Advanced Usage
--------------

**Switch evaluation task**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.omni_config.task.activity_name=picking_up_trash

**Random offline instance sampling**

Use the scene template and instance directory from :ref:`behavior-public-instances`. At each reset, each environment randomly selects a cached layout from that directory and may select the same layout again.

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           instance_file_format: tro_state
           instance_resample_mode: offline
       rollout_epoch: 5

**Adjust parallel scale**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.total_num_envs=4 \
     env.eval.num_env_subprocess=2

**Evaluate from a training config**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_ppo_openpi_pi05 \
     rollout.model.model_path=/path/to/model

FAQ
---

- **Data download:** BEHAVIOR assets are large; complete Isaac Sim, OmniGibson asset, and license setup per :doc:`../../examples/embodied/behavior` before evaluation.
- **Isaac Sim startup failure:** Clear paths from older Isaac Sim installations with ``unset ISAAC_PATH EXP_PATH CARB_APP_PATH`` and activate the installed environment.
- **Headless mode:** ``run_eval.sh`` sets ``OMNIGIBSON_HEADLESS=1`` by default.
- **Out of memory:** Lower ``total_num_envs`` or ``num_env_subprocess``.
- **Blurry or blocky rendering:** The GPU lacks Ray Tracing; use RTX 30/40 series or newer.
- **Very slow startup:** First load of a large scene is expensive; keep ``partial_scene_load: true`` to load only task-relevant rooms.
- **Fewer video frames than expected:** ``skip_intermediate_renders: True`` skips intermediate chunk frames and keeps only observations consumed by the policy.
- **Instance load failure:** JSON filenames under ``activity_instance_dir`` must match ``activity_name``, ``activity_definition_id``, and ``scene_model``; see ``rlinf/envs/sim/behavior/instance_loader.py``.
- **Step count validation error:** ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks``.
