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

BEHAVIOR depends on Isaac Sim 4.5 and has additional GPU and driver requirements; see the `Isaac Sim requirements <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html>`_ in the training doc. Key points:

- A GPU with Ray Tracing support (e.g. RTX 30/40 series) is recommended. GPUs without RT (A100, H100, etc.) produce poor rendering quality with visible artifacts.
- Hopper and newer GPUs require NVIDIA driver 570 or later.

You can also run evaluation inside the official Docker image ``rlinf/rlinf:agentic-rlinf0.3-behavior``; see :doc:`../../examples/embodied/behavior`.

**Environment variables**

Set ``ISAAC_PATH`` and OmniGibson data paths before every run (``run_eval.sh`` auto-fills derived variables such as ``OMNIGIBSON_DATASET_PATH``, ``EXP_PATH``, and ``CARB_APP_PATH``):

.. code-block:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

BEHAVIOR assets exceed 30 GB; see the "Resource download" section in :doc:`../../examples/embodied/behavior` for download and license setup.

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
     - π₀.₅

If ``evaluations/behavior/<config>.yaml`` is missing, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` with the same name (e.g. ``behavior_ppo_openpi_pi05_eval``). Fallback configs include ``actor`` / ``algorithm`` sections but still work for evaluation when ``runner.only_eval: True``.

End-to-End Workflow
-------------------

**Step 1: Activate the environment and set paths**

.. code-block:: bash

   source .venv/bin/activate
   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

**Step 2: Prepare the model**

Recommended checkpoint: `RLinf/RLinf-Pi0-Behavior <https://huggingface.co/RLinf/RLinf-Pi0-Behavior>`_ (download commands in the training doc). Third-party OpenPI weights (e.g. OpenPI-Comet) must be converted to PyTorch format before setting ``rollout.model.model_path``.

**Step 3: Edit the config**

Copy or edit the target YAML and set at least ``rollout.model.model_path``. Generic ``env.eval`` fields are documented in :doc:`../reference/configuration` (:ref:`env-eval-fields`); BEHAVIOR-specific fields and the evaluation protocol are covered in :ref:`behavior-eval-config` below.

The OpenPI fields in ``behavior_openpi_pi05_eval.yaml`` must match training (``action_dim: 23``, ``num_action_chunks: 32``, ``openpi.config_name: pi05_behavior``, etc.).

**Step 4: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

**Step 5: Check results**

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs and videos.

.. _behavior-eval-config:

Evaluation Configuration
--------------------------

BEHAVIOR evaluation runs **one task per launch** (selected by ``omni_config.task.activity_name``). A single run does not automatically sweep all 50 tasks. The fields below control parallel scale, trajectory length, and initial scene instances.

Evaluation protocol
~~~~~~~~~~~~~~~~~~~

BEHAVIOR-1K defines 50 household tasks (names listed in ``rlinf/envs/behavior/behavior_task.jsonl``). The ``behavior_r1pro`` preset defaults to ``turning_on_radio`` on scene ``house_double_floor_lower``.

Each evaluation trajectory is determined by:

- ``omni_config.task.activity_name``: task name (language instruction and BDDL definition);
- ``omni_config.task.activity_definition_id``: task definition variant (usually ``0``);
- ``omni_config.task.activity_instance_id`` and ``instance_resample_mode``: initial object layout and robot pose.

``instance_resample_mode`` supports three values:

- ``disabled`` (default): every reset loads the fixed instance for ``activity_instance_id``; if ``activity_instance_dir`` is set, the matching JSON is read from that directory.
- ``offline``: every reset **randomly** picks a cached instance from ``activity_instance_dir`` (download official ``2025-challenge-task-instances`` or generate files with ``instance_generator.py``).
- ``online``: online object resampling on reset (requires ``online_object_sampling: True`` and ``use_presampled_robot_pose: False``; slower startup).

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
     - Global parallel env count. Each BEHAVIOR env uses roughly **10 GiB** VRAM; the example defaults to ``8``.
   * - ``rollout_epoch``
     - Number of eval rounds with the same config; metrics are averaged. The example defaults to ``2``.
   * - ``max_episode_steps``
     - Max steps per trajectory. The π₀.₅ example uses ``4096`` (the preset default ``2000`` may be too short for long-horizon tasks).
   * - ``max_steps_per_rollout_epoch``
     - Total interaction steps per rollout round; **must be divisible by** ``rollout.model.num_action_chunks``. Without ``auto_reset``, usually equals ``max_episode_steps``.
   * - ``num_env_subprocess``
     - Isaac sim subprocesses per env worker (default ``1``). Increasing this can reduce stepping bottlenecks but multiplies VRAM and process overhead; ``total_num_envs`` must be divisible by ``num_env_subprocess × pipeline_stage_num``.
   * - ``skip_intermediate_obs_in_chunk``
     - When ``True``, skips intermediate observations inside action chunks for faster stepping; saved videos only contain chunk-boundary frames.

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

For full field descriptions (``partial_scene_load``, ``instance_generator.py``, etc.), see "behavior_r1pro.yaml key settings" in :doc:`../../examples/embodied/behavior`.

GPU and cluster placement
~~~~~~~~~~~~~~~~~~~~~~~~~

BEHAVIOR stepping is slow; allocate enough GPUs to env workers and share or split placement with rollout:

.. code-block:: yaml

   cluster:
     component_placement:
       rollout,env: all          # share all GPUs (example default)

You can also place env and rollout on separate GPUs to reduce memory pressure; see "Key cluster settings" in the training doc.

Advanced Usage
--------------

**Switch evaluation task**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.omni_config.task.activity_name=picking_up_trash

**Random offline instance sampling**

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances
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

   bash evaluations/run_eval.sh behavior behavior_ppo_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

FAQ
---

- **Data download:** BEHAVIOR assets are large; complete Isaac Sim, OmniGibson asset, and license setup per :doc:`../../examples/embodied/behavior` before evaluation.
- **ISAAC_PATH not set:** ``run_eval.sh`` defaults to ``/path/to/isaac-sim``; Isaac Sim will fail to start without a valid path.
- **Headless mode:** ``run_eval.sh`` sets ``OMNIGIBSON_HEADLESS=1`` by default.
- **Out of memory:** Lower ``total_num_envs`` or ``num_env_subprocess``; each env uses about 10 GiB VRAM.
- **Blurry or blocky rendering:** The GPU lacks Ray Tracing; use RTX 30/40 series or newer.
- **Very slow startup:** First load of a large scene is expensive; keep ``partial_scene_load: true`` to load only task-relevant rooms.
- **Fewer video frames than expected:** ``skip_intermediate_obs_in_chunk: True`` skips intermediate chunk frames and keeps only observations consumed by the policy.
- **Instance load failure:** JSON filenames under ``activity_instance_dir`` must match ``activity_name``, ``activity_definition_id``, and ``scene_model``; see ``rlinf/envs/behavior/instance_loader.py``.
- **Step count validation error:** ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks``.
