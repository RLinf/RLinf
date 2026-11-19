BEHAVIOR-1K Evaluation
======================

BEHAVIOR-1K is a large-scale household scene simulation benchmark built on OmniGibson and Isaac Sim. It tasks a dual-arm R1 Pro robot with manipulation skills such as pick-and-place, stacking, and tidying. RLinf supports parallel evaluation of OpenPI and other VLA policies in BEHAVIOR environments and reports metrics such as ``eval/success_once``.

Related training doc: :doc:`../../examples/embodied/behavior`

Environment Setup
-----------------

**Install dependencies**

.. code-block:: bash

   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

``evaluations/behavior/`` currently ships an OpenPI π₀.₅ example only. Training also supports OpenVLA-OFT; you can derive an eval YAML from ``examples/embodiment/config/`` (see :doc:`../reference/configuration`).

**Hardware and Isaac Sim**

Evaluation uses OmniGibson 3.9.3 and Isaac Sim 5.1.

- Use an NVIDIA GPU with RT cores. NVIDIA lists 16 GB of GPU memory as the minimum for Isaac Sim 5.1. A100 and H100 GPUs lack RT cores and are unsupported.
- NVIDIA lists Linux driver 580.65.06 as a tested version. Check the supported driver versions and operating systems in the `Isaac Sim 5.1 requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.

The RLinf BEHAVIOR Docker image with OmniGibson 3.9.3 is pending release.

**Environment variables**

Set the BEHAVIOR data root in the terminal where you launch evaluation.

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

OmniGibson looks for ``behavior-1k-assets/`` (scene and object assets), ``omnigibson-robot-assets/`` (robot assets), and ``omnigibson.key`` (the asset key) under this root. Keep the 2026 cached layouts and task statistics in ``2026-challenge-task-instances/`` under the same root.

The installation command above installs Isaac Sim 5.1 as a Python package. The package sets ``ISAAC_PATH`` when it is imported. Use a shell with ``ISAAC_PATH``, ``EXP_PATH``, and ``CARB_APP_PATH`` unset when running the installation command.

Download and license setup commands are in :doc:`../../examples/embodied/behavior`.

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

**Step 2: Prepare the model**

Recommended checkpoint: `RLinf/RLinf-Pi0-Behavior <https://huggingface.co/RLinf/RLinf-Pi0-Behavior>`_ (download commands in the training doc). Third-party OpenPI weights (e.g. OpenPI-Comet) must be converted to PyTorch format before setting ``rollout.model.model_path``.

**Step 3: Edit the config**

Copy or edit the target YAML and set at least ``rollout.model.model_path``. Generic ``env.eval`` fields are documented in :doc:`../reference/configuration` (:ref:`env-eval-fields`); BEHAVIOR-specific fields and the evaluation protocol are covered in :ref:`behavior-eval-config` below.

The OpenPI fields in ``behavior_openpi_pi05_eval.yaml`` must match training (``action_dim: 23``, ``num_action_chunks: 32``, ``openpi.config_name: pi05_behavior``, etc.). ``num_action_chunks`` is the env-executed chunk; the network horizon comes from official ``TrainConfig.model.action_horizon`` for ``pi05_behavior`` (**32**) unless ``openpi.action_horizon`` is set.

**Step 4: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

The command above runs the radio task on base layout ``0`` with the default configuration. To evaluate the first 2026 public test instance, set the task's cached-instance directory and select instance ID ``301``. Public test index ``0`` refers to layout ``301``. The layout determines the starting object states and robot pose.

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     +env.eval.omni_config.scene.scene_instance=house_double_floor_lower_task_turning_on_radio_0_0_template \
     "env.eval.omni_config.task.activity_instance_dir=${OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances" \
     env.eval.omni_config.task.activity_instance_id=301

**Step 5: Check results**

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs and videos.

.. _behavior-eval-config:

Evaluation Configuration
--------------------------

BEHAVIOR evaluation runs **one task per launch** (selected by ``omni_config.task.activity_name``). A single run does not automatically sweep all 100 tasks. The fields below control parallel scale, trajectory length, and initial scene instances.

Evaluation protocol
~~~~~~~~~~~~~~~~~~~

The 2026 challenge covers 100 tasks. RLinf provides instructions for all 100 tasks. The ``behavior_r1pro`` preset defaults to ``turning_on_radio`` on scene ``house_double_floor_lower``.

Each evaluation trajectory is determined by:

- ``omni_config.task.activity_name``: task name (language instruction and BDDL definition);
- ``omni_config.task.activity_definition_id``: task definition variant (usually ``0``);
- ``omni_config.task.activity_instance_id`` and ``instance_resample_mode``: initial object layout and robot pose.

``instance_resample_mode`` supports three values:

- ``disabled`` (default): every reset loads the fixed instance for ``activity_instance_id``; if ``activity_instance_dir`` is set, the matching JSON is read from that directory.
- ``offline``: each reset randomly selects a cached layout from the task directory you set in ``activity_instance_dir``.
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
     - Number of environments across all workers. The π₀.₅ evaluation defaults to four environments and one pipeline stage. With one environment worker, all four share one simulator process.
   * - ``rollout_epoch``
     - Number of evaluation rounds. The π₀.₅ example runs four rounds.
   * - ``max_episode_steps``
     - Leave unset in the π₀.₅ evaluation to use the official timeout for the task. The SDK checks a zero-based counter against 1.5 times the mean human demonstration length, rounded down. Set an integer to specify the number of actions directly.
   * - ``max_steps_per_rollout_epoch``
     - Leave unset to request enough whole action chunks to reach the episode limit. With the default fixed episode, only the remaining actions in the last chunk are executed. An explicit budget must be divisible by ``rollout.model.num_action_chunks``.
   * - ``num_env_subprocess``
     - Number of simulator processes per environment worker (default 1). When there are fewer processes than pipeline stages, stages share processes. Only the active stage receives actions and advances its episode counters. Each process handling those actions advances physics for all its scenes. The worker’s environment count must be divisible by both the process count and the stage count.
   * - ``skip_intermediate_renders``
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
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
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
- **Headless mode:** ``run_eval.sh`` sets ``OMNIGIBSON_HEADLESS=1`` by default.
- **Out of memory:** Reduce the number of parallel environments or simulator processes per GPU. Each simulator process uses additional GPU memory.
- **Blurry or blocky rendering:** The GPU lacks Ray Tracing; use RTX 30/40 series or newer.
- **Very slow startup:** First load of a large scene is expensive; keep ``partial_scene_load: true`` to load only task-relevant rooms.
- **Fewer video frames than expected:** ``skip_intermediate_renders: True`` skips intermediate chunk frames and keeps only observations consumed by the policy.
- **Instance load failure:** JSON filenames under ``activity_instance_dir`` must match ``activity_name``, ``activity_definition_id``, and ``scene_model``; see ``rlinf/envs/sim/behavior/instance_loader.py``.
- **Step count validation error:** ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks``.
