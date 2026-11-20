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

The installer selects a BEHAVIOR-1K fork based on Stanford’s v3.9.4 release. It runs on Isaac Sim 5.1 and supports sending actions to selected scenes in a shared simulator.

- Use an NVIDIA GPU with RT cores. NVIDIA lists 16 GB of GPU memory as the minimum for Isaac Sim 5.1. A100 and H100 GPUs lack RT cores and are unsupported.
- NVIDIA lists Linux driver 580.65.06 as a tested version. Check the supported driver versions and operating systems in the `Isaac Sim 5.1 requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.

The RLinf Docker image for the BEHAVIOR-1K v3.9.4 fork is pending release.

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

By default this command evaluates the radio task on base layout 0. To use a public layout such as 301, set the fields in :ref:`behavior-public-instances` and rerun the same command.

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
- ``online``: resample objects on reset. This requires one environment per simulator process, ``online_object_sampling: True``, and ``use_presampled_robot_pose: False``.

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
     - Leave unset to use the official limit for the selected task, or enter the maximum number of actions in one episode.
   * - ``max_steps_per_rollout_epoch``
     - Leave unset to request enough whole action chunks to reach the episode limit. With the default fixed episode, only the remaining actions in the last chunk are executed. An explicit budget must be divisible by ``rollout.model.num_action_chunks``.
   * - ``num_env_subprocess``
     - Simulator processes per environment worker (default 1). The environment count on each worker must be divisible by both its process count and its pipeline stage count.
   * - ``skip_intermediate_renders``
     - Set to ``True`` to render and collect observations only on the last executed step of each action chunk. Videos keep one frame per chunk.

Sharing a simulator across pipeline stages saves the startup and GPU memory cost of extra processes. Only the active stage receives new actions, but all scenes in the process advance in physics. A waiting robot can keep moving under its previous command. Use the default ``rollout.pipeline_stage_num: 1`` to step all environments in each process together.

.. _behavior-public-instances:

Select a public instance
~~~~~~~~~~~~~~~~~~~~~~~~

Merge the following settings into ``evaluations/behavior/behavior_openpi_pi05_eval.yaml``. For the radio task, template 0 loads the scene. Public instance 301 supplies the starting layout for evaluation.

.. code-block:: yaml

   env:
     eval:
       omni_config:
         scene:
           scene_model: house_double_floor_lower
           scene_instance: house_double_floor_lower_task_turning_on_radio_0_0_template
         task:
           activity_name: turning_on_radio
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
           activity_instance_id: 301

To run instance 302 of the same task, change only ``activity_instance_id`` to ``302``. For another task, set the task and scene names, its full scene template, and its cached-instance directory. See :doc:`../../examples/embodied/behavior` for the other scene settings.

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
