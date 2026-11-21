BEHAVIOR-1K Evaluation
======================

Evaluate OpenPI π₀.₅ on the BEHAVIOR-1K task ``turning_on_radio``. Start with the default scene or select a public 2026 challenge instance to measure success from a specific starting layout.

Set Up the Simulator
--------------------

The RLinf installer sets up its BEHAVIOR-1K v3.9.4 fork with Isaac Sim 5.1. The simulator requires an NVIDIA GPU with RT cores and at least 16 GB of GPU memory. See the `Isaac Sim 5.1 requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_ for supported drivers and operating systems.

Run the installer from the RLinf repository root. If this shell points to another Isaac Sim installation, clear its paths first.

.. code-block:: bash

   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

Set ``OMNIGIBSON_DATA_PATH`` to the directory containing your assets and key. For a new installation, follow :ref:`Download the BEHAVIOR assets <behavior-sim-assets>` to get the scenes, robot models, asset key and 2026 public instances.

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

Download the Model and Run the Radio Task
-----------------------------------------

From the repository root, download the `π₀.₅ PT50 checkpoint <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_ used for the radio evaluation and pass the downloaded directory to the launcher.

.. code-block:: bash

   MODEL_DIR="$PWD/checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 --local-dir "$MODEL_DIR"

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path="$MODEL_DIR"

The default run evaluates instance 0 of ``turning_on_radio``. The terminal reports ``eval/success_once``, the fraction of episodes that succeeded at least once. See :doc:`evaluation results <../reference/results>` for logs and videos. To evaluate public instance 301, edit the evaluation YAML as shown in :ref:`behavior-public-instances` and rerun the launcher.

.. _behavior-eval-config:

Configure Evaluation
--------------------

Default Task and Scene
~~~~~~~~~~~~~~~~~~~~~~

The evaluation config ``evaluations/behavior/behavior_openpi_pi05_eval.yaml`` inherits these defaults from ``examples/embodiment/config/env/behavior_r1pro.yaml``. With ``activity_instance_dir: null``, evaluation uses the built-in layout. ``partial_scene_load: true`` loads only the rooms needed for the task.

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_name: turning_on_radio
           activity_definition_id: 0
           activity_instance_id: 0
           activity_instance_dir: null
           instance_file_format: tro_state
           instance_resample_mode: disabled
         scene:
           scene_model: house_double_floor_lower
           partial_scene_load: true

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

Sample Cached Instances at Random
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep the template and instance directory from the 301 example. This command runs five evaluation rounds. At each reset, each environment draws a layout from that directory and may draw the same layout again.

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path="$MODEL_DIR" \
     env.eval.omni_config.task.instance_resample_mode=offline \
     env.eval.rollout_epoch=5

For online layout sampling, under ``env.eval.omni_config.task``, set ``instance_resample_mode: online``, ``online_object_sampling: True`` and ``use_presampled_robot_pose: False``. Set ``activity_instance_dir: null`` and use one environment per simulator process.

Parallel Environments, Step Limits, and Video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the fields below under ``env.eval`` in the evaluation YAML. See :ref:`env-eval-fields` for other evaluation settings.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - BEHAVIOR guidance
   * - ``total_num_envs``
     - Number of environments across all workers (default 4).
   * - ``rollout_epoch``
     - Number of evaluation rounds. The π₀.₅ example runs four rounds.
   * - ``max_episode_steps``
     - Leave unset to use the official limit for the selected task, or enter the maximum number of actions in one episode.
   * - ``max_steps_per_rollout_epoch``
     - Leave unset to request enough whole action chunks to reach the episode limit. With the default fixed episode, only the remaining actions in the last chunk are executed. An explicit budget must be divisible by ``rollout.model.num_action_chunks``.
   * - ``num_env_subprocess``
     - Simulator processes per environment worker (default 1). The environment count on each worker must be divisible by both its process count and its pipeline stage count.
   * - ``skip_intermediate_renders``
     - With ``True``, the simulator executes every action and renders the observation at the end of each chunk. For a full 32-action chunk, a recording contains the frame from step 32. To record every action step, enable video and set this field to ``False``.

With the default ``rollout.pipeline_stage_num: 1``, all environments in a simulator process receive their actions together. If two stages share a process, stepping stage A also advances the physics clock for stage B. A robot in stage B can keep moving under its previous command while it waits for the next one.

To place model inference and simulation on separate GPUs, set ``cluster.component_placement`` as described in :doc:`component placement <../../concepts/placement>`.

**Evaluate a training config**

To evaluate from a training config, replace the config name in the launch command with ``behavior_ppo_openpi_pi05``. If the evaluation directory has no matching config, the launcher reads it from ``examples/embodiment/config/``.
