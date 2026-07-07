Configuration Reference
=======================

Eval configs are Hydra YAML files under ``evaluations/<benchmark>/``. The core structure (using ``libero_spatial_openpi_pi05_eval.yaml`` as an example):

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval      # Environment preset
     - model/pi0_5@rollout.model        # Model preset
     - override hydra/job_logging: stdout

   hydra:
     searchpath:
       - file://${oc.env:EMBODIED_PATH}/config/

   runner:
     task_type: embodied_eval   # Must be embodied_eval
     only_eval: True            # Evaluation only, no training
     ckpt_path: null            # Optional: load a .pt checkpoint
     logger:
       log_path: "../results"

   cluster:
     component_placement:
       env,rollout: all          # GPU placement for env and rollout

   env:
     eval:
       total_num_envs: 500       # Number of parallel environments
       rollout_epoch: 1          # Number of eval epochs
       max_episode_steps: 240
       auto_reset: True
       is_eval: True
       video_cfg:
         save_video: True

   rollout:
     generation_backend: "huggingface"
     model:
       model_path: "/path/to/model"   # Required: model weights path
       model_type: "openpi"

.. _env-eval-fields:

env.eval Field Reference
------------------------

The fields below live under ``env.eval`` and control parallelism, trajectory length, and test-set coverage for embodied evaluation.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Role and recommended settings
   * - ``total_num_envs``
     - Total number of parallel environments, evenly distributed across env workers. Higher values improve throughput but use more GPU/RAM. Set to the total init-state count when resources allow; use a smaller value with ``auto_reset`` when memory is limited.
   * - ``rollout_epoch``
     - Number of evaluation rollout epochs. Each epoch traverses the test set under the same seed; multiple epochs are averaged for lower variance. Use ``1`` for full coverage; use ``2`` or more for stabler metrics.
   * - ``max_episode_steps``
     - Maximum interaction steps per trajectory before forced truncation. Should meet the benchmark's minimum step requirement and match the model's training config.
   * - ``max_steps_per_rollout_epoch``
     - Per-env step budget within one ``rollout_epoch``. Chunk steps = ``max_steps_per_rollout_epoch / rollout.model.num_action_chunks``; this value **must be divisible by** ``num_action_chunks``. Without ``auto_reset``, usually equals ``max_episode_steps``; with ``auto_reset``, set to an integer multiple of ``max_episode_steps`` to run multiple trajectories serially per env per epoch.
   * - ``auto_reset``
     - Whether to reset and load the next init state when an episode ends (success or truncation). ``True`` enables full test-set coverage with fewer parallel envs; ``False`` yields one trajectory per env per epoch, with ``finish_rollout`` advancing init states between epochs.
   * - ``ignore_terminations``
     - Whether to ignore early termination on task success. When ``True``, episodes only end at ``max_episode_steps``; success is recorded in ``success_once`` / ``success_at_end`` without early reset, keeping trajectory lengths uniform for parallel eval. Recommended ``True`` for eval configs.
   * - ``use_fixed_reset_state_ids``
     - Use pre-assigned reset state IDs instead of random sampling. Set ``True`` for evaluation so each trajectory maps to a deterministic initial condition.
   * - ``use_ordered_reset_state_ids``
     - Traverse init states in a fixed order. When ``is_eval=True``, some environments (e.g. LIBERO) enforce ordered traversal internally; set ``True`` explicitly to reuse the same ordering outside eval. On ``auto_reset``, the next state ID is taken in order.
   * - ``is_eval``
     - Evaluation mode flag; **must** be ``True``. Enables the ordered init-state list and ordered state assignment on ``auto_reset``.

Fields You Must Customize
-------------------------

1. ``rollout.model.model_path``: Local model directory or HuggingFace cache path.
2. Resource-related fields under ``env.eval``: ``total_num_envs``, ``max_episode_steps``, ``assets_path`` (RoboTwin), etc.
3. ``cluster.component_placement``: Adjust ``env`` and ``rollout`` placement for your GPUs.
4. **Real-robot eval:** Configure Franka IP and node topology in ``cluster.node_groups`` (see ``realworld/realworld_eval.yaml``).

Deriving from a Training Config
-------------------------------

Copy the matching YAML from ``examples/embodiment/config/`` or ``tests/e2e_tests/embodied/``, remove training sections (``algorithm``, ``actor``, etc.), keep ``env.eval`` and ``rollout``, and set:

- ``runner.task_type: embodied_eval``
- ``runner.only_eval: True``

Config Fallback
---------------

If ``evaluations/<benchmark>/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` with the same config name. See :doc:`cli`.
