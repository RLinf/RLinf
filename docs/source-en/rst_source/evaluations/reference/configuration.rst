Configuration
=============

Evaluation configs are Hydra YAML files under ``evaluations/<benchmark>/``. Minimal core fields every eval config sets:

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval
     - model/pi0_5@rollout.model
   runner:
     task_type: embodied_eval
     only_eval: True
   cluster:
     component_placement:
       env,rollout: all
   env:
     eval:
       total_num_envs: 500
   rollout:
     model:
       model_path: "./RLinf-Pi05-LIBERO-SFT"

.. dropdown:: Full Example YAML (libero_spatial_openpi_pi05_eval.yaml)

   .. code-block:: yaml

      defaults:
        - env/libero_spatial@env.eval
        - model/pi0_5@rollout.model
        - override hydra/job_logging: stdout

      hydra:
        run:
          dir: .
        output_subdir: null
        searchpath:
          - file://${oc.env:EMBODIED_PATH}/config/

      cluster:
        num_nodes: 1
        component_placement:
          env,rollout: all

      runner:
        task_type: embodied_eval
        logger:
          log_path: "../results"
          project_name: rlinf
          experiment_name: "libero_spatial_openpi_pi05_eval"
          logger_backends: ["tensorboard"]
        max_epochs: 1
        max_steps: -1
        only_eval: True
        val_check_interval: -1
        save_interval: -1
        resume_dir: null
        ckpt_path: null

      env:
        group_name: "EnvGroup"
        eval:
          rollout_epoch: 1
          total_num_envs: 500
          auto_reset: True
          ignore_terminations: True
          max_episode_steps: 240
          max_steps_per_rollout_epoch: 240
          group_size: 1
          use_fixed_reset_state_ids: True
          is_eval: True
          video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

      rollout:
        group_name: "RolloutGroup"
        rollout_backend: "huggingface"
        recompute_logprobs: False
        unnorm_key: libero_10
        enable_offload: False
        pipeline_stage_num: 1
        model:
          model_path: "RLinf/RLinf-Pi05-LIBERO-SFT"
          model_type: "openpi"
          num_action_chunks: 5
          num_steps: 3
          openpi:
            task: eval
            noise_level: 0.3

.. _env-eval-fields:

env.eval Field Reference
------------------------

Fields under ``env.eval`` control parallelism, trajectory length, and test-set coverage during embodied evaluation:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Role and recommended settings
   * - ``total_num_envs``
     - Global parallel environment count, split evenly across env workers. Higher values raise throughput at higher GPU/RAM cost (LIBERO uses roughly 2 GiB host RAM per env on ``libero_spatial``). Match the init-state count when memory allows, or use a smaller value with ``auto_reset: True``.
   * - ``rollout_epoch``
     - Number of evaluation rollout epochs. Each epoch traverses the test set under the same seed and averages results. Use ``1`` for single-pass coverage or ``2`` or higher for lower variance.
   * - ``max_episode_steps``
     - Maximum interaction steps per trajectory before truncation. Match the benchmark's required horizon and training config.
   * - ``max_steps_per_rollout_epoch``
     - Per-env step budget within one ``rollout_epoch``. Must be divisible by ``rollout.model.num_action_chunks``. Without ``auto_reset``, usually equals ``max_episode_steps``; with ``auto_reset``, set to an integer multiple of ``max_episode_steps`` to run multiple episodes serially per env.
   * - ``auto_reset``
     - Reset and load the next init state when an episode ends. ``True`` covers the full test set with fewer parallel envs; ``False`` produces one trajectory per env per epoch.
   * - ``ignore_terminations``
     - Ignore early termination on task success so episodes run to ``max_episode_steps`` while recording ``success_once`` and ``success_at_end``. Keeps trajectory lengths uniform for parallel eval; recommended ``True`` unless the benchmark requires early stopping (such as SIMPLE).
   * - ``use_fixed_reset_state_ids``
     - Use pre-assigned reset state IDs instead of random sampling. Set ``True`` for deterministic evaluation.
   * - ``use_ordered_reset_state_ids``
     - Traverse init states in fixed order. Some environments (such as LIBERO) enforce ordered traversal when ``is_eval: True``; set ``True`` explicitly to keep the same order on ``auto_reset``.
   * - ``is_eval``
     - Evaluation mode flag; **must** be ``True`` to enable deterministic init-state ordering.

Fields You Must Customize
-------------------------

1. ``rollout.model.model_path``: Local model directory or Hugging Face ``owner/repo`` ID (downloaded into the Hugging Face cache automatically on first run).
2. ``env.eval`` resource settings: ``total_num_envs``, ``max_episode_steps``, and benchmark asset paths (for example ``assets_path`` on RoboTwin).
3. ``cluster.component_placement``: Match ``env`` and ``rollout`` placement to your GPUs.
4. **Real-robot evaluation:** Set robot IPs and node topology in ``cluster.node_groups`` (see :doc:`../guides/realworld`).

Deriving from a Training Config
-------------------------------

Training-only configs under ``examples/embodiment/config/`` bind the model preset to ``@actor.model`` and leave ``rollout.model`` without ``model_type`` or ``num_action_chunks``, so passing them directly to ``run_eval.sh`` fails eval validation. Copy the training YAML into ``evaluations/<benchmark>/``, remove ``algorithm`` and ``actor``, rebind the model default from ``@actor.model`` to ``@rollout.model``, and set eval mode:

.. code-block:: yaml

   defaults:
     - env/behavior_r1pro@env.eval
     - model/pi0_5@rollout.model        # rebind from @actor.model
   runner:
     task_type: embodied_eval
     only_eval: True

.. note::

   SuperMimic configs under ``evaluations/supermimic/`` are standalone evaluation configs that load ``full_weights.pt`` through ``runner.ckpt_path``, select tasks via ``env.eval.eval_tasks``, and configure exports via ``runner.eval_export``. See :doc:`../guides/supermimic`.

Config Fallback
---------------

If ``evaluations/<benchmark>/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to ``examples/embodiment/config/`` for eval-ready configs that define ``rollout.model`` (with ``model_type`` and ``num_action_chunks``). See :doc:`cli`.
