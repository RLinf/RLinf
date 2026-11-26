Evaluate DROID tasks in Isaac Lab-Arena
=======================================

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Install RLinf
      :link: ../../start/installation
      :link-type: doc

   .. grid-item-card:: Prepare Arena and the DROID checkpoint
      :link: ../../examples/embodied/isaaclab_arena
      :link-type: doc

Setup and Run
-------------

Use the shell from the Arena example. In a new shell, activate the RLinf environment and export ``MODEL_DIR`` and the Isaac Sim license variables again. If you use an external Arena installation, also export ``ISAACLAB_ARENA_PYTHON`` with the same absolute interpreter path.

Set ``env.eval.init_params.environment`` to an Arena graph YAML or a registered environment name. Graphs define the scene, embodiment and task together. Use an absolute path for your own graph, or a path relative to the Arena checkout for a bundled graph. Select the ``droid_abs_joint_pos`` embodiment.

.. tab-set::

   .. tab-item:: Registered task

      For example, switch from the banana graph to Arena's registered maple-table task and choose the object to pick up:

      .. code-block:: bash

         bash evaluations/run_eval.sh isaaclab_arena_pi05_eval \
             rollout.model.model_path="$MODEL_DIR" \
             env.eval.init_params.environment=pick_and_place_maple_table \
             +env.eval.init_params.environment_args.pick_up_object=banana_ycb_robolab

      Arena uses the task description as the prompt by default. Set ``env.eval.init_params.language_instruction`` to supply your own.

   .. tab-item:: Short rollout

      Check the default banana task with one environment. The policy predicts 15 actions per call. A 30-step rollout budget lets an episode reach its 16-step limit if it has not already ended.

      .. code-block:: bash

         bash evaluations/run_eval.sh isaaclab_arena_pi05_eval \
             rollout.model.model_path="$MODEL_DIR" \
             env.eval.total_num_envs=1 env.eval.max_episode_steps=16 \
             env.eval.max_steps_per_rollout_epoch=30

Each environment contributes one completed episode per evaluation epoch. A drop and a timeout both count as unsuccessful. ``eval/episode_len`` counts executed steps through the first terminal action. Set ``env.eval.video_cfg.save_video=true`` to record the camera frames.

The banana task runs for up to 40 seconds at 15 Hz (600 steps). For a shorter protocol set both limits, as above. The evaluation budget must cover the episode limit and be divisible by the policy chunk length. At the next evaluation epoch the worker rebuilds the scene with the next placement seed, so earlier terminations do not change the next round's placements.

.. _isaaclab-arena-single-action-ppo:

Train with Single-Action PPO
----------------------------

Copy ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi_pi05.yaml`` to ``arena_droid_ppo.yaml`` in the same directory.

In the copy’s ``defaults``, replace both occurrences of ``env/polaris_droid_tapeintocontainer`` with ``env/isaaclab_arena``. Keep the ``@env.train`` and ``@env.eval`` suffixes and merge the following fields into the matching sections.

.. dropdown:: Single-action PPO settings
   :open:

   .. code-block:: yaml

      env:
        train:
          auto_reset: true
          ignore_terminations: false
          max_episode_steps: 600
          max_steps_per_rollout_epoch: 600
        eval:
          auto_reset: false
          ignore_terminations: false
          max_episode_steps: 600
          max_steps_per_rollout_epoch: 600
      actor:
        model:
          model_path: ${oc.env:MODEL_DIR}
          action_dim: 8
          num_action_chunks: 1
          openpi:
            task: rl
            pytorch_numerics: true
            config_name: pi05_droid_robolab

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh arena_droid_ppo

Training executes one action per observation and returns the pre-reset observation for timeout bootstrapping. Standalone evaluation supports 15-action chunks. Arena samples each layout independently. Keep ``env.train.group_size=1`` and leave ``use_fixed_reset_state_ids`` and ``use_ordered_reset_state_ids`` disabled in both environments. Grouped GRPO requires shared placements.

Set ``env.train.seed`` before launching training to reproduce the sequence of initial layouts. Each training ``reset()`` draws the next layout. Only evaluation accepts ``reset(seed=...)``. Keep ``data_collection.enabled=false`` for both Arena training and evaluation.
