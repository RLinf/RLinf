RL Token: Bootstrapping Online RL with Vision-Language-Action Models
====================================================================

**RL Token: Bootstrapping Online RL with Vision-Language-Action Models** trains
a compact reinforcement-learning policy on top of a frozen VLA feature model.
In RLinf configs and code, this workflow is abbreviated as **RLT**. It has two
stages:

1. Train a VLA checkpoint together with an RLT token transformer on
   demonstration data.
2. Freeze that feature model and train a lightweight off-policy actor-critic
   policy using the extracted RLT state.

The checked-in examples target Franka peg insertion and the ManiSkill
``PegInsertionSideWideClearance-v1`` joint-control simulation. The pipeline is
not tied to either task. Reuse the same two-stage structure when the
demonstrations, environment config, action shape, state semantics, and OpenPI
dataconfig stay aligned.

Official project page: `Precise Manipulation with Efficient Online RL <https://www.pi.website/research/rlt>`_.

Overview
--------

RLT separates representation learning from online RL control.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Stage 1
      :text-align: center

      VLA SFT + RLT token transformer

   .. grid-item-card:: Stage 2
      :text-align: center

      Compact off-policy actor-critic

   .. grid-item-card:: State
      :text-align: center

      ``z_rl`` + proprio + reference chunk

   .. grid-item-card:: Deployment
      :text-align: center

      Franka real robot / ManiSkill simulation

| **You'll do:** prepare demonstrations -> train Stage 1 -> point Stage 2 at
  the Stage 1 checkpoint -> launch actor-critic training -> monitor replay-buffer and task
  success metrics.
| **Prerequisites:** `OpenPI π₀.₅ base weights <https://huggingface.co/lerobot/pi05_base>`__, plus the environment for your example—either :doc:`Franka real-world <../embodied/franka>` or :doc:`ManiSkill simulation <../embodied/maniskill>`.

Provided Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Stage
     - Config
     - Purpose
   * - Franka Stage 1
     - ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``
     - SFT π₀.₅ together with the RLT token transformer on Franka demonstrations.
   * - Franka Stage 2
     - ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml``
     - Run real-world RLT actor-critic training with the frozen Stage 1 feature model.
   * - Dual Franka Stage 2
     - ``examples/embodiment/config/realworld_dual_franka_rlt_stage2_ac_mlp.yaml``
     - Train the Stage 2 MLP on one host that drives both arms, from a ``pi05_dualfranka_tcp_rot6d`` Stage 1 checkpoint.
   * - ManiSkill Stage 1
     - ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``
     - Jointly train the ManiSkill OpenPI base policy and RLT token transformer.
   * - ManiSkill Stage 2
     - ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``
     - Run simulated RLT actor-critic training with automatic ``rlt_policy_switch`` and transition replay.
   * - ManiSkill Stage 2 (TD3)
     - ``examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml``
     - Run the simulated TD3-MLP variant with the same frozen Stage 1 feature model and replay route.

Installation
------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI + LIBERO / ManiSkill

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

RLT uses ``model_type: openpi`` in the configs below. The install target is still ``openpi`` because the vendored PyTorch model shares the OpenPI runtime and the RLT Stage 1 dataloader keeps using the OpenPI data pipeline for ManiSkill and real-world compatibility.

How RLT Works
-------------

Stage 1: Learn the RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 starts from a VLA checkpoint and optimizes two objectives from the
same demonstration batch:

1. The normal VLA action objective, reported as ``vla_loss``.
2. The RLT token objective, reported as ``rlt_loss``.

The total Stage 1 loss is:

.. code:: text

   total_loss = rlt_loss + rlt_alpha * vla_loss

The OpenPI model exposes the VLA prefix hidden states. The RLT token
transformer reads those prefix states and produces a compact vector ``z_rl``.
Stage 2 uses ``z_rl`` as the learned RL representation rather than training the
actor-critic directly on image observations.

Keep ``repo_id``, ``config_name``, and normalization stats consistent with the Stage 2 feature-model config. ``openpi_data`` belongs under ``actor.model``.

.. dropdown:: Advanced configuration (optional)

   Key Stage 1 defaults in ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``:

   .. code:: yaml

      data:
        train_data_paths: "/path/to/data"

      actor:
        model:
          openpi_data:
            repo_id: "realworld_peg_insertion_rlt_stage1"
          model_type: "openpi"
          is_lora: False
          model_path: "/path/to/pi05_base_openpi"
          num_action_chunks: 20
          action_dim: 7
          num_steps: 4
          add_value_head: False
          openpi:
            task: sft
            config_name: "pi05_franka_state"
            num_images_in_input: 1
            # Network prediction horizon (Pi0Config.action_horizon), distinct from num_action_chunks.
            action_horizon: 20
            action_chunk: ${actor.model.num_action_chunks}
            action_env_dim: ${actor.model.action_dim}
            num_steps: ${actor.model.num_steps}
            model_action_dim: 32
            paligemma_variant: "gemma_2b"
            action_expert_variant: "gemma_300m"
            max_token_len: 200
            discrete_state_input: True
            use_rlt: True
            rlt_alpha: 1.0
            rlt_prefix_seq_len: 1024
            rlt_image_only: False
            rlt_use_mask: True

.. note::

   The ManiSkill joint example does not rely on manual ``state_indices``
   slicing. The ``pi05_rlt_maniskill_joint`` dataconfig maps the LeRobot
   ``state`` field into OpenPI ``observation.state``. Stage 2 rollout also uses
   the processed OpenPI ``observation.state`` as ``proprio``. Stage 1, Stage 2,
   and OpenPI normalization therefore see the same state semantics.

Stage 2: Train the Actor-Critic Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 2 freezes the Stage 1 feature model and trains only the compact RLT MLP
actor and critic.

.. note::

   The current Stage 2 implementation is not standard maximum-entropy SAC.

During rollout:

1. The environment returns raw observations and task metadata.
2. ``rollout.rlt_feature_model`` runs the frozen Stage 1 model and converts the
   raw observation into:

   - ``z_rl``: the compact RLT representation.
   - ``proprio``: the selected robot or simulator state.
   - ``ref_chunk``: the VLA reference action chunk.

3. The Stage 2 actor consumes ``ref_chunk``, ``z_rl``, and ``proprio``.
4. The replay buffer stores RLT transitions:

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = action actually sent to the environment
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

For the provided real-robot config, ``keyboard_reward_wrapper:
rlt_policy_switch`` adds an ``rlt_switch_flags`` flag. Before the operator presses
``b``, the executed action is the VLA ``ref_chunk``; after ``b`` is pressed,
the executed action switches to the Stage 2 actor.

For the ManiSkill joint config, ``env.*.rlt_policy_switch`` automatically
produces ``rlt_switch_flags`` (actor/ref phase) and ``intervene_flag`` (expert
takeover request) from task information. The HF rollout route uses those flags
to choose the actor action, VLA ``ref_chunk``, or expert action at whole-chunk
granularity, then writes ``record_transition``, ``actor_switch``, and
``intervention_requested`` into ``forward_inputs`` for replay and monitoring.

The critic is trained with a TD target over chunked rewards. Rewards inside
the action chunk are discounted and then bootstrapped with the next-state Q
value:

.. code:: text

   target_q = discounted_chunk_reward + gamma ** chunk_horizon * Q_target(next_obs, next_action)

If the episode terminates and ``bootstrap_type`` is ``standard``, the bootstrap
term is removed.

.. dropdown:: Advanced configuration (optional)

   Key Stage 2 defaults in ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml``:

   .. code:: yaml

      algorithm:
        loss_type: rlt_ac
        q_weight: 0.1
        bc_weight: 5
        reference_dropout_prob: 0.5
        gamma: 0.96
        entropy_tuning:
          alpha_type: fixed_alpha
          initial_alpha: 0.0

      rollout:
        collect_transitions: True
        rlt_feature_model:
          model_type: "openpi"
          precision: bf16
          is_lora: False
          num_action_chunks: 20
          action_dim: 7
          num_steps: 4
          add_value_head: False
          model_path: "/path/to/stage1/checkpoint/actor"
          openpi_data:
            repo_id: "realworld_peg_insertion_rlt_stage1"
            norm_stats_path: /path/to/lerobot_dataset/norm_stats.json
          openpi:
            task: eval
            config_name: "pi05_franka_state"
            num_images_in_input: 1
            action_horizon: 20
            action_chunk: ${actor.model.ref_num_action_chunks}
            action_env_dim: ${rollout.rlt_feature_model.action_dim}
            num_steps: ${rollout.rlt_feature_model.num_steps}
            model_action_dim: 32
            paligemma_variant: "gemma_2b"
            action_expert_variant: "gemma_300m"
            max_token_len: 200
            discrete_state_input: True
            state_indices: []      # keep the full raw state, e.g. 19D
            use_rlt: True
            rlt_prefix_seq_len: 1024
            rlt_image_only: False
            rlt_use_mask: True

      actor:
        model:
          model_type: "rlt_mlp_policy"
          precision: fp32
          add_value_head: False
          add_q_head: True
          q_head_type: "default"
          fixed_std: 0.002
          is_lora: False
          z_dim: 2048
          proprio_dim: 19
          action_dim: 7
          num_action_chunks: 10
          ref_num_action_chunks: 20

      env:
        train:
          keyboard_reward_wrapper: rlt_policy_switch

The Stage 2 actor loss is:

.. code:: text

   actor_loss = -q_weight * Q(obs, pi(obs)) + bc_weight * BC(pi(obs), target_action)

The BC target is the VLA reference action for normal policy steps. If a human
intervention action is stored for a step, the BC target for that step becomes
the human action. ManiSkill disables ``expert_takeover`` by default, so the
default simulated route uses only the VLA reference and actor actions.

Run the Provided Franka Example
-------------------------------

Data: Collect Franka Demonstrations and Compute Normalization Stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 expects a Franka demonstration dataset in LeRobot format; the dataset
directory should directly contain ``data/`` and ``meta/``. On the controller
node, follow the data-collection flow in the :doc:`Franka real-world guide
<../embodied/franka>` to prepare the robot and target pose, and export LeRobot
data from the collection config:

.. code:: yaml

   env:
     data_collection:
       enabled: True
       export_format: lerobot_dataset

Then launch collection:

.. code:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

The LeRobot shard is ``logs/<timestamp>/lerobot_dataset/rank_0/id_0/``,
which directly contains ``data/`` and ``meta/``. After collection, place that
shard on the training node and compute
normalization statistics for the RLT OpenPI dataconfig. ``repo_id`` should
match ``actor.openpi_data.repo_id`` and
``rollout.rlt_feature_model.openpi_data.repo_id`` in the Stage 1 / Stage 2
configs:

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_franka_state \
       --repo-id realworld_peg_insertion_rlt_stage1

Then point ``data.train_data_paths`` in the Stage 1 config at that LeRobot dataset directory. If the checkpoint does not embed norm stats, set ``norm_stats_path`` in ``openpi_data`` to point at ``norm_stats.json``.

Stage 1: Train the RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the dataset and base model paths in ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``:

.. code:: yaml

   data:
     train_data_paths: /path/to/lerobot_dataset

   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"

Launch SFT:

.. code:: bash

   bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05

The saved checkpoint directory should look like:

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Use this ``actor`` directory as ``rollout.rlt_feature_model.model_path`` in Stage 2. Do not put the Stage 1 checkpoint under ``actor.model.model_path``; that field does not load the Stage 1 feature model.

Stage 2: Run RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the Stage 1 checkpoint path, robot IP, and target pose in ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml``:

.. code:: yaml

   rollout:
     rlt_feature_model:
       model_path: /path/to/stage1/checkpoint/actor
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"

   env:
     train:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]

Update ``cluster.node_groups`` with your GPU and Franka node ranks and ``robot_ip``, then launch the async run from the master node:

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_rlt_stage2_ac_mlp

The default keyboard module implements the key phase switch used by RLT: press ``b`` to enter the Stage 2 actor-controlled phase. Other behavior can be customized in the real-robot policy-switch wrapper.

Dual Franka Stage 2
~~~~~~~~~~~~~~~~~~~

Use ``examples/embodiment/config/realworld_dual_franka_rlt_stage2_ac_mlp.yaml``
when both arms and the GPU are on one host and the frozen Stage 1 checkpoint
is ``pi05_dualfranka_tcp_rot6d``. The single-arm file above is the peg-insertion
recipe: 7-dimensional actions and one camera. This file is 20-dimensional
tcp_rot6d. The arms execute 10 steps; the Stage 1 reference horizon is 20.

Replace the arm IPs, camera serials, gripper paths, ``TASK_DESCRIPTION``, and
target poses. Point ``rollout.rlt_feature_model.model_path`` at the dual-arm
Stage 1 checkpoint and set ``openpi_data.repo_id`` to that checkpoint's dataset
id. Leave ``keyboard_device`` null and export ``RLINF_KEYBOARD_DEVICE``, or set
the evdev path under both ``env.train`` and ``env.eval``. Keep
``hold_current_when_inactive: False``. After the Pico grip is released, that
arm holds the last teleop command until the current action chunk ends, then
returns to the policy. USB-keyboard ``b`` switches the rollout from the Stage 1
policy to the Stage 2 actor. Pico A/B/X/Y stay gripper buttons.

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dual_franka_rlt_stage2_ac_mlp

Run the ManiSkill Joint Example
-------------------------------

The ManiSkill joint example uses ``PegInsertionSideWideClearance-v1``,
two-view RGB observations, the first 9 Panda qpos dimensions, and 8D
``pd_joint_delta_pos`` actions. Each Stage 2 action is a 10-step chunk. The
language instruction is fixed to ``insert the peg in the hole``.

Data: Prepare the Joint-Control LeRobot Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Stage 1 and Stage 2 OpenPI feature model uses the
``pi05_rlt_maniskill_joint`` dataconfig. The LeRobot dataset must contain at
least:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``image``
     - Main-view RGB image.
   * - ``wrist_image``
     - Wrist RGB image.
   * - ``state``
     - Panda joint proprioception. The current example uses a 9D state.
   * - ``actions``
     - 8D ``pd_joint_delta_pos`` actions.
   * - ``task``
     - Language instruction. You can also set ``default_prompt`` to ``insert the peg in the hole``.

You can download the reference dataset `RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ <https://huggingface.co/datasets/RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ>`__ or regenerate it with ManiSkill's Panda motion-planning solver, then compute normalization statistics with matching ``--config-name`` and ``--repo-id``:

.. tab-set::

   .. tab-item:: Download Reference Dataset

      .. code-block:: bash

         export HF_LEROBOT_HOME=./lerobot_root
         hf download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
             --repo-type dataset \
             --local-dir ./lerobot_root/maniskill_peginsertionside_joint
         python toolkits/lerobot/calculate_norm_stats.py \
             --config-name pi05_rlt_maniskill_joint \
             --repo-id maniskill_peginsertionside_joint

   .. tab-item:: Regenerate with Solver

      .. code-block:: bash

         export HF_LEROBOT_HOME=./lerobot_root
         python toolkits/lerobot/collect_maniskill_peg_lerobot_joint.py \
             --repo-id maniskill_peginsertionside_joint \
             --num-episodes 400 \
             --seed 0 \
             --max-attempts 4000 \
             --overwrite
         python toolkits/lerobot/calculate_norm_stats.py \
             --config-name pi05_rlt_maniskill_joint \
             --repo-id maniskill_peginsertionside_joint

.. note::

   The collection script under ``toolkits/lerobot`` depends on ManiSkill's PegInsertionSide Panda motion-planning solver. If importing the solver fails, check that your ManiSkill installation includes the motion-planning examples.

.. warning::

   The Stage 1 checkpoint, Stage 2 ``rollout.rlt_feature_model``, and OpenPI assets must use the same ``norm_stats.json``. If the SFT base policy and Stage 2 load stats from different ``repo_id`` directories, the scale of VLA reference actions shifts. Set ``openpi_data.norm_stats_path`` to that ``norm_stats.json`` when you need an explicit path.

Stage 1: Jointly Train the ManiSkill OpenPI + RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the dataset, base model, and ``norm_stats.json`` paths in ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``:

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: ./lerobot_root/maniskill_peginsertionside_joint
         weight: 1.0

   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: ./lerobot_root/maniskill_peginsertionside_joint/norm_stats.json

Launch training:

.. code:: bash

   bash examples/sft/run_vla_sft.sh maniskill_rlt_stage1_sft_openpi_pi05

The saved FSDP checkpoint usually looks like:

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Point this ``actor`` directory directly at ``rollout.rlt_feature_model.model_path`` in Stage 2.

Stage 2: Run ManiSkill RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the Stage 1 checkpoint and ``norm_stats.json`` paths in ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``:

.. code:: yaml

   rollout:
     rlt_feature_model:
       model_path: /path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_step/actor
       openpi_data:
         norm_stats_path: ./lerobot_root/maniskill_peginsertionside_joint/norm_stats.json

Launch training:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_ac_mlp

To run the TD3-MLP variant, apply the same Stage 1 checkpoint settings to ``maniskill_rlt_stage2_td3_mlp.yaml`` and launch:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_td3_mlp

This variant is configured for ManiSkill simulation. It keeps the frozen Stage 1 features and transition replay path, while replacing the Stage 2 policy and update objective with a direct TD3 actor and twin-Q critic.

This config starts actor, rollout, and ManiSkill env workers. The rollout side freezes ``rollout.rlt_feature_model`` and only synchronizes the Stage 2 MLP actor. Before ``ready_for_online``, the ManiSkill route executes the VLA ``ref_chunk``. After ``algorithm.rlt_schedule.warmup_post_collect_updates``, the actor can take over in the automatic critical phase.

Tune ManiSkill Critical Phase and Intervention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ManiSkill switch uses peg-insertion task info to enter the critical phase (switching from the VLA reference policy to the Stage 2 actor) and, if enabled, to request expert takeover after repeated no-progress actor chunks. The shipped config already includes working defaults.

.. dropdown:: Advanced configuration (optional)

   Critical-phase schedule, auto gate, and optional expert takeover settings in ``maniskill_rlt_stage2_ac_mlp.yaml`` (comments are documentation-only):

   .. code:: yaml

      algorithm:
        rlt_schedule:
          enable: True                    # Use the ManiSkill rollout/update budget.
          warmup_post_collect_updates: 30000  # Updates before actor rollout starts.
          train_every_transitions: 5       # Add update budget per N recorded transitions.
        actor_weight_schedule:
          enable: True                    # Ramp BC/Q weights instead of fixed weights.
          warmup_updates: 20000           # Keep warmup BC/Q weights for this many updates.
          ramp_updates: 50000             # Smoothly move to online BC/Q weights.

      env:
        train:
          wrap_obs_mode: rlt_openpi_joint
          rlt_policy_switch:
            enable: True
            task_mode: full_task          # VLA before critical phase, actor inside it.
            trigger_mode: auto            # Compute critical phase from task info.
            latch_until_done: True        # Keep actor control after entering critical phase.
            auto_gate:
              require_grasp: True         # Enter only after the peg is grasped.
              require_not_success: True   # Do not switch after task success.
              near_hole_x_min: -0.16      # Larger value narrows the x entry gate.
              near_hole_yz_margin: 1.5    # Larger value widens the y/z entry gate.
            expert_takeover:
              enable: True                # Train-only simulated expert intervention.
              trigger_mode: stalled_progress  # Wait for repeated no-progress chunks.
              gate:
                near_hole_x_min: -0.10    # Start stall checks only near the hole.
                near_hole_yz_margin: 2.0  # Start stall checks only when y/z is close.
                stuck_chunks_before_takeover: 3  # No-progress chunks before takeover.
                min_x_progress: 0.003     # Forward progress that resets stall count.
                min_yz_progress: 0.0015   # Alignment progress that resets stall count.
                min_score_progress: 0.002 # Combined-score progress that resets stalls.
                progress_yz_weight: 1.0   # y/z penalty in the combined progress score.

        eval:
          wrap_obs_mode: rlt_openpi_joint
          rlt_policy_switch:
            enable: True
            task_mode: full_task
            trigger_mode: auto
            expert_takeover:
              enable: False               # Eval uses base policy + actor, never expert.

Optional: Enable ManiSkill Expert Takeover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ManiSkill expert takeover path is disabled by default. Enable it when you want the rollout route to replace actor actions with a stronger SFT expert in the critical phase and store those expert actions as intervention targets. Keep the expert's OpenPI dataconfig and norm stats aligned with the Stage 2 dataset. The expert is only used for train rollout; eval rollout runs without expert takeover and measures the learned actor.

.. dropdown:: Advanced configuration (optional)

   .. code:: yaml

      rollout:
        expert_model:
          model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_step/actor
          precision: null
          openpi:
            use_rlt: False

      env:
        train:
          rlt_policy_switch:
            expert_takeover:
              enable: True
              trigger_mode: stalled_progress
              gate:
                stuck_chunks_before_takeover: 3
                min_x_progress: 0.003
                min_yz_progress: 0.0015
                min_score_progress: 0.002
                progress_yz_weight: 1.0

Replay Buffer Behavior
----------------------

For ``loss_type: rlt_ac``, the replay buffer does not store raw image
observations as the RL state. The rollout worker returns RLT features, and the
learner side stores those features as transitions:

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = action chunk actually sent to the environment
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

This means:

- Steps before a real-robot policy switch are still useful. Their executed
  action is the VLA reference action, and their transition is stored in the
  same replay buffer.
- Steps after the switch use the actor action and are also stored with the
  same RLT observation format.
- The ManiSkill route chooses actor actions or the VLA ``ref_chunk`` at whole
  chunk granularity. The replay ``action`` is always the executed action.
- The ManiSkill learner splits rollout chunks into 1-sample transition
  trajectories before adding them to RLinf ``TrajectoryReplayBuffer``.
- ``sample_window_size`` controls the recent transition window sampled from the
  replay buffer. It does not need to match ``max_steps_per_rollout_epoch``.
- ``max_steps_per_rollout_epoch`` controls how many environment steps are
  collected before the rollout worker flushes a batch to training.

Monitoring
----------

For metric definitions, see :doc:`Training metrics <../../reference/metrics>`.
Useful RLT signals:

- Stage 1 SFT:

  - ``vla_loss``: the OpenPI action-prediction loss.
  - ``rlt_loss``: the RLT token reconstruction/compression loss.

- Stage 2 actor-critic:

  - ``critic/critic_loss``: Q-function TD loss.
  - ``actor/actor_loss``: combined ``-Q + BC`` actor objective.
  - ``q_pi`` and ``q_value_*``: learned Q-values for actor and critic heads.
  - ``actor/bc_loss``, ``actor/bc_ref_loss``, ``actor/bc_human_loss``: BC regularization terms.
  - ``actor/human_mask_ratio``: fraction of the current training batch marked as expert intervention (from ``trajectory.intervene_flags``).
  - ``train/replay_buffer/size``: number of stored replay transitions.
  - ``env/success_once`` and ``env/episode_len``: task outcome metrics.
  - ``eval/success_once``: success rate on fixed eval reset ids.

  ManiSkill rollout / replay diagnostics (logged from trajectories received by the actor):

  - ``train/replay/record_transition_rate``: fraction of collected steps saved as RLT transitions (from ``forward_inputs.record_transition``).
  - ``train/replay/actor_switch_rate``: fraction of collected steps where the actor/student action actually controlled the env (from ``forward_inputs.actor_switch``).
  - ``train/replay/intervention_requested_rate``: fraction of steps where the env requested expert takeover (from ``forward_inputs.intervention_requested``).
  - ``train/replay/intervention_rate``: fraction of steps where the route actually applied expert actions (from ``trajectory.intervene_flags``).
  - ``train/replay/transition_count``, ``train/replay/reward_mean``, ``train/replay/reward_positive_rate``, ``train/replay/done_rate``: ManiSkill transition-replay ingest stats for the current collect step.

  RLT schedule / learner backlog (ManiSkill ``algorithm.rlt_schedule.enable``):

  - ``train/rlt/ready_for_online``: whether learner ``update_step`` has passed ``warmup_post_collect_updates``.
  - ``train/rlt/actor_updates_run``, ``train/rlt/critic_updates_run``, and ``train/rlt/pending_update_budget``: actor/critic updates executed in the step and remaining learner backlog.
  - ``train/rlt/updates_to_run``, ``train/rlt/should_train``, and ``train/rlt/skip_reason``: scheduled training for the step and why training may be skipped.
  - ``train/rlt/global_transitions_since_train`` and ``train/rlt/global_total_transitions_added``: global replay ingest counters since the last training burst.

Experimental Results
--------------------

The RL Token training result on the peg_insertion task in RLinf is shown below.

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
     <div style="flex: 0.5; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT_real.png" style="width: 100%;"/>
       <p><em>RL Token training result on the RLinf peg_insertion task</em></p>
     </div>
   </div>

Practical Notes
---------------

- Keep Stage 1 and Stage 2 data settings consistent: ``repo_id``,
  ``config_name``, ``action_dim``, ``proprio_dim``, ``ref_num_action_chunks``,
  and ``z_dim`` must agree. To keep the full raw state, use
  ``state_indices: []``.
- ``rollout.rlt_feature_model`` should point to the Stage 1 checkpoint, while
  ``actor.model`` is the Stage 2 MLP policy updated by the actor-critic
  worker.
- ``rollout.model`` is the synced Stage 2 MLP copy on rollout workers. Keep ``actor.model.model_path: null`` for scratch Stage 2 training; use ``runner.resume_dir`` to resume a Stage 2 run or ``runner.ckpt_path`` to load a single Stage 2 weight file.
- Do not configure ``actor.model.model_path`` for Stage 1. ``actor.model`` only
  describes the Stage 2 MLP input/output shape and Q-head settings.
- Stage 2 MLP settings are defined inline under ``actor.model`` in each Stage 2
  YAML, not in a separate model defaults file.
- ``keyboard_reward_wrapper: rlt_policy_switch`` is only needed for
  operator-controlled critical-phase switching.
- The ManiSkill joint example uses ``env.*.rlt_policy_switch``. Do not use the
  real-robot keyboard wrapper there.
- ManiSkill ``proprio`` comes from the processed OpenPI ``observation.state``.
  When you add a new simulator dataconfig, check the dataset ``state``, OpenPI
  transform, and Stage 2 ``proprio_dim`` together.
- Stage 1, Stage 2, and checkpoint assets must load ``norm_stats.json`` from the same data semantics and ``repo_id``. Prefer setting ``openpi_data.norm_stats_path`` explicitly so Stage 1 runs still work when the checkpoint does not embed norm stats.
- ``rollout.rlt_feature_model.model_path`` should point to the Stage 1 FSDP ``actor`` directory, for example ``.../checkpoints/global_step_<step>/actor``. Do not convert the RLT Stage 1 checkpoint to a bare ``model.safetensors`` for Stage 2, because the RLT token module is stored in the full wrapper checkpoint.
- To add a simulator example, create a simulator environment config, keep
  ``loss_type: rlt_ac`` and ``rollout.rlt_feature_model``, and replace the
  real-robot phase-switching logic with simulator-appropriate behavior.
