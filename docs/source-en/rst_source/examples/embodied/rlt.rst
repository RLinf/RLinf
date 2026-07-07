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
| **Prerequisites:** download or prepare the
  `OpenPI π₀.₅ checkpoint <https://huggingface.co/lerobot/pi05_base>`__, and
  prepare the :doc:`Franka real-world setup <../embodied/franka>` or the
  :doc:`ManiSkill simulator setup <../embodied/maniskill>`.

Provided Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Stage
     - Config
     - Purpose
   * - Franka Stage 1
     - ``examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml``
     - SFT π₀.₅ together with the RLT token transformer on Franka demonstrations.
   * - Franka Stage 2
     - ``examples/embodiment/config/rlt_stage2_ac_mlp.yaml``
     - Run real-world RLT actor-critic training with the frozen Stage 1 feature model.
   * - ManiSkill Stage 1
     - ``examples/sft/config/rlt_stage1_maniskill_joint_alpha1.yaml``
     - Jointly train the ManiSkill OpenPI base policy and RLT token transformer.
   * - ManiSkill Stage 2
     - ``examples/embodiment/config/rlt_stage2_maniskill_joint_ac.yaml``
     - Run simulated RLT actor-critic training with automatic ``rlt_policy_switch`` and transition replay.

Installation
------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

Please switch to the OpenPI virtual environment via the built-in ``switch_env`` utility:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

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

Important Stage 1 fields:

.. code:: yaml

   # examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml
   data:
     train_data_paths: "/path/to/data"

   actor:
     openpi_data:
       repo_id: "realworld_peg_insertion_rlt_stage1"
     model:
       model_type: "openpi"
       is_lora: False
       model_path: "/path/to/model"
       num_action_chunks: 20
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         use_rlt: True
         rlt_alpha: 1.0
         rlt_prefix_seq_len: 1024
         rlt_image_only: False
         rlt_use_mask: True

Keep ``repo_id`` and ``config_name`` consistent with the normalization stats and
the Stage 2 feature-model config.

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
granularity, then writes ``record_transition`` and ``intervention_requested`` into
``forward_inputs`` for replay and monitoring.

The critic is trained with a TD target over chunked rewards. Rewards inside
the action chunk are discounted and then bootstrapped with the next-state Q
value:

.. code:: text

   target_q = discounted_chunk_reward + gamma ** chunk_horizon * Q_target(next_obs, next_action)

If the episode terminates and ``bootstrap_type`` is ``standard``, the bootstrap
term is removed.

Important Stage 2 fields:

.. code:: yaml

   # examples/embodiment/config/rlt_stage2_ac_mlp.yaml
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
     model:
       model_path: null
       precision: ${actor.model.precision}
       action_dim: ${actor.model.action_dim}
       num_action_chunks: ${actor.model.num_action_chunks}
       ref_num_action_chunks: ${actor.model.ref_num_action_chunks}
     rlt_feature_model:
       model_type: "openpi"
       model_path: "/path/to/stage1/checkpoint"
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         action_chunk: ${actor.model.ref_num_action_chunks}
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
       export_format: "lerobot"

Then launch collection:

.. code:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

After collection, place the LeRobot dataset on the training node and compute
normalization statistics for the RLT OpenPI dataconfig. ``repo_id`` should
match ``actor.openpi_data.repo_id`` and
``rollout.rlt_feature_model.openpi_data.repo_id`` in the Stage 1 / Stage 2
configs:

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_franka_state \
       --repo-id realworld_peg_insertion_rlt_stage1

Then point ``data.train_data_paths`` in the Stage 1 config at that LeRobot
dataset directory.

Stage 1: Train the RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the Stage 1 config paths before launch:

.. code:: yaml

   data:
     train_data_paths: /path/to/lerobot_dataset

   actor:
     openpi_data:
       repo_id: "realworld_peg_insertion_rlt_stage1"
     model:
       model_path: /path/to/model
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         rlt_prefix_seq_len: 1024

Launch SFT:

.. code:: bash

   bash examples/sft/run_vla_sft.sh rlt_stage1_sft_openpi_pi05

The saved checkpoint directory should look like:

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>

Use this directory as ``rollout.rlt_feature_model.model_path`` in Stage 2.
Do not put the Stage 1 checkpoint under ``rollout.model.model_path`` or
``actor.model.model_path``; those fields do not load the Stage 1 feature model.

Stage 2: Run RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the Stage 2 config:

.. code:: yaml

   rollout:
     model:
       model_path: null
     rlt_feature_model:
       model_path: /path/to/stage1/checkpoint
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         state_indices: []
         rlt_prefix_seq_len: 1024

   cluster:
     node_groups:
       - label: <gpu_node_group>
         node_ranks: <gpu_node_rank>
       - label: <env_node_group>
         node_ranks: <env_node_rank>
         hardware:
           type: <robot_type>
           configs:
             - robot_ip: <robot_ip>

   env:
     train:
       keyboard_reward_wrapper: rlt_policy_switch
       override_cfg:
         target_ee_pose: [<target_x>, <target_y>, <target_z>, <target_roll>, <target_pitch>, <target_yaw>]

Launch the async run from the master node:

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh rlt_stage2_ac_mlp

The default keyboard module implements the key phase switch used by RLT: press
``b`` to enter the Stage 2 actor-controlled phase. Other behavior can be
customized for the task in
``rlinf/envs/realworld/common/wrappers/keyboard_rlt_policy_switch_wrapper.py``.

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

You can start from the reference dataset
`RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ
<https://huggingface.co/datasets/RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ>`__.
It contains successful ManiSkill ``PegInsertionSideWideClearance-v1``
demonstrations, uses the ``pd_joint_delta_pos`` action space, and follows the
joint-control LeRobot fields above.

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   huggingface-cli download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
       --repo-type dataset \
       --local-dir ${HF_LEROBOT_HOME}/maniskill_peginsertionside_joint

You can also regenerate the dataset with the collection script in this branch.
The script first runs ManiSkill's Panda motion-planning solver, converts
``pd_joint_pos`` solver actions into ``pd_joint_delta_pos`` actions, and saves
only episodes that replay successfully:

.. code:: bash

   python toolkits/lerobot/collect_maniskill_peg_lerobot_joint.py \
       --repo-id maniskill_peginsertionside_joint \
       --num-episodes 400 \
       --seed 0 \
       --max-attempts 4000 \
       --overwrite

.. note::

   The collection script lives under ``toolkits/lerobot`` and is intended for
   RLT joint data preparation. It depends on ManiSkill's
   PegInsertionSide Panda motion-planning solver. If importing the solver fails,
   first check that your ManiSkill installation includes the motion-planning
   examples.

When you compute normalization statistics, keep ``--config-name`` and
``--repo-id`` aligned with the training config:

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_rlt_maniskill_joint \
       --repo-id maniskill_peginsertionside_joint

.. warning::

   The Stage 1 checkpoint, Stage 2 ``rollout.rlt_feature_model``, and OpenPI
   assets must use the same ``norm_stats.json``. If the SFT base policy and
   Stage 2 load stats from different ``repo_id`` directories, the scale of VLA
   reference actions shifts.

Stage 1: Jointly Train the ManiSkill OpenPI + RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the data path in
``examples/sft/config/rlt_stage1_maniskill_joint_alpha1.yaml``:

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/maniskill_peginsertionside_joint
         weight: 1.0

   actor:
     model:
       model_path: /path/to/pi05_base
       openpi:
         config_name: pi05_rlt_maniskill_joint
     openpi_data:
       repo_id: maniskill_peginsertionside_joint
       default_prompt: insert the peg in the hole

Launch training:

.. code:: bash

   bash examples/sft/run_vla_sft.sh rlt_stage1_maniskill_joint_alpha1

Stage 2 uses this Stage 1 actor directory as
``rollout.rlt_feature_model.model_path``. The directory must contain VLA
weights, RLT token transformer weights, and matching OpenPI assets.

Stage 2: Run ManiSkill RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the Stage 1 checkpoint in
``examples/embodiment/config/rlt_stage2_maniskill_joint_ac.yaml``:

.. code:: yaml

   env:
     train:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto
     eval:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto

   rollout:
     rlt_feature_model:
       model_path: /path/to/rlt_stage1_maniskill_joint_alpha1/checkpoints/global_step_<step>/actor
       openpi_data:
         repo_id: maniskill_peginsertionside_joint
       openpi:
         config_name: pi05_rlt_maniskill_joint
     expert_model:
       model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_<step>/actor
       precision: null
       openpi:
         use_rlt: False

Launch training:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh rlt_stage2_maniskill_joint_ac

This config starts actor, rollout, and ManiSkill env workers. The rollout side
freezes ``rollout.rlt_feature_model`` and only synchronizes the Stage 2 MLP
actor. Before ``ready_for_online``, the ManiSkill route executes the VLA
``ref_chunk``. After ``algorithm.rlt_schedule.warmup_post_collect_updates``,
the actor can take over in the automatic critical phase.

Tune ManiSkill Critical Phase and Intervention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ManiSkill switch is simulator-specific. It uses peg-insertion task info to
turn on ``rlt_switch_flags`` / ``in_critical_phase`` and, if enabled, to request
expert takeover after repeated no-progress actor chunks. The comments below are
documentation-only annotations; keep the checked-in YAML clean.

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
       rlt_policy_switch:
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
       rlt_policy_switch:
         expert_takeover:
           enable: False               # Eval uses base policy + actor, never expert.

Optional: Enable ManiSkill Expert Takeover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ManiSkill expert takeover path is disabled by default. Enable it when you
want the rollout route to replace actor actions with a stronger SFT expert in
the critical phase and store those expert actions as intervention targets:

.. code:: yaml

   rollout:
     expert_model:
       model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_<step>/actor
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

You can replace ``rollout.expert_model.model_path`` with a more fully trained
joint-control SFT checkpoint. Keep the expert's OpenPI dataconfig and norm
stats aligned with the Stage 2 dataset. The expert is only used for train
rollout; eval rollout runs without expert takeover and measures the learned actor.

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

  - ``train/replay/actor_switch_rate``: fraction of collected steps where the route executed the actor instead of VLA ``ref_chunk`` (from ``forward_inputs.record_transition``; includes schedule warmup masking).
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
- ``rollout.model`` is the synced Stage 2 MLP copy on rollout workers. Keep
  ``rollout.model.model_path: null`` for scratch Stage 2 training; use
  ``runner.resume_dir`` to resume a Stage 2 run or ``runner.ckpt_path`` to load
  a single Stage 2 weight file.
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
- Stage 1, Stage 2, and checkpoint assets must load ``norm_stats.json`` from
  the same data semantics and ``repo_id``.
- To add a simulator example, create a simulator environment config, keep
  ``loss_type: rlt_ac`` and ``rollout.rlt_feature_model``, and replace the
  real-robot phase-switching logic with simulator-appropriate behavior.
