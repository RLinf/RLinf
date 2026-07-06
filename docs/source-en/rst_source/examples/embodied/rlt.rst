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

The checked-in example currently targets Franka peg insertion, while the
pipeline itself is not tied to that task. A simulator version can reuse the
same Stage 1 / Stage 2 split once the environment, action shape, state
selection, and data paths are swapped.

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

      Real robot now, simulator-ready layout

| **You'll do:** prepare demonstrations -> train Stage 1 -> point Stage 2 at
  the Stage 1 checkpoint -> launch actor-critic training -> monitor replay-buffer and task
  success metrics.
| **Prerequisites:** prepare the OpenPI π₀.₅ checkpoint and prepare the
  :doc:`Franka real-world setup <../embodied/franka>`.

Provided Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Stage
     - Config
     - Purpose
   * - Stage 1
     - ``examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml``
     - SFT pi0.5 together with the RLT token transformer.
   * - Stage 2
     - ``examples/embodiment/config/rlt_stage2_ac_mlp.yaml``
     - Run RLT Stage 2 actor-critic training with the frozen Stage 1 feature model.

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
the executed action switches to the Stage 2 actor. Simulator configs can omit
this wrapper or replace it with an automatic switching rule.

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
the human action.

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

Replay Buffer Behavior
----------------------

For ``loss_type: rlt_ac``, the replay buffer does not store raw image
observations as the RL state. The environment worker waits for the rollout
worker to return RLT features and stores those features as transitions.

This means:

- Steps before a real-robot policy switch are still useful. Their executed
  action is the VLA reference action, and their transition is stored in the
  same replay buffer.
- Steps after the switch use the actor action and are also stored with the
  same RLT observation format.
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

  - ``train/sac/critic_loss``: Q-function TD loss.
  - ``train/sac/actor_loss``: combined ``-Q + BC`` actor objective.
  - ``q_pi`` and ``q_value_*``: learned Q-values for actor and critic heads.
  - ``bc_loss``, ``bc_ref_loss``, ``bc_human_loss``: BC regularization terms.
  - ``train/replay_buffer/size``: number of stored replay transitions.
  - ``env/success_once`` and ``env/episode_len``: task outcome metrics.

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
- ``keyboard_reward_wrapper: rlt_policy_switch`` is only needed for
  operator-controlled critical-phase switching.
- To add a simulator example, create a simulator environment config, keep
  ``loss_type: rlt_ac`` and ``rollout.rlt_feature_model``, and replace the
  real-robot phase-switching logic with simulator-appropriate behavior.
