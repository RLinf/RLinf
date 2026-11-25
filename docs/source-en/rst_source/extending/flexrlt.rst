:orphan:

FlexRLT Pluggable Components
============================

Replace the frozen VLA feature extractor, actor/critic heads, or off-policy update rule in :doc:`FlexRLT <../examples/embodied/flexrlt>` without changing the rollout loop or replay pipeline.

At a Glance
-----------

Each FlexRLT layer has its own config field and registration hook; swap only the row you need:

.. list-table::
   :header-rows: 1
   :widths: 20 28 24 28

   * - Layer
     - Config Field
     - Registration / Contract
     - Reference Code
   * - Frozen feature model
     - ``rollout.prefix_feature_model``
     - ``register_model(..., category="embodied")`` + ``PrefixFeatureModel``
     - ``rlinf/models/embodiment/prefix_features.py``
   * - Trainable policy
     - ``actor.model.model_type``
     - ``register_model(..., obs_contract=ObsContract.PREFIX)``
     - ``rlinf/models/embodiment/mlp_policy/prefix_mlp_policy.py``
   * - Actor & critic heads
     - ``actor.model.actor_head.name``, ``critic_head.name``
     - ``@register_prefix_actor`` / ``@register_prefix_critic``
     - ``rlinf/models/embodiment/mlp_policy/heads.py``
   * - Update rule
     - ``algorithm.name`` (``loss_type: prefix_off_policy``)
     - ``@register_prefix_off_policy_algorithm``
     - ``rlinf/algorithms/prefix_off_policy/``

During rollout, the frozen VLA maps raw observations to ``{z_rl, proprio, ref_chunk}``; replay stores those features with the executed action chunk so online updates train only the compact actor and critic heads.

Implement Your Extension
------------------------

Pick the tab for the component you want to replace (save out-of-tree examples in ``flexrlt_ext.py`` at the repository root):

.. tab-set::

   .. tab-item:: 1. Feature Model (VLA)
      :sync: flexrlt-ext-vla

      Wrap any frozen VLA with ``z_dim`` and ``extract_prefix_obs(env_obs) -> PrefixObs``, then register its builder via ``register_model(..., category="embodied")`` (see :doc:`new_model_fsdp` for registration details):

      .. code-block:: python

         import torch
         from rlinf.models.embodiment.prefix_features import pool_prefix

         class MyBaseVLAAdapter(torch.nn.Module):
             def __init__(self, base_vla):
                 super().__init__()
                 self.base_vla = base_vla
                 self.z_dim = base_vla.prefix_width

             @torch.no_grad()
             def extract_prefix_obs(self, env_obs):
                 observation = self.base_vla.encode_observation(env_obs)
                 hidden, mask = self.base_vla.encode_prefix(observation)
                 z_rl = pool_prefix(hidden, mask, mode="masked_mean")
                 ref_chunk = self.base_vla.sample_action_chunk(observation)
                 proprio = torch.as_tensor(env_obs["states"], device=z_rl.device)
                 return {
                     "z_rl": z_rl.float(),
                     "proprio": proprio.float(),
                     "ref_chunk": ref_chunk.float(),
                 }

      - ``z_rl`` has shape ``[B, z_dim]``, ``proprio`` has ``[B, proprio_dim]``, and ``ref_chunk`` has ``[B, ref_num_action_chunks, action_dim]``.
      - At startup, rollout validates ``z_dim`` against ``actor.model.z_dim``.

   .. tab-item:: 2. Actor & Critic Heads
      :sync: flexrlt-ext-heads

      Subclass ``PrefixActorHead`` or ``PrefixCriticHead``, declare capabilities, and register each head under its YAML name. Built-in heads are ``fixed_std_mlp`` (AC), ``deterministic_mlp`` (TD3), ``multi_q_mlp``, and ``cross_q_mlp``.

      .. code-block:: python

         import torch
         from torch import nn
         from rlinf.models.embodiment.mlp_policy.heads import (
             ActorHeadCapabilities, CriticHeadCapabilities,
             PrefixActorHead, PrefixCriticHead,
             register_prefix_actor, register_prefix_critic,
         )

         class CustomMLPActor(PrefixActorHead):
             capabilities = ActorHeadCapabilities(
                 distribution="deterministic", supports_action_noise=True,
             )

             def __init__(self, state_dim, reference_dim, action_dim, cfg):
                 super().__init__()
                 hidden = int(cfg.get("hidden_dim", 256))
                 self.net = nn.Sequential(
                     nn.Linear(state_dim + reference_dim, hidden), nn.ReLU(),
                     nn.Linear(hidden, action_dim),
                 )

             def mean_action(self, state, reference):
                 return self.net(torch.cat([state, reference], dim=-1)).clamp(-1, 1)

             def forward(self, state, reference, *, deterministic,
                         apply_reference_dropout, reference_dropout_prob, apply_action_noise):
                 if apply_reference_dropout:
                     keep = torch.rand(reference.shape[0], 1, device=reference.device)
                     reference = reference * (keep >= float(reference_dropout_prob or 0.0)).to(reference.dtype)
                 action = self.mean_action(state, reference)
                 if apply_action_noise and not deterministic:
                     action = (action + torch.randn_like(action) * 0.1).clamp(-1, 1)
                 return action, torch.zeros_like(action)

         @register_prefix_actor("custom_mlp", capabilities=CustomMLPActor.capabilities)
         def make_custom_mlp_actor(state_dim, reference_dim, action_dim, cfg):
             return CustomMLPActor(state_dim, reference_dim, action_dim, cfg)

         class CustomTwinCritic(PrefixCriticHead):
             capabilities = CriticHeadCapabilities(fixed_num_q_heads=2)

             def __init__(self, state_dim, action_dim, cfg):
                 super().__init__()
                 self.num_q_heads = 2
                 width = int(cfg.get("hidden_dim", 256))
                 self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, width), nn.ReLU(), nn.Linear(width, 1))
                 self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, width), nn.ReLU(), nn.Linear(width, 1))

             def forward(self, state, actions, *, next_state=None, next_actions=None):
                 x = torch.cat([state, actions], dim=-1)
                 return torch.cat([self.q1(x), self.q2(x)], dim=-1)

         @register_prefix_critic("custom_twin", capabilities=CustomTwinCritic.capabilities)
         def make_custom_twin_critic(state_dim, action_dim, cfg):
             return CustomTwinCritic(state_dim, action_dim, cfg)

   .. tab-item:: 3. Update Rule
      :sync: flexrlt-ext-algo

      Subclass ``PrefixOffPolicyAlgorithm``, declare compatible head ``requirements``, and override ``actor_loss``, ``target_actions``, or ``critic_loss`` (the base ``critic_loss`` handles chunk-discounted Bellman MSE):

      .. code-block:: python

         from rlinf.algorithms.prefix_off_policy import (
             PrefixAlgorithmRequirements, PrefixOffPolicyAlgorithm,
             register_prefix_off_policy_algorithm,
         )
         from rlinf.models.embodiment.base_policy import ForwardType

         @register_prefix_off_policy_algorithm("my_algorithm")
         class MyAlgorithm(PrefixOffPolicyAlgorithm):
             requirements = PrefixAlgorithmRequirements(
                 actor_distribution="deterministic",
                 requires_action_noise=True,
             )

             def actor_loss(self, context, batch):
                 return self._actor_loss(
                     context, batch,
                     actor_agg_q="min",
                     apply_reference_dropout=False,
                     apply_action_noise=True,
                 )

             def target_actions(self, context, next_obs):
                 actions, _, _ = context.target_model(forward_type=ForwardType.SAC, obs=next_obs)
                 return actions

Configure and Launch
--------------------

Save a child config ``my_flexrlt_custom.yaml`` under ``examples/embodiment/config/`` inheriting from the ManiSkill preset in :doc:`FlexRLT <../examples/embodied/flexrlt>`:

.. code-block:: yaml

   defaults:
     - maniskill_prefix_stage2_td3_mlp_steam
     - _self_

   algorithm:
     name: my_algorithm
   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

Export ``RLINF_EXT_MODULE`` so every Ray worker loads your registrations, then launch:

.. code-block:: bash

   export RLINF_EXT_MODULE=flexrlt_ext
   bash examples/embodiment/run_embodiment.sh my_flexrlt_custom

Reference Details
-----------------

.. dropdown:: Custom policy contract (``ObsContract.PREFIX``)

   To replace ``prefix_policy`` itself (for example with a recurrent actor), register a ``BasePolicy`` subclass with ``obs_contract=ObsContract.PREFIX`` satisfying:

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - Requirement
        - Consumer
      * - Accept ``actor.model.actor_head`` and ``critic_head`` names
        - Config validation and algorithm capability checks.
      * - ``forward(ForwardType.SAC, obs=...) -> (actions, logprobs, None)``
        - Actor loss and target actions (flat ``[B, num_action_chunks * action_dim]``).
      * - ``ForwardType.SAC_Q`` (and ``CROSSQ_Q`` if supported)
        - Critic loss and Q term of actor loss.
      * - Store critic parameters under ``q_head``
        - FSDP worker routes ``q_head`` / ``encoder`` / ``state_proj`` to the critic optimizer and remaining parameters to the actor optimizer.
      * - ``predict_action_batch(..., return_obs=True)``
        - Rollout: return ``[B, num_action_chunks, action_dim]`` actions and place ``action``, ``z_rl``, ``proprio``, ``ref_chunk`` in ``forward_inputs``.

.. dropdown:: Prefix pooling modes & built-in AC / TD3 rules

   - **Prefix pooling** (set alongside ``openpi``): ``prefix.image_only: true`` (default) selects image tokens; ``prefix.pool`` accepts ``masked_mean`` (default without RLT), ``mean``, ``last``, or ``rlt_token`` (Stage 1 token transformer).
   - **Built-in ``ac``**: stochastic actor (``fixed_std_mlp``), online next-action bootstrap, ``algorithm.actor_agg_q: q1``.
   - **Built-in ``td3``**: deterministic actor with action noise (``deterministic_mlp``), target policy smoothing (``target_policy_noise`` default ``0.1``, ``target_noise_clip`` default ``0.5``), coupled actor + target updates every ``critic_actor_ratio`` (default ``2``) critic steps.
   - Both rules use minimum-of-first-two-Q critic targets (``algorithm.agg_q: min``) and ``algorithm.entropy_tuning.alpha_type: fixed_alpha``.

Run EXPO with Frozen Proposals
------------------------------

Use EXPO when the learned policy should select among frozen VLA proposals and small edits of those proposals. Start from ``examples/embodiment/config/maniskill_prefix_expo_openpi_pi05_async.yaml``. It uses the existing OpenPI Prefix features, a stochastic ``residual_mlp`` editor, and ten ``multi_q_mlp`` critics. The example uses ManiSkill joint actions and the full pick-and-insert task.

Prepare the OpenPI/ManiSkill environment and two GPUs, then set ``rollout.prefix_feature_model.model_path`` and ``rollout.prefix_feature_model.openpi_data.norm_stats_path`` to a matching joint-action SFT checkpoint and its normalization statistics. From the activated environment, run:

.. code:: bash

   bash examples/embodiment/run_async.sh maniskill_prefix_expo_openpi_pi05_async

This launches asynchronous collection and training, with TensorBoard metrics and evaluation videos. Adjust placement, environment counts, and epoch-based evaluation/checkpoint intervals in the example before launching.

Proposal Selection and Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At each observation, the frozen model samples ``algorithm.num_candidates`` action chunks. The editor makes one bounded stochastic edit of each of the first ``algorithm.num_edits`` proposals. With the example's values of eight and eight, the selector scores sixteen choices using the minimum of two randomly selected target critics and executes the highest-scoring chunk. Training and evaluation both use stochastic edits and critic-pair selection.

The critic target sums rewards within the executed chunk and applies ``gamma`` once to the next-state value. Next-action selection uses the online editor and target critics; an independently drawn target-critic pair evaluates the selected action for the backup. Terminal transitions stop bootstrapping. Time limits retain bootstrap when ``algorithm.bootstrap_time_limits: true``. EXPO requires ``backup_entropy: false`` and trains every online critic against the same target.

The actor edits the action actually recorded in replay and minimizes ``alpha * log_prob - mean(Q)`` over all online critics. Temperature training uses the same replay-action conditioning. With ``target_entropy: null``, the target is the sum of ``log(edit_scale_i) - 0.5`` over all coordinates of the executed chunk. The edit density includes both the tanh Jacobian and the scale Jacobian.

``ref_chunk`` and ``candidates`` use the coordinates accepted by the environment adapter. OpenPI applies its existing output transform to each proposal. Set ``actor_head.edit_scale`` in those coordinates: it accepts a positive scalar or a vector repeated across the flattened chunk. A value of ``0.02`` is an action-coordinate bound, whose physical meaning depends on the environment controller.

Collection, Replay, and Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``rollout.policy_routing: full_task`` to collect the whole task, with the routing gate disabled and no ``expert_model`` configured. The example executes the frozen reference while collecting at least 4,000 chunk transitions and performing 20,000 learner warmup updates. The existing Prefix schedule updates the actor and temperature every twenty critic updates, including during warmup. After warmup, rollout enables EXPO selection and the learner budgets twenty critic minibatch updates per new transition. The number of Q heads does not multiply this update count. The learner catches up in groups capped by ``rlt_schedule.max_updates_per_train_step``.

EXPO extends the feature call to ``extract_prefix_obs(env_obs, num_candidates=K)``. An adapter must return ``candidates`` shaped ``[B, K, L, A]`` in addition to the usual fields, with its first proposal equal to ``ref_chunk``. Rollout keeps the executed horizon and stores candidates for both the current and next observation, including terminal observations. The EXPO objectives depend only on these tensors, rewards, and termination flags. Task initialization, controller decoding, and geometry stay in environment adapters.

Rollout weight synchronization publishes the online editor together with the target critics. Checkpoint saving uses the existing SAC implementation; resume training with ``runner.resume_dir``. For standalone evaluation, combine the saved online actor and target critic weights from the same actor checkpoint directory:

.. code:: python

   from pathlib import Path

   import torch

   from rlinf.algorithms.prefix_off_policy.base import rollout_actor_target_critic

   actor_dir = Path("/path/to/actor/checkpoint")
   online = torch.load(
       actor_dir / "model_state_dict/full_weights.pt",
       map_location="cpu",
       weights_only=True,
   )
   target = torch.load(
       actor_dir / "sac_components/target_model/checkpoint_rank_0.pt",
       map_location="cpu",
       weights_only=True,
   )
   torch.save(rollout_actor_target_critic(online, target), actor_dir / "expo_eval.pt")

Set ``runner.only_eval: true`` and point ``runner.ckpt_path`` to the resulting ``expo_eval.pt`` when running ``evaluations/eval_embodied_agent.py``. Loading only the online weights would change the critics used for candidate selection.

To extend another algorithm with these behaviors, declare ``trains_entropy_temperature`` or ``rollout_uses_target_critic`` in its requirements. Override ``alpha_loss`` for temperature training, ``predict_action_batch`` for a custom rollout selector, and ``rollout_state_dict`` to compose the weights it uses. The existing Prefix worker continues to own replay, optimization, target updates, and weight synchronization.

EEF Rotation Inputs
^^^^^^^^^^^^^^^^^^^

For normalized EEF rot6d actions, register heads that undo rotation normalization and apply Gram–Schmidt before evaluation. The built-in ``eef_rot6d_residual_mlp`` and ``eef_rot6d_multi_q_mlp`` heads do this while keeping translation and gripper inputs normalized. The editor adds its bounded residual to the original normalized reference; its entropy is measured before rotation projection. Pair ``prefix.action_space: normalized_model`` with the environment's ``normalized_actions_stats_path``. See :ref:`maniskill-eef-preinsert` for the task and configuration.
