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
