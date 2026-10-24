Prefix Fine-Tune: Frozen VLA Features and Pluggable Policies
============================================================

Prefix Fine-Tune trains a pluggable compact policy on frozen VLA features. This page follows the rollout, policy, and algorithm call path through the current implementation, then shows how to add a VLA adapter, MLP head, or off-policy algorithm. Read :doc:`../examples/embodied/rlt` for the RL Token Stage 1 transformer and task workflow.

Prefix Fine-Tune gives the VLA representation and the online RL policy separate lifecycles. The rollout worker loads a frozen VLA feature model, while the actor worker trains only compact actor and critic heads. The workers meet through the fixed ``{z_rl, proprio, ref_chunk}`` contract, so the vision-language encoder, policy heads, and update algorithm can be replaced independently.

Overview
--------

One Prefix policy step follows this path:

1. The environment provides images, language, state, and task metadata.
2. ``rollout.prefix_feature_model`` calls the VLA's ``extract_prefix_obs`` method.
3. The VLA returns ``z_rl``, ``proprio``, and ``ref_chunk``.
4. The rollout ``prefix_policy`` builds the state and calls the registered actor head to produce an action chunk.
5. The learner ``prefix_off_policy`` worker receives transitions, writes them to replay, computes critic targets and the actor objective, and synchronizes the rollout policy.

The default ``z_rl`` is the masked mean of the VLM prefix hidden states. ``prefix.pool: mean``, ``last``, and ``rlt_token`` provide other representation paths. ``rlt_token`` reads the token transformer trained in Stage 1; its training procedure is documented in :doc:`../examples/embodied/rlt`.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Frozen feature model
      :text-align: center

      ``PrefixFeatureModel`` contract

   .. grid-item-card:: Prefix observation
      :text-align: center

      ``z_rl``, ``proprio``, ``ref_chunk``

   .. grid-item-card:: Trainable policy
      :text-align: center

      ``PrefixPolicy`` and registered heads

   .. grid-item-card:: Update runtime
      :text-align: center

      ``prefix_off_policy`` worker

Current Implementation
----------------------

How the frozen VLA creates a Prefix observation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rlinf.models.embodiment.prefix.contracts`` defines the smallest feature-model interface:

.. code:: python

   from typing import Any, Protocol
   from rlinf.models.embodiment.prefix.types import PrefixObs

   class PrefixFeatureModel(Protocol):
       @property
       def z_dim(self) -> int: ...

       def extract_prefix_obs(
           self, env_obs: dict[str, Any]
       ) -> PrefixObs: ...

``PrefixObs`` is a TypedDict with three batch-first tensors. Here ``z_dim`` is the original feature-model output width; rollout appends proprio history to ``z_rl`` when state history is enabled.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - key
     - shape
     - meaning
   * - ``z_rl``
     - ``[B, z_dim]``
     - Frozen VLA Prefix representation.
   * - ``proprio``
     - ``[B, proprio_dim]``
     - Current robot or simulator state used by the policy.
   * - ``ref_chunk``
     - ``[B, ref_num_action_chunks, action_dim]``
     - Reference action chunk sampled from the frozen VLA and passed to the actor as a condition.

OpenPI's ``Pi0Eval.extract_prefix_obs`` is the current implementation. It converts environment inputs to an OpenPI observation, runs the image and language Prefix through ``build_prefix_cache``, and lets ``_encode_stage2_z`` select pooling or the RLT token path. ``_sample_actions_from_prefix_cache`` produces the reference chunk, and the configured state becomes ``proprio``. The rollout worker puts this model in ``eval`` mode and disables gradients during initialization.

Prefix pooling
^^^^^^^^^^^^^^

``rlinf.models.embodiment.prefix.pool.pool_prefix`` accepts ``hidden`` (``[B, T, D]``) and an optional boolean mask, then returns ``[B, D]``:

* ``masked_mean`` averages valid tokens and is the default for Prefix Fine-Tune.
* ``mean`` averages the complete token sequence.
* ``last`` reads the last valid token; without a mask it reads the final sequence position.
* ``rlt_token`` is a configuration marker. OpenPI routes it to ``rlt_module.encode_flat`` instead of calling ``pool_prefix`` directly.

``resolve_prefix_pool`` resolves the final mode from ``prefix.pool``, ``stage2_z_source``, ``use_rlt``, and ``rlt_use_mask``. Set ``prefix.pool`` explicitly so the representation source can be inspected in both Stage 1 and Stage 2 configurations.

State history
^^^^^^^^^^^^^

With ``actor.model.state_history.enable: true``, ``StateHistoryBuffer`` keeps the latest ``steps`` proprio values at the policy decision rate and appends the flattened history to ``z_rl``. ``pad`` accepts ``zero`` and ``repeat``. Episode reset clears history for the selected environments. ``get_prefix_feature_dim`` computes the policy width:

.. code:: text

   runtime_z_dim = z_dim + steps * proprio_dim  # state_history.enable=true
   runtime_z_dim = z_dim                         # state_history.enable=false

Rollout and actor use the same ``actor.model.state_history`` configuration, which keeps the feature width identical.

PrefixPolicy forward contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rlinf.models.embodiment.prefix.policy.PrefixPolicy`` inherits ``BasePolicy`` and exposes off-policy forward types:

* ``ForwardType.SAC`` calls the actor head and returns ``actions, logprobs, None``.
* ``ForwardType.SAC_Q`` calls the critic head and returns one value per Q head.
* ``ForwardType.CROSSQ_Q`` evaluates current and next state values when the critic supports CrossQ.
* ``predict_action_batch`` reshapes flat actions to ``[B, num_action_chunks, action_dim]`` and returns the action and Prefix observation in ``forward_inputs``.

The policy state is ``concat(flatten(z_rl), flatten(proprio))``. The reference is reshaped by action dimension and truncated to ``num_action_chunks`` before it conditions the actor and supplies the BC target. ``ref_num_action_chunks`` may exceed the executed ``num_action_chunks``, so the feature model can expose a longer reference window.

PrefixPolicy does not own VLA parameters. The rollout worker owns the VLA, while PrefixPolicy contains the actor and critic heads. The FSDP worker identifies critic parameters through the stable ``q_head`` path.

Minimal Configuration
---------------------

The following fragment uses an OpenPI masked-mean feature model, a stochastic MLP actor, and a twin-Q critic. Fill in paths and environment fields for the selected task.

.. code:: yaml

   algorithm:
     loss_type: prefix_off_policy
     name: ac
     gamma: 0.99
     q_weight: 1.0
     bc_weight: 1.0

   rollout:
     prefix_feature_model:
       model_type: openpi
       model_path: /path/to/base-or-sft-vla
       precision: bf16
       prefix:
         pool: masked_mean
         image_only: false
       openpi_data:
         repo_id: my_lerobot_dataset
         norm_stats_path: /path/to/norm_stats.json
       openpi:
         task: eval
         config_name: pi05_franka_state

   actor:
     model:
       model_type: prefix_policy
       z_dim: 2048
       proprio_dim: 19
       action_dim: 7
       num_action_chunks: 10
       ref_num_action_chunks: 20
       actor_head:
         name: fixed_std_mlp
         hidden_dims: [256, 256, 256]
         activation: tanh
         fixed_std: 0.002
       critic_head:
         name: twin_q_mlp
         hidden_dims: [256, 256, 256]

``actor.model.z_dim`` must equal the base ``z_rl`` width produced by the feature model. State history adds its width automatically. Keep ``actor.model.proprio_dim``, ``action_dim``, ``num_action_chunks``, and the feature-model and environment action contracts aligned.

``rlinf/config.py`` validates the training path. ``loss_type: prefix_off_policy`` requires ``actor.model.model_type: prefix_policy``, both head names, and a registered ``algorithm.name``. The validation rejects incompatible actor distributions, action-noise capabilities, and Q-head counts before the worker starts.

Extending the BaseVLA Feature Model
-----------------------------------

Use the feature-model contract when adding a VLA. BaseVLA here refers to an existing VLA base or vendor model; RLinf currently has no common class with that name. The ``PrefixFeatureModel`` Protocol fixes the boundary with the rollout worker. The concrete object must support ``eval``, ``requires_grad_``, ``to``, and ``extract_prefix_obs``. A route that executes base VLA actions also needs the policy's action API.

Implement an adapter
^^^^^^^^^^^^^^^^^^^^

An adapter converts ``env_obs`` to VLA inputs, runs the Prefix encoder, samples the reference action chunk, and returns three float32 batch tensors. The skeleton uses illustrative preprocessing, encoding, and action-sampling methods; implement those operations against the vendor's actual API.

.. code:: python

   import torch
   from rlinf.models.embodiment.prefix.pool import pool_prefix

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

The ``env_obs`` input is defined by the environment wrapper, while the returned keys must stay equal to ``PREFIX_OBS_KEYS``. The last dimension of ``ref_chunk`` must match the environment action dimension, and its length must cover ``num_action_chunks``. Keep proprio semantics and normalization aligned with ``actor.model.proprio_dim`` and the training data.

Connect the model factory
^^^^^^^^^^^^^^^^^^^^^^^^^

The rollout worker constructs ``rollout.prefix_feature_model`` through ``get_model``. A custom adapter needs a lazy builder in ``rlinf/models/__init__.py``, a model type in ``SupportedModel`` in ``rlinf/config.py``, and embodied-model validation there. The returned object must satisfy both the rollout PyTorch-module lifecycle and the Prefix feature contract. A route that executes base VLA actions also needs the policy's action API.

After connecting the adapter, reuse the existing ``prefix_policy``, ``prefix_off_policy`` worker, and replay schema. Test batch size, device and dtype, reset-time state semantics, and action-chunk shapes.

Extending MLP Actor and Critic Heads
------------------------------------

Prefix heads receive the assembled state and reference/action tensors. An actor's ``forward`` returns flat actions and per-dimension log probabilities, while ``mean_action`` returns the noise-free action used by BC. A critic returns ``[..., num_q_heads]``; a CrossQ critic can also return next-state Q values.

Register an actor head
^^^^^^^^^^^^^^^^^^^^^^

Subclass ``PrefixActorHead``, declare ``ActorHeadCapabilities``, and use ``register_prefix_actor``. The builder arguments are ``state_dim``, ``reference_dim``, ``action_dim``, and the head's Hydra config.

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.prefix.heads.base import (
       ActorHeadCapabilities, PrefixActorHead,
   )
   from rlinf.models.embodiment.prefix.heads.registry import register_prefix_actor

   class CustomMLPActor(PrefixActorHead):
       capabilities = ActorHeadCapabilities(
           distribution="deterministic", supports_action_noise=True,
       )

       def __init__(self, state_dim, reference_dim, action_dim, cfg):
           super().__init__()
           hidden_dim = int(cfg.get("hidden_dim", 256))
           self.net = nn.Sequential(
               nn.Linear(state_dim + reference_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, action_dim),
           )

       def mean_action(self, state, reference):
           return self.net(torch.cat([state, reference], dim=-1)).clamp(-1, 1)

       def forward(
           self, state, reference, *, deterministic,
           apply_reference_dropout, reference_dropout_prob,
           apply_action_noise,
       ):
           if apply_reference_dropout:
               probability = float(reference_dropout_prob or 0.0)
               keep = torch.rand(reference.shape[0], 1, device=reference.device)
               reference = reference * (keep >= probability).to(reference.dtype)
           action = self.mean_action(state, reference)
           if apply_action_noise and not deterministic:
               action = (action + torch.randn_like(action) * 0.1).clamp(-1, 1)
           return action, torch.zeros_like(action)

   @register_prefix_actor(
       "custom_mlp", capabilities=CustomMLPActor.capabilities,
   )
   def make_custom_mlp_actor(state_dim, reference_dim, action_dim, cfg):
       return CustomMLPActor(state_dim, reference_dim, action_dim, cfg)

Built-in actor names are ``fixed_std_mlp`` and ``deterministic_mlp``. AC requires a stochastic actor. TD3 requires a deterministic actor with action-noise support. Capability metadata keeps validation and algorithm composition consistent.

Register a critic head
^^^^^^^^^^^^^^^^^^^^^^

Subclass ``PrefixCriticHead``, set ``CriticHeadCapabilities`` and ``num_q_heads``, and use ``register_prefix_critic``. ``multi_q_mlp`` supports a configurable number of Q estimates, ``twin_q_mlp`` fixes two independent Q networks, and ``cross_q_mlp`` adds CrossQ evaluation.

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.prefix.heads.base import (
       CriticHeadCapabilities, PrefixCriticHead,
   )
   from rlinf.models.embodiment.prefix.heads.registry import register_prefix_critic

   class CustomTwinCritic(PrefixCriticHead):
       capabilities = CriticHeadCapabilities(fixed_num_q_heads=2)

       def __init__(self, state_dim, action_dim, cfg):
           super().__init__()
           self.num_q_heads = 2
           width = int(cfg.get("hidden_dim", 256))
           def q_network():
               return nn.Sequential(
                   nn.Linear(state_dim + action_dim, width),
                   nn.ReLU(),
                   nn.Linear(width, 1),
               )
           self.q1 = q_network()
           self.q2 = q_network()

       def forward(self, state, actions, *, next_state=None, next_actions=None):
           del next_state, next_actions
           inputs = torch.cat([state, actions], dim=-1)
           return torch.cat([self.q1(inputs), self.q2(inputs)], dim=-1)

   @register_prefix_critic(
       "custom_twin", capabilities=CustomTwinCritic.capabilities,
   )
   def make_custom_twin_critic(state_dim, action_dim, cfg):
       return CustomTwinCritic(state_dim, action_dim, cfg)

Place the new head in ``rlinf.models.embodiment.prefix.heads`` and import it from the registry's ``_load_builtin_heads``. Configuration uses the registered name:

.. code:: yaml

   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

Extending the Prefix Algorithm
------------------------------

``PrefixOffPolicyAlgorithm`` separates the actor objective, critic target action, and Q aggregation rules. Subclass it, implement ``requirements``, ``actor_forward_kwargs``, ``target_actions``, and ``aggregate_actor_q``, then register the class with ``register_prefix_off_policy_algorithm``.

.. code:: python

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
           uses_target_actor=True,
       )

       def actor_forward_kwargs(self):
           return {"apply_action_noise": True}

       def target_actions(self, context, next_obs):
           actions, _, _ = context.target_model(
               forward_type=ForwardType.SAC,
               obs=next_obs,
           )
           return actions

       def aggregate_actor_q(self, all_q_values):
           return self._min_twin_q(all_q_values)

``PrefixOffPolicyAlgorithm`` already implements chunk-reward discounting, termination bootstrap, BC targets, human-intervention masks, BC/Q weight schedules, and critic MSE. The subclass supplies algorithm semantics. Import the module from the registry's ``_load_builtin_algorithms``; select it with ``algorithm.name: my_algorithm`` while keeping ``loss_type: prefix_off_policy``.

The built-in ``ac`` uses a stochastic actor and Q1 aggregation. ``td3`` uses a deterministic actor, a target actor, action noise, and ``min``, ``q1``, or ``mean`` actor-Q aggregation. The algorithm requirements are checked against the head capabilities.

EXPO edit actor and candidate selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The built-in ``expo`` algorithm trains a stochastic edit actor around the VLA reference chunk. Set ``actor_head.name: expo_edit_mlp`` to use ``ExpoEditMLPActor``. It encodes the concatenation of policy state and reference action, samples a ``tanh`` edit, and returns ``reference + edit_scale * edit`` clipped to the action range. The actor returns the per-dimension transformed log probability used by the entropy objective. ``mean_action`` uses the deterministic mean edit for supervised calls.

``expo`` requires a stochastic actor and at least two Q estimates. ``multi_q_mlp`` provides the Q ensemble through ``critic_head.num_q_heads``. The critic target always contains the unmodified reference action and ``algorithm.expo_num_edit_samples`` sampled edits. ``algorithm.expo_num_min_qs`` selects a random subset of Q heads for a conservative candidate score, and the best candidate supplies the chunk-level bootstrap value. Rewards inside a chunk use discounted aggregation, followed by ``gamma ** chunk_horizon`` for the next-state bootstrap.

The actor objective is ``alpha * log_prob(edit) - q_weight * Q(edit) + bc_weight * BC``, where ``Q(edit)`` averages the ensemble estimates. Set ``bc_weight`` above zero when the run should retain a reference or intervention BC term. ``entropy_tuning.alpha_type: softplus`` enables automatic temperature training; ``initial_alpha`` and ``target_entropy`` define its starting value and target. ``actor_tau`` controls the EXPO actor target update independently from the critic ``tau``.

Rollout can score a candidate set before sending an action to the environment. ``actor.model.expo_candidate_samples`` adds sampled edits to the reference candidate, and the rollout policy selects the candidate with the highest minimum of its first two Q estimates. Keep this value at zero to disable rollout-time selection while retaining EXPO training.

The current Prefix integration covers the edit MLP, candidate Q selection, Q ensemble target, chunk-level EXPO objective, and temperature training. Real-Time EXPO latency handling, asynchronous VLA inference, and noise filtering are outside this implementation.

Training and Running
--------------------

The repository includes ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml`` as a Prefix Fine-Tune example. Its current settings use ``use_rlt: false`` with masked-mean VLM Prefix features, ``expo_edit_mlp``, a ten-head ``multi_q_mlp`` ensemble, and ``algorithm.name: expo``. Replace ``rollout.prefix_feature_model.model_path``, the OpenPI ``repo_id``, ``norm_stats_path``, and environment asset paths, then run:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

The rollout configuration sets ``stage2_z_source: vlm_prefix`` and ``prefix_pool: masked_mean``, so ``z_rl`` comes from VLM pooling while the RLT token module remains disabled. STEAM routing selects among the base VLA, Stage 2 actor, and expert during rollout; the Prefix runtime still owns the heads and algorithm. The same file contains the EXPO candidate and temperature settings described above.

Stage 2 replay uses a fixed observation contract:

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = action chunk actually sent to the environment
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

The critic discounts rewards inside a chunk with ``gamma`` and bootstraps the next-state Q with ``gamma ** chunk_horizon``. The actor uses ``-q_weight * Q + bc_weight * BC``. The default BC target is ``ref_chunk``; a step marked with an intervention flag uses the executed human action.

Implementation and Checkpoints
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Component
     - Code location
   * - ``PrefixObs`` and feature contract
     - ``rlinf/models/embodiment/prefix/contracts.py``, ``types.py``
   * - pooling, state history, config parsing
     - ``rlinf/models/embodiment/prefix/pool.py``, ``history.py``, ``config.py``
   * - trainable policy
     - ``rlinf/models/embodiment/prefix/policy.py``
   * - actor/critic heads and registry
     - ``rlinf/models/embodiment/prefix/heads/``
   * - off-policy algorithms
     - ``rlinf/algorithms/prefix_off_policy/``
   * - EXPO edit actor and algorithm
     - ``rlinf/models/embodiment/prefix/heads/mlp.py``, ``rlinf/algorithms/prefix_off_policy/expo.py``
   * - FSDP actor worker
     - ``rlinf/workers/actor/fsdp_prefix_off_policy_worker.py``

When debugging, compare the feature model ``z_dim``, the policy width ``z_dim + history.extra_dim``, and the environment width ``num_action_chunks * action_dim``. Then check that Stage 1 and Stage 2 use the same OpenPI dataconfig and ``norm_stats.json``. An RLT-token checkpoint must include the complete ``rlt_module``; its configuration and run steps are in :doc:`../examples/embodied/rlt`.
