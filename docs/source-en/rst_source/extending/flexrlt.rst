FlexRLT: Frozen VLA Features and Pluggable Policies
===================================================

Add a feature model, policy head, or update rule to FlexRLT. Download the checkpoints, launch ManiSkill training, and calibrate STEAM from :doc:`FlexRLT <../examples/embodied/flexrlt>`. The examples below extend that ManiSkill configuration.

Here, *prefix* means the image and language hidden states that the VLA computes before generating actions. The VLA stays frozen during online RL. A separate actor learns to use those features, the robot state, and the VLA's reference actions to produce an action chunk.

Overview
--------

The main choice is whether to use the VLA's existing features directly or first learn an RL-specific representation. With ``prefix.pool: masked_mean``, you can start from a base or SFT VLA checkpoint and average its valid prefix tokens into ``z_rl``. The RL Token route instead uses ``prefix.pool: rlt_token`` and needs the token transformer trained in Stage 1. The ManiSkill RLT presets in :doc:`../examples/embodied/rlt` use that learned representation with the same actor-critic training path.

Online updates train the actor and critic heads. Replay stores ``{z_rl, proprio, ref_chunk}``, so learner batches reuse the extracted features. The frozen VLA extracts features and generates reference actions during rollout.

How the Pieces Fit
------------------

Four independent choices make up a FlexRLT run. Each one has its own config field and its own registration point, so you change only the part you are actually replacing.

.. list-table::
   :header-rows: 1
   :widths: 20 26 22 32

   * - Choice
     - Config field
     - Registered by
     - Code
   * - Frozen feature model
     - ``rollout.prefix_feature_model``
     - ``register_model(..., category="embodied")``
     - any module implementing ``extract_prefix_obs``
   * - Trainable policy
     - ``actor.model.model_type``
     - ``register_model(..., obs_contract=ObsContract.PREFIX)``
     - ``rlinf/models/embodiment/mlp_policy/prefix_mlp_policy.py``
   * - Actor and critic heads
     - ``actor.model.actor_head.name``, ``critic_head.name``
     - ``register_prefix_actor`` / ``register_prefix_critic``
     - ``rlinf/models/embodiment/mlp_policy/heads.py``
   * - Update rule
     - ``algorithm.name`` (with ``loss_type: prefix_off_policy``)
     - ``register_prefix_off_policy_algorithm``
     - ``rlinf/algorithms/prefix_off_policy/``

The layer boundaries follow what each piece owns:

* The **feature model** owns how raw observations become ``{z_rl, proprio, ref_chunk}``. It stays frozen during online updates.
* The **policy** owns how those three tensors become a policy state and a reference input. ``prefix_policy`` builds the two heads and exposes the ``BasePolicy`` forward types the worker calls.
* The **heads** own the networks. Actor and critic are separate objects behind separate registries, so you can pair a stochastic actor with a CrossQ critic, or swap one side only.
* The **algorithm** owns the losses and the update schedule. Its ``requirements`` select compatible head ``capabilities``.

Register a policy with ``obs_contract=ObsContract.PREFIX`` to match the ``prefix_off_policy`` observation contract. Its interface follows :ref:`flexrlt-policy-contract`.

Extending it
^^^^^^^^^^^^

Pick the row you need and change only that row:

* **A new VLA** — write an adapter with ``z_dim`` and ``extract_prefix_obs``, register it as an embodied model, and point ``rollout.prefix_feature_model`` at it. The policy, heads, algorithm, and replay format stay as they are. See :ref:`Extending the BaseVLA Feature Model <flexrlt-feature-model>`.
* **A new network** — subclass ``PrefixActorHead`` or ``PrefixCriticHead``, declare its capabilities, and register it. The rollout runtime, replay format, and training loop are untouched. See :ref:`Extending MLP Actor and Critic Heads <flexrlt-heads>`.
* **A new update rule** — subclass ``PrefixOffPolicyAlgorithm``, implement ``actor_loss`` and ``target_actions``, and declare ``requirements``. Chunk discounting, BC targets, and the weight schedule stay shared. See :ref:`Extending the FlexRLT Algorithm <flexrlt-algorithm>`.
* **A different policy over the same features**: for example a recurrent actor. Subclass ``BasePolicy``, meet the contract below, register it with ``obs_contract=ObsContract.PREFIX``, and set ``actor.model.model_type``. The algorithm, replay format, and rollout runtime stay as they are.

.. _flexrlt-policy-contract:

Policy Contract
^^^^^^^^^^^^^^^

The algorithm, the FSDP worker, and the rollout runtime call the policy directly, so a replacement for ``prefix_policy`` must provide what they use:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Requirement
     - Used by
   * - Accept ``actor.model.actor_head`` and ``critic_head`` with registered ``name`` values
     - Config validation and the algorithm read both head names to check the actor distribution, action noise, and Q-head count. The policy should build those heads, or at least behave as their capabilities declare.
   * - ``forward(forward_type=ForwardType.SAC, obs=...)`` returns ``actions, logprobs, None``
     - Actor loss and target actions. ``actions`` is flat ``[B, num_action_chunks * action_dim]``; the call also receives ``deterministic``, ``apply_action_noise``, ``apply_reference_dropout``, and ``reference_dropout_prob``.
   * - ``ForwardType.SAC_Q`` returns ``[B, num_q_heads]``; ``ForwardType.CROSSQ_Q`` returns ``current_q, next_q``
     - Critic loss and the Q term of the actor loss. ``CROSSQ_Q`` is needed only when the critic head declares ``supports_cross_q``. Both accept ``detach_encoder``.
   * - Critic parameters live under a ``q_head`` attribute
     - The worker puts parameters whose names contain ``q_head``, ``encoder``, ``encoders``, or ``state_proj`` in the critic optimizer and the rest in the actor optimizer. Give actor-only modules names distinct from those critic-owned names.
   * - ``predict_action_batch(env_obs, mode=..., return_obs=True)``
     - Rollout. Return actions as ``[B, num_action_chunks, action_dim]`` and put ``action`` plus the ``z_rl``, ``proprio``, and ``ref_chunk`` tensors in ``forward_inputs``, which become the replay transition.

For a recurrent actor, include its state in the observation returned by the feature model.

Replay and Updates
------------------

Routing may execute a base, actor, or expert action. Its ``ActionPayload`` carries the executable actions and optional feedback. The gate declares the feedback keys in ``rollout_info_keys``; a gate-enabled environment must implement ``RolloutInfoReceiver.set_rollout_infos``, which the env worker checks at startup.

An environment declares calibration labels through ``TrajectoryInfoProvider.trajectory_info_spec``. Those labels and gate diagnostics travel in ``Trajectory.trajectory_infos``, separate from ``forward_inputs``. The trace writer reads them before replay drops them. Replay pairs the action actually sent to the environment with its current and next feature observations:

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = action chunk actually sent to the environment
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

The critic discounts rewards within each chunk by ``gamma`` and bootstraps the next-state Q with ``gamma ** chunk_horizon``. The actor objective is ``-q_weight * Q + bc_weight * BC``. BC normally targets ``ref_chunk``; steps marked as interventions use the executed intervention action.

How the Frozen VLA Creates a Prefix Observation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rollout worker passes raw environment observations to ``extract_prefix_obs``. The feature model reports its output width as ``z_dim`` and returns a ``PrefixObs``. This interface is defined in ``rlinf/models/embodiment/prefix_features.py``, next to ``base_policy.py``. It is orthogonal to model architecture, so any VLA can implement it:

.. code:: python

   from typing import Any, Protocol
   from rlinf.models.embodiment.prefix_features import (
       PrefixFeatureModel,
       PrefixObs,
   )

   class PrefixFeatureModel(Protocol):
       @property
       def z_dim(self) -> int: ...

       def extract_prefix_obs(
           self, env_obs: dict[str, Any]
       ) -> PrefixObs: ...

``PrefixObs`` contains three batch-first tensors. The table shows what the feature model returns:

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

OpenPI's ``Pi0Eval`` implements this interface through ``OpenPiPrefixMixin``. It converts the environment input to an OpenPI observation, computes the image and language prefix once, then uses that cache for both ``z_rl`` and reference action generation. The selected robot state becomes ``proprio``. ``PrefixRolloutRuntime`` puts the feature model in ``eval`` mode and disables its gradients.

Prefix Pooling
^^^^^^^^^^^^^^

Choose which tokens contribute to ``z_rl`` before choosing how to reduce them. ``prefix.image_only: true`` selects image tokens; ``false`` retains image and language tokens. Set ``prefix.pool`` beside it, at the same level as the model's ``openpi`` block:

* ``masked_mean`` averages valid tokens and is the default when ``openpi.use_rlt`` is false.
* ``mean`` averages the complete selected token sequence, including masked positions.
* ``last`` reads the last valid token. The mask defaults to all sequence positions.
* ``rlt_token`` runs the Stage 1 ``rlt_module.encode_flat`` instead of pooling.

``pool_prefix`` accepts ``hidden`` with shape ``[B, T, D]`` and an optional boolean mask shaped ``[B, T]`` or ``[B, T, 1]``; the three pooling modes return ``[B, D]``. When ``prefix.pool`` is unset, ``resolve_prefix_pool`` chooses ``rlt_token`` if ``openpi.use_rlt`` is true and ``masked_mean`` otherwise. ``prefix.image_only`` defaults to true. Set both fields explicitly when comparing representations.

At startup, rollout checks the feature model's ``z_dim`` against ``actor.model.z_dim``. It also checks each extracted ``z_rl`` width against the feature model's declared width.

.. _flexrlt-feature-model:

Extending the BaseVLA Feature Model
-----------------------------------

To use another VLA, adapt its observations and outputs to ``PrefixObs``. BaseVLA means the vendor or base model you adapt. The concrete adapter must support ``eval``, ``requires_grad_``, ``to``, and ``extract_prefix_obs``; routes that call the VLA's action API also need that API.

Implement an Adapter
^^^^^^^^^^^^^^^^^^^^

The skeleton below extracts prefix features and a reference chunk from one observation. Its preprocessing, encoding, and sampling methods are illustrative: implement those calls with your VLA's actual API.

.. code:: python

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

``extract_prefix_obs`` accepts the observation dictionary supplied by the environment wrapper and returns the three ``PREFIX_OBS_KEYS`` as float32 batch tensors. The example pools ``hidden`` with its mask for ``z_rl``, samples actions for ``ref_chunk``, and reads the state for ``proprio``. Check the reference action dimension and horizon against the environment, and keep state normalization consistent with the training data.

Connect the Model Factory
^^^^^^^^^^^^^^^^^^^^^^^^^

The rollout worker creates ``rollout.prefix_feature_model`` through ``get_model``. Register a builder with ``register_model`` from ``rlinf.models`` and set ``category="embodied"``. The builder receives the model config and ``torch_dtype`` and returns your adapter. Registration also adds the model type to config validation.

Make the registration run in every worker process: import in-tree builders from ``rlinf/models/__init__.py`` or load an external registration module with ``RLINF_EXT_MODULE``. See :doc:`new_model_fsdp` for the model registration workflow. You can then keep the existing ``prefix_policy``, algorithm, and replay format.

.. _flexrlt-heads:

Extending MLP Actor and Critic Heads
------------------------------------

Replace a head when you want a different network over the same features. The actor receives the assembled state and reference; its ``forward`` returns flat actions and per-dimension log probabilities, while ``mean_action`` returns a noise-free prediction for supervised forwards. The critic receives state and flat actions, and returns one value per Q head. A joint CrossQ critic also returns next-state Q values.

Save the actor, critic, and algorithm examples below in ``flexrlt_ext.py`` at the repository root.

Register an Actor Head
^^^^^^^^^^^^^^^^^^^^^^

Subclass ``PrefixActorHead`` and declare its behavior in ``ActorHeadCapabilities``. The registry passes ``state_dim``, ``reference_dim``, ``action_dim``, and the head config to the builder. This example registers a deterministic MLP with reference dropout and action noise:

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.mlp_policy.heads import (
       ActorHeadCapabilities,
       PrefixActorHead,
       register_prefix_actor,
   )

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

``register_prefix_actor`` makes this actor available under its YAML name. Its ``mean_action`` supplies the noise-free prediction used by the policy's SFT forward; ``forward`` applies the requested dropout and exploration behavior before returning actions and log probabilities. Built-in actors are ``fixed_std_mlp`` for AC and ``deterministic_mlp`` for TD3. AC requires a stochastic actor; TD3 requires a deterministic actor with action-noise support.

Register a Critic Head
^^^^^^^^^^^^^^^^^^^^^^

A critic declares ``CriticHeadCapabilities`` and ``num_q_heads``. The example below evaluates two independent Q networks on the same state-action input and registers them as the example critic:

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.mlp_policy.heads import (
       CriticHeadCapabilities,
       PrefixCriticHead,
       register_prefix_critic,
   )

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

Its ``forward`` returns ``[B, 2]``. ``multi_q_mlp`` builds ``num_q_heads`` independent Q heads from the shared ``MultiQHead``, and ``cross_q_mlp`` evaluates current and next state jointly with batch renormalization. A two-head TD3 critic is ``multi_q_mlp`` with ``num_q_heads: 2``. Choose a head whose capabilities satisfy the algorithm.

In-tree heads live in ``rlinf/models/embodiment/mlp_policy/heads.py`` and register when that module is imported, which happens as soon as the policy or the algorithm is built. For an out-of-tree head, register it in a module loaded through ``RLINF_EXT_MODULE`` so every worker process runs the registration. Select the registered names in the config:

.. code:: yaml

   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

This replaces the two networks while retaining the feature model and training worker. Both AC and TD3 can use ``multi_q_mlp`` when its Q-head count meets their requirements.

.. _flexrlt-algorithm:

Extending the FlexRLT Algorithm
-------------------------------

Change the algorithm when the actor objective, critic target, or next-action rule changes. The three entry points are ``critic_loss``, ``actor_loss``, and ``target_actions``. The base ``critic_loss`` already handles chunk-discounted Bellman targets and critic MSE; inherit it unless those rules differ. Declare the required head behavior in ``requirements`` and register the subclass.

This example reuses the shared BC/Q objective with minimum-Q aggregation and gets bootstrap actions from the target actor:

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
       )

       def actor_loss(self, context, batch):
           return self._actor_loss(
               context,
               batch,
               actor_agg_q="min",
               apply_reference_dropout=False,
               apply_action_noise=True,
           )

       def target_actions(self, context, next_obs):
           actions, _, _ = context.target_model(
               forward_type=ForwardType.SAC,
               obs=next_obs,
           )
           return actions

``actor_loss`` returns the shared helper's loss, entropy statistic, and metrics. ``target_actions`` passes ``next_obs`` to the worker-owned ``context.target_model`` and returns the flat actions for the critic target. Both it and ``context.model`` use the FSDP-wrapped ``forward`` shown above. ``PrefixPolicy`` handles the ordinary-Q or CrossQ backend.

The optional ``_actor_loss`` helper shares intervention-aware BC targets, BC/Q weight schedules, and Q aggregation. Implement ``actor_loss`` directly for a different objective. Load their registrations in the driver and workers through ``RLINF_EXT_MODULE``.

Configure and Run the Extension
-------------------------------

Prepare the environment and model paths from the :doc:`ManiSkill recipe <../examples/embodied/flexrlt>`, then create ``examples/embodiment/config/maniskill_flexrlt_custom.yaml``:

.. code:: yaml

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

Launch from the repository root. The child config inherits the environment and checkpoints from the preset:

.. code:: bash

   export RLINF_EXT_MODULE=flexrlt_ext
   bash examples/embodiment/run_embodiment.sh maniskill_flexrlt_custom

For a head-only replacement, retain the original ``algorithm.name``. For a feature-model replacement, register the adapter's builder and set ``rollout.prefix_feature_model.model_type`` and its checkpoint path in the child config. Its tensors must meet the observation contract above.

Built-in Update Rules
---------------------

The built-in ``ac`` uses the online stochastic actor for next actions and requires ``algorithm.actor_agg_q: q1``. ``td3`` keeps the fixed-variance Gaussian actor in the `RLT paper <https://arxiv.org/html/2604.23073>`_ (Section IV-B, equations 4–5): actor updates sample actions by default, and both the Q objective and BC use that sample. The actor optimizes Q1; the critic target uses the minimum of the first two Q heads. ``algorithm.actor_update_action_noise: false`` selects mean actions for actor updates instead.

For target policy smoothing, ``td3`` obtains the target actor's mean action, adds one clipped Gaussian sample, and clips the result to [-1, 1]. ``algorithm.target_policy_noise`` defaults to the actor head's ``noise_sigma`` (0.1 for ``deterministic_mlp``); ``algorithm.target_noise_clip`` defaults to 0.5. The presets use 0.1 for both action-sampling scales. Setting either target-noise parameter to zero disables smoothing.

The actor and both target networks update together after every ``algorithm.critic_actor_ratio`` critic updates, starting with the second critic update at the default ratio of 2. This matches RLT's two-critic-to-one-actor schedule and the coupled target updates in the `TD3 reference implementation <https://github.com/sfujim/TD3/blob/master/TD3.py>`_. If set, ``algorithm.target_update_freq`` must equal ``critic_actor_ratio``, and ``algorithm.target_update_type`` must be ``all``. BC, reference dropout, and the BC/Q weight schedule remain shared with AC.

The critic target in either algorithm is the minimum of the first two Q heads. Extra ``multi_q_mlp`` heads regress to the same target, so leave ``algorithm.agg_q`` unset or set it to ``min``. Both algorithms use ``algorithm.entropy_tuning.alpha_type: fixed_alpha``. These settings are checked when the algorithm is built.
