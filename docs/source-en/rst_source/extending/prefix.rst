Prefix Fine-Tune: Frozen VLA Features and Pluggable Policies
============================================================

Add a feature model, policy head, or update rule to Prefix Fine-Tune. Download the checkpoints, launch ManiSkill training, and calibrate STEAM from :doc:`Prefix Fine-Tune <../examples/embodied/prefix>`. This page starts from that running config and shows the observation contract and the registration points you change.

Here, *prefix* means the image and language hidden states that the VLA computes before generating actions. The VLA stays frozen during online RL. A separate actor learns to use those features, the robot state, and the VLA's reference actions to produce an action chunk.

Overview
--------

The main choice is whether to use the VLA's existing features directly or first learn an RL-specific representation. With ``prefix.pool: masked_mean``, you can start from a base or SFT VLA checkpoint and average its valid prefix tokens into ``z_rl``. The RL Token route instead uses ``prefix.pool: rlt_token`` and needs the token transformer trained in Stage 1. Follow :doc:`../examples/embodied/rlt` if you want that learned representation; both routes use the same actor-critic training path.

Online updates train only the actor and critic heads, rather than backpropagating through the VLA. Replay stores ``{z_rl, proprio, ref_chunk}`` instead of the image observations, so learner batches reuse the extracted features. The VLA still runs during rollout to extract features and generate the reference action chunk; freezing it does not remove that inference work.

The RLT-specific AC and TD3 policy/worker implementations are replaced by one ``prefix_policy`` model and one ``prefix_off_policy`` worker. Choose the actor head, critic head, and algorithm separately in the config. Chunk-reward discounting, BC targets, and weight schedules are shared by AC and TD3. This is useful when comparing a new representation or head without copying the training loop. Evaluate pooled and learned features on your task before choosing between them.

Run the Example
---------------

The ManiSkill peg-insertion preset, checkpoint layout, routing settings, and STEAM calibration commands are documented in :doc:`Prefix Fine-Tune <../examples/embodied/prefix>`. Use that page to launch ``maniskill_prefix_stage2_td3_mlp_steam``. The sections below explain the data the run stores and the code you change to add a component.

Replay and Updates
------------------

Routing may execute a base, actor, or expert action. Its ``ActionPayload`` carries the executable actions and optional feedback. The gate declares the feedback keys in ``rollout_info_keys``; a gate-enabled environment must implement ``RolloutInfoReceiver.set_rollout_infos``, which the env worker checks at startup.

An environment declares calibration labels through ``TrajectoryInfoProvider.trajectory_info_spec``. Those labels and gate diagnostics travel in ``Trajectory.trajectory_infos``, separate from ``forward_inputs``. The trace writer reads them before replay drops them. Replay pairs the action actually sent to the environment with its current and next feature observations:

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = action chunk actually sent to the environment
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

The critic discounts rewards within each chunk by ``gamma`` and bootstraps the next-state Q with ``gamma ** chunk_horizon``. The actor objective is ``-q_weight * Q + bc_weight * BC``. BC normally targets ``ref_chunk``; steps marked as interventions use the executed intervention action.

Minimal Configuration
---------------------

When adapting the example, keep the frozen VLA under ``rollout.prefix_feature_model`` and the trainable heads under ``actor.model``. The fragment below shows that split for AC with a stochastic MLP actor and twin-Q critic. Add the task's environment, placement, optimizer, and rollout settings from a complete preset before launching it.

.. code:: yaml

   algorithm:
     loss_type: prefix_off_policy
     name: ac
     gamma: 0.99
     q_weight: 1.0
     bc_weight: 1.0
     entropy_tuning:
       alpha_type: fixed_alpha
       initial_alpha: 0.0

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

``actor.model.z_dim`` is the feature model's output width before state history is added. Match ``proprio_dim`` to the returned robot state, ``action_dim`` to one environment action, and ``num_action_chunks`` to the executed chunk length. The reference must cover at least that many actions; ``ref_num_action_chunks`` may be longer.

Config validation requires ``loss_type: prefix_off_policy``, ``actor.model.model_type: prefix_policy``, both registered head names, and a registered ``algorithm.name``. It checks the actor distribution, action-noise support, and Q-head count before workers start.

Current Implementation
----------------------

The frozen VLA and the trainable policy meet at a three-tensor observation. This section follows that observation from feature extraction to actor and critic calls; use the same shapes when adding another VLA or head.

How the Frozen VLA Creates a Prefix Observation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rollout worker passes raw environment observations to ``extract_prefix_obs``. The feature model reports its output width as ``z_dim`` and returns a ``PrefixObs``. This interface is defined in ``rlinf.models.embodiment.prefix.contracts``:

.. code:: python

   from typing import Any, Protocol
   from rlinf.models.embodiment.prefix.types import PrefixObs

   class PrefixFeatureModel(Protocol):
       @property
       def z_dim(self) -> int: ...

       def extract_prefix_obs(
           self, env_obs: dict[str, Any]
       ) -> PrefixObs: ...

``PrefixObs`` contains three batch-first tensors. The table shows the feature model's output, before rollout appends state history:

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

OpenPI's ``Pi0Eval`` implements this interface through ``OpenPiPrefixMixin``, which adds no parameters or checkpoint keys. It converts the environment input to an OpenPI observation, computes the image and language prefix once, then uses that cache for both ``z_rl`` and reference action generation. The selected robot state becomes ``proprio``. ``PrefixRolloutRuntime`` puts the feature model in ``eval`` mode and disables its gradients.

Prefix Pooling
^^^^^^^^^^^^^^

Choose which tokens contribute to ``z_rl`` before choosing how to reduce them. ``prefix.image_only: true`` selects image tokens; ``false`` retains image and language tokens. Set ``prefix.pool`` beside it, at the same level as the model's ``openpi`` block:

* ``masked_mean`` averages valid tokens and is the default when ``openpi.use_rlt`` is false.
* ``mean`` averages the complete selected token sequence, including masked positions.
* ``last`` reads the last valid token, or the final sequence position if no mask is supplied.
* ``rlt_token`` runs the Stage 1 ``rlt_module.encode_flat`` instead of pooling.

``pool_prefix`` accepts ``hidden`` with shape ``[B, T, D]`` and an optional boolean mask; the three pooling modes return ``[B, D]``. When ``prefix.pool`` is unset, ``resolve_prefix_pool`` chooses ``rlt_token`` if ``openpi.use_rlt`` is true and ``masked_mean`` otherwise. ``prefix.image_only`` defaults to true. Set both fields explicitly when comparing representations.

At startup, rollout checks the feature model's ``z_dim`` against ``actor.model.z_dim``. It also checks each extracted ``z_rl`` width against the feature model's declared width.

State History
^^^^^^^^^^^^^

For a policy that needs recent motion as well as the current observation, set ``actor.model.state_history.enable: true``. ``StateHistoryBuffer`` saves the most recent ``steps`` proprio observations at the policy decision rate and appends their flattened values to ``z_rl``. ``pad`` selects ``zero`` or ``repeat`` padding, and an episode reset clears that environment's history.

.. code:: text

   runtime_z_dim = z_dim + steps * proprio_dim  # state_history.enable=true
   runtime_z_dim = z_dim                         # state_history.enable=false

``get_prefix_feature_dim`` computes this runtime width. Rollout and actor both use ``actor.model.state_history``, so leave ``actor.model.z_dim`` at the original feature-model width.

History is indexed by mode, environment rank, pipeline stage, and source-row offset. Terminal observations read the preceding episode's history before current observations reset or advance it. Terminal reads do not commit history, and evaluation keeps its history separate from training. The STEAM gate uses the same source identity for image history and routing latches, so splitting or reordering a request does not reset other environments.

PrefixPolicy Forward Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PrefixPolicy`` concatenates the flattened ``z_rl`` and ``proprio`` into the policy state. It truncates ``ref_chunk`` to the executed ``num_action_chunks`` and flattens it for the actor's reference input and BC target. It contains the actor and critic heads. ``PrefixRolloutRuntime`` owns the frozen VLA, optional expert and gate, and train/eval history. The FSDP worker identifies critic parameters through ``q_head``.

Calls through its ``BasePolicy`` interface return:

* ``PrefixForwardType.ACTOR``: ``actions, logprobs, None`` from the actor head.
* ``PrefixForwardType.CRITIC``: ``current_q, next_q``, with one value per Q head. ``next_q`` is ``None`` for a non-joint critic.
* ``predict_action_batch``: actions reshaped to ``[B, num_action_chunks, action_dim]``, with the action and feature observation in ``forward_inputs``.

``PrefixPolicy`` selects the ordinary-Q or joint CrossQ critic internally. Algorithms make the same actor and critic requests for either backend. All training calls go through the FSDP-wrapped ``forward``.

Extending the BaseVLA Feature Model
-----------------------------------

To use another VLA, adapt its observations and outputs to ``PrefixObs``. The existing OpenPI adapter is one implementation, not a requirement of the policy heads. Here, BaseVLA means your vendor or base model, not an RLinf base class. The concrete adapter must support ``eval``, ``requires_grad_``, ``to``, and ``extract_prefix_obs``; routes that call the VLA's action API also need that API.

Implement an Adapter
^^^^^^^^^^^^^^^^^^^^

The skeleton below extracts prefix features and a reference chunk from one observation. Its preprocessing, encoding, and sampling methods are illustrative: implement those calls with your VLA's actual API.

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

``extract_prefix_obs`` accepts the observation dictionary supplied by the environment wrapper and returns the three ``PREFIX_OBS_KEYS`` as float32 batch tensors. The example pools ``hidden`` with its mask for ``z_rl``, samples actions for ``ref_chunk``, and reads the state for ``proprio``. Check the reference action dimension and horizon against the environment, and keep state normalization consistent with the training data.

Connect the Model Factory
^^^^^^^^^^^^^^^^^^^^^^^^^

The rollout worker creates ``rollout.prefix_feature_model`` through ``get_model``. Register a builder with ``register_model`` from ``rlinf.models`` and set ``category="embodied"``. The builder receives the model config and ``torch_dtype`` and returns your adapter. Registration also adds the model type to config validation.

Make the registration run in every worker process: import in-tree builders from ``rlinf/models/__init__.py`` or load an external registration module with ``RLINF_EXT_MODULE``. See :doc:`new_model_fsdp` for the model registration workflow. You can then keep the existing ``prefix_policy``, algorithm, and replay format. Test the adapter's batch shapes, device and dtype, reset behavior, and action chunks through that rollout path.

Extending MLP Actor and Critic Heads
------------------------------------

Replace a head when you want a different network over the same features. The actor receives the assembled state and reference; its ``forward`` returns flat actions and per-dimension log probabilities, while ``mean_action`` returns a noise-free prediction for supervised forwards. The critic receives state and flat actions, and returns one value per Q head. A joint CrossQ critic also returns next-state Q values.

Register an Actor Head
^^^^^^^^^^^^^^^^^^^^^^

Subclass ``PrefixActorHead`` and declare its behavior in ``ActorHeadCapabilities``. The registry passes ``state_dim``, ``reference_dim``, ``action_dim``, and the head config to the builder. This example registers a deterministic MLP with reference dropout and action noise:

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

``register_prefix_actor`` makes ``custom_mlp`` available to the model builder. Its ``mean_action`` supplies the noise-free prediction used by the policy's SFT forward; ``forward`` applies the requested dropout and exploration behavior before returning actions and log probabilities. Built-in actors are ``fixed_std_mlp`` for AC and ``deterministic_mlp`` for TD3. AC requires a stochastic actor; TD3 requires a deterministic actor with action-noise support.

Register a Critic Head
^^^^^^^^^^^^^^^^^^^^^^

A critic declares ``CriticHeadCapabilities`` and ``num_q_heads``. The example below evaluates two independent Q networks on the same state-action input and registers them as ``custom_twin``:

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

Its ``forward`` returns ``[B, 2]``. The built-in ``multi_q_mlp`` allows a configurable head count, ``twin_q_mlp`` fixes it at two, and ``cross_q_mlp`` supports joint current/next-state evaluation. Choose a head whose capabilities satisfy the algorithm.

Import in-tree heads from the registry's ``_load_builtin_heads`` so each worker process registers them. Select the registered names in the config:

.. code:: yaml

   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

This replaces the two networks while retaining the feature model and training worker. Both AC and TD3 can use ``multi_q_mlp`` when its Q-head count meets their requirements.

Extending the Prefix Algorithm
------------------------------

Change the algorithm when the actor objective, critic target, or next-action rule changes. The three entry points are ``critic_loss``, ``actor_loss``, and ``target_actions``. The base ``critic_loss`` already handles chunk-discounted Bellman targets and critic MSE; inherit it unless those rules differ. Declare the required head behavior in ``requirements`` and register the subclass.

This example reuses the shared BC/Q objective with minimum-Q aggregation and gets bootstrap actions from the target actor:

.. code:: python

   from rlinf.algorithms.prefix_off_policy import (
       PrefixAlgorithmRequirements, PrefixOffPolicyAlgorithm,
       register_prefix_off_policy_algorithm,
   )
   from rlinf.models.embodiment.prefix import PrefixForwardType

   @register_prefix_off_policy_algorithm("my_algorithm")
   class MyAlgorithm(PrefixOffPolicyAlgorithm):
       requirements = PrefixAlgorithmRequirements(
           actor_distribution="deterministic",
           requires_action_noise=True,
           uses_target_actor=True,
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
               forward_type=PrefixForwardType.ACTOR,
               obs=next_obs,
           )
           return actions

``actor_loss`` returns the shared helper's loss, entropy statistic, and metrics. ``target_actions`` passes ``next_obs`` to the worker-owned ``context.target_model`` and returns the flat actions for the critic target. Both it and ``context.model`` are wrapped training models: call them through ``forward`` as shown, rather than bypassing FSDP via ``.module``. ``PrefixPolicy`` handles the ordinary-Q or CrossQ backend.

The optional ``_actor_loss`` helper shares intervention-aware BC targets, BC/Q weight schedules, and Q aggregation between AC and TD3. Implement ``actor_loss`` directly for a different objective. Import the algorithm module from the registry's ``_load_builtin_algorithms`` and select ``algorithm.name: my_algorithm`` with ``loss_type: prefix_off_policy``.

The built-in ``ac`` uses the online stochastic actor for next actions and requires ``algorithm.actor_agg_q: q1``. ``td3`` uses noise-free target actions, rollout exploration noise, and ``min``, ``q1``, or ``mean`` actor-Q aggregation. It does not apply target policy smoothing. ``algorithm.actor_update_action_noise`` separately controls noise during actor updates. The critic target in either algorithm is the minimum of the first two Q heads. Extra ``multi_q_mlp`` heads regress to the same target, so leave ``algorithm.agg_q`` unset or set it to ``min``. Both algorithms require ``algorithm.entropy_tuning.alpha_type: fixed_alpha``; they do not train an entropy temperature. These settings are checked when the algorithm is built.

Implementation and Checkpoints
------------------------------

Use these files to trace a config choice to its implementation:

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
   * - Rollout lifecycle and replay/scheduling
     - ``rlinf/algorithms/prefix_off_policy/rollout.py``, ``replay.py``, ``schedule.py``
   * - OpenPI feature mixin
     - ``rlinf/models/embodiment/openpi/prefix.py``
   * - FSDP actor worker
     - ``rlinf/workers/actor/fsdp_prefix_off_policy_worker.py``

For shape errors, compare the feature model's ``z_dim``, the policy's runtime width ``z_dim + history.extra_dim``, and the environment's flat action width ``num_action_chunks * action_dim``. For action-scale errors, check the OpenPI data config and ``norm_stats.json`` used by each checkpoint. An RLT-token feature model must also load the complete ``rlt_module`` from its Stage 1 checkpoint; see :doc:`../examples/embodied/rlt`.
