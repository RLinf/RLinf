Embodied Data Interface
=======================

The embodied data interface separates runtime communication from Actor-facing
training data:

.. code-block:: text

   EnvOutput(obs, EnvTransition)
       -> PolicyInput (+ previous EnvPart)
       -> actions tensor

   PolicyPart + completed EnvPart
       -> TrajectoryCollector
       -> Trajectory / episode shard / pipeline micro-batch

See :doc:`../../concepts/trajectory_collector` for the lifecycle, data
ownership, and execution-mode semantics.

Shape notation
--------------

The class field comments use the following symbols:

* ``B`` is the routed environment batch. Before decoupled merging, it is
  normally ``env.train.total_num_envs / env_world_size /
  rollout.pipeline_stage_num``.
* ``C`` is ``actor.model.num_action_chunks``; eval-only runs use the Rollout
  model config instead.
* ``A`` is ``actor.model.action_dim`` and ``D = C * A``. With
  ``E = env.train.rollout_epoch``, a full trajectory normally has
  ``T = E * env.train.max_steps_per_rollout_epoch / C`` chunks.

Environment and policy messages
-------------------------------

``EnvOutput`` composes observations with one ``EnvTransition``; it does not
duplicate reward and boundary fields. ``PolicyInput`` carries observations and
may piggyback the preceding ``EnvPart``. Rollout sends only the action tensor
back to Env. The full ``PolicyOutput`` stays on the trajectory path because it
also contains log-probabilities, values, versions, and training inputs.

.. autoclass:: rlinf.data.schema.embodied_types.EnvOutput
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.EnvTransition
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyInput
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyOutput
   :members:
   :member-order: bysource

Trajectory parts
----------------

``PolicyPart`` and ``EnvPart`` are the only Actor channel input types. They are
matched by ``TrajectoryKey``. ``TrajectorySource`` preserves source size and
offset when channel routing splits a batch. Env creates an incomplete
``EnvPart`` after stepping; Rollout adds optional terminal inference data and
publishes the same type.

.. autoclass:: rlinf.data.schema.embodied_types.TrajectoryKey
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.TrajectorySource
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyPart
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.EnvPart
   :members:
   :member-order: bysource

Collection
----------

``TrajectoryPlan`` validates the configured output mode and derives rollout
geometry. ``TrajectoryCollector`` is the one public channel collector for every
embodied mode.

.. autoclass:: rlinf.data.schema.embodied_trajectory.TrajectoryMode
   :members:

.. autoclass:: rlinf.data.schema.embodied_trajectory.TrajectoryPlan
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_trajectory.TrajectoryCollector
   :members:
   :member-order: bysource

Actor output
------------

``TrajectoryStep`` resolves one joined policy/environment pair, including
reward, intervention, and transition semantics. ``Trajectory`` then stacks
steps into ``[T, B, ...]`` Actor tensors. Boundary and final-value sequences may
have ``T + E`` entries.

.. autoclass:: rlinf.data.schema.embodied_types.TrajectoryStep
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.Trajectory
   :members:
   :member-order: bysource
