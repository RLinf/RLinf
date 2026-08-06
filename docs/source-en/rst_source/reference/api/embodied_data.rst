Embodied Data Interface
========================

This section describes the core data structures used during rollout and training
in embodied settings: `EnvOutput`, `PolicyOutput`, `ChunkStepResult`,
`EmbodiedTrajectoryBuilder`, and `Trajectory`. Together, they connect environment
outputs, policy communication, chunk-step accumulation, trajectory construction,
and training batches.

Relationships
-------------

- `EnvOutput`: raw environment outputs per chunk step (obs, reward, done, etc.).
- `PolicyOutput`: policy/rollout-worker outputs for one communication round
  (actions, log-probabilities, values, etc.).
- `ChunkStepResult`: env-side per-chunk package combining policy outputs with
  reward and termination signals.
- `EmbodiedTrajectoryBuilder`: accumulates chunk-step results and transitions.
- `Trajectory`: aggregated trajectory tensors (typically `[T, B, ...]`).

Typical flow::

   EnvOutput -> PolicyOutput -> ChunkStepResult -> EmbodiedTrajectoryBuilder -> Trajectory

`EmbodiedTrajectoryBuilder.to_splited_trajectories()` can split trajectories along the
batch dimension for Channel distribution to multiple Actor/Trainer workers.

EnvOutput
---------

`EnvOutput` describes environment-side outputs, including observations and
episode-termination signals. During initialization, tensors are moved to CPU
and made contiguous.

.. autoclass:: rlinf.data.schema.embodied_types.EnvOutput
   :members:
   :member-order: bysource

PolicyOutput
------------

`PolicyOutput` is the message sent from the rollout worker to the env worker for
one communication round. It carries actions and optional training signals such
as log-probabilities, values, intervene flags, and forward inputs.

.. autoclass:: rlinf.data.schema.embodied_types.PolicyOutput
   :members:
   :member-order: bysource

ChunkStepResult
---------------

`ChunkStepResult` represents per-step inference results and training signals,
including actions, log-probabilities, value estimates, and extra forward inputs.
Tensors are moved to CPU on initialization.

.. autoclass:: rlinf.data.schema.embodied_types.ChunkStepResult
   :members:
   :member-order: bysource

EmbodiedTrajectoryBuilder
--------------------------

`EmbodiedTrajectoryBuilder` accumulates chunk-step results and transitions during
rollout, and provides conversion utilities:

- `append_step_result()`: append chunk-step results
- `append_transitions()`: append current/next transition observations
- `to_trajectory()`: concatenate into trajectory tensors
- `to_splited_trajectories()`: split trajectories along the batch dimension

.. autoclass:: rlinf.data.schema.embodied_trajectory_builder.EmbodiedTrajectoryBuilder
   :members:
   :member-order: bysource

Trajectory
----------

`Trajectory` is the final trajectory representation for training. It includes
actions, rewards, termination flags, observations, and model forward inputs.
The typical tensor shape is `[T, B, ...]`, where **T is the chunk-step count**
and **B is the number of parallel environments** (batch dimension).

.. autoclass:: rlinf.data.schema.embodied_types.Trajectory
   :members:
   :member-order: bysource
