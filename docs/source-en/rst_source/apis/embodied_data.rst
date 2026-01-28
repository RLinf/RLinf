Embodied Data Interface
========================

This section describes the core data structures used during rollout and training
in embodied settings: `EnvOutput`, `ChunkStepResult`, `EmbodiedRolloutResult`,
and `Trajectory`. Together, they connect environment outputs, chunk-step
accumulation, trajectory construction, and training batches.

Relationships
-------------

- `EnvOutput`: raw environment outputs per chunk step (obs, reward, done, etc.).
- `ChunkStepResult`: model inference outputs and reward signals per chunk step.
- `EmbodiedRolloutResult`: accumulates chunk-step results and transitions.
- `Trajectory`: aggregated trajectory tensors (typically `[T, B, ...]`).

`EmbodiedRolloutResult.to_splited_trajectories()` can split trajectories along the
batch dimension for Channel distribution to multiple Actor/Trainer workers.

EnvOutput
---------

`EnvOutput` describes environment-side outputs, including observations and
episode-termination signals. During initialization, tensors are moved to CPU
and made contiguous.

.. autoclass:: rlinf.data.embodied_io_struct.EnvOutput
   :members:
   :member-order: bysource

ChunkStepResult
---------------

`ChunkStepResult` represents per-step inference results and training signals,
including actions, log-probabilities, value estimates, and extra forward inputs.
Tensors are moved to CPU on initialization.

.. autoclass:: rlinf.data.embodied_io_struct.ChunkStepResult
   :members:
   :member-order: bysource

EmbodiedRolloutResult
---------------------

`EmbodiedRolloutResult` accumulates chunk-step results and transitions during
rollout, and provides conversion utilities:

- `append_step_result()`: append chunk-step results
- `append_transitions()`: append current/next transition observations
- `to_trajectory()`: concatenate into trajectory tensors
- `to_splited_trajectories()`: split trajectories along the batch dimension

.. autoclass:: rlinf.data.embodied_io_struct.EmbodiedRolloutResult
   :members:
   :member-order: bysource

Trajectory
----------

`Trajectory` is the final trajectory representation for training. It includes
actions, rewards, termination flags, observations, and model forward inputs.
The typical tensor shape is `[T, B, ...]`, where **T is the chunk-step count**
and **B is the number of parallel environments** (batch dimension).

.. autoclass:: rlinf.data.embodied_io_struct.Trajectory
   :members:
   :member-order: bysource
