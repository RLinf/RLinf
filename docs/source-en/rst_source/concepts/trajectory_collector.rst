Understand the Trajectory Collector
===================================

Use this page to trace embodied rollout data from one action chunk to the
Actor-facing training batch. Focus on the fields your algorithm produces and
consumes; the Collector owns routing, matching, and completion state.

Core Model
----------

One action chunk is split into two independently produced parts:

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Part
     - Owner
     - Core Data
   * - ``PolicyPart``
     - Policy inference
     - Observation, action, log-probability, value, model version, and retained
       forward inputs.
   * - ``EnvPart``
     - Environment interaction
     - Reward, termination state, intervention result, and optional boundary
       observation or compact RLT features.

Both parts carry the same ``TrajectoryKey``:

.. code-block:: text

   (step_id, epoch_id, env_rank, stage_id, chunk_id)

The key identifies the chunk. ``TrajectorySource`` adds batch size and offset
when routing splits or merges a source batch.

The Actor Channel sees this flow:

.. code-block:: text

   PolicyPart -----------+
                         |
                         v
                   join by key
                         |
                         v
   EnvPart --------------+----> accumulate ----> flush ----> Actor output

The Environment does not write the Actor Channel directly. It attaches an
``EnvPart`` to the next ``PolicyInput``. Rollout adds optional terminal
inference data and publishes that same ``EnvPart`` so the Actor Channel has one
producer and one ordering boundary.

Complete One Action Chunk
-------------------------

Follow one normal inferred chunk:

1. Env sends ``PolicyInput`` with the current observation and a new
   ``TrajectoryKey``.
2. Rollout runs policy inference, returns the action tensor to Env, and
   immediately publishes a ``PolicyPart`` containing the full ``PolicyOutput``
   to the Actor Channel.
3. Env executes the action chunk and records rewards, boundary flags, the
   post-action observation, and intervention data.
4. Env stores that ``EnvTransition`` in ``EnvPart``. The accompanying
   ``PolicyInput.obs`` normally carries the same post-action observation, so
   ``EnvPart.next_obs`` is set only when a terminal or decoupled boundary needs
   an explicit override. Chunk zero also carries ``initial_transition``.
5. Rollout resolves the effective next observation from the request or its
   override and completes the ``EnvPart``. When requested, it runs terminal
   inference and adds the final value. RLT keeps only the next-state features
   required by the algorithm, not the complete next forward-input dictionary.
6. ``ChunkJoiner`` matches both parts by key. Arrival order does not matter.
7. ``TrajectoryCollector`` passes the joined chunk to the output strategy
   selected by ``TrajectoryPlan``.

The temporal meaning is:

.. code-block:: text

   initial_transition             EnvPart.transition
         |                                |
         v                                v
      state_t ---- PolicyPart.action ----> state_t+1

Only chunk zero owns ``initial_transition``. This keeps boundary sequences
aligned without a separate start event.

From Chunks to Actor Output
---------------------------

The Collector performs three core operations.

Join
~~~~

``ChunkJoiner`` restores routed source fragments and stores policy and
environment parts until both sides of a key are present. A joined chunk contains
one source id, one ``PolicyPart``, and one ``EnvPart``.

Accumulate
~~~~~~~~~~

``TrajectoryStep.from_parts`` converts each joined pair into one complete
training step. The embodied data types own reward composition, intervention
updates, transition extraction, and CPU conversion. ``TrajectoryAccumulator``
only retains these complete steps in rollout order.

``Trajectory.from_steps`` later materializes the temporal layout. Action-owned
fields normally have one entry per chunk. Boundary fields include the state
before the first action, and value fields may include the final bootstrap value:

.. code-block:: text

   actions:      [a0, a1, ...]
   rewards:      [r0, r1, ...]
   dones:        [d_before, d0, d1, ...]
   prev_values:  [V(s0), V(s1), ..., V(s_final)]

This separation keeps field-specific behavior next to ``EnvTransition``,
``TrajectoryStep``, and ``Trajectory`` instead of embedding it in Collector or
accumulator control flow.

Flush
~~~~~

An output strategy defines the smallest completion scope that can be emitted.
After a scope completes, it materializes contiguous tensors, removes its
accumulator state, and returns queue-ready items.

Output Modes
------------

.. list-table::
   :header-rows: 1
   :widths: 20 32 48

   * - Mode
     - Completion Scope
     - Actor Output
   * - ``rollout``
     - One training step for one logical source
     - ``Trajectory`` shards on the default key. Sources flush independently.
   * - ``pipeline``
     - One ``(step_id, epoch_id)`` across all sources
     - Preprocessed, advantage-ready micro-batches keyed to Actor ranks.
   * - ``lerobot``
     - One training step for one logical source
     - Completed episode shards. In-progress episode state persists across
       training steps.

Synchronous, asynchronous, and decoupled runners use the same two parts and the
same Collector. Their service loops differ, but the chunk data contract does
not.

What Algorithm Developers Need to Own
-------------------------------------

Start from the training tensor your algorithm needs, then follow its ownership
path.

.. list-table::
   :header-rows: 1
   :widths: 27 31 42

   * - Algorithm Need
     - Produce It In
     - Data-Type Path
   * - Policy statistic such as log-probability or value
     - ``PolicyOutput`` in Rollout
     - Resolve it in ``TrajectoryStep.from_parts`` and materialize it with
       ``Trajectory.from_steps``.
   * - Environment or reward-model signal
     - ``EnvTransition`` in Env
     - Combine sources in ``EnvTransition.compute_rewards``.
   * - Current and next observation
     - ``PolicyPart.obs`` and the next ``PolicyInput.obs`` or boundary override
     - Rollout resolves the effective next observation before publishing
       ``EnvPart``. Retain it only when ``rollout.collect_transitions`` is
       enabled.
   * - Executed intervention action
     - ``EnvTransition.intervene_actions`` and ``intervene_flags``
     - Apply it through ``TrajectoryStep.apply_interventions``.
   * - RLT transition feature
     - ``PolicyOutput.forward_inputs`` and compact ``EnvPart.next_rlt_obs``
     - Rollout extracts only ``z_rl``, ``proprio``, and ``ref_chunk`` for the
       next state. ``TrajectoryStep`` applies executed interventions before
       retaining the transition.
   * - Pipeline training target
     - Algorithm advantage/loss configuration
     - Compute advantages and returns before Actor-specific micro-batch packing.

For a new algorithm, check these five points:

1. Add the field to ``PolicyOutput`` or ``EnvTransition`` according to who
   owns it.
2. Populate it in Rollout or Env and update that type's ``split``/``merge``
   methods when the field follows the batch dimension.
3. Resolve cross-part semantics in ``TrajectoryStep.from_parts``.
4. Materialize temporal fields in ``Trajectory.from_steps`` or the
   mode-specific Actor output.
5. Keep pipeline preprocessing aligned with the non-pipeline Actor path.

Most algorithms do not need to change ``TrajectoryKey``, ``TrajectorySource``,
fragment restoration, dispatcher selection, or completion scopes. Change those
only when the algorithm changes data ownership or when a batch is allowed to
flush.

Minimal End-to-End Flow
-----------------------

With one Env source, one chunk, one rollout epoch, and one Actor:

1. Bootstrap produces the pre-action observation and boundary state.
2. Rollout publishes one ``PolicyPart`` and sends its action to Env.
3. Env executes the action and attaches one ``EnvPart`` to the next request.
4. Rollout completes and publishes that ``EnvPart`` with the same key.
5. The Joiner creates one complete chunk.
6. ``TrajectoryStep`` resolves the pair; the Accumulator retains that complete
   step without inspecting its fields.
7. The rollout scope is complete, so the Collector emits one ``Trajectory`` on
   the default queue key.
8. Actor converts the received trajectory into a training batch, then computes
   advantages and returns in the non-pipeline path.

With more chunks, steps 2–6 repeat before flush. With more independent sources,
rollout mode may flush each source as soon as it completes. Pipeline mode waits
for every source in the epoch because routing and optional advantage
normalization operate on Actor-specific batches.

Read the Interfaces
-------------------

Use :doc:`channel` to understand Collector and Dispatcher execution. Use
:doc:`../reference/api/embodied_data` for exact data-class and Collector APIs.
The complete collector and accumulation implementation lives in
``rlinf/data/schema/embodied_trajectory.py``.
