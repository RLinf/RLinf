Embodied Training Pipeline Mode
================================

This page introduces RLinf's ``use_training_pipeline`` mode for synchronous
embodied training. Its goal is to start actor training as early as possible so
that the system can take advantage of two kinds of overlap:

- overlap between ``env -> actor`` communication and actor training
- when ``rollout_epoch > 1``, overlap between later ``rollout <-> env`` data
  generation and actor training

As a result, this mode is especially useful in multi-node, low-bandwidth, or
long-rollout scenarios where actor would otherwise spend a noticeable amount of
time waiting.

This feature currently targets the synchronous embodied training path used by
``examples/embodiment/train_embodied_agent.py``.


Why This Mode Exists
------------------------------

The default synchronous embodied training flow can be simplified as:

1. env and rollout finish one full rollout round.
2. env sends the rollout data to actor as a whole.
3. actor waits for the full rollout batch, then computes advantages and starts training.

This flow is simple and easy to reason about, but when ``env -> actor``
communication is slow, actor can spend a long time waiting for data.

The core idea of ``use_training_pipeline`` is to send trainable micro-batches
to actor as early as possible, so actor can start training while the remaining
data is still being transferred. When ``rollout_epoch > 1``, actor training can
also overlap with data generation in later rollout epochs.


How It Works
------------------------------

This mode does not change the training entrypoint, and it does not change the
basic env-rollout interaction pattern. The main change is how data is organized
on the ``env -> actor`` path.

The high-level flow is:

1. env and rollout run one normal ``rollout_epoch``.
2. Once the current ``rollout_epoch`` finishes, env flushes immediately instead
   of waiting for the whole step to finish.
3. env splits data according to the actor's local ``micro_batch_size``.
4. env computes the required ``advantages`` / ``returns`` for each split and
   converts it into a train-ready micro-batch.
5. actor starts training as soon as one micro-batch arrives.
6. If ``update_epoch > 1``, only the first update epoch overlaps with transfer;
   later epochs are replayed locally after all data has arrived.

In practice, this mode mainly optimizes:

- ``env -> actor`` communication
- the first actor update epoch
- when ``rollout_epoch > 1``, actor training together with env-rollout data
  generation in later rollout epochs

It is **not the same thing as** ``rollout.pipeline_stage_num``, and it does not
optimize the ``rollout <-> env`` inference pipeline.


Two Types of Overlap
------------------------------

This mode actually has two different layers of overlap:

1. Overlap between ``env -> actor`` communication and actor training

   This is the most direct layer. While env is still sending multiple pipeline
   micro-batches for the current ``rollout_epoch``, actor can already start
   training on the earlier micro-batches it has received.

2. Overlap between later ``rollout <-> env`` interaction and actor training

   When ``rollout_epoch > 1``, env flushes immediately after each
   ``rollout_epoch`` finishes. This means actor can start training on data from
   rollout epoch ``k`` while env and rollout are already generating data for
   rollout epoch ``k+1`` and later epochs.

This second layer matters because the mode is not only hiding a chunk of network
transfer time. It can also overlap actor training with ongoing env-rollout data
generation in later rollout epochs.


How To Enable It
------------------------------

Enable it in your embodied training config:

.. code-block:: yaml

   runner:
     use_training_pipeline: true
     pipeline_schedule: global_batch

The most common way is either to set it directly in YAML or override it on the
command line:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-path examples/embodiment/config \
     --config-name <your_config> \
     +runner.use_training_pipeline=true \
     +runner.pipeline_schedule=global_batch

``runner.pipeline_schedule`` currently supports two values:

- ``global_batch``: schedule work at global-batch granularity, closer to the
  original optimizer-step boundary
- ``micro_batch``: schedule work at micro-batch granularity, more aggressive
  and better at filling pipeline bubbles


Current Support
------------------------------

``use_training_pipeline`` currently supports these ``adv_type`` values:

- ``gae``
- ``raw``
- ``grpo``

``use_training_pipeline`` currently does **not** support:

- ``grpo_dynamic``

Common pairings are:

- ``gae``: usually paired with ``actor_critic`` or ``decoupled_actor_critic``
- ``raw``: usually paired with ``actor``
- ``grpo``: usually paired with ``actor``

This mode is also not recommended for:

- ``embodied_sac`` / ``embodied_dagger`` / ``embodied_nft`` and other non-current
  synchronous on-policy paths
- custom training flows that rely on full global shuffle or a full global
  normalization stage


Limitations And Notes
------------------------------

1. ``runner.pipeline_schedule`` must be set explicitly.

   The currently supported values are ``global_batch`` and ``micro_batch``.

2. Extra global ``normalize_advantages`` is not used for ``gae`` and ``raw``.

   The goal of this mode is to start training early, so actor does not wait for
   the full rollout to arrive just to run another global normalization pass.
   For ``grpo``, the group-relative normalization behavior is still handled by
   the advantage definition itself.

3. ``shuffle_rollout`` is not an exact full-rollout global shuffle in this mode.

   With ``pipeline_schedule=global_batch``, shuffle only happens among ready
   global batches. With ``pipeline_schedule=micro_batch``, shuffle only happens
   among ready micro-batches. This is closer to schedule-local reordering than
   to the exact full-step global shuffle used by the default synchronous path.

4. Gains usually decrease when ``update_epoch > 1``.

   Only the first update epoch participates in this pipeline overlap. The
   remaining ``update_epoch - 1`` epochs are replayed locally after all data has
   arrived, so they no longer overlap with ``env -> actor`` communication or
   with data generation in later rollout epochs.

5. This mode usually shows clearer benefit when ``rollout_epoch > 1``.

   The current implementation flushes to actor after each ``rollout_epoch``
   instead of waiting for the full step. When ``rollout_epoch > 1``, actor can
   usually receive its first trainable micro-batch earlier and build a more
   effective overlap window.

6. ``micro_batch_size`` must align with the rollout shape.

   In the current implementation, env flushes once per ``rollout_epoch``, so one
   actor-local pipeline micro-batch roughly corresponds to:

   .. code-block:: text

      env_batch_per_pipeline_micro_batch
      = actor.micro_batch_size / n_train_chunk_steps

   If you use group-based advantages such as ``grpo``, this
   ``env_batch_per_pipeline_micro_batch`` must also be divisible by
   ``group_size`` so that a group is not split across different pipeline batches.


Difference From ``pipeline_stage_num``
---------------------------------------

These two features solve different problems:

``runner.use_training_pipeline``
  Optimizes overlap between ``env -> actor`` communication and actor training.

``rollout.pipeline_stage_num``
  Optimizes the interaction and inference pipeline between rollout and env.

They can coexist, but they are not the same mechanism and do not replace each other.


When It Is Worth Trying
------------------------------

This mode is a good fit when:

- actor and env are placed on different machines
- network bandwidth is limited and ``env -> actor`` transfer is noticeably slow
- ``rollout_epoch > 1``

If your main bottleneck is neither ``env -> actor`` communication nor the
synchronous waiting between later rollout epochs and actor, or if your training
setup strongly depends on full global shuffle / full global normalization
semantics, the default synchronous path is usually simpler and easier to keep
aligned with existing experiments.
