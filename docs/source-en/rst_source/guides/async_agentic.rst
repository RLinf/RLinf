Asynchronous Agentic RL
=======================

This page explains how asynchronous agentic RL works in RLinf: how rollout
and training overlap, what the two admission limits control, and how to size
them. If you want a runnable configuration rather than the mechanism, start
from :doc:`Asynchronous Search-R1 Training
</rst_source/examples/agentic/searchr1/async>` and come back here when the
limits need tuning.

Why It Helps
------------

In a synchronous agent run, the two expensive sides of the loop idle in
turns. While the actor trains on a batch of trajectories, the rollout
engines sit with nothing to generate. When the actor is done, the engines
fill up again — but the actor then waits for the slowest trajectory of the
batch before it can start the next step. Multi-turn agentic rollout makes
this worse: a trajectory that keeps calling tools holds the whole batch
hostage at the end of each step.

Asynchronous training removes the mutual waiting by decoupling the two
sides. Prompts flow into the rollout engines continuously; finished
trajectories wait in a buffer; the actor consumes the oldest
``rollout_batch_size`` groups whenever it is ready for the next step.
Generation and training then run at their own pace instead of taking turns.

The measured effect is close to the ideal overlap. On the reference run
(steps 2–19, TensorBoard), the synchronous run's ``time/step`` of 278.6 s
splits into ``time/rollout`` 181.9 + ``time/training`` 93.3 +
``time/sync_weights`` 1.4. The asynchronous run's ``time/step`` of 139.0 s
splits into ``time/training`` 91.7 + ``time/wait_data`` 46.0 +
``time/sync_weights`` 1.3. Training takes almost the same time on both
sides; what disappears is the time each side spent waiting for the other.

How It Fits Together
--------------------

Four components cooperate, connected by two channels.

The rollout worker runs the SGLang server backend and holds the generation
engines. The agent loop drives the multi-turn interaction: it sends
generation requests, parses tool calls, queries the tools, and appends the
tool responses, until each trajectory finishes. The
``RolloutBufferWorker`` owns the dataloader and the scheduling state. It
reads prompts from the ``dataloader_channel``, decides which of them to
issue based on the admission limits, collects finished groups from the
``agent_output_channel``, and at every step hands the oldest
``rollout_batch_size`` groups to the actor. The actor trains on those
groups and syncs the new weights back to the rollout engines.

The buffer worker is the hub that makes the overlap safe. Because it — not
the runner — owns the dataloader, it can keep issuing prompts while the
actor trains. Because it counts both in-flight and buffered groups, it
bounds how much data can accumulate between the two sides. And because it
hands off the *oldest* groups first, the data the actor sees ages in a
controlled way, which brings us to the two limits.

Admission: Two Limits
---------------------

The buffer worker admits a new prompt only when both conditions hold, and
each condition answers a different question.

The first condition — the number of in-flight groups stays below
``rollout_buffer.max_inflight_groups`` — decides how much work you give the
rollout engines. Its default is derived as
``ceil(max_running_requests x instances x 1.2 / group_size)``: the engines'
request capacity converted to query groups, with a 1.2 margin for groups
waiting on tools or on their slowest member.

The second condition — the number of in-flight groups *plus* the groups
already sitting in the buffer stays below
``rollout_buffer.max_buffered_groups`` — decides how old the training data
can get. Its default is 2 × ``rollout_batch_size``.

The second limit counts the in-flight groups, not just the buffered ones.
This is deliberate. A group that is in flight today is a buffered group
tomorrow: once its trajectories finish, it enters the buffer
unconditionally. If admission counted only the buffered groups, the
in-flight portion would be unconstrained, and whenever training became the
bottleneck the buffer would keep growing without bound — one measurement of
exactly that misconfiguration showed the buffer piling up to 338 and 369
groups and the data staleness reaching 8. Counting in-flight groups toward
the limit keeps the buffer within ``max_buffered_groups``, which in turn
bounds the staleness.

Admission re-checks whenever capacity might have freed up: when a result
arrives, when a group is dropped, and when a batch is handed off to the
actor.

Sizing the Limits
-----------------

Which limit you should size carefully depends on which side of the loop is
the bottleneck.

When generation is the bottleneck, the actor is always starved and the
buffer drains as fast as it fills. Here the constraint on
``max_buffered_groups`` is arithmetic: keep it at least
``max_inflight_groups + rollout_batch_size``. Below that, the buffer and
the in-flight quota compete for the same budget, and at the end of each
step the buffered groups crowd out in-flight ones, starving the engines
that were supposed to be saturated.

When training is the bottleneck, generation outruns consumption and the
buffer is what keeps the pipeline moving. Keep ``max_buffered_groups`` at
least 2 × ``rollout_batch_size`` so the next batch can generate while the
current one trains. At 1 ×, the next batch can only start generating after
the current one is handed off, and every step waits for a whole batch —
including its slowest group — to finish generating: generation and training
serialize again, just with extra steps. Measured on Qwen with
``rollout_batch_size`` 64, the median step time at 1×B is about 11 s
against 3.94 s at 2×B, and the price of the faster option is staleness
rising from about 1 to a peak mean of 2.09.

You rarely have to guess which side you are on: read
``rollout/buffer/admission_wait_inflight_limit`` and
``rollout/buffer/admission_wait_buffer_full``, the fraction of the step
window during which admission was blocked by each limit. Whichever is high
is the limit that is currently binding.

And the in-flight limit is not a bigger-is-better knob. Raising it past the
point where the engines are busy only queues more prompts inside them and
ages the data. On the GLM reference run, raising ``max_inflight_groups``
from 77 to 160 moved the running requests per DP rank from about 75 to 128
(with another 33–78 queued), left the throughput unchanged, and raised the
mean staleness from about 1.3 to about 2.0 — the GPUs were already busy at
75 requests. The measured plots behind this are on the
:doc:`Search-R1 async page </rst_source/examples/agentic/searchr1/async>`
under "Tuning the Limits".

Staleness
---------

Because the actor trains on buffered data while the engines generate with
newer weights, each trained group is a few weight versions old. The
``rollout/data_staleness/*`` series measure exactly that: for the groups of
each step, the actor's weight version at step start minus the earliest
version a group was generated with — reported as the mean, the max, and
per-value ratios. It decomposes into ``rollout/version_span/mean`` (how
many versions passed while the group itself was still generating) and
``rollout/queue_steps/mean`` (how long its finished groups waited in the
buffer before training).

A bounded staleness is the price of the overlap, and the buffer limit is
what bounds it. If you want a hard bound, ``agentloop.max_actor_accept_staleness``
drops groups staler than the threshold before training; the drop frees
admission capacity for other prompts, and the dropped prompt itself comes
around again in a later epoch. The cost is the generation compute already
spent on the dropped groups, so set the threshold above what the buffer
limit itself produces (about ``max_buffered_groups / rollout_batch_size``
steps of waiting when training is the bottleneck), or you will drop the
oldest groups systematically rather than the exceptional ones. The actor
itself only has to wait for data when over-stale drops prevent the buffer
from filling a whole batch — with the limits sized as above, admission
blocks long before that happens.

Weight Sync and ``pause_mode``
------------------------------

After each training step, the new weights are synced into the rollout
engines while generation is in full flight. ``rollout.pause_mode`` decides
what happens to the requests that are mid-generation during the sync:

- ``retract`` (the default) returns the in-flight requests to the engine's
  queue and syncs the weights. The tokens already generated are kept; the
  KV cache is flushed with the old weights and recomputed after the pause,
  so the cost of the switch is one re-prefill per retracted request.
- ``in_place`` pauses the engine's event loop: the running requests stay in
  place with their KV cache and resume decoding with the new weights once
  the sync completes.
- ``drain`` waits for all in-flight requests to finish, then syncs, then
  lets new requests in.

The default is ``retract`` for a reason specific to multi-turn rollout:
with ``in_place``, the engine cannot flush its cache while requests are
running, so a request that started before the sync and continues afterwards
reuses its old-version prefix KV cache against the new weights — and the
resulting staleness has no bound the scheduler can see. ``retract`` pays a
re-prefill per request instead, buying a version boundary the rest of the
system can reason about.

What Asynchrony Cannot Fix
--------------------------

Asynchrony removes the idle time between generation and training; it does
not make either side faster. When generation is the bottleneck, the step
time is set by the rollout throughput no matter how you tune the limits: the
GLM reference configuration generates about 250k tokens per step at about
2000 tok/s, which puts a floor of roughly 125–140 s on the step, while
training only needs about 85 s. Admission tuning cannot lower that floor.
To go faster you have to change the GPU split between the actor and the
rollout — more rollout capacity, or a smaller model to generate.

Limitations
-----------

Asynchronous agentic RL currently requires the combination it was built on:
a Megatron actor, the multi-turn agent loop, the SGLang server backend, and
a disaggregated placement of actor and rollout. The trained policy takes
the data as it comes: there is no off-policy correction, the rollout
logprob is used directly as the PPO old logprob, and the staleness bounds
described above are the guardrail that keeps the gap small.
