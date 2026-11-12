Asynchronous Agentic RL
=======================

This page shows how to turn on asynchronous agentic RL in RLinf and which
settings you may need to change, then explains what happens underneath: how
rollout and training overlap, what the two admission limits control, and how
stale the training data gets. If you want a complete runnable configuration,
start from :doc:`Asynchronous Search-R1 Training
</rst_source/examples/agentic/searchr1/async>`; come back here when a default
needs changing and you want to know which one and why.

Turning It On
-------------

Asynchronous training is a runner switch on top of an agentic configuration,
plus two requirements on how the job is deployed. The actor and the rollout
must sit on separate GPUs, because both run at the same time, and the rollout
must use the SGLang server backend, because the weight sync pauses and resumes
generation through the server's HTTP routes. Starting from a synchronous
Search-R1 configuration, these are the lines that change:

.. code-block:: yaml

   runner:
     async_mode: True

   cluster:
     component_placement:
       actor: 0-7             # actor and rollout on separate GPUs
       rollout,reward: 8-15

   rollout:
     rollout_backend: sglang
     sglang:
       backend_type: server

   agentloop:
     task_name: searchr1      # required in async mode

The ``reward`` entry is there because ``component_placement`` always expects
one; Search-R1 computes its rewards inside the agent loop and launches no
reward worker. A configuration that misses one of these requirements is
rejected when the configuration is validated, before any worker starts.

Everything else has a default. The settings below are the ones you are most
likely to touch; the rest of the page explains when.

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Setting
     - Default
     - What it controls
   * - ``rollout_buffer.max_inflight_groups``
     - ``ceil(max_running_requests x instances x 1.2 / group_size)``
     - How much work the rollout engines get
   * - ``rollout_buffer.max_buffered_groups``
     - 2 × ``data.rollout_batch_size``
     - How old the training data can get
   * - ``agentloop.max_actor_accept_staleness``
     - unset (staleness is measured, not capped)
     - Drops groups older than this before training
   * - ``rollout_buffer.max_dropped_groups_per_step``
     - 10 × ``data.rollout_batch_size``
     - Stops the run when groups are dropped faster than training uses them
   * - ``rollout.pause_mode``
     - ``retract``
     - What happens to in-flight requests during a weight sync

The two metrics to watch first are ``rollout/buffer/admission_wait_inflight_limit``
and ``rollout/buffer/admission_wait_buffer_full``. They show which of the two
limits is holding generation back, and the sections below explain what to do
about each.

Why It Helps
------------

To see what the limits are for, start with what asynchrony changes. In a
synchronous agent run, the two expensive sides of the loop take turns being
idle. While the actor trains on a batch of trajectories, the rollout engines
have nothing to generate. When the actor is done, the engines fill up again,
and the actor then waits for the slowest trajectory of the batch before it can
start the next step. Multi-turn rollout makes this worse: a trajectory that
keeps calling tools holds up the whole batch at the end of every step.

Asynchronous training removes the waiting by decoupling the two sides. Prompts
flow into the rollout engines continuously, finished trajectories wait in a
buffer, and the actor takes the oldest ``rollout_batch_size`` groups whenever
it is ready for the next step.

The measured effect is close to an ideal overlap. On the reference run (steps
2–19, TensorBoard), the synchronous run's ``time/step`` of 278.6 s splits into
``time/rollout`` 181.9 + ``time/training`` 93.3 + ``time/sync_weights`` 1.4.
The asynchronous run's ``time/step`` of 139.0 s splits into ``time/training``
91.7 + ``time/wait_data`` 46.0 + ``time/sync_weights`` 1.3. Training takes
almost the same time in both; what disappears is the time each side spent
waiting for the other.

How It Fits Together
--------------------

Four components take part, and it helps to name them before following the data
between them:

- the **rollout worker**, which runs the SGLang server backend and holds the
  generation engines;
- the **agent loop**, which drives each multi-turn trajectory: it sends
  generation requests, parses tool calls, queries the tools, and appends the
  tool responses until the trajectory ends;
- the **buffer worker**, which owns the dataloader and decides which prompts
  to issue and which finished groups to train on;
- the **actor**, which trains and then syncs its new weights back to the
  rollout engines.

A prompt's path through them is short. The buffer worker reads it from the
dataloader, and when the admission limits allow, issues it to the agent loop
through the ``dataloader_channel``. The agent loop runs the group's
trajectories against the rollout engines and returns the finished group
through the ``agent_output_channel``. The buffer worker holds finished groups
in its buffer and, at every step, hands the oldest ``rollout_batch_size`` of
them to the actor.

The buffer worker is what keeps this overlap under control. Because it, not
the runner, owns the dataloader, it can keep issuing prompts while the actor
trains. Because it counts both the groups in flight and the groups in its
buffer, it bounds how much data piles up between the two sides. And because it
hands off the oldest groups first, the data the actor sees ages at a rate the
limits determine.

The Two Limits
--------------

The buffer worker issues a new prompt only while both limits hold. They
answer different questions: one sets how much work the engines get, the other
how old the training data may become. Admission re-checks them whenever
capacity may have freed up: when a finished group arrives, when a group is
dropped, and when a batch is handed to the actor.

The In-Flight Limit: Engine Load
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first limit keeps the number of in-flight groups (issued, not yet
returned) below ``rollout_buffer.max_inflight_groups``. The default converts the
engines' request capacity into query groups,
``ceil(max_running_requests x instances x 1.2 / group_size)``, with a 1.2
margin for groups whose trajectories are waiting on a tool or on the slowest
member of the group.

Raising it helps only while the engines still have idle capacity. Past the
point where they are busy, more in-flight groups just queue inside the engines
and age the data. On the GLM reference run, raising ``max_inflight_groups``
from 77 to 160 moved the running requests per DP rank from about 75 to 128
(with another 33–78 queued), left the throughput unchanged, and raised the
mean staleness from about 1.3 to about 2.0: the GPUs were already busy at 75
requests. The plots are on the :doc:`Search-R1 async page
</rst_source/examples/agentic/searchr1/async>` under "Tuning the Limits".

The Buffer Limit: Data Age
~~~~~~~~~~~~~~~~~~~~~~~~~~

The second limit keeps the in-flight groups *plus* the groups already in the
buffer below ``rollout_buffer.max_buffered_groups``, 2 × ``rollout_batch_size``
by default. In-flight groups count because a group in flight now is a buffered
group later: once its trajectories finish, it enters the buffer
unconditionally. A cap on the buffer alone would leave the in-flight part
unbounded, and when training is the bottleneck the buffer would keep growing.
That misconfiguration was measured once, with the buffer reaching 338 and 369
groups and the staleness reaching 8. Counting in-flight groups keeps the buffer
within ``max_buffered_groups``, and that is what bounds the staleness.

How to size it depends on which side of the loop is the bottleneck. When
generation is the bottleneck, the actor is always waiting and the buffer
empties as fast as it fills; keep ``max_buffered_groups`` at least
``max_inflight_groups + rollout_batch_size``, or the buffered groups take
budget from the in-flight ones at the end of each step and the engines run
short of work. When training is the bottleneck, the buffer is what keeps
generation busy; keep it at least 2 × ``rollout_batch_size`` so the next batch
generates while the current one trains. At 1 ×, the next batch can only start
after the current one is handed off, and every step waits for a whole batch,
slowest group included, to finish generating. On a scaled-down Qwen2.5-1.5B
setup with ``rollout_batch_size`` 64, the median step time was about 11 s at
1×B against 3.94 s at 2×B, and the staleness rose from about 1 to a peak mean
of 2.09.

Which Limit Is Binding
~~~~~~~~~~~~~~~~~~~~~~

You rarely need to work out the bottleneck by hand.
``rollout/buffer/admission_wait_inflight_limit`` and
``rollout/buffer/admission_wait_buffer_full`` report the fraction of each step
during which admission was blocked by each limit. A high
``admission_wait_inflight_limit`` means the engines are as loaded as you
allowed; check GPU utilization before raising the limit. A high
``admission_wait_buffer_full`` means training is the bottleneck and the buffer
limit is what bounds the staleness.

Staleness
---------

By default staleness is measured, not capped. ``agentloop.max_actor_accept_staleness``
is unset, and the bound comes from the buffer limit above: when training is the
bottleneck, the oldest buffered group has waited about
``max_buffered_groups / rollout_batch_size`` steps.

The ``rollout/data_staleness/*`` metrics report it. For each group trained in a
step, staleness is the actor's weight version at the start of the step minus
the earliest version the group was generated with, reported as the mean, the
max, and per-value ratios. It splits into two parts:
``rollout/version_span/mean``, the versions that passed while the group itself
was generating, and ``rollout/queue_steps/mean``, the versions it waited in the
buffer after finishing.

Set ``agentloop.max_actor_accept_staleness`` when you need a hard cap. Groups
older than the cap are dropped before training, and the drop frees admission
capacity, so another prompt is issued in their place; the dropped prompt comes
back in a later epoch. The cost is the generation already spent on the dropped
group, so put the cap above what the buffer limit produces on its own (the
``max_buffered_groups / rollout_batch_size`` steps above). A cap below that
drops the oldest groups of every batch instead of the rare outliers, and the
configuration check warns about it.

Weight Sync During Generation
-----------------------------

After each training step, the new weights go into the rollout engines while
generation is in full flight. ``rollout.pause_mode`` decides what happens to the
requests that are mid-generation during the sync:

- ``retract`` (the default) returns the in-flight requests to the engine's
  queue and syncs the weights. The tokens already generated are kept; the KV
  cache is flushed with the old weights and recomputed after the pause, so the
  cost of the switch is one re-prefill per retracted request.
- ``in_place`` pauses the engine's event loop: the running requests keep their
  KV cache and resume decoding with the new weights once the sync completes.
- ``drain`` waits for all in-flight requests to finish, then syncs, then lets
  new requests in.

The default is ``retract`` for a reason specific to multi-turn rollout. With
``in_place`` the engine cannot flush its cache while requests are running, so a
request that started before the sync and continues afterwards reuses its
old-version prefix KV cache against the new weights, and that staleness is
invisible to the scheduler. ``retract`` pays a re-prefill per request and in
exchange gives every request a clear version boundary.

What Asynchrony Cannot Fix
--------------------------

Asynchrony removes the idle time between generation and training; it does not
make either side faster. When generation is the bottleneck, the step time is
set by the rollout throughput however you tune the limits: the GLM reference
configuration generates about 250k tokens per step at about 2000 tok/s, which
puts a floor of roughly 125–140 s on the step, while training needs only about
85 s. To go faster you have to change the GPU split between the actor and the
rollout, giving the rollout more capacity or generating with a smaller model.

Limitations
-----------

Asynchronous agentic RL runs only on the combination it was built for: a
Megatron actor, the multi-turn agent loop, the SGLang server backend, and a
disaggregated placement of actor and rollout. There is no off-policy
correction: the rollout logprob is used directly as the PPO old logprob, so
the staleness bounds described above are what keeps the training data close to
the current policy.
