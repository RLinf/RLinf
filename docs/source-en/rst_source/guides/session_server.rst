Session Server
==============

The session server collects agent trajectories on the rollout nodes and
assembles them into training data. Agents send their generate requests to it.
It forwards each request to an inference engine and records every turn.

It also provides two features:

- **Separate control and data planes.** The buffer worker schedules groups
  using only their metadata. The trajectories stay on the rollout nodes until
  the actor fetches them.
- **Routing through an SGLang router.** You can use two of the router's own
  policies: ``manual`` pins each group to one engine, and ``cache_aware``
  routes by prefix affinity.

The session server works only with the asynchronous runner. If you have not
used that runner, read :doc:`Asynchronous Agentic RL <async_agentic>` first.
For a full recipe, see the :ref:`Session Server Path
<searchr1-async-session-server>` of the asynchronous Search-R1 example.

Enable the Session Server
-------------------------

Add two blocks to an asynchronous agentic config:

.. code-block:: yaml

   session_server:
     enable: true          # agents talk to session servers
     num_servers: 1        # session servers, spread over the rollout nodes
     payload_shards: 1     # trajectory stores per session server

   rollout:
     router:
       policy: manual      # or cache_aware

Only ``enable`` is required; the other values above are the defaults.

At launch, the entry script starts three components on the rollout nodes:

- the **session servers**, HTTP services that the agent loop talks to;
- one **payload channel** per shard, the store that holds the recorded
  trajectories;
- one **SGLang router**, which every engine registers with.

Example configuration:

.. code-block:: text

   examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml

Requirements
------------

The session server path requires:

- ``runner.async_mode: True`` and a Megatron actor;
- ``rollout.sglang.backend_type: server``;
- the Search-R1 agent (``reward.reward_type: searchr1``);
- ``rollout.pause_mode`` of ``retract`` or ``in_place``, not ``drain``;
- ``cluster.auto_scheduler`` off;
- the ``sglang_router`` package.

Config validation checks each of these at startup.

How Trajectories Are Collected
------------------------------

Each group goes through four steps.

1. **Open.** The agent loop opens one session per trajectory. All sessions of
   a group go to the same session server.
2. **Generate.** For each turn, the session server forwards the request to
   the router and records the engine's output: tokens, logprobs, and weight
   version. It answers the agent first and writes the record to the payload
   channel in the background.
3. **Close.** The agent loop closes each trajectory and reports the rewards.
   It also reports how many tokens of each turn it kept. For example,
   Search-R1 cuts a turn at ``</search>``.
4. **Deliver.** When the group is complete, the session server sends its
   metadata to the buffer worker. At hand-off, the actor fetches the records
   from the payload channel and assembles the training sequences.

The session server also checks that each turn's prompt extends the previous
turn. A rewritten history shows up in the ``prefix_*`` metrics.

Control and Data Planes
-----------------------

On the default path, the buffer worker receives each group's full training
data and holds it until hand-off. One process both schedules the groups and
moves their data.

The session server path splits the two:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Plane
     - What travels
     - Who handles it
   * - Control
     - Group metadata: rewards, weight versions, token counts, and where the
       data is stored
     - Buffer worker: admission, dropping, hand-off. Runner: training and
       weight sync.
   * - Data
     - Generate requests and recorded trajectories
     - Session server, router, and engines; the payload channel until the
       actor fetches the data

Scheduling does not change. Admission, buffer limits, staleness, and
dropping work as on the default path, but use only the metadata.

The buffer worker is a single process on node 0 (see the
:doc:`multi-node guide <multi_node>`). On this path, its memory does not grow
with the training data. That matters when groups are large, for example with
routed-expert replay (R3) or long contexts.

Weight sync and other control calls go from each rollout worker straight to
its engine. They never pass through the router.

Choose a Routing Policy
-----------------------

The session servers forward every request through the router. RLinf
supports two of the router's policies:

- ``manual`` (default): the router picks the least loaded engine for the
  group's first request. Later requests with the same group id go to the
  same engine.
- ``cache_aware``: for each request, the router picks the engine whose cache
  best matches the request prefix. It balances load only when the gap between
  engines grows large. Requests of one group usually share a prefix, but the
  router does not guarantee they reach the same engine.

When all prompts share a long prefix, ``cache_aware`` piles new groups onto
one engine. Search-R1 prompts all start with the same instructions:

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentic_rl/async/session_server_router_load.png
   :align: center
   :width: 85%

   Running requests per rollout instance (Qwen2.5-1.5B, 12 instances,
   64 × 4 per step, 20 steps). With ``cache_aware``, instance 5 averages 27.5
   and peaks at 79; the others stay near 4.8. With ``manual``, every instance
   stays between 6 and 7.5.

Use ``manual`` for workloads with a shared prefix. ``cache_aware`` may suit
prompts that differ from the start; no run has measured that case yet.

With one rollout instance, the router only adds a hop. The setup still works,
and the run logs a warning.

Dropped Groups and Failures
---------------------------

When the buffer worker drops a group, the payload channel deletes its data.
The same holds for groups that an agent aborts or that a failed write voids.

Errors on the request path fail the run:

- The session server resends a request only if it could not connect to the
  router. That request never reached an engine, so no turn is sampled twice.
- If a generate request times out or the engine aborts it, the run fails.

A failed payload write is the one recoverable error. The session server voids
that group, and the buffer worker drops it.

Choose a Path
-------------

Use the **default path** (the channel path) when:

- you run a synchronous or evaluation runner, or need ``pause_mode: drain``;
- each group's data is small and the run fits on a few nodes;
- you want no extra processes or ports.

Use the **session server path** when:

- a group's trajectories share a long prefix;
- each group's data is large, or node 0 is short of memory.

Deploy on Multiple Nodes
------------------------

With several rollout nodes, the engines listen on ``0.0.0.0`` instead of
loopback, so the router can reach engines on other nodes.

The engines also serve RLinf's control endpoints, which accept pickled
request bodies. To protect them, the engines accept requests from other
machines on four paths only: ``/generate``, ``/health``, ``/server_info``,
and ``/model_info``. Everything else gets a 403 before the body is read.
Requests from the same machine are not restricted.

Idle HTTP connections follow one rule: the client must close them before the
server does. Otherwise a client may reuse a connection the server has just
closed. Config validation enforces this through ``keep_alive_s`` and
``engine_keep_alive_s``, so the defaults rarely need changing.

.. warning::

   Set ``NCCL_IB_HCA`` to the same value in the shell that runs ``ray start``
   and in the shell that launches ``train.py``. RLinf sets each worker's
   environment variables to those of the ``train.py`` process.

.. _session-server-metrics:

Metrics to Watch
----------------

The session servers report under ``rollout/session_server/`` at every step.
Counts and times cover the step; bytes and queue sizes are snapshots.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - How to read it
   * - ``forward_count``, ``forward_time_total_s``, ``forward_time_avg_s``
     - Generate requests forwarded in the step, with their total and mean
       latency.
   * - ``prefix_continuous``, ``prefix_rewritten``, ``prefix_underreported``
     - How each prompt relates to the previous turn. The last two should
       stay at 0. Non-zero means the agent changes its history between
       turns.
   * - ``payload_write_failed``
     - Groups voided by a failed write. Should stay at 0.
   * - ``pending_payload_bytes``
     - Bytes still being written. If it stays near
       ``max_pending_payload_bytes``, raise the budget or add
       ``num_servers`` or ``payload_shards``.
   * - ``payload_queued_groups``, ``payload_queued_bytes``
     - Data waiting in the payload channels. ``payload_queued_groups``
       should equal or slightly exceed ``rollout/buffer/size``. A gap that
       keeps growing means data is not being released.

Check Correctness
-----------------

Turn on ``session_server.verify_assembly`` the first time you use a new
agent. The agent loop then also builds each group as on the default path.
The actor compares both versions field by field before training:

- token ids, prompt and response lengths, and rollout logprobs;
- rewards, the sequence-to-trajectory mapping, and end flags;
- extra training fields;
- the sampling parameters of every forwarded request.

Any difference fails the run and names the field. Both versions come from
the same generation, so the check covers recording and assembly, not
sampling. It costs throughput, so turn it off for training runs.

On Search-R1, about 2,900 groups passed this check. They came from two
5-step runs: Qwen2.5-1.5B on 12 instances, and Qwen2.5-3B on 8.

Configuration Reference
-----------------------

``session_server``:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Key
     - Default
     - What it does
   * - ``enable``
     - ``False``
     - Turns the session server path on.
   * - ``num_servers``
     - ``1``
     - Number of session servers, spread round-robin over the rollout nodes.
   * - ``payload_shards``
     - ``1``
     - Payload channels per session server, one process each. Raise it if
       writes become a CPU bottleneck.
   * - ``port``
     - ``8030``
     - Port of the first session server. Server ``i`` uses ``port + i``.
   * - ``generate_timeout_s``
     - ``600``
     - Limit on one forwarded request, including pauses for weight sync.
       Exceeding it fails the run.
   * - ``per_task.<task>.generate_timeout_s``
     - ``generate_timeout_s``
     - Per-task override of the timeout above.
   * - ``payload_put_timeout_s``
     - ``300``
     - Limit on one payload write. Exceeding it voids the group.
   * - ``max_pending_payload_bytes``
     - 512 MiB
     - Budget for bytes still being written. Past it, each response waits for
       its own write. This slows the agents but keeps memory bounded.
   * - ``keep_alive_s``
     - ``5``
     - How long the session servers keep an idle connection open, in whole
       seconds.
   * - ``engine_keep_alive_s``
     - ``120``
     - How long the engines keep an idle connection open, in whole seconds.
       Must exceed the router's 50 s idle pool timeout.
   * - ``verify_assembly``
     - ``False``
     - Checks each group against the default-path assembly; see
       `Check Correctness`_.

``rollout.router`` (separate from the ``router_server_args`` block in
:doc:`SGLang Server & Router <sglang_server>`):

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Key
     - Default
     - What it does
   * - ``policy``
     - ``manual``
     - ``manual`` or ``cache_aware``; see `Choose a Routing Policy`_.
   * - ``assignment_mode``
     - ``min_load``
     - How ``manual`` picks an engine for a new group.
   * - ``request_timeout_secs``
     - ``1800``
     - The router's own timeout. Must be at least as long as every generate
       timeout.
   * - ``max_idle_secs``
     - ``600``
     - How long the router remembers a group's engine after its last
       request.
   * - ``disable_health_check``
     - ``True``
     - Must stay ``True``. The router's health probes generate tokens.
   * - ``disable_retries``
     - ``True``
     - Stops the router from resending requests.

Use It from Another Runner
--------------------------

The session server does not depend on asynchronous training. RLinf connects
only the asynchronous runner to it. Another runner, such as the synchronous
agent runner, must:

- launch the router and the session servers, and register the engines.
  The agent runner has a startup hook for this. When the hook returns the
  session server URLs, the agent loop switches to this path;
- watch the session servers, so a failure ends the run;
- send the session servers the loaded weight versions after each sync;
- collect the session server metrics every step;
- pass the payload channels to the actor;
- turn each group's metadata into a delivery the actor can fetch, and delete
  the data of every group it drops.

On the asynchronous path, the buffer worker does the last item. A
synchronous runner should not use the buffer worker, because it admits groups
ahead of training. You must also relax the ``runner.async_mode`` check in
config validation. Only the asynchronous runner has run this path in tests.

Limitations
-----------

- Requires the asynchronous runner, a Megatron actor, the SGLang server
  backend, and the Search-R1 agent.
- Agents can attach ``extra_fields_train``. Other per-turn, per-trajectory,
  and per-group extra fields are rejected.
- ``agentloop.print_outputs`` works only with ``verify_assembly`` on. The
  agent loop prints while it assembles, which on this path happens only for
  the check.
- HTTP is the only transport (``session_server.type: http``).
- If a session server dies, the run ends. Nothing restarts it.
