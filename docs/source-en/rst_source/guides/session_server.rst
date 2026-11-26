Session Server
==============

The Session Server collects multi-turn agent trajectories directly on the rollout nodes and assembles them into training data. Instead of streaming full trajectories through the Buffer Worker on node 0, the Session Server keeps trajectory payloads on the rollout side, forwards generation requests through an SGLang router, and exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.

If you are new to asynchronous agentic training, read :doc:`Asynchronous Agentic RL <async_agentic>` first.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Item
     - Summary
   * - **When to enable**
     - Trajectories share long prefixes, groups have large payloads (for example long contexts or R3), node 0 memory is tight, or your agents call the OpenAI chat completions API.
   * - **Keep default when**
     - You use a synchronous/evaluation runner, need ``pause_mode: drain``, or run small jobs that do not need extra server processes.
   * - **What starts**
     - ``session_server.num_servers`` HTTP servers on rollout nodes, ``payload_shards`` payload channels per server, and one shared SGLang router.

Enable and Launch
-----------------

Add ``session_server.enable: true`` to an asynchronous agentic config:

.. code-block:: yaml

   session_server:
     enable: true

Startup config validation requires:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Setting / Package
     - Requirement
   * - ``runner.async_mode`` & Actor
     - ``True`` with a Megatron actor
   * - ``rollout.sglang.backend_type``
     - ``server``
   * - ``reward.reward_type``
     - ``searchr1`` (Search-R1 agent)
   * - ``rollout.pause_mode``
     - ``retract`` or ``in_place`` (not ``drain``)
   * - ``cluster.auto_scheduler``
     - ``False``
   * - Python package
     - ``sglang_router`` installed

Example: Search-R1
~~~~~~~~~~~~~~~~~~

``examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml`` inherits ``train_qwen2.5_async.yaml`` (Megatron actor on node 0, 8 TP1 SGLang rollout instances on node 1) and enables the Session Server with ``rollout.router.policy: manual``.

Start a 2-node Ray cluster (:doc:`multi_node`) and the retrieval server (:doc:`Search-R1 </rst_source/examples/agentic/searchr1/index>`), then launch from the head node with your local paths:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server \
     'data.train_data_paths=["./data/searchr1_train.jsonl"]' \
     rollout.model.model_path=./models/Qwen2.5-3B-Instruct \
     tools.search.server_addr=0.0.0.0:8000

How It Works & Routing
----------------------

The Session Server splits scheduling (**control plane**) from trajectory storage (**data plane**):

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Plane
     - Traffic
     - Owners
   * - **Control**
     - Group metadata (rewards, weight versions, token counts, payload refs)
     - Buffer Worker (admission, dropping, hand-off) & Runner (training, weight sync)
   * - **Data**
     - Generate requests and recorded turn payloads
     - Session Server, SGLang router, rollout engines, and payload channels (until the Actor fetches them)

Each group flows through four steps:

1. **Open:** The agent loop opens one session per trajectory, pinning all sessions of a group to the same Session Server.

2. **Generate:** The Session Server forwards each turn through the router, records tokens, logprobs, and weight version, replies to the agent first, and writes to the payload channel in the background.

3. **Close:** The agent loop closes each trajectory, reports rewards, and records kept token lengths (for example, Search-R1 cuts at ``</search>``).

4. **Deliver:** Once the group completes, metadata goes to the Buffer Worker; at hand-off, the Actor fetches the payloads and assembles training sequences.

Weight sync and control calls still go straight from each rollout worker to its engine.

.. note::

   **Failure semantics:** Only connection failures to the router are retried (before reaching an engine). An isolated payload write failure or timeout voids and drops that single group; a generate timeout, engine abort, or crashed payload channel stops the run.

Choose a Routing Policy
~~~~~~~~~~~~~~~~~~~~~~~

Set ``rollout.router.policy`` to pick how the router assigns requests to rollout engines:

- ``manual`` (default): picks the least-loaded engine for a group's first request and pins later requests of that group to the same engine. **Use this when prompts share a long prefix**, such as Search-R1.
- ``cache_aware``: picks the engine whose KV cache best matches the prefix. When all prompts share the same system/instruction prefix, ``cache_aware`` piles new groups onto one engine instead of balancing load:

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentic_rl/async/session_server_router_load.png
   :align: center
   :width: 85%

   Running requests per rollout instance (Qwen2.5-1.5B, 12 instances, 64 × 4 per step). Under ``cache_aware``, instance 5 peaks at 79 while others sit near 4.8; under ``manual``, every instance stays between 6 and 7.5.

With a single rollout instance, the router still works and logs a warning because it only adds a hop.

Multi-Node Deployment
---------------------

- **Network scope:** Rollout engines listen on ``0.0.0.0`` so the router can reach remote nodes. Off-host requests are limited to ``/generate``, ``/health``, ``/server_info``, and ``/model_info``; other control endpoints return HTTP 403 remotely (loopback stays unrestricted).

- **Idle connections:** Clients must close idle HTTP connections before servers do. Config validation enforces this via ``keep_alive_s`` and ``engine_keep_alive_s``; keep the defaults unless needed.

.. warning::

   Export the same ``NCCL_IB_HCA`` value in both the shell running ``ray start`` and the shell launching ``train.py``. RLinf propagates the ``train.py`` process environment to workers.

Monitor, Verify, and Reference
------------------------------

.. _session-server-metrics:

Metrics to Watch
~~~~~~~~~~~~~~~~

Reported every step under ``rollout/session_server/`` (counts/times are per-step; bytes/queues are snapshots):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - How to read it
   * - ``forward_count``, ``forward_time_total_s``, ``forward_time_avg_s``
     - Forwarded generate requests and total/mean latency (mean is reported only when ``forward_count > 0``).
   * - ``prefix_continuous``, ``prefix_rewritten``, ``prefix_underreported``
     - Turn prefix continuity. ``prefix_underreported`` must stay 0; token-ID agents should keep ``prefix_rewritten`` at 0 (chat agents may raise it).
   * - ``payload_write_failed``
     - Groups voided by payload write failures (should stay 0).
   * - ``pending_payload_bytes``
     - In-flight payload bytes. If near ``max_pending_payload_bytes``, raise the budget or add ``num_servers`` / ``payload_shards``.
   * - ``payload_queued_groups``, ``payload_queued_bytes``
     - Queued payload data. ``payload_queued_groups`` should track ``rollout/buffer/size``; a growing gap indicates leaked payloads.
   * - ``chat_reasoning_responses``, ``chat_tool_call_responses``, ``chat_tool_call_parse_failed``
     - Chat completions responses with parsed reasoning, tool calls, or unparseable tool-call blocks.

Verify Assembly
~~~~~~~~~~~~~~~

When onboarding a new token-ID agent, run a short check with ``session_server.verify_assembly=true``. The Actor compares every assembled group field by field (tokens, lengths, logprobs, rewards, extra fields, and sampling params) against the direct-channel path and fails on any mismatch. Turn it off for production runs because it reduces throughput.

.. dropdown:: Configuration reference (session_server & rollout.router)

   ``session_server``:

   .. list-table::
      :header-rows: 1
      :widths: 34 16 50

      * - Key
        - Default
        - Description
      * - ``enable``
        - ``False``
        - Enable the Session Server data plane.
      * - ``num_servers``
        - ``1``
        - Number of Session Server workers, placed round-robin over rollout nodes.
      * - ``payload_shards``
        - ``1``
        - Payload channel processes per server; raise if payload writes become CPU-bound.
      * - ``port``
        - ``8030``
        - Base port (server ``i`` uses ``port + i``).
      * - ``generate_timeout_s``
        - ``600``
        - Per-request timeout including weight-sync pauses; exceeding fails the run.
      * - ``per_task.<task>.generate_timeout_s``
        - ``generate_timeout_s``
        - Per-task override for the generate timeout.
      * - ``payload_put_timeout_s``
        - ``300``
        - Timeout for one payload write; exceeding voids the group.
      * - ``max_pending_payload_bytes``
        - 512 MiB
        - Backpressure budget for background writes; past it, replies wait for writes to finish.
      * - ``keep_alive_s``
        - ``5``
        - Session Server idle HTTP keep-alive in whole seconds.
      * - ``engine_keep_alive_s``
        - ``120``
        - Rollout engine idle HTTP keep-alive in whole seconds (must exceed the router's 50 s pool timeout).
      * - ``verify_assembly``
        - ``False``
        - Field-by-field assembly check against the direct-channel path (token-ID agents only).
      * - ``chat_template_kwargs``
        - ``{}``
        - Fixed chat template kwargs for ``/v1/chat/completions`` (for example ``clear_thinking``).

   ``rollout.router`` (separate from ``router_server_args`` in :doc:`SGLang Server & Router <sglang_server>`):

   .. list-table::
      :header-rows: 1
      :widths: 34 16 50

      * - Key
        - Default
        - Description
      * - ``policy``
        - ``manual``
        - ``manual`` or ``cache_aware``.
      * - ``assignment_mode``
        - ``min_load``
        - How ``manual`` picks an engine for a new group.
      * - ``request_timeout_secs``
        - ``1800``
        - Router request timeout (must be ``>=`` every generate timeout).
      * - ``max_idle_secs``
        - ``600``
        - How long the router remembers a group-to-engine binding after its last request.
      * - ``disable_health_check``
        - ``True``
        - Must stay ``True`` (router health probes generate tokens).
      * - ``disable_retries``
        - ``True``
        - Prevents the router from silently resampling requests.

.. dropdown:: Limitations

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Area
        - Limitation
      * - Runner & backend
        - Requires the async runner, Megatron actor, SGLang server backend, and Search-R1 agent. To wire another runner, see :doc:`/rst_source/extending/session_server_runner`.
      * - Chat streaming
        - No token-by-token streaming: ``stream: true`` returns the full turn in one chunk followed by ``[DONE]``.
      * - Chat parameters
        - Does not support ``tool_choice: required``, named functions, or ``response_format``; ignores ``seed``.
      * - Assembly check
        - ``session_server.verify_assembly`` supports token-ID agent loops only and rejects chat completions requests.
      * - Extra fields & logging
        - Only ``extra_fields_train`` is accepted; ``agentloop.print_outputs`` prints only when ``verify_assembly`` is enabled.
      * - Fault tolerance
        - If a Session Server exits, the run terminates without automatic restart.

Related Extensions
------------------

- **Connect a black-box OpenAI API agent:** See :doc:`Black-Box Agent </rst_source/extending/openai_agent>` to point ``AsyncOpenAI`` at ``/sessions/<session id>/v1/chat/completions``.
- **Wire Session Server into a custom runner:** See :doc:`Connect the Session Server to a Runner </rst_source/extending/session_server_runner>` when building a runner other than the built-in async runner.
