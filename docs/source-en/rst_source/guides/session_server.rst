Session Server
==============

The session server collects agent trajectories on the rollout nodes and
assembles them into training data. Agents send their generate requests to it.
It forwards each request to an inference engine and records every turn.

It also provides:

- **Separate control and data planes.** The buffer worker schedules groups
  using only their metadata. The trajectories stay on the rollout nodes until
  the actor fetches them.
- **Routing through an SGLang router.** See `Choose a Routing Policy`_.
- **An OpenAI chat completions endpoint.** Agents built on the OpenAI API
  can send messages instead of token ids; see
  `Use the OpenAI Chat Completions Endpoint`_.

If you have not used the asynchronous runner, read :doc:`Asynchronous
Agentic RL <async_agentic>` first.
`Enable the Session Server`_ includes a worked example on Search-R1.

When to Use the Session Server
------------------------------

Without the session server, the agent loop sends its generate requests to
the rollout workers through channels, and the buffer worker holds each
group's full training data until hand-off.

Keep that default when:

- you run a synchronous or evaluation runner, or need ``pause_mode: drain``;
- each group's data is small and the run fits on a few nodes;
- you want no extra processes or ports.

Enable the session server when:

- a group's trajectories share a long prefix;
- each group's data is large, or node 0 is short of memory;
- your agents use the OpenAI chat completions API.

Enable the Session Server
-------------------------

Add this block to an asynchronous agentic config:

.. code-block:: yaml

   session_server:
     enable: true

At launch, three components start on the rollout nodes:

- the **session servers**, HTTP services that the agent loop sends its
  requests to;
- one **payload channel** per shard, the store that holds the recorded
  trajectories;
- one **SGLang router**, which every engine registers with.

Enabling it requires:

- ``runner.async_mode: True`` and a Megatron actor;
- ``rollout.sglang.backend_type: server``;
- the Search-R1 agent (``reward.reward_type: searchr1``);
- ``rollout.pause_mode`` of ``retract`` or ``in_place``, not ``drain``;
- ``cluster.auto_scheduler`` off;
- the ``sglang_router`` package.

Config validation checks each of these at startup.

Example: Search-R1
^^^^^^^^^^^^^^^^^^

This example runs the asynchronous Search-R1 recipe on two nodes with
Qwen2.5-3B-Instruct and the session server enabled.

The config is
``examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml``.
It inherits ``train_qwen2.5_async.yaml`` and adds
``session_server.enable: true`` and ``rollout.router.policy: manual``.

The parent config places the Megatron actor on node 0 and the SGLang server
rollout (8 instances at TP1) on node 1. Each step uses
``rollout_batch_size`` 512 × ``group_size`` 4.

Start a Ray cluster on the two nodes as described in :doc:`multi_node`. Then
start the retrieval server as described in :doc:`Search-R1
</rst_source/examples/agentic/searchr1/index>` and fill in three values in
``train_qwen2.5_async.yaml``:

.. code-block:: yaml

   data:
     train_data_paths: ["/path/to/data/searchr1_train.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
   tools:
     search:
       server_addr: 0.0.0.0:8000  # the retrieval server

Start training from the head node:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server

The results below come from three 100-step runs on 2×8 A100-80G: two on this
config and one on ``train_qwen2.5_async.yaml`` without the session server.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - With the session server
     - Without
   * - Trajectory reward, mean of the last 20 steps
     - 0.376 / 0.375
     - 0.372
   * - Mean data staleness
     - 2.0
     - 2.0

Training behaves the same with and without the session server. In both session
server runs, no payload write failed, and
``rollout/session_server/payload_queued_groups`` tracked
``rollout/buffer/size``. See :ref:`Metrics to Watch <session-server-metrics>` for
how to read these series.

How It Works
------------

How Trajectories Are Collected
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^

With the session server, scheduling and data movement split:

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
dropping work as they do without the session server, but use only the
metadata.

The buffer worker is a single process on node 0. With the session server, its
memory does not grow with the training data. That matters when groups are
large, for example with routed-expert replay (R3) or long contexts.

Weight sync and other control calls go from each rollout worker straight to
its engine.

Dropped Groups and Failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the buffer worker drops a group, the payload channel deletes its data.
The same holds for groups that an agent aborts or that a failed write voids.

Errors while serving a request fail the run:

- The session server resends a request only if it could not connect to the
  router. That request never reached an engine, so no turn is sampled twice.
- If a generate request times out or the engine aborts it, the run fails.

An isolated payload write failure or timeout is the one recoverable error:
the session server voids that group, and the buffer worker drops it. If the
payload channel worker dies or its communication fails, later writes on it
cannot succeed, so the session server stops and fails the run.

Choose a Routing Policy
-----------------------

Use ``manual`` when the prompts share a long prefix, as Search-R1's do.

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

With one rollout instance, the router only adds a hop. The setup still works,
and the run logs a warning.

Deploy on Multiple Nodes
------------------------

With the session server, the engines listen on ``0.0.0.0``, so the router
can reach engines on other nodes.

The engines also serve RLinf's control endpoints, which accept pickled
request bodies. To protect them, the engines accept requests from other
machines on four paths only: ``/generate``, ``/health``, ``/server_info``,
and ``/model_info``. Everything else gets a 403 before the body is read.
Requests over loopback are not restricted.

Idle HTTP connections follow one rule: the client must close them before the
server does. Otherwise a client may reuse a connection the server has just
closed. Config validation enforces this through ``keep_alive_s`` and
``engine_keep_alive_s``, so the defaults rarely need changing.

.. warning::

   Set ``NCCL_IB_HCA`` to the same value in the shell that runs ``ray start``
   and in the shell that launches ``train.py``. RLinf sets each worker's
   environment variables to those of the ``train.py`` process.

Monitor and Check
-----------------

.. _session-server-metrics:

Metrics to Watch
^^^^^^^^^^^^^^^^

The session servers report under ``rollout/session_server/`` at every step.
Counts and times cover the step; bytes and queue sizes are snapshots.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - How to read it
   * - ``forward_count``, ``forward_time_total_s``, ``forward_time_avg_s``
     - Generate requests forwarded in the step, with their total and mean
       latency. The average is reported only in steps with at least one
       forward.
   * - ``prefix_continuous``, ``prefix_rewritten``, ``prefix_underreported``
     - How each prompt relates to the previous turn.
       ``prefix_underreported`` should stay at 0. ``prefix_rewritten``
       counts turns whose history the agent changed; agents that send token
       ids should keep it at 0, while chat agents may raise it.
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
   * - ``chat_reasoning_responses``, ``chat_tool_call_responses``,
       ``chat_tool_call_parse_failed``
     - Chat completions responses with parsed reasoning, with parsed tool
       calls, and with a tool-call block the parser could not read.

Check Correctness
^^^^^^^^^^^^^^^^^

Turn on ``session_server.verify_assembly`` the first time you use a new
agent. The agent loop then also builds each group as it does without the
session server. The actor compares both versions field by field before
training:

- token ids, prompt and response lengths, and rollout logprobs;
- rewards, the sequence-to-trajectory mapping, and end flags;
- extra training fields;
- the sampling parameters of every forwarded request.

Any difference fails the run and names the field. Both versions come from
the same generation, so the check covers recording and assembly, not
sampling. It costs throughput, so turn it off for training runs.

On Search-R1, about 2,900 groups passed this check. They came from two
5-step runs: Qwen2.5-1.5B on 12 instances, and Qwen2.5-3B on 8.

Use the OpenAI Chat Completions Endpoint
----------------------------------------

Each session server also serves the OpenAI chat completions API at
``/sessions/<session id>/v1/chat/completions``. The session server renders
the messages into token ids with the chat template of the model family set
by ``agentloop.agentic_tokenizer``. It sends them to the engine the same way
as a ``generate`` request and returns an OpenAI ``chat.completion``
response, with reasoning and tool calls parsed into ``reasoning_content``
and ``tool_calls``. Reasoning is split out only for models whose tokenizer
has reasoning markers, such as Qwen3 and GLM-4.7.

Set ``agentloop.agentic_tokenizer`` to the model family, and
``rollout.sglang.tool_call_parser`` to the parser it pairs with. Startup
checks the pair:

.. code-block:: yaml

   agentloop:
     agentic_tokenizer: qwen     # glm47 for GLM-4.7
   rollout:
     sglang:
       tool_call_parser: qwen25  # glm47 for GLM-4.7

For other models, set ``agentic_tokenizer: default``. It renders the messages
but does not parse tool calls, and ``tool_call_parser`` can stay unset. Without
``agentloop.agentic_tokenizer``, chat requests fail with 500.

To use it, write an agent loop that runs each trajectory itself, as in the
example below. The agent loop still opens and closes the sessions, reports
rewards, and drops failed groups; your code only runs one trajectory on a
session that is already open:

.. code-block:: python

   from openai import APIStatusError, AsyncOpenAI, BadRequestError

   from rlinf.workers.agent.agent_loop import (
       GroupDiscarded,
       MultiAgentLoopWorker,
   )


   class MyChatAgentLoopWorker(MultiAgentLoopWorker):
       async def run_session_trajectory(self, binding, input_ids, answer):
           base_url = self.session_server_client.session_base_url(
               binding.group_id, binding.session_id
           )
           prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
           messages = build_messages(prompt)  # your prompt
           message = None
           num_responses = 0
           async with AsyncOpenAI(
               base_url=base_url,
               api_key="EMPTY",
               max_retries=0,  # a retried request is sampled again
           ) as client:
               for _ in range(MAX_TURNS):
                   try:
                       response = await client.chat.completions.create(
                           model="policy", messages=messages, tools=TOOLS
                       )
                   except BadRequestError as exc:
                       if exc.code != "context_length_exceeded":
                           raise
                       break  # no room left for another turn
                   except APIStatusError as exc:
                       if exc.status_code != 410:
                           raise
                       # The session server dropped the group.
                       raise GroupDiscarded("group_discarded") from exc
                   num_responses += 1
                   message = response.choices[0].message
                   messages.append(message.model_dump(exclude_none=True))
                   if not message.tool_calls:
                       break
                   for call in message.tool_calls:
                       result = await run_tool(call)  # your tool
                       messages.append(
                           {"role": "tool", "tool_call_id": call.id, "content": result}
                       )
           if num_responses == 0:
               # Not even the first request fit: drop the whole group.
               raise GroupDiscarded("context_length_exceeded")
           content = (message.content or "") if message is not None else ""
           reward = score(content, answer)  # your reward
           close_fields = {
               # One value per trained response: one agent per trajectory.
               "extra_fields_train": {"idx_to_sub_traj": [0] * num_responses}
           }
           return None, close_fields, reward

``input_ids`` holds the token ids of the dataset prompt, with the chat
template already applied. Decoded without special tokens, the text still
holds the template's role lines, such as ``system`` and ``user``, so
``build_messages`` takes the task out of it before building the messages.

Launch this class in your entry script instead of the agent loop that sends
token ids, the way ``examples/agent/searchr1/train.py`` launches the Search-R1
agent loop. Keep ``reward.reward_type: searchr1`` in the config: startup
validation still requires it even when your agent loop scores trajectories
itself.
``idx_to_sub_traj`` is needed when ``algorithm.loss_scales`` includes
``agent_level``.

A 410 means the session server has already dropped the group, for example
after a failed payload write. The example drops the whole group, as the agent
loop does for any group it cannot finish.

Set ``max_retries=0`` on the client. A retried request is sampled again,
and only one of the two answers is trained.

When a request no longer fits in ``runner.seq_length``, the example ends the
trajectory and scores the last answer; if not even the first request fits, it
drops the whole group.

The session server reuses the tokens of earlier turns. A turn's prompt is the
previous turn's prompt and output tokens, exactly as the engine produced
them, followed by the new messages. Rendering the history again from text
could give different tokens. The session server handles a request in one of
three ways:

- **Continue.** The request starts with the stored conversation and adds tool
  or user messages. Only the new messages are rendered, so the prefix cache
  still hits. With ``clear_thinking: true``, a new user message makes the
  server render the whole conversation, because the template then drops
  earlier reasoning.
- **Retry.** The request repeats the previous request, so the agent never
  used the previous answer. The server rolls the conversation back to before
  that turn. The abandoned turn stays in the payload but is not trained.
- **Rewrite.** Any other request is rendered in full from its messages.
  For example, an agent that shortens or edits an earlier answer to save
  context sends such a request.
  ``prefix_rewritten`` counts these requests, and their data is trained as
  usual.

When a request continues the conversation, the session server uses its own
stored copy of the earlier turns, including ``reasoning_content``. A client
that drops reasoning from its history still continues the conversation.

A turn's prompt and answer together must fit in ``runner.seq_length``, the
length the actor trains on. The endpoint lowers ``max_tokens`` to the remaining
length, and rejects a request whose prompt alone does not fit with a 400 and
the code ``context_length_exceeded``.

The endpoint also rejects the following with a 400 and an OpenAI error body:

- ``n`` above 1;
- ``temperature``, ``top_p``, or ``top_k`` different from the config;
- ``tool_choice`` other than ``auto`` or ``none``;
- ``response_format``;
- ``chat_template_kwargs`` different from the config.

A session handles one request at a time, on either endpoint. A request or a
close sent while another request is still running gets a 409.

Configuration and Limitations
-----------------------------

Configuration Reference
^^^^^^^^^^^^^^^^^^^^^^^

``session_server``:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Key
     - Default
     - What it does
   * - ``enable``
     - ``False``
     - Turns the session server on.
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
     - Checks each group against the assembly built without the session
       server; see `Check Correctness`_.
   * - ``chat_template_kwargs``
     - ``{}``
     - Chat template arguments for the chat completions endpoint, such as
       GLM's ``clear_thinking``. Requests with other values are rejected.

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

Limitations
^^^^^^^^^^^

- Requires the asynchronous runner, a Megatron actor, the SGLang server
  backend, and the Search-R1 agent. To connect the session server to another
  runner, see :doc:`/rst_source/extending/session_server_runner`.
- The chat completions endpoint has no real streaming. With
  ``stream: true``, the whole turn comes back in one chunk, followed by
  ``[DONE]``.
- The chat completions endpoint does not support ``tool_choice: required``,
  named functions, or ``response_format``, and it ignores ``seed``.
- ``session_server.verify_assembly`` works only with agent loops that send
  token ids. With it on, the chat completions endpoint rejects every request.
- Agents can attach ``extra_fields_train``. Other per-turn, per-trajectory,
  and per-group extra fields are rejected.
- With the session server, ``agentloop.print_outputs`` prints only when
  ``verify_assembly`` is on.
- If a session server dies, the run ends. Nothing restarts it.
