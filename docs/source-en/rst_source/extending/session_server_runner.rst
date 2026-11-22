Connect the Session Server to a Runner
======================================

Use this page to connect the session server to a runner other than the
asynchronous agent runner, ``AsyncAgentRunner``, the only runner RLinf wires
it into. The steps below follow the order the asynchronous runner uses. Once
they are in place, the agent loop sends its generate requests to the session
servers, and the actor trains on the groups they collect.

Before You Start
----------------

- The asynchronous Search-R1 example with the session server runs on your
  cluster. See :doc:`/rst_source/guides/session_server`.
- Your runner subclasses ``AgentRunner``.

The Files You Edit
------------------

1. The entry script, ``examples/agent/searchr1/train.py``. With
   ``session_server.enable``, it creates ``session_server.num_servers``
   ``SessionServerWorker`` groups and one ``SGLangRouterWorker`` group, and
   passes them to the runner. Reuse this block as it is.
2. Your runner, as described in the next section.
3. Config validation. ``validate_session_server_cfg`` in ``rlinf/config.py``
   requires ``runner.async_mode``. Relax that check for your runner.

Wire the Session Server into the Runner
---------------------------------------

``rlinf/runners/async_agent_runner.py`` shows each step in the asynchronous
runner.

1. **Create the payload channels.** In ``__init__``, call
   ``create_payload_channels`` once for each session server, on that server's
   node, with ``num_shards`` set to ``cfg.session_server.payload_shards``.
2. **Start the router and the session servers.** Override
   ``AgentRunner._start_session_servers``, which runs once the rollout engines
   are up. Call ``init_router``, register every engine URL from
   ``rollout.get_server_url`` with ``register_server``, and read the router
   URL with ``get_router_url``. Then, for each session server, dispatch
   ``SessionServerWorker.serve`` without waiting for it, call
   ``wait_until_serving``, and read its address with ``get_server_address``.
   Return the session server URLs, the client's generate timeout, and its
   keep-alive. ``AgentRunner`` passes them to the agent loop.
3. **Watch the session servers.** ``serve`` returns only when the server
   stops. Keep its handle and poll it while you wait for the actor, so that a
   session server that stops ends the run instead of leaving the actor waiting
   for data.
4. **Send the loaded weight versions.** After each weight sync, call
   ``set_loaded_versions`` on every session server with the version each
   engine loaded. The session server uses the oldest one as the lower bound of
   each turn's weight version.
5. **Collect the metrics.** Once per step, call ``get_metrics`` on every
   session server and sum the values across servers. The call also removes
   expired records of discarded groups.
6. **Deliver the groups.** When a group finishes, its session server puts a
   ``GroupMetadata`` on ``agent_output_channel``. When it drops a group, it
   puts a ``DiscardedGroup`` there instead.

   - For each group you train on, put a ``GroupDelivery`` with the group's
     ``payload_ref`` on the actor's input channel, and pass the payload
     channels to ``run_training``.
   - For each finished group you do not train on, for example one filtered
     out for equal rewards, call ``remove_key`` on the payload channel its
     ``payload_ref`` names. Otherwise its data stays there for the rest of the
     run.
   - A ``DiscardedGroup`` needs no cleanup. The session server has already
     dropped its data.

   In the asynchronous runner, the buffer worker (``RolloutBufferWorker``)
   does this step. A synchronous runner should not reuse it, because it admits
   groups ahead of training.

Test It
-------

- Add unit tests to ``tests/unit_tests/test_session_server.py``. It drives
  session servers against a mock SGLang server.
- Run your runner for a few steps with ``session_server.verify_assembly:
  true``. The actor compares every group with the data the agent loop
  assembles without the session server, and the run fails at the first
  difference.
- For the asynchronous runner, the same check runs with
  ``bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server``
  after ``session_server.verify_assembly: true`` is added to that config.
