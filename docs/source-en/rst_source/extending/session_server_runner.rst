Connect the Session Server to a Runner
======================================

This page shows how to connect the session server (see
:doc:`/rst_source/guides/session_server`) to a runner other than
``AsyncAgentRunner``: the calls the entry script, the runner, and the buffer
worker make, and the two things the session server needs from a runner. The
asynchronous runner, which already uses the session server, is the worked
example.

Before You Start
----------------

- The asynchronous Search-R1 example with the session server runs on your
  cluster. See :doc:`/rst_source/guides/session_server`.

What the Session Server Needs from a Runner
-------------------------------------------

1. **A group id and a task name for every group.** The session server keys a
   group's sessions and payload by its id, so the id must be unique for the
   whole run. Each ``RolloutRequest`` carries them in ``group_ids`` and
   ``task_names``, and the agent loop passes them on only when it runs with
   ``run_agentloop_rollout_continuously``.
2. **A full batch for the actor.** The actor takes ``rollout_batch_size``
   groups each step. A group can end without data for the actor: the agent
   aborts it, or a filter drops it. Each such group needs a replacement.

In the asynchronous runner, the buffer worker (``RolloutBufferWorker``)
provides both; another runner has to provide them itself.

The Files
---------

1. ``examples/agent/searchr1/train.py`` launches the session servers and the
   router, and passes them to the runner.
2. ``rlinf/runners/async_agent_runner.py`` creates the payload channels,
   starts the router and session servers, and polls them each step.
3. ``rlinf/workers/rollout_buffer/rollout_buffer_worker.py`` issues the group
   ids, and delivers or releases each finished group.
4. ``validate_session_server_cfg`` in ``rlinf/config.py`` accepts the session
   server only with ``runner.async_mode``. To use another runner, allow it in
   this check.

Example
-------

Each block below is trimmed from the file named in its first line. The
numbered comments match the steps in the next section.

The entry script launches the session servers and the router:

.. code-block:: python

   # examples/agent/searchr1/train.py
   if OmegaConf.select(cfg, "session_server.enable", default=False):
       from rlinf.workers.rollout.session_server.server_worker import (
           SessionServerWorker,
       )
       from rlinf.workers.rollout.sglang.router_worker import SGLangRouterWorker

       rollout_node_ranks = list(
           dict.fromkeys(
               placement.cluster_node_rank for placement in rollout_placements
           )
       )
       # 1. One group per session server, round-robin over the rollout nodes,
       # and one router group on the first rollout node.
       num_servers = cfg.session_server.num_servers
       session_servers = [
           SessionServerWorker.create_group(cfg, server_index).launch(
               cluster,
               name=(
                   cfg.session_server.group_name
                   if num_servers == 1
                   else f"{cfg.session_server.group_name}_{server_index}"
               ),
               placement_strategy=NodePlacementStrategy(
                   [rollout_node_ranks[server_index % len(rollout_node_ranks)]]
               ),
           )
           for server_index in range(num_servers)
       ]
       session_router = SGLangRouterWorker.create_group(
           cfg, cfg.rollout.router
       ).launch(
           cluster,
           name="SessionServerRouter",
           placement_strategy=NodePlacementStrategy([rollout_node_ranks[0]]),
       )
   ...
   if session_servers is not None:
       runner_kwargs["session_servers"] = session_servers
       runner_kwargs["session_router"] = session_router

The runner's part, with each numbered comment matching a step below:

.. code-block:: python

   # rlinf/runners/async_agent_runner.py
   # At startup
   # 2. Payload channels on each session server's node, one per shard.
   if session_servers is not None:
       for server_index, server in enumerate(session_servers):
           self._session_payload_channels.append(
               create_payload_channels(
                   f"SessionPayload-{server_index}",
                   num_shards=cfg.session_server.payload_shards,
                   producer=server,
               )
           )

   # In _start_session_servers, which AgentRunner calls once the engines are up
   # 3. The router first, then each session server behind it.
   self._session_router.init_router().wait()
   instance_urls = [
       url for url in self.rollout.get_server_url().wait() if url is not None
   ]
   for url in instance_urls:
       self._session_router.register_server(url).wait()
   router_url = self._session_router.get_router_url().wait()[0]
   session_urls = []
   for server, payload_channels in zip(
       self._session_servers, self._session_payload_channels
   ):
       self._session_serve_handles.append(
           server.serve(router_url, self.agent_output_channel, payload_channels)
       )
       server.wait_until_serving().wait()
       session_urls.append(f"http://{server.get_server_address().wait()[0]}")
   ...
   return session_urls, client_timeout_s, keepalive_timeout_s

   # Before the training loop (after the initial weight sync)
   # 4. Push the initial loaded versions, flatten all payload channels,
   # and start the agent loop and buffer worker for the whole training.
   for server in self._session_servers or []:
       server.set_loaded_versions(self._rollout_loaded_versions).wait()
   payload_channels = [
       channel
       for server_channels in self._session_payload_channels
       for channel in server_channels
   ]
   agent_handle = self.agent_loop.run_agentloop_rollout_continuously(
       input_channel=self.dataloader_channel,
       output_channel=self.agent_output_channel,
       agent_metric_channel=self.agent_metric_channel,
   )
   buffer_handle = self.rollout_buffer.run_continuously(
       agent_output_channel=self.agent_output_channel,
       dataloader_channel=self.dataloader_channel,
       rollout_channel=self.rollout_channel,
       payload_channels=payload_channels,
   )

   # Every step
   # 7. Each step trains on the delivered groups and watches the serve handles.
   actor_handle: Handle = self.actor.run_training(
       input_channel=self.rollout_channel,
       version=self.global_steps,
       payload_channels=payload_channels,
   )
   watched_handles = [
       agent_handle,
       buffer_handle,
       *self._session_serve_handles,
   ]
   metrics = self._wait_actor(actor_handle, watched_handles)
   # 8. After each weight sync.
   for server in self._session_servers or []:
       server.set_loaded_versions(self._rollout_loaded_versions).wait()

   # 9. Every step, summed over the servers.
   per_server_metrics = [
       server.get_metrics().wait()[0]
       for server in self._session_servers
   ]
   session_server_metrics = {
       key: sum(metrics[key] for metrics in per_server_metrics)
       for key in per_server_metrics[0]
   }

   # At shutdown
   # 10. The session servers stop before the router.
   for server in self._session_servers:
       server.stop().wait()
   for serve_handle in self._session_serve_handles:
       serve_handle.wait()
   self._session_router.stop().wait()

The buffer worker issues the groups, then delivers or releases each one:

.. code-block:: python

   # rlinf/workers/rollout_buffer/rollout_buffer_worker.py
   # For each prompt
   # 5. Every prompt gets a group id and a task name.
   await self.buffer.wait_for_admission_slot()
   group_id, _ = self.buffer.issue_group(sample_idx, self.task_name)
   request.group_ids = [group_id]
   request.task_names = [self.task_name]
   await dataloader_channel.put(request, async_op=True).async_wait()

   # For each finished or dropped group
   # 6. Each finished or dropped group.
   result = await agent_output_channel.get(async_op=True).async_wait()
   if isinstance(result, DiscardedGroup):
       self.buffer.drop_group(
           result.group_id,
           result.reason,
           self.max_dropped_groups_per_step,
           already_buffered=False,
       )
       continue
   if isinstance(result, GroupMetadata):
       metadata = result
       result = None
   if self.reject_equal_reward and rewards_all_equal(metadata.rewards):
       await self._release_payload(metadata)
       self.buffer.drop_group(
           metadata.group_id,
           "equal_reward",
           self.max_dropped_groups_per_step,
           already_buffered=False,
       )
       continue
   self.buffer.record_result(result, metadata)

   # When a batch is full
   for metadata, result in self.buffer.release_groups():
       if result is None:
           delivery = GroupDelivery(
               group_id=metadata.group_id,
               task_name=metadata.task_name,
               payload_ref=metadata.payload_ref,
           )
           await rollout_channel.put(delivery, async_op=True).async_wait()

   # To release a group's payload
   ref = metadata.payload_ref
   channel = payload_channel_for(ref.locator, self._payload_channels)
   await channel.remove_key(key=ref.key, async_op=True).async_wait()

The Calls in Order
------------------

1. **Launch the session servers and the router.** The entry script launches
   ``session_server.num_servers`` ``SessionServerWorker`` groups, spread over
   the rollout nodes, and one ``SGLangRouterWorker`` group, and passes them to
   the runner as ``session_servers`` and ``session_router``.
2. **Create the payload channels.** ``create_payload_channels`` takes a base
   name, the number of shards, and the session server's worker group, and
   returns one channel per shard, placed on that server's node. The runner
   creates them in ``__init__``, before any session server starts, because
   ``serve`` needs them.
3. **Start the router and the session servers.** ``AgentRunner`` calls
   ``_start_session_servers`` once the rollout engines are up. In it, the
   runner:

   - starts an empty router with ``init_router``;
   - registers each engine URL from ``rollout.get_server_url`` with
     ``register_server``;
   - reads the router's address with ``get_router_url``;
   - calls ``serve`` on each session server and keeps the handle, since
     ``serve`` runs until the server stops;
   - calls ``wait_until_serving``, which returns once the server listens
     or raises its startup error;
   - reads the server's address with ``get_server_address``.

   The method returns the session server URLs, the client's generate
   timeout, and its keep-alive, and ``AgentRunner`` passes them to
   ``agent_loop.init_worker`` as ``session_server_urls``,
   ``session_server_generate_timeout_s``, and
   ``session_server_keepalive_timeout_s``. A runner that does not subclass
   ``AgentRunner`` must pass these three arguments to
   ``agent_loop.init_worker`` itself.
4. **Push initial weight versions and start the loops.** After the initial
   weight sync, the runner calls ``set_loaded_versions`` on every session
   server (requests fail with HTTP 500 before this first call). It then
   flattens ``self._session_payload_channels`` across all servers into one
   list and starts the agent loop with ``run_agentloop_rollout_continuously``
   and the buffer worker with ``run_continuously``. Both run for the whole
   training.
5. **Issue the groups.** The buffer worker reserves an id for each prompt
   with ``issue_group`` and puts the prompt on the agent loop's input channel
   as a ``RolloutRequest`` that carries the id and the task name.
6. **Deliver or release each finished group.** A session server puts a
   ``GroupMetadata`` on the agent output channel when a group finishes, and a
   ``DiscardedGroup`` when it drops one. A ``DiscardedGroup`` needs no
   cleanup. A finished group that is filtered out has its payload removed
   with ``remove_key``, on the channel that ``payload_channel_for`` finds for
   its ``payload_ref``; otherwise the data stays there for the rest of the
   run. Each dropped group frees its admission slot, so the buffer worker
   issues another prompt. Once a batch is full, each group goes to the actor
   as a ``GroupDelivery`` with its ``payload_ref``.
7. **Train and watch the session servers.** ``run_training`` takes all
   payload channels, so the actor can fetch each delivered group's data.
   While the actor trains, the runner polls the ``serve`` handles with the
   others, so a session server that stops ends the run.
8. **Update the loaded weight versions.** After each later weight sync, the
   runner calls ``set_loaded_versions`` on every session server with the
   version each engine loaded. The session server uses the oldest one as the
   lower bound of each turn's weight version.
9. **Collect the metrics.** Every step, the runner calls ``get_metrics`` on
   every session server and sums the values. The call also removes expired
   records of discarded groups.
10. **Stop.** At shutdown, the runner stops every session server and waits
    for its ``serve`` handle before it stops the router, so no request is in
    flight when the router goes away.

Run It
------

``examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml``
selects this wiring with ``session_server.enable: true``. Run it as described
in :doc:`/rst_source/guides/session_server`:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server

Test It
-------

- Add unit tests to ``tests/unit_tests/test_session_server.py``. It drives
  session servers against a mock SGLang server. Run it with
  ``pytest tests/unit_tests/test_session_server.py``. The file needs the
  agentic environment, with fastapi and SGLang installed; without them, every
  case is skipped.
- Run a few steps with ``session_server.verify_assembly: true`` added to the
  config. The actor compares every group with the data the agent loop
  assembles without the session server, and the run fails at the first
  difference.
