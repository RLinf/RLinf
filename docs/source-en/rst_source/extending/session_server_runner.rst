:orphan:

Connect the Session Server to a Runner
======================================

Wire the agent :doc:`Session Server </rst_source/guides/session_server>` into a runner other than ``AsyncAgentRunner``. Read this guide only if you are writing a custom runner; for standard asynchronous agent training, enable ``session_server.enable: true`` directly on ``AsyncAgentRunner``.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Layer / File
     - Responsibility
   * - Entry script (``examples/agent/searchr1/train.py``)
     - Launch ``SessionServerWorker`` groups and one ``SGLangRouterWorker``, then pass them to the runner.
   * - Runner (``rlinf/runners/async_agent_runner.py``)
     - Create payload channels, start router and session servers, sync loaded weight versions, poll serve handles, and stop cleanly.
   * - Buffer worker (``rlinf/workers/rollout_buffer/rollout_buffer_worker.py``)
     - Issue unique group IDs, drop aborted/filtered groups, delete stale payloads, and deliver full batches to the actor.
   * - Config check (``rlinf/config.py``)
     - Allow your custom runner mode in ``validate_session_server_cfg``.

What a Runner Must Guarantee
----------------------------

1. **Unique group ID and task name per group.** Session servers key every session and payload by ``group_id``. Attach unique ``group_ids`` and ``task_names`` on each ``RolloutRequest`` and run the agent loop via ``run_agentloop_rollout_continuously``.
2. **Full actor batches with payload cleanup.** When a group is aborted or filtered out (for example by ``reject_equal_reward``), delete its payload via ``remove_key`` and admit a replacement prompt so the actor still receives ``rollout_batch_size`` groups.

Lifecycle in Four Phases
------------------------

``AsyncAgentRunner`` coordinates the session server in this order:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Phase
     - Required Calls
   * - **1. Launch**
     - Entry script launches ``session_server.num_servers`` ``SessionServerWorker`` groups round-robin across rollout nodes plus one ``SGLangRouterWorker`` on the first rollout node, passing ``session_servers`` and ``session_router`` to the runner. Runner creates ``payload_shards`` payload channels per server via ``create_payload_channels`` inside ``__init__``.
   * - **2. Startup**
     - After rollout engines come up, call ``init_router()``, register each engine URL from ``rollout.get_server_url()``, start each session server with ``serve(router_url, agent_output_channel, payload_channels)`` (keep the handles), call ``wait_until_serving()``, and pass ``session_server_urls`` plus client timeouts to ``agent_loop.init_worker()``. After the initial weight sync, call ``set_loaded_versions()`` on every server (requests return HTTP 500 until then) and start the agent loop and buffer worker.
   * - **3. Per-step loop**
     - Pass the flattened ``payload_channels`` into ``actor.run_training()`` while polling ``_session_serve_handles`` so a crashed server stops the run. After every weight sync, call ``set_loaded_versions()``. Each step, call ``get_metrics()`` on every server and sum the counts.
   * - **4. Shutdown**
     - Stop all session servers and await their ``serve`` handles **before** calling ``_session_router.stop()``, ensuring no request is in flight when the router exits.

Reference Wiring
----------------

The trimmed excerpts below show how ``AsyncAgentRunner`` implements each phase:

.. tab-set::

   .. tab-item:: 1. Entry Script
      :sync: ss-runner-entry

      .. code-block:: python

         # examples/agent/searchr1/train.py
         if OmegaConf.select(cfg, "session_server.enable", default=False):
             from rlinf.workers.rollout.session_server.server_worker import SessionServerWorker
             from rlinf.workers.rollout.sglang.router_worker import SGLangRouterWorker

             rollout_node_ranks = list(
                 dict.fromkeys(p.cluster_node_rank for p in rollout_placements)
             )
             num_servers = cfg.session_server.num_servers
             session_servers = [
                 SessionServerWorker.create_group(cfg, idx).launch(
                     cluster,
                     name=(
                         cfg.session_server.group_name
                         if num_servers == 1
                         else f"{cfg.session_server.group_name}_{idx}"
                     ),
                     placement_strategy=NodePlacementStrategy(
                         [rollout_node_ranks[idx % len(rollout_node_ranks)]]
                     ),
                 )
                 for idx in range(num_servers)
             ]
             session_router = SGLangRouterWorker.create_group(
                 cfg, cfg.rollout.router
             ).launch(
                 cluster,
                 name="SessionServerRouter",
                 placement_strategy=NodePlacementStrategy([rollout_node_ranks[0]]),
             )
             runner_kwargs["session_servers"] = session_servers
             runner_kwargs["session_router"] = session_router

   .. tab-item:: 2. Runner
      :sync: ss-runner-runner

      .. code-block:: python

         # rlinf/runners/async_agent_runner.py
         # 1) Create payload channels in __init__
         for idx, server in enumerate(session_servers):
             self._session_payload_channels.append(
                 create_payload_channels(
                     f"SessionPayload-{idx}",
                     num_shards=cfg.session_server.payload_shards,
                     producer=server,
                 )
             )

         # 2) Start router, then session servers (_start_session_servers)
         self._session_router.init_router().wait()
         for url in [u for u in self.rollout.get_server_url().wait() if u is not None]:
             self._session_router.register_server(url).wait()
         router_url = self._session_router.get_router_url().wait()[0]
         for server, channels in zip(self._session_servers, self._session_payload_channels):
             self._session_serve_handles.append(
                 server.serve(router_url, self.agent_output_channel, channels)
             )
             server.wait_until_serving().wait()

         # 3) Push weight versions, train, and poll handles each step
         for server in self._session_servers:
             server.set_loaded_versions(self._rollout_loaded_versions).wait()
         payload_channels = [ch for group in self._session_payload_channels for ch in group]
         actor_handle = self.actor.run_training(
             input_channel=self.rollout_channel,
             version=self.global_steps,
             payload_channels=payload_channels,
         )
         self._wait_actor(actor_handle, [agent_handle, buffer_handle, *self._session_serve_handles])

         # 4) Shutdown: stop servers before router
         for server in self._session_servers:
             server.stop().wait()
         for handle in self._session_serve_handles:
             handle.wait()
         self._session_router.stop().wait()

   .. tab-item:: 3. Buffer Worker
      :sync: ss-runner-buffer

      .. code-block:: python

         # rlinf/workers/rollout_buffer/rollout_buffer_worker.py
         # Issue unique group_id + task_name
         await self.buffer.wait_for_admission_slot()
         group_id, _ = self.buffer.issue_group(sample_idx, self.task_name)
         request.group_ids = [group_id]
         request.task_names = [self.task_name]
         await dataloader_channel.put(request, async_op=True).async_wait()

         # Handle finished or discarded groups
         result = await agent_output_channel.get(async_op=True).async_wait()
         if isinstance(result, DiscardedGroup):
             self.buffer.drop_group(
                 result.group_id, result.reason,
                 self.max_dropped_groups_per_step, already_buffered=False,
             )
         elif isinstance(result, GroupMetadata):
             if self.reject_equal_reward and rewards_all_equal(result.rewards):
                 channel = payload_channel_for(result.payload_ref.locator, self._payload_channels)
                 await channel.remove_key(key=result.payload_ref.key, async_op=True).async_wait()
                 self.buffer.drop_group(
                     result.group_id, "equal_reward",
                     self.max_dropped_groups_per_step, already_buffered=False,
                 )
             else:
                 self.buffer.record_result(None, result)

Run and Verify
--------------

Launch the reference Search-R1 session server config (see :doc:`/rst_source/guides/session_server` for cluster setup):

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server \
     session_server.verify_assembly=true

Confirm two checks before turning ``verify_assembly`` off for full training:

1. No assembly mismatch occurs (the actor compares every group field-by-field against the non-session-server path and aborts on the first difference).
2. ``rollout/session_server/payload_write_failed`` stays at ``0`` and ``rollout/session_server/payload_queued_groups`` tracks ``rollout/buffer/size`` without unbounded growth.
