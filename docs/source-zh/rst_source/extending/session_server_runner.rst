将 session server 接入 runner
=============================

本页以异步 runner（``AsyncAgentRunner``）为例，说明把 session server（见 :doc:`/rst_source/guides/session_server`）接入其他 runner 时，入口脚本、runner 和 buffer worker 要做哪些调用，以及 session server 对 runner 的两项要求。

准备工作
--------

- 带 session server 的异步 Search-R1 示例能在你的集群上运行，见 :doc:`/rst_source/guides/session_server`。

session server 对 runner 的要求
-------------------------------

1. **每个组都有组 id 和 task name。** session server 按组 id 管理一个组的 session 和 payload，所以组 id 在整个运行期间不能重复。组 id 和 task name 写在每个 ``RolloutRequest`` 的 ``group_ids`` 和 ``task_names`` 里，只有用 ``run_agentloop_rollout_continuously`` 运行时，agent loop 才会把它们往下传。
2. **actor 每一步都拿到完整的一批。** actor 每一步要取 ``rollout_batch_size`` 个组。有的组结束时没有交给 actor 的数据，比如 agent 中止了它，或者被过滤掉了。每出现一个这样的组，都要补上一个。

异步 runner 里，这两件事都由 buffer worker（``RolloutBufferWorker``）完成；其他 runner 要自己完成。

涉及的文件
----------

1. ``examples/agent/searchr1/train.py``：创建 session server 和 router，传给 runner。
2. ``rlinf/runners/async_agent_runner.py``：创建 payload channel、启动 router 和 session server，并在每一步轮询。
3. ``rlinf/workers/rollout_buffer/rollout_buffer_worker.py``：分配组 id，交付或释放每个完成的组。
4. ``rlinf/config.py`` 里的 ``validate_session_server_cfg``：只有开启 ``runner.async_mode`` 时才允许使用 session server；要接其他 runner，需要在这项检查里放行。

示例
----

下面每一段代码都删减自它第一行注明的文件，带编号的注释对应下一节的步骤。

入口脚本创建 session server 和 router：

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
       # 1. 每个 session server 一个组，轮流放在各 rollout 节点上；
       # router 一个组，放在第一个 rollout 节点上。
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

runner 侧的代码，每条编号注释对应下面一步：

.. code-block:: python

   # rlinf/runners/async_agent_runner.py
   # 启动时
   # 2. 每个 session server 所在节点上的 payload channel，每个 shard 一个。
   if session_servers is not None:
       for server_index, server in enumerate(session_servers):
           self._session_payload_channels.append(
               create_payload_channels(
                   f"SessionPayload-{server_index}",
                   num_shards=cfg.session_server.payload_shards,
                   producer=server,
               )
           )

   # 在 _start_session_servers 里（引擎启动后由 AgentRunner 调用）
   # 3. 先启动 router，再启动挂在它后面的每个 session server。
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

   # 进入训练循环前（首次权重同步之后）
   # 4. 下发初始权重版本、展平全部 payload channel，并启动整个训练期间运行的循环。
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

   # 每一步
   # 7. 每一步用交付的组训练，并监视 serve handle。
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
   # 8. 每次权重同步之后。
   for server in self._session_servers or []:
       server.set_loaded_versions(self._rollout_loaded_versions).wait()

   # 9. 每一步把各 server 的值相加。
   per_server_metrics = [
       server.get_metrics().wait()[0]
       for server in self._session_servers
   ]
   session_server_metrics = {
       key: sum(metrics[key] for metrics in per_server_metrics)
       for key in per_server_metrics[0]
   }

   # 结束时
   # 10. 先停 session server，再停 router。
   for server in self._session_servers:
       server.stop().wait()
   for serve_handle in self._session_serve_handles:
       serve_handle.wait()
   self._session_router.stop().wait()

buffer worker 下发组，再交付或释放每个组：

.. code-block:: python

   # rlinf/workers/rollout_buffer/rollout_buffer_worker.py
   # 每个 prompt
   # 5. 每个 prompt 都分配组 id 和 task name。
   await self.buffer.wait_for_admission_slot()
   group_id, _ = self.buffer.issue_group(sample_idx, self.task_name)
   request.group_ids = [group_id]
   request.task_names = [self.task_name]
   await dataloader_channel.put(request, async_op=True).async_wait()

   # 每个完成或丢弃的组
   # 6. 每个完成或丢弃的组。
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

   # 攒满一批时
   for metadata, result in self.buffer.release_groups():
       if result is None:
           delivery = GroupDelivery(
               group_id=metadata.group_id,
               task_name=metadata.task_name,
               payload_ref=metadata.payload_ref,
           )
           await rollout_channel.put(delivery, async_op=True).async_wait()

   # 释放一个组的 payload
   ref = metadata.payload_ref
   channel = payload_channel_for(ref.locator, self._payload_channels)
   await channel.remove_key(key=ref.key, async_op=True).async_wait()

按调用顺序说明
--------------

1. **创建 session server 和 router。** 入口脚本创建 ``session_server.num_servers`` 个 ``SessionServerWorker`` 组，分散在各 rollout 节点上，再创建一个 ``SGLangRouterWorker`` 组，作为 ``session_servers`` 和 ``session_router`` 传给 runner。
2. **创建 payload channel。** ``create_payload_channels`` 接收基础名称、shard 数和 session server 的 worker 组，返回每个 shard 一个 channel，放在这个 server 所在的节点上。runner 在 ``__init__`` 里、任何 session server 启动之前创建它们，因为 ``serve`` 要用到。
3. **启动 router 和 session server。** rollout 引擎启动之后，``AgentRunner`` 会调用 ``_start_session_servers``。在这个方法里，runner：

   - 用 ``init_router`` 启动一个空的 router；
   - 用 ``register_server`` 逐个加入 ``rollout.get_server_url`` 返回的引擎地址；
   - 用 ``get_router_url`` 读取 router 的地址；
   - 对每个 session server 调用 ``serve`` 并保留 handle，因为 ``serve`` 会一直运行到 server 停止；
   - 调用 ``wait_until_serving``，它在 server 开始监听时返回，启动失败时抛出启动错误；
   - 用 ``get_server_address`` 读取 server 的地址。

   这个方法最后返回 session server 的地址、客户端的 generate 超时和 keep-alive，``AgentRunner`` 会把它们作为 ``session_server_urls``、``session_server_generate_timeout_s`` 和 ``session_server_keepalive_timeout_s`` 传给 ``agent_loop.init_worker``。如果 runner 不继承 ``AgentRunner``，需要自行把这三个参数传给 ``agent_loop.init_worker``。
4. **下发初始权重版本并启动循环。** 首次权重同步后，runner 先对每个 session server 调用 ``set_loaded_versions``\ （首次调用前所有请求都会返回 HTTP 500），再把所有 server 的 ``self._session_payload_channels`` 展平为一个列表，然后用 ``run_agentloop_rollout_continuously`` 启动 agent loop，用 ``run_continuously`` 启动 buffer worker，两者在整个训练期间一直运行。
5. **下发组。** buffer worker 用 ``issue_group`` 为每个 prompt 分配组 id，再把 prompt 包成带组 id 和 task name 的 ``RolloutRequest``，放到 agent loop 的输入 channel 上。
6. **交付或释放每个完成的组。** 一个组完成时，session server 往 agent output channel 放一个 ``GroupMetadata``；丢弃一个组时，放的是 ``DiscardedGroup``，这种组不用清理。已完成但被过滤掉的组，要删掉它的 payload：用 ``payload_channel_for`` 找到 ``payload_ref`` 所在的 channel，调用 ``remove_key``，否则数据会一直留在 channel 里。每丢掉一个组，就腾出一个名额，buffer worker 会补发新的 prompt。一批凑满后，每个组以带着 ``payload_ref`` 的 ``GroupDelivery`` 交给 actor。
7. **训练并监视 session server。** ``run_training`` 接收全部 payload channel，actor 据此取出每个交付组的数据。actor 训练时，runner 把 ``serve`` 的 handle 和其他 handle 一起轮询。这样 session server 停止时，训练会随之结束。
8. **更新已加载的权重版本。** 之后每次权重同步后，runner 再次对每个 session server 调用 ``set_loaded_versions``，传入各引擎加载的版本。session server 用其中最旧的版本作为每一轮权重版本的下界。
9. **收集指标。** 每一步，runner 对每个 session server 调用 ``get_metrics``，把各 server 的值相加。这次调用还会删除过期的丢弃组记录。
10. **停止。** 结束时，runner 先停掉每个 session server 并等它的 ``serve`` handle 返回，再停 router。这样停 router 时，已经没有请求在处理。

运行
----

``examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml`` 用 ``session_server.enable: true`` 开启上面的接入。按 :doc:`/rst_source/guides/session_server` 的说明运行：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server

测试
----

- 单测加在 ``tests/unit_tests/test_session_server.py`` 里，这个文件用模拟的 SGLang server 驱动 session server，运行命令是 ``pytest tests/unit_tests/test_session_server.py``。这个文件需要装有 fastapi 和 SGLang 的 agentic 环境，否则所有用例都会跳过。
- 在配置里加上 ``session_server.verify_assembly: true`` 跑几步。actor 会把每个组和 agent loop 不经 session server 装配的数据逐项比对，出现第一处不一致时训练就会失败。
