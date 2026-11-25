:orphan:

将 Session Server 接入自定义 Runner
====================================

本文以 ``AsyncAgentRunner`` 为参考，说明如何把 :doc:`Session Server </rst_source/guides/session_server>` 接入自定义 Runner。如果直接沿用 ``AsyncAgentRunner``，只需在配置中开启 ``session_server.enable: true``，无需阅读本页。

快速总览
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 层级 / 文件
     - 职责
   * - 入口脚本（``examples/agent/searchr1/train.py``）
     - 启动 ``SessionServerWorker`` 组与单个 ``SGLangRouterWorker``，并传入 Runner。
   * - Runner（``rlinf/runners/async_agent_runner.py``）
     - 创建 payload channel、启动 router 与 session server、同步权重版本、轮询服务句柄并在退出时依次停止。
   * - Buffer Worker（``rlinf/workers/rollout_buffer/rollout_buffer_worker.py``）
     - 分配唯一组 ID、清理中止或被过滤组的 payload，并向 Actor 交付完整批次。
   * - 配置校验（``rlinf/config.py``）
     - 在 ``validate_session_server_cfg`` 中放行目标 Runner 模式。

Runner 必须满足的两项契约
-------------------------

1. **每个组带唯一组 ID 与任务名。** Session Server 按 ``group_id`` 索引会话与 payload：在每个 ``RolloutRequest`` 上填写全局唯一的 ``group_ids`` 和 ``task_names``，并通过 ``run_agentloop_rollout_continuously`` 运行 Agent Loop。
2. **保证完整训练批次并回收丢弃数据。** 某个组被中止或过滤（如 ``reject_equal_reward``）时，必须通过 ``remove_key`` 删除其 payload 并补发新 prompt，确保 Actor 每步仍收到 ``rollout_batch_size`` 个组。

四阶段生命周期
--------------

``AsyncAgentRunner`` 按以下四个阶段编排 Session Server：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 阶段
     - 关键调用
   * - **1. 启动进程**
     - 入口脚本按轮询在 rollout 节点启动 ``session_server.num_servers`` 个 ``SessionServerWorker`` 组，并在首个 rollout 节点启动单个 ``SGLangRouterWorker``，传给 Runner；Runner 在 ``__init__`` 中调用 ``create_payload_channels`` 为每个 server 创建 ``payload_shards`` 个 payload channel。
   * - **2. 初始化服务**
     - Rollout 引擎就绪后，调用 ``init_router()``，把 ``rollout.get_server_url()`` 逐个注册进 router；对每个 session server 调用 ``serve(router_url, agent_output_channel, payload_channels)``\ （保留句柄）并 ``wait_until_serving()``，再把 ``session_server_urls`` 与超时参数传给 ``agent_loop.init_worker()``。首次权重同步后，调用 ``set_loaded_versions()``\ （调用前请求返回 HTTP 500），随后启动 Agent Loop 与 Buffer Worker。
   * - **3. 每步循环**
     - 将展平后的 ``payload_channels`` 传给 ``actor.run_training()``，同时轮询 ``_session_serve_handles`` 以在任一服务异常退出时终止训练；每次权重同步后调用 ``set_loaded_versions()``；每步调用 ``get_metrics()`` 汇总指标。
   * - **4. 清理退出**
     - 先停止全部 session server 并等待其 ``serve`` 句柄返回，**最后**\ 再调用 ``_session_router.stop()``，避免 router 退出时仍有在途请求。

参考实现
--------

以下为 ``AsyncAgentRunner`` 三个核心文件的精简接入代码：

.. tab-set::

   .. tab-item:: 1. 入口脚本
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
         # 1) 在 __init__ 中创建 payload channel
         for idx, server in enumerate(session_servers):
             self._session_payload_channels.append(
                 create_payload_channels(
                     f"SessionPayload-{idx}",
                     num_shards=cfg.session_server.payload_shards,
                     producer=server,
                 )
             )

         # 2) 先启动 router，再启动各 session server（_start_session_servers）
         self._session_router.init_router().wait()
         for url in [u for u in self.rollout.get_server_url().wait() if u is not None]:
             self._session_router.register_server(url).wait()
         router_url = self._session_router.get_router_url().wait()[0]
         for server, channels in zip(self._session_servers, self._session_payload_channels):
             self._session_serve_handles.append(
                 server.serve(router_url, self.agent_output_channel, channels)
             )
             server.wait_until_serving().wait()

         # 3) 同步权重版本、启动训练并轮询句柄
         for server in self._session_servers:
             server.set_loaded_versions(self._rollout_loaded_versions).wait()
         payload_channels = [ch for group in self._session_payload_channels for ch in group]
         actor_handle = self.actor.run_training(
             input_channel=self.rollout_channel,
             version=self.global_steps,
             payload_channels=payload_channels,
         )
         self._wait_actor(actor_handle, [agent_handle, buffer_handle, *self._session_serve_handles])

         # 4) 退出：先停 session server，再停 router
         for server in self._session_servers:
             server.stop().wait()
         for handle in self._session_serve_handles:
             handle.wait()
         self._session_router.stop().wait()

   .. tab-item:: 3. Buffer Worker
      :sync: ss-runner-buffer

      .. code-block:: python

         # rlinf/workers/rollout_buffer/rollout_buffer_worker.py
         # 分配唯一 group_id 与 task_name
         await self.buffer.wait_for_admission_slot()
         group_id, _ = self.buffer.issue_group(sample_idx, self.task_name)
         request.group_ids = [group_id]
         request.task_names = [self.task_name]
         await dataloader_channel.put(request, async_op=True).async_wait()

         # 处理完成或丢弃的组
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

运行与验证
----------

使用参考 Search-R1 配置并开启装配校验（集群准备见 :doc:`/rst_source/guides/session_server`）：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server \
     session_server.verify_assembly=true

正式训练关闭 ``verify_assembly`` 前，确认以下两项通过：

1. 无装配差异报错（Actor 会将每个组与不经过 Session Server 的结果逐字段比对，首次不一致即终止）。
2. ``rollout/session_server/payload_write_failed`` 保持为 ``0``，且 ``rollout/session_server/payload_queued_groups`` 随 ``rollout/buffer/size`` 同步波动、无持续堆积。
