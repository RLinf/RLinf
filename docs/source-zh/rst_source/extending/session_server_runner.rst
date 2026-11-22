将 session server 接入 runner
=============================

本页介绍如何在异步 agent runner（``AsyncAgentRunner``）以外的 runner 中接入 session server。RLinf 目前只在异步 runner 里接入了 session server，下面的步骤按异步 runner 的顺序排列。做完之后，agent loop 会把 generate 请求发给 session server，actor 用 session server 收集的组来训练。

准备工作
--------

- 带 session server 的异步 Search-R1 示例能在你的集群上运行，见 :doc:`/rst_source/guides/session_server`。
- 你的 runner 继承自 ``AgentRunner``。

要改的文件
----------

1. 入口脚本 ``examples/agent/searchr1/train.py``。开启 ``session_server.enable`` 时，入口脚本创建 ``session_server.num_servers`` 个 ``SessionServerWorker`` 组和一个 ``SGLangRouterWorker`` 组，一起传给 runner。这一段可以直接复用。
2. 你的 runner，按下一节的步骤接入。
3. 配置校验。``rlinf/config.py`` 里的 ``validate_session_server_cfg`` 要求开启 ``runner.async_mode``，需要为你的 runner 放开这项检查。

在 runner 中接入 session server
-------------------------------

每一步都可以对照 ``rlinf/runners/async_agent_runner.py`` 里异步 runner 的写法。

1. **创建 payload channel。** 在 ``__init__`` 里为每个 session server 调用一次 ``create_payload_channels``，放在这个 server 所在的节点上，``num_shards`` 取 ``cfg.session_server.payload_shards``。
2. **启动 router 和 session server。** 覆盖 ``AgentRunner._start_session_servers``，它在 rollout 引擎启动之后调用。先调用 ``init_router``，用 ``register_server`` 注册 ``rollout.get_server_url`` 返回的每个引擎地址，再用 ``get_router_url`` 取得 router 的地址。然后对每个 session server 发出 ``SessionServerWorker.serve``，不等它返回，调用 ``wait_until_serving`` 等它开始监听，再用 ``get_server_address`` 取得它的地址。最后返回 session server 的地址、客户端的 generate 超时和 keep-alive，``AgentRunner`` 会把它们交给 agent loop。
3. **监视 session server。** ``serve`` 只在 server 停止时返回。保留它的 handle，在等待 actor 时轮询。这样 session server 停止时训练会立即结束，actor 不会一直等数据。
4. **发送已加载的权重版本。** 每次权重同步后，对每个 session server 调用 ``set_loaded_versions``，传入各引擎加载的版本。session server 用其中最旧的版本作为每一轮权重版本的下界。
5. **收集指标。** 每一步对每个 session server 调用一次 ``get_metrics``，把各 server 的值相加。这次调用还会删除过期的作废组记录。
6. **交付组。** 一个组完成时，session server 往 ``agent_output_channel`` 放一个 ``GroupMetadata``；丢弃一个组时，放的是 ``DiscardedGroup``。

   - 要训练的组：把带着该组 ``payload_ref`` 的 ``GroupDelivery`` 放到 actor 的输入 channel 上，并把 payload channel 传给 ``run_training``。
   - 已完成但不训练的组（比如 reward 全部相同而被过滤掉的组）：对它的 ``payload_ref`` 指向的 payload channel 调用 ``remove_key``，否则这个组的数据会一直留在 channel 里。
   - ``DiscardedGroup`` 不用清理，session server 已经删掉了它的数据。

   异步 runner 里，这一步由 buffer worker（``RolloutBufferWorker``）完成。同步 runner 不要复用它，因为它会提前投放，让生成领先于训练。

测试
----

- 单测加在 ``tests/unit_tests/test_session_server.py`` 里，这个文件用模拟的 SGLang server 驱动 session server。
- 用你的 runner 开启 ``session_server.verify_assembly: true`` 跑几步。actor 会把每个组和 agent loop 不经 session server 装配的数据逐项比对，出现第一处不一致时训练就会失败。
- 异步 runner 可以用同样的方法检查：在配置里加上 ``session_server.verify_assembly: true`` 后运行 ``bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server``。
