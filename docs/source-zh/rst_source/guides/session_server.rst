Session Server
==============

Session server 在 rollout 节点上收集 agent 的轨迹，并整理成训练数据。agent 把 generate 请求发给 session server，session server 转发给推理引擎，并记下每一轮的结果。

session server 还有两个特点：

- **控制面与数据面分离。** buffer worker 只按组的元数据调度。actor 取走之前，轨迹一直留在 rollout 节点上。
- **经 SGLang router 路由。** 可以使用 router 自带的两种策略：``manual`` 把每个组固定在同一个引擎上，``cache_aware`` 按前缀亲和选引擎。

目前只有异步 runner 接入了 session server；如果还不熟悉异步 runner，请先读 :doc:`异步 agentic RL <async_agentic>`。完整配方见 Search-R1 异步训练中的 :ref:`Session Server 路径 <searchr1-async-session-server>`。

开启 session server
-------------------

在异步 agentic 配置里加两个配置块：

.. code-block:: yaml

   session_server:
     enable: true          # agent 改为向 session server 发请求
     num_servers: 1        # session server 个数，分布在各 rollout 节点上
     payload_shards: 1     # 每个 session server 的 payload channel 数

   rollout:
     router:
       policy: manual      # 或 cache_aware

只有 ``enable`` 必须设置，上面其他值都是默认值。

入口脚本还会在 rollout 节点上启动三个组件：

- **session server**：HTTP 服务，agent loop 向它发请求
- **payload channel**：每个分片一个，存放记录下来的轨迹
- **SGLang router**：只有一个，所有引擎都注册到它上面

示例配置：

.. code-block:: text

   examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml

前提条件
--------

session server 路径要求：

- ``runner.async_mode: True``，actor 使用 Megatron
- ``rollout.sglang.backend_type: server``
- 使用 Search-R1 agent（``reward.reward_type: searchr1``）
- ``rollout.pause_mode`` 为 ``retract`` 或 ``in_place``，不能是 ``drain``
- 关闭 ``cluster.auto_scheduler``
- 安装 ``sglang_router`` 包

启动时，配置校验会逐项检查。

轨迹怎样收集
------------

每个组依次经过四步。

1. **建立 session。** agent loop 为每条轨迹建立一个 session。同一组的 session 都在同一个 session server 上。
2. **生成。** session server 把每一轮请求转发给 router，并记下引擎的输出：token、logprob 和权重版本。它先回复 agent，再在后台把记录写进 payload channel。
3. **收尾。** agent loop 关闭每条轨迹并上报 reward，同时报告每一轮实际保留了多少 token。比如 Search-R1 会在 ``</search>`` 处截断一轮。
4. **交付。** 组完成后，session server 把元数据发给 buffer worker。buffer worker 交付这个组时，actor 从 payload channel 取出记录，装配成训练序列。

session server 还会检查每一轮的 prompt 是否接续上一轮。如果 agent 改写了历史，``prefix_*`` 指标会反映出来。

控制面与数据面
--------------

默认路径上，buffer worker 收到每个组的完整训练数据，一直存到交付。负责调度的进程同时在搬运数据。

session server 路径把两者分开：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 面
     - 传递的内容
     - 由谁处理
   * - 控制面
     - 组的元数据：reward、权重版本、token 数，以及数据存放的位置
     - buffer worker：投放、丢弃、交付。runner：训练和权重同步。
   * - 数据面
     - generate 请求和记录下来的轨迹
     - session server、router 和引擎；数据存放在 payload channel 里，直到 actor 取走

调度逻辑不变。投放、缓冲上限、staleness 和丢弃规则都和默认路径一样，只是改为依据元数据。

buffer worker 是 node 0 上的单个进程（见 :doc:`多机指南 <multi_node>`）。在这条路径上，它的内存不随训练数据增长。开启 routed-expert replay（R3）或上下文很长时，每组数据量大，这一点的好处最明显。

各 rollout worker 把权重同步等控制调用直接发给自己的引擎，不经过 router。

选择路由策略
------------

session server 的所有请求都经 router 转发。RLinf 支持 router 的两种策略：

- ``manual``\ （默认）：组的第一个请求到来时，router 选当前负载最低的引擎。之后同一组 id 的请求都发到这个引擎。
- ``cache_aware``：对每个请求，router 选缓存与请求前缀匹配最好的引擎。只有引擎之间负载差距很大时，才按负载分配。同一组的请求通常前缀相同，但 router 不保证它们落在同一个引擎上。

如果所有 prompt 共享一段长前缀，``cache_aware`` 会把新组都堆到一个引擎上。Search-R1 的 prompt 都以相同的指令开头：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentic_rl/async/session_server_router_load.png
   :align: center
   :width: 85%

   每个 rollout 实例的运行中请求数（Qwen2.5-1.5B，12 个实例，每步 64 × 4，20 步）。``cache_aware`` 下实例 5 平均 27.5 个、峰值 79 个，其余实例在 4.8 个左右。``manual`` 下各实例都在 6 到 7.5 个之间。

有共享前缀的工作负载请用 ``manual``。prompt 从开头就各不相同时，``cache_aware`` 可能更合适，但这种情况还没有实测。

只有一个 rollout 实例时，经过 router 只是多转发一次。这种配置可以运行，启动时打印一条 warning。

丢弃与失败
----------

buffer worker 丢弃一个组时，payload channel 会删除这个组的数据。agent 中止的组、因写入失败而作废的组也一样。

请求路径上的错误会让 run 失败：

- 只有连不上 router 时，session server 才会重发请求。这种请求没有到达引擎，同一轮不会重复采样。
- generate 请求超时或被引擎中止，run 失败。

唯一可以恢复的错误是 payload 写入失败。session server 作废这个组，buffer worker 丢弃它。

选择路径
--------

以下情况用 **默认路径**\ （channel 路径）：

- 使用同步或评测 runner，或者需要 ``pause_mode: drain``
- 每组数据量小，run 只占几个节点
- 不想部署额外的进程和端口

以下情况用 **session server 路径**：

- 同组轨迹共享长前缀
- 每组数据量大，或者 node 0 内存紧张

多节点部署
----------

有多个 rollout 节点时，引擎改为监听 ``0.0.0.0``，不再只监听回环地址，这样 router 才能访问其他节点上的引擎。

引擎上还有 RLinf 自己的控制路由，它们接受 pickle 格式的请求体。为了不让这些控制路由对外开放，来自其他机器的请求只能访问四个路径：``/generate``、``/health``、``/server_info`` 和 ``/model_info``。访问其余路径的请求在读取请求体之前就会被拒绝，返回 403。来自本机的请求不受限制。

空闲 HTTP 连接必须由客户端先断开，否则客户端可能复用服务端刚关掉的连接。配置校验会检查 ``keep_alive_s`` 和 ``engine_keep_alive_s`` 是否满足这条规则，通常保持默认值即可。

.. warning::

   执行 ``ray start`` 的 shell 和启动 ``train.py`` 的 shell 要把 ``NCCL_IB_HCA`` 设成相同的值。RLinf 会用 ``train.py`` 进程的环境变量覆盖各 worker 的环境变量。

.. _session-server-metrics:

监控指标
--------

session server 每一步都在 ``rollout/session_server/`` 下上报指标。计数和耗时只统计当前这一步；字节数和队列长度取上报时的快照。

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 指标
     - 含义
   * - ``forward_count``、``forward_time_total_s``、``forward_time_avg_s``
     - 这一步转发的 generate 请求数，以及总耗时和平均耗时。
   * - ``prefix_continuous``、``prefix_rewritten``、``prefix_underreported``
     - 每一轮 prompt 与上一轮的关系。后两个应当一直为 0。不为 0 说明 agent 在轮与轮之间改动了历史。
   * - ``payload_write_failed``
     - 因写入失败而作废的组数。应当一直为 0。
   * - ``pending_payload_bytes``
     - 还在写入的字节数。如果一直接近 ``max_pending_payload_bytes``，调大这个上限，或者增加 ``num_servers`` 或 ``payload_shards``。
   * - ``payload_queued_groups``、``payload_queued_bytes``
     - payload channel 里等待取走的数据。``payload_queued_groups`` 应当等于或略大于 ``rollout/buffer/size``。如果差值不断增大，说明有数据没有释放。

检查正确性
----------

第一次在新的 agent 上使用时，请开启 ``session_server.verify_assembly``。开启后，agent loop 还会按默认路径的方式再构建一遍每个组。actor 在训练前逐项比对两份结果：

- token id、prompt 与回复长度、rollout logprob
- reward、序列到轨迹的映射、轨迹结束标记
- 额外的训练字段
- 每个转发请求所用的采样参数

任何一项不一致，run 都会失败，并报出字段名。两份结果来自同一次生成，所以检查的是记录和装配，不是采样本身。这项检查会降低吞吐，正式训练时请关闭。

在 Search-R1 上，约 2,900 个组通过了这项检查。它们来自两次 5 步的 run：Qwen2.5-1.5B 在 12 个实例上，Qwen2.5-3B 在 8 个实例上。

配置参考
--------

``session_server``：

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - 配置项
     - 默认值
     - 作用
   * - ``enable``
     - ``False``
     - 开启 session server 路径。
   * - ``num_servers``
     - ``1``
     - session server 个数，按轮询分布到各 rollout 节点。
   * - ``payload_shards``
     - ``1``
     - 每个 session server 的 payload channel 数，每个 payload channel 是一个独立进程。写入成为 CPU 瓶颈时调大。
   * - ``port``
     - ``8030``
     - 第一个 session server 的端口。第 ``i`` 个使用 ``port + i``。
   * - ``generate_timeout_s``
     - ``600``
     - 单次转发请求的时间上限，包括权重同步期间的暂停。超过即 run 失败。
   * - ``per_task.<task>.generate_timeout_s``
     - ``generate_timeout_s``
     - 按 task 覆盖上面的超时。
   * - ``payload_put_timeout_s``
     - ``300``
     - 单次 payload 写入的时间上限。超过即作废这个组。
   * - ``max_pending_payload_bytes``
     - 512 MiB
     - 尚未写完的字节数上限。超过后，session server 等这一轮写完才回复 agent，宁可放慢 agent，也不让内存继续增长。
   * - ``keep_alive_s``
     - ``5``
     - session server 保持空闲连接的时长，单位为整秒。
   * - ``engine_keep_alive_s``
     - ``120``
     - 引擎保持空闲连接的时长，单位为整秒。必须大于 router 连接池 50 秒的空闲超时。
   * - ``verify_assembly``
     - ``False``
     - 把每个组与默认路径的装配结果逐项比对；见 `检查正确性`_。

``rollout.router``\ （与 :doc:`SGLang Server & Router <sglang_server>` 中的 ``router_server_args`` 相互独立）：

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - 配置项
     - 默认值
     - 作用
   * - ``policy``
     - ``manual``
     - ``manual`` 或 ``cache_aware``；见 `选择路由策略`_。
   * - ``assignment_mode``
     - ``min_load``
     - ``manual`` 为新组选引擎的方式。
   * - ``request_timeout_secs``
     - ``1800``
     - router 自身的超时。不能小于任何一个 generate 超时。
   * - ``max_idle_secs``
     - ``600``
     - router 保留组与引擎对应关系的时长，从组的最后一个请求起算。
   * - ``disable_health_check``
     - ``True``
     - 必须保持 ``True``。router 的健康检查会生成 token。
   * - ``disable_retries``
     - ``True``
     - 禁止 router 重发请求。

在其他 runner 上使用
--------------------

session server 本身不依赖异步训练，目前只有异步 runner 接入了它。其他 runner（比如同步的 agent runner）要接入，需要完成以下工作：

- 启动 router 和 session server，并注册各引擎。agent runner 已为此留好启动钩子。钩子返回 session server 的 URL 后，agent loop 改走这条路径。
- 监视 session server，出错时结束 run。
- 每次权重同步后，把各引擎加载的版本发给 session server。
- 每一步收集一次 session server 的指标。
- 把 payload channel 传给 actor。
- 把每个组的元数据转成 actor 可以取数的交付对象，并删除被丢弃组的数据。

在异步路径上，最后一项由 buffer worker 完成。同步 runner 不应使用 buffer worker：buffer worker 会超量投放，让生成领先于训练，而同步训练不允许这样。还要去掉配置校验中要求 ``runner.async_mode`` 的检查。这些改动都没有在异步 runner 之外测试过。

限制
----

- 要求异步 runner、Megatron actor、SGLang server 后端和 Search-R1 agent。
- agent 可以附带 ``extra_fields_train``。按轮、按轨迹或按组附带的其他额外字段，session server 一律拒绝。
- ``agentloop.print_outputs`` 只在开启 ``verify_assembly`` 时生效。agent loop 在装配时打印，而在这条路径上只有检查时才装配。
- 只支持 HTTP（``session_server.type: http``）。
- session server 退出时 run 会结束，不会自动重启。
