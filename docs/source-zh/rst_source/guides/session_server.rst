Session Server
==============

Session Server 直接在 rollout 节点上收集多轮 Agent 轨迹并装配为训练数据。开启后，轨迹 payload 留在 rollout 侧，不再经由 node 0 上的 Buffer Worker 中转；请求通过 SGLang Router 转发，并同时提供 OpenAI 兼容的 ``/v1/chat/completions`` 接口。

如尚未了解异步 Agentic 训练，请先阅读 :doc:`异步 Agentic RL <async_agentic>`。

概览
----

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 维度
     - 说明
   * - **适用场景**
     - 同组轨迹共享长前缀、每组数据量大（如长上下文或 R3）、node 0 内存紧张，或 Agent 直接调用 OpenAI chat completions 接口。
   * - **保持默认**
     - 使用同步或评测 Runner、需要 ``pause_mode: drain``，或小规模训练不希望额外部署服务进程。
   * - **启动组件**
     - Rollout 节点上的 ``session_server.num_servers`` 个 HTTP 服务、每个服务 ``payload_shards`` 个 payload channel，以及 1 个共享 SGLang Router。

开启与启动
----------

在异步 Agentic 配置中添加：

.. code-block:: yaml

   session_server:
     enable: true

启动校验要求满足以下条件：

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 配置项 / 依赖
     - 要求
   * - ``runner.async_mode`` 与 Actor
     - ``True``，且 Actor 使用 Megatron
   * - ``rollout.sglang.backend_type``
     - ``server``
   * - ``reward.reward_type``
     - ``searchr1``\ （Search-R1 Agent）
   * - ``rollout.pause_mode``
     - ``retract`` 或 ``in_place``\ （不支持 ``drain``）
   * - ``cluster.auto_scheduler``
     - ``False``
   * - Python 包
     - 已安装 ``sglang_router``

示例：Search-R1
~~~~~~~~~~~~~~~

``examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml`` 继承 ``train_qwen2.5_async.yaml``\ （node 0 运行 Megatron Actor，node 1 运行 8 个 TP1 SGLang Rollout 实例），并开启 Session Server 与 ``rollout.router.policy: manual``。

按 :doc:`多节点部署 <multi_node>` 启动双机 Ray 集群并按 :doc:`Search-R1 </rst_source/examples/agentic/searchr1/index>` 启动检索服务后，在 head 节点传入本地路径直接启动：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server \
     'data.train_data_paths=["./data/searchr1_train.jsonl"]' \
     rollout.model.model_path=./models/Qwen2.5-3B-Instruct \
     tools.search.server_addr=0.0.0.0:8000

工作原理与路由策略
------------------

Session Server 将调度（**控制面**）与轨迹存储（**数据面**）分离：

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - 平面
     - 传递内容
     - 负责组件
   * - **控制面**
     - 组元数据（reward、权重版本、token 数、payload 引用）
     - Buffer Worker（准入、丢弃、交付）与 Runner（训练、权重同步）
   * - **数据面**
     - Generate 请求与记录的多轮轨迹 payload
     - Session Server、SGLang Router、推理引擎与 payload channel（直到 Actor 拉取）

每个组依次经过四个步骤：

1. **建立会话：** Agent Loop 为每条轨迹打开一个 session，同组全部 session 固定路由到同一个 Session Server。

2. **生成记录：** Session Server 经 Router 转发每轮请求，记录 token、logprob 与权重版本，先回复 Agent，再在后台写入 payload channel。

3. **收尾关闭：** Agent Loop 关闭轨迹、上报 reward，并记录每轮保留的 token 长度（例如 Search-R1 在 ``</search>`` 处截断）。

4. **交付装配：** 整组完成后元数据发往 Buffer Worker；交付时 Actor 按引用拉取 payload 并装配训练序列。

权重同步等控制调用仍由各 Rollout Worker 直接发给对应引擎，不经过 Router。

.. note::

   **故障语义：** 仅连不上 Router 时才会重试（请求尚未到达引擎）。单次 payload 写入失败或超时只会作废并丢弃当前组；generate 超时、引擎中止或 payload channel 崩溃会直接终止训练。

选择路由策略
~~~~~~~~~~~~

通过 ``rollout.router.policy`` 指定 Router 的引擎分配策略：

- ``manual``\ （默认）：为每个组的首轮请求选择负载最低的引擎，同组后续请求固定发往同一引擎。**所有 prompt 共享长前缀时（如 Search-R1）必须选此项。**
- ``cache_aware``：按 KV cache 前缀匹配选择引擎。若所有 prompt 以相同系统指令开头，新组会集中堆到单个引擎上，导致严重负载不均：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentic_rl/async/session_server_router_load.png
   :align: center
   :width: 85%

   各 Rollout 实例的并发请求数（Qwen2.5-1.5B，12 实例，每步 64 × 4）。``cache_aware`` 下实例 5 峰值达 79、其余仅约 4.8；``manual`` 下各实例均保持在 6–7.5。

仅有单个 Rollout 实例时仍可运行，启动时会提示一条多经过一跳 Router 的 warning。

多节点部署
----------

- **网络访问控制：** Rollout 引擎监听 ``0.0.0.0`` 供跨节点 Router 访问。非本机请求仅放行 ``/generate``、``/health``、``/server_info`` 与 ``/model_info``，其余控制接口直接返回 HTTP 403（回环地址不受限）。

- **空闲连接保活：** 客户端必须先于服务端关闭空闲连接；配置校验会检查 ``keep_alive_s`` 与 ``engine_keep_alive_s``，通常保持默认值即可。

.. warning::

   执行 ``ray start`` 与启动 ``train.py`` 的两个 shell 中必须设置相同的 ``NCCL_IB_HCA``，因为 RLinf 会把 ``train.py`` 的环境变量同步到各 Worker。

监控、校验与配置参考
--------------------

.. _session-server-metrics:

监控指标
~~~~~~~~

每步在 ``rollout/session_server/`` 下上报（计数与耗时为单步统计，字节数与队列长度为瞬时快照）：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 指标
     - 观察要点
   * - ``forward_count``、``forward_time_total_s``、``forward_time_avg_s``
     - 单步转发请求数及总/平均耗时（仅 ``forward_count > 0`` 时上报平均耗时）。
   * - ``prefix_continuous``、``prefix_rewritten``、``prefix_underreported``
     - 前缀连续性。``prefix_underreported`` 应始终为 0；token-ID Agent 的 ``prefix_rewritten`` 应为 0（Chat Agent 可能不为 0）。
   * - ``payload_write_failed``
     - 因 payload 写入失败作废的组数（应始终为 0）。
   * - ``pending_payload_bytes``
     - 在途写入字节数。长期接近 ``max_pending_payload_bytes`` 时，调大该上限或增加 ``num_servers`` / ``payload_shards``。
   * - ``payload_queued_groups``、``payload_queued_bytes``
     - 队列中待拉取的数据。``payload_queued_groups`` 应与 ``rollout/buffer/size`` 基本一致，差值持续扩大说明存在未释放数据。
   * - ``chat_reasoning_responses``、``chat_tool_call_responses``、``chat_tool_call_parse_failed``
     - 解析出推理内容、工具调用及解析失败的 chat completions 响应数。

装配正确性校验
~~~~~~~~~~~~~~

首次接入新的 token-ID Agent 时，建议先开启 ``session_server.verify_assembly=true`` 短跑几步：Actor 会在训练前将 Session Server 装配结果与直连 channel 路径逐字段比对（token、长度、logprob、reward、额外字段与采样参数），不一致即报错并指出字段名。该开关影响吞吐，正式训练请关闭。

.. dropdown:: 完整配置参考（session_server 与 rollout.router）

   ``session_server``：

   .. list-table::
      :header-rows: 1
      :widths: 34 16 50

      * - 配置项
        - 默认值
        - 说明
      * - ``enable``
        - ``False``
        - 开启 Session Server 数据面。
      * - ``num_servers``
        - ``1``
        - Session Server 数量，按轮询分布到各 rollout 节点。
      * - ``payload_shards``
        - ``1``
        - 每个 server 的 payload channel 进程数；写入出现 CPU 瓶颈时调大。
      * - ``port``
        - ``8030``
        - 起始端口（第 ``i`` 个 server 使用 ``port + i``）。
      * - ``generate_timeout_s``
        - ``600``
        - 单次转发超时（含权重同步暂停）；超时终止训练。
      * - ``per_task.<task>.generate_timeout_s``
        - ``generate_timeout_s``
        - 按任务覆盖上述超时。
      * - ``payload_put_timeout_s``
        - ``300``
        - 单次 payload 写入超时；超时作废该组。
      * - ``max_pending_payload_bytes``
        - 512 MiB
        - 后台写入反压预算；超过后回复需等待写入完成以限制内存。
      * - ``keep_alive_s``
        - ``5``
        - Session Server 空闲连接保活秒数（整数）。
      * - ``engine_keep_alive_s``
        - ``120``
        - 引擎空闲连接保活秒数（须大于 Router 的 50 秒连接池超时）。
      * - ``verify_assembly``
        - ``False``
        - 与直连 channel 装配结果逐项比对（仅限 token-ID Agent）。
      * - ``chat_template_kwargs``
        - ``{}``
        - 固定传给 ``/v1/chat/completions`` 模板的参数（如 ``clear_thinking``）。

   ``rollout.router``\ （与 :doc:`SGLang Server 与 Router <sglang_server>` 中的 ``router_server_args`` 相互独立）：

   .. list-table::
      :header-rows: 1
      :widths: 34 16 50

      * - 配置项
        - 默认值
        - 说明
      * - ``policy``
        - ``manual``
        - ``manual`` 或 ``cache_aware``。
      * - ``assignment_mode``
        - ``min_load``
        - ``manual`` 为新组选择引擎的策略。
      * - ``request_timeout_secs``
        - ``1800``
        - Router 请求超时（须 ``>=`` 所有 generate 超时）。
      * - ``max_idle_secs``
        - ``600``
        - 组最后一次请求后，Router 保留组与引擎绑定关系的时长。
      * - ``disable_health_check``
        - ``True``
        - 须保持 ``True``\ （Router 健康检查会触发 token 生成）。
      * - ``disable_retries``
        - ``True``
        - 禁止 Router 静默重发请求。

.. dropdown:: 当前限制

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 维度
        - 限制说明
      * - Runner 与后端
        - 仅支持异步 Runner、Megatron Actor、SGLang Server 后端与 Search-R1 Agent；接入自定义 Runner 见 :doc:`/rst_source/extending/session_server_runner`。
      * - 流式输出
        - 不支持逐 token 流式返回：``stream: true`` 时整轮结果以单个 chunk 返回，随后发送 ``[DONE]``。
      * - Chat 参数
        - 不支持 ``tool_choice: required``、指定函数与 ``response_format``，且忽略 ``seed``。
      * - 装配校验
        - ``session_server.verify_assembly`` 仅支持 token-ID Agent，开启后拒绝 chat completions 请求。
      * - 额外字段与日志
        - 仅允许 ``extra_fields_train``；``agentloop.print_outputs`` 仅在开启 ``verify_assembly`` 时生效。
      * - 容错
        - 任一 Session Server 异常退出时训练直接结束，不自动重启。

相关扩展
--------

- **接入外部黑盒 Agent：** 参见 :doc:`黑盒 Agent </rst_source/extending/openai_agent>`，将 ``AsyncOpenAI`` 直接指向 ``/sessions/<session id>/v1/chat/completions``。
- **将 Session Server 接入自定义 Runner：** 如需在内置异步 Runner 之外的自建 Runner 中编排 Session Server，参见 :doc:`将 Session Server 接入自定义 Runner </rst_source/extending/session_server_runner>`。
