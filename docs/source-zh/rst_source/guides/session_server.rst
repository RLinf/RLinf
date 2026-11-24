Session Server
==============

Session server 在 rollout 节点上收集 agent 的轨迹，并整理成训练数据。agent 把 generate 请求发给 session server，session server 转发给推理引擎，并记下每一轮的结果。

session server 还提供：

- **控制面与数据面分离。** buffer worker 只按组的元数据调度。actor 取走之前，轨迹一直留在 rollout 节点上。
- **经 SGLang router 路由。** 见 `选择路由策略`_。
- **OpenAI chat completions 接口。** 基于 OpenAI API 编写的 agent 可以发送消息，不必发送 token id；见 `使用 OpenAI chat completions 接口`_。

如果还不熟悉异步 runner，请先读 :doc:`异步 agentic RL <async_agentic>`。`开启 session server`_ 一节附有一个 Search-R1 的完整示例。

何时使用 session server
-----------------------

不开启 session server 时，agent loop 经 channel 把 generate 请求发给 rollout worker，buffer worker 保存每个组的完整训练数据，直到交付。

以下情况不必开启：

- 使用同步或评测 runner，或者需要 ``pause_mode: drain``；
- 每组数据量小，一次训练只占几个节点；
- 不想多部署进程和端口。

以下情况建议开启 session server：

- 同组轨迹共享长前缀；
- 每组数据量大，或者 node 0 内存紧张；
- agent 使用 OpenAI chat completions 接口。

开启 session server
-------------------

在异步 agentic 配置里加这个配置块：

.. code-block:: yaml

   session_server:
     enable: true

启动时，rollout 节点上会启动三个组件：

- **session server**：HTTP 服务，agent loop 向它发请求
- **payload channel**：每个分片一个，存放记录下来的轨迹
- **SGLang router**：只有一个，所有引擎都注册到它上面

开启 session server 需要满足以下条件：

- ``runner.async_mode: True``，actor 使用 Megatron
- ``rollout.sglang.backend_type: server``
- 使用 Search-R1 agent（``reward.reward_type: searchr1``）
- ``rollout.pause_mode`` 为 ``retract`` 或 ``in_place``，不能是 ``drain``
- 关闭 ``cluster.auto_scheduler``
- 安装 ``sglang_router`` 包

启动时，配置校验会逐项检查。

示例：Search-R1
^^^^^^^^^^^^^^^

下面的例子在两台机器上运行异步 Search-R1 训练，模型用 Qwen2.5-3B-Instruct，并开启 session server。

配置是 ``examples/agent/searchr1/config/train_qwen2.5_async_session_server.yaml``，它继承 ``train_qwen2.5_async.yaml``，加上 ``session_server.enable: true`` 和 ``rollout.router.policy: manual`` 两项。

``train_qwen2.5_async.yaml`` 里，Megatron actor 放在 node 0，SGLang server rollout（8 个实例，TP1）放在 node 1。每步用 ``rollout_batch_size`` 512 × ``group_size`` 4。

先按 :doc:`多机指南 <multi_node>` 在两台机器上启动 Ray 集群。再按 :doc:`Search-R1 </rst_source/examples/agentic/searchr1/index>` 启动检索服务，然后填好 ``train_qwen2.5_async.yaml`` 里的三项：

.. code-block:: yaml

   data:
     train_data_paths: ["/path/to/data/searchr1_train.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
   tools:
     search:
       server_addr: 0.0.0.0:8000  # 检索服务地址

然后在 head 节点上启动训练：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_async_session_server

下面的结果来自 2×8 A100-80G 上三次 100 步的训练：两次用这份配置，一次用 ``train_qwen2.5_async.yaml`` 且不开启 session server。

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 指标
     - 开启 session server
     - 不开启
   * - 轨迹 reward，最后 20 步均值
     - 0.376 / 0.375
     - 0.372
   * - 平均 data staleness
     - 2.0
     - 2.0

无论是否开启 session server，训练表现都一致。两次开启 session server 的训练都没有 payload 写入失败，``rollout/session_server/payload_queued_groups`` 始终与 ``rollout/buffer/size`` 同步变化。这些指标的含义见 :ref:`监控指标 <session-server-metrics>`。

工作原理
--------

轨迹怎样收集
^^^^^^^^^^^^

每个组依次经过四步。

1. **建立 session。** agent loop 为每条轨迹建立一个 session。同一组的 session 都在同一个 session server 上。
2. **生成。** session server 把每一轮请求转发给 router，并记下引擎的输出：token、logprob 和权重版本。它先回复 agent，再在后台把记录写进 payload channel。
3. **收尾。** agent loop 关闭每条轨迹并上报 reward，同时报告每一轮实际保留了多少 token。比如 Search-R1 会在 ``</search>`` 处截断一轮。
4. **交付。** 组完成后，session server 把元数据发给 buffer worker。buffer worker 交付这个组时，actor 从 payload channel 取出记录，装配成训练序列。

session server 还会检查每一轮的 prompt 是否接续上一轮。如果 agent 改写了历史，``prefix_*`` 指标会反映出来。

控制面与数据面
^^^^^^^^^^^^^^

开启 session server 后，调度和搬运数据分开：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 平面
     - 传递的内容
     - 由谁处理
   * - 控制面
     - 组的元数据：reward、权重版本、token 数，以及数据存放的位置
     - buffer worker：投放、丢弃、交付。runner：训练和权重同步。
   * - 数据面
     - generate 请求和记录下来的轨迹
     - session server、router 和引擎；数据存放在 payload channel 里，直到 actor 取走

调度逻辑不变。投放、缓冲上限、staleness 和丢弃规则都和不开启 session server 时一样，只是改为依据元数据。

buffer worker 是 node 0 上的单个进程。开启 session server 后，buffer worker 的内存不随训练数据增长。每组数据量越大，好处越明显，例如开启 routed-expert replay（R3）时，或者上下文很长时。

各 rollout worker 把权重同步等控制调用直接发给自己的引擎，不经过 router。

丢弃与失败
^^^^^^^^^^

buffer worker 丢弃一个组时，payload channel 会删除这个组的数据。agent 中止的组、因写入失败而作废的组也一样。

处理请求时出错，训练会失败：

- 只有连不上 router 时，session server 才会重发请求。这种请求没有到达引擎，同一轮不会重复采样。
- generate 请求超时或被引擎中止，训练失败。

唯一可以恢复的错误是单次 payload 写入失败或超时：session server 作废这个组，buffer worker 丢弃它。如果 payload channel 进程退出或通信失败，后续写入都无法成功，session server 会停止服务并让训练失败。

选择路由策略
------------

所有 prompt 共享长前缀时用 ``manual``，Search-R1 就属于这种情况。

session server 的所有请求都经 router 转发。RLinf 支持 router 的两种策略：

- ``manual``\ （默认）：组的第一个请求到来时，router 选当前负载最低的引擎。之后同一组 id 的请求都发到这个引擎。
- ``cache_aware``：对每个请求，router 选缓存与请求前缀匹配最好的引擎。只有引擎之间负载差距很大时，才按负载分配。同一组的请求通常前缀相同，但 router 不保证它们落在同一个引擎上。

如果所有 prompt 共享一段长前缀，``cache_aware`` 会把新组都堆到一个引擎上。Search-R1 的 prompt 都以相同的指令开头：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentic_rl/async/session_server_router_load.png
   :align: center
   :width: 85%

   每个 rollout 实例的运行中请求数（Qwen2.5-1.5B，12 个实例，每步 64 × 4，20 步）。``cache_aware`` 下实例 5 平均 27.5 个、峰值 79 个，其余实例在 4.8 个左右。``manual`` 下各实例都在 6 到 7.5 个之间。

只有一个 rollout 实例时，经过 router 只是多转发一次。这种配置可以运行，启动时打印一条 warning。

多节点部署
----------

开启 session server 后，引擎监听 ``0.0.0.0``，这样 router 才能访问其他节点上的引擎。

引擎上还有 RLinf 自己的控制接口，接受 pickle 格式的请求体。为了不让这些接口对外开放，来自其他机器的请求只能访问四个路径：``/generate``、``/health``、``/server_info`` 和 ``/model_info``。访问其余路径的请求在读取请求体之前就会被拒绝，返回 403。通过回环地址发来的请求不受限制。

空闲 HTTP 连接必须由客户端先断开，否则客户端可能复用服务端刚关掉的连接。配置校验会检查 ``keep_alive_s`` 和 ``engine_keep_alive_s`` 是否满足这条规则，通常保持默认值即可。

.. warning::

   执行 ``ray start`` 的 shell 和启动 ``train.py`` 的 shell 要把 ``NCCL_IB_HCA`` 设成相同的值。RLinf 会用 ``train.py`` 进程的环境变量覆盖各 worker 的环境变量。

监控与检查
----------

.. _session-server-metrics:

监控指标
^^^^^^^^

session server 每一步都在 ``rollout/session_server/`` 下上报指标。计数和耗时只统计当前这一步；字节数和队列长度取上报时的快照。

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 指标
     - 含义
   * - ``forward_count``、``forward_time_total_s``、``forward_time_avg_s``
     - 这一步转发的 generate 请求数，以及总耗时和平均耗时。平均耗时只在有转发的步里上报。
   * - ``prefix_continuous``、``prefix_rewritten``、``prefix_underreported``
     - 每一轮 prompt 与上一轮的关系。``prefix_underreported`` 应当一直为 0。``prefix_rewritten`` 统计 agent 改动了历史的轮；发 token id 的 agent 应当一直为 0，chat agent 则可能不为 0。
   * - ``payload_write_failed``
     - 因写入失败而作废的组数。应当一直为 0。
   * - ``pending_payload_bytes``
     - 还在写入的字节数。如果一直接近 ``max_pending_payload_bytes``，调大这个上限，或者增加 ``num_servers`` 或 ``payload_shards``。
   * - ``payload_queued_groups``、``payload_queued_bytes``
     - payload channel 里等待取走的数据。``payload_queued_groups`` 应当等于或略大于 ``rollout/buffer/size``。如果差值不断增大，说明有数据没有释放。
   * - ``chat_reasoning_responses``、``chat_tool_call_responses``、``chat_tool_call_parse_failed``
     - 解析出推理内容、解析出 tool call、以及 tool-call 块解析失败的 chat completions 回答数。

检查正确性
^^^^^^^^^^

第一次在新的 agent 上使用时，请开启 ``session_server.verify_assembly``。开启后，agent loop 还会再构建一遍每个组，做法与不开启 session server 时相同。actor 在训练前逐项比对两份结果：

- token id、prompt 与回复长度、rollout logprob
- reward、序列到轨迹的映射、轨迹结束标记
- 额外的训练字段
- 每个转发请求所用的采样参数

任何一项不一致，训练都会失败，并报出字段名。两份结果来自同一次生成，所以检查的是记录和装配，不是采样本身。这项检查会降低吞吐，正式训练时请关闭。

在 Search-R1 上，约 2,900 个组通过了这项检查。这些组来自两次 5 步的训练：Qwen2.5-1.5B 在 12 个实例上，Qwen2.5-3B 在 8 个实例上。

使用 OpenAI chat completions 接口
---------------------------------

每个 session server 也在 ``/sessions/<session id>/v1/chat/completions`` 提供 OpenAI chat completions 接口。session server 用 ``agentloop.agentic_tokenizer`` 指定的模型族的 chat 模板把消息渲染成 token id，再像 ``generate`` 请求那样发给引擎。返回时用 OpenAI 的 ``chat.completion`` 格式，推理与工具调用分别解析到 ``reasoning_content`` 和 ``tool_calls``。只有词表里带推理标记的模型（Qwen3、GLM-4.7 等）才拆分推理内容。

``agentloop.agentic_tokenizer`` 填模型族，``rollout.sglang.tool_call_parser`` 填与之配套的 parser，启动时会校验这一对：

.. code-block:: yaml

   agentloop:
     agentic_tokenizer: qwen     # GLM-4.7 填 glm47
   rollout:
     sglang:
       tool_call_parser: qwen25  # GLM-4.7 填 glm47

其他模型填 ``agentic_tokenizer: default``，它只渲染消息，不解析 tool call，``tool_call_parser`` 可以不填。不设 ``agentloop.agentic_tokenizer`` 时，chat 请求返回 500。

使用时，写一个自行执行每条轨迹的 agent loop，写法见下面的示例。agent loop 仍负责创建和关闭 session、上报 reward、丢弃失败的组；你的代码只需在已经打开的 session 上执行一条轨迹：

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
           messages = build_messages(prompt)  # 你的 prompt
           message = None
           num_responses = 0
           async with AsyncOpenAI(
               base_url=base_url,
               api_key="EMPTY",
               max_retries=0,  # 重发的请求会被重新采样
           ) as client:
               for _ in range(MAX_TURNS):
                   try:
                       response = await client.chat.completions.create(
                           model="policy", messages=messages, tools=TOOLS
                       )
                   except BadRequestError as exc:
                       if exc.code != "context_length_exceeded":
                           raise
                       break  # 没有空间再跑一轮
                   except APIStatusError as exc:
                       if exc.status_code != 410:
                           raise
                       # session server 已经丢弃了这个组。
                       raise GroupDiscarded("group_discarded") from exc
                   num_responses += 1
                   message = response.choices[0].message
                   messages.append(message.model_dump(exclude_none=True))
                   if not message.tool_calls:
                       break
                   for call in message.tool_calls:
                       result = await run_tool(call)  # 你的工具
                       messages.append(
                           {"role": "tool", "tool_call_id": call.id, "content": result}
                       )
           if num_responses == 0:
               # 第一轮请求就放不下：丢弃整个组。
               raise GroupDiscarded("context_length_exceeded")
           content = (message.content or "") if message is not None else ""
           reward = score(content, answer)  # 你的打分
           close_fields = {
               # 每个训练的回答给一个值：一条轨迹一个 agent。
               "extra_fields_train": {"idx_to_sub_traj": [0] * num_responses}
           }
           return None, close_fields, reward

``input_ids`` 是数据集 prompt 的 token id，已经应用过 chat 模板。去掉特殊 token 解码后，文字里仍有模板的角色行（如 ``system``、``user``），所以 ``build_messages`` 要先从中取出任务，再构造消息。

在入口脚本里启动这个类，替换原来发送 token id 的 agent loop，写法参照 ``examples/agent/searchr1/train.py`` 启动 Search-R1 agent loop 的方式。即使由自己的 agent loop 打分，配置里也必须保留 ``reward.reward_type: searchr1``，否则启动校验会拒绝该配置。``algorithm.loss_scales`` 里包含 ``agent_level`` 时需要 ``idx_to_sub_traj``。

返回 410 说明 session server 已经丢弃了这个组，比如 payload 写入失败之后。示例随即丢弃整个组，和 agent loop 处理其他无法完成的组一样。

客户端要设置 ``max_retries=0``：重发的请求会被重新采样，两次回答只有一次参与训练。

请求放不下 ``runner.seq_length`` 时，示例结束这条轨迹，用最后一条回答打分；如果第一轮请求就放不下，示例会丢弃整个组。

session server 会复用之前各轮的 token：一轮的 prompt 是上一轮的 prompt 加上输出 token，再接上新消息；这些输出 token 与引擎生成的完全一致。如果按文本重新渲染历史，得到的 token 可能不同。session server 按以下三种情况处理请求：

- **接续**：请求以已保存的对话开头，后面追加 tool 或 user 消息。只渲染新消息，因此前缀缓存仍能命中。如果 ``clear_thinking`` 为 true，新增 user 消息时会重新渲染整段对话，因为此时模板会去掉之前各轮的思考内容。
- **重发**：请求与上一次请求相同，说明 agent 没有用到上一轮的回答。session server 把对话回退到那一轮之前。被放弃的那一轮仍留在 payload 中，但不参与训练。
- **改写**：其他请求按其中的完整消息重新渲染。例如 agent 为了节省上下文截短或修改了之前的回答，就会发出这类请求。``prefix_rewritten`` 会统计这类请求，它们的数据照常参与训练。

请求接续对话时，session server 使用自己保存的前几轮内容，包括 ``reasoning_content``。所以客户端从历史里去掉思考内容，对话照样能接续。

一轮的 prompt 和回答的总长度不能超过 ``runner.seq_length``，也就是 actor 训练的长度。接口会把 ``max_tokens`` 降到剩余长度；prompt 本身放不下时返回 400，错误码是 ``context_length_exceeded``。

以下情况也会返回 400 和 OpenAI 格式的错误：

- ``n`` 大于 1
- ``temperature``、``top_p`` 或 ``top_k`` 与配置值不同
- ``tool_choice`` 不是 ``auto`` 或 ``none``
- 给出了 ``response_format``
- ``chat_template_kwargs`` 与配置不一致

一个 session 同一时间只处理一个请求，两个端点都是如此。上一个请求还在处理时，新的请求或 close 会收到 409。

配置参考与限制
--------------

配置参考
^^^^^^^^

``session_server``：

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - 配置项
     - 默认值
     - 作用
   * - ``enable``
     - ``False``
     - 开启 session server。
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
     - 单次转发请求的时间上限，包括权重同步期间的暂停。超过即训练失败。
   * - ``per_task.<task>.generate_timeout_s``
     - ``generate_timeout_s``
     - 按任务覆盖上面的超时。
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
     - 把每个组与不开启 session server 时的装配结果逐项比对；见 `检查正确性`_。
   * - ``chat_template_kwargs``
     - ``{}``
     - chat completions 接口使用的 chat 模板参数，例如 GLM 的 ``clear_thinking``。请求中给出其他值会被拒绝。

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

限制
^^^^

- 要求异步 runner、Megatron actor、SGLang server 后端和 Search-R1 agent。要接入其他 runner，见 :doc:`/rst_source/extending/session_server_runner`。
- chat completions 接口不支持真正的流式输出。``stream: true`` 时，整轮结果在一个 chunk 中返回，随后是 ``[DONE]``。
- chat completions 接口不支持 ``tool_choice: required``、指定函数和 ``response_format``，并忽略 ``seed``。
- ``session_server.verify_assembly`` 只适用于发送 token id 的 agent loop。开启它时，chat completions 接口会拒绝所有请求。
- agent 可以附带 ``extra_fields_train``；其他按轮、按轨迹或按组的额外字段，session server 一律拒绝。
- 使用 session server 时，``agentloop.print_outputs`` 只在开启 ``verify_assembly`` 后才打印。
- session server 退出时训练会结束，不会自动重启。
