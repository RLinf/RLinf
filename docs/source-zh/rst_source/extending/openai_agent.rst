.. _session-server-openai-endpoint:

黑盒 Agent
==========

**黑盒 Agent** 指在 RLinf 之外独立编排、仅通过 OpenAI 兼容 API（``/v1/chat/completions``）调用模型的 Agent（例如基于 OpenAI SDK、AutoGen 或自有框架的已有 Agent）。开启 :doc:`Session Server <../guides/session_server>` 后，只需把客户端地址指向 RLinf 暴露的 OpenAI 接口，即可免重写接入并自动收集训练轨迹。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 项目
     - 说明
   * - 扩展点
     - 继承 ``MultiAgentLoopWorker`` 并覆写 ``run_session_trajectory(self, binding, input_ids, answer)``
   * - 接口地址
     - ``/sessions/<session id>/v1/chat/completions``\ （通过 ``self.session_server_client.session_base_url(...)`` 获取）
   * - 必需配置
     - ``session_server.enable: true``、``agentloop.agentic_tokenizer`` 与配套的 ``rollout.sglang.tool_call_parser``
   * - 前置条件
     - 已按 :doc:`Session Server <../guides/session_server>` 完成异步 Agentic 环境与服务配置

1. 配置 Tokenizer 与工具解析器
------------------------------

开启 Session Server，在 ``agentloop.agentic_tokenizer`` 指定模型族，并在 ``rollout.sglang.tool_call_parser`` 填对应的 SGLang 解析器（启动时校验匹配关系）：

.. code-block:: yaml

   session_server:
     enable: true
   agentloop:
     agentic_tokenizer: qwen     # GLM-4.7 填 glm47；不解析工具调用填 default
   rollout:
     sglang:
       tool_call_parser: qwen25  # GLM-4.7 填 glm47；default 时省略

- 响应采用 OpenAI ``chat.completion`` 格式：带推理标记的模型（如 Qwen3、GLM-4.7）会将推理内容拆入 ``reasoning_content``，工具调用解析到 ``tool_calls``。
- 其他模型填 ``agentic_tokenizer: default``\ （仅渲染消息，不解析工具调用）；未设置 ``agentloop.agentic_tokenizer`` 时 chat 请求返回 HTTP 500。

2. 用 AsyncOpenAI 驱动轨迹
--------------------------

继承 ``MultiAgentLoopWorker`` 并覆写 ``run_session_trajectory``，在已打开的 session 上执行一条轨迹。基类自动管理 session 开关、reward 上报与失败组回收：

.. code-block:: python

   from openai import APIStatusError, AsyncOpenAI, BadRequestError

   from rlinf.workers.agent.agent_loop import GroupDiscarded, MultiAgentLoopWorker


   class MyChatAgentLoopWorker(MultiAgentLoopWorker):
       async def run_session_trajectory(self, binding, input_ids, answer):
           base_url = self.session_server_client.session_base_url(
               binding.group_id, binding.session_id
           )
           prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
           messages = build_messages(prompt)  # 从已应用模板的文本中提取任务
           message, num_responses = None, 0

           async with AsyncOpenAI(
               base_url=base_url, api_key="EMPTY", max_retries=0
           ) as client:
               for _ in range(MAX_TURNS):
                   try:
                       response = await client.chat.completions.create(
                           model="policy", messages=messages, tools=TOOLS
                       )
                   except BadRequestError as exc:
                       if exc.code != "context_length_exceeded":
                           raise
                       break
                   except APIStatusError as exc:
                       if exc.status_code != 410:
                           raise
                       raise GroupDiscarded("group_discarded") from exc

                   num_responses += 1
                   message = response.choices[0].message
                   messages.append(message.model_dump(exclude_none=True))
                   if not message.tool_calls:
                       break
                   for call in message.tool_calls:
                       result = await run_tool(call)  # 调用已有工具或黑盒 Agent 步骤
                       messages.append(
                           {"role": "tool", "tool_call_id": call.id, "content": result}
                       )

           if num_responses == 0:
               raise GroupDiscarded("context_length_exceeded")
           reward = score((message.content or "") if message else "", answer)
           close_fields = {
               "extra_fields_train": {"idx_to_sub_traj": [0] * num_responses}
           }
           return None, close_fields, reward

客户端必须遵守的契约：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 规则
     - 说明
   * - 关闭客户端重试
     - 必须设 ``max_retries=0``；重发的请求会被重新采样，两次输出中只有一次参与训练。
   * - 处理 410 与超长错误
     - HTTP 410 表示该组已被 Session Server 丢弃（抛出 ``GroupDiscarded``）；``context_length_exceeded``\ （HTTP 400）表示 prompt 已超出 ``runner.seq_length``\ （用最后一轮打分，首轮即超长则丢弃整组）。
   * - 单 session 串行请求
     - 上一轮未结束前再发请求或 close 会返回 HTTP 409。
   * - Prompt 解码与额外字段
     - ``input_ids`` 已套用 chat 模板，``build_messages`` 需先从中提取任务文本；若 ``algorithm.loss_scales`` 含 ``agent_level``，需返回 ``idx_to_sub_traj``。

3. 启动与验证
-------------

参照 ``examples/agent/searchr1/train.py`` 在入口脚本中启动上述自定义 Agent Loop 类；即便由自定义 Worker 自行打分，配置里也须保留 ``reward.reward_type: searchr1`` 以通过启动校验。

启动后在 ``rollout/session_server/`` 下核对关键指标（完整列表见 :ref:`监控指标 <session-server-metrics>`）：

- ``chat_reasoning_responses`` 与 ``chat_tool_call_responses`` 确认推理内容与工具调用正常解析。
- ``chat_tool_call_parse_failed`` 排查无法解析的 tool-call 输出块。
- ``prefix_rewritten`` 统计客户端改写历史消息（而非单纯追加消息）的轮次。

.. note::

   ``session_server.verify_assembly`` 仅支持 token-ID Agent Loop；使用 ``/v1/chat/completions`` 时请保持为 ``False``。

.. dropdown:: 前缀 Token 复用机制与拒绝参数

   为保持与引擎输出完全一致的 token 并命中前缀缓存，Session Server 会复用已存的历史 token（即使客户端丢弃了 ``reasoning_content`` 也可继续）：

   - **接续：** 仅渲染新增的 user/tool 消息并拼接到历史 token 后（若 ``clear_thinking: true``，新增 user 消息会触发整段重渲染）。
   - **重发：** 重复发送上一轮请求时自动回退上一轮（放弃的轮次留在 payload 中，不参与训练）。
   - **改写：** 修改历史消息时整段重渲染，计入 ``prefix_rewritten`` 并正常训练。

   以下不受支持的 OpenAI 请求参数会返回 HTTP 400：

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - 参数
        - 拒绝条件
      * - ``n``
        - 大于 1
      * - ``temperature``、``top_p``、``top_k``
        - 与配置值不一致
      * - ``tool_choice``
        - 非 ``auto`` 或 ``none``
      * - ``response_format``
        - 任何取值（暂不支持）
      * - ``chat_template_kwargs``
        - 与配置值不一致
