白盒 Agent
==========

**白盒 Agent** 指把多轮交互与工具调用逻辑直接写在 RLinf 内，可精确控制每轮提示词、工具解析与 token 损失掩码（例如 Search-R1 或自定义标签工具 Agent）。本指南以计算器（``<calc>...</calc>``）为例，依次实现工具解析器、``ToolWorker``、``AgentLoopWorker`` 并接入训练。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - 组件
     - 扩展点
     - 职责
   * - 工具解析器
     - ``@register_toolcall_parser``
     - 从模型输出文本中提取工具名和参数；无调用时返回空列表以结束轨迹。
   * - ``ToolWorker``
     - 继承 ``ToolWorker``
     - 在独立进程中接收工具请求并回传结果，避免 I/O 或沙箱阻塞推理。
   * - ``AgentLoopWorker``
     - 继承 ``AgentLoopWorker``
     - 控制单条轨迹的多轮生成、工具调用、长度预算与 token 损失掩码。

整体调用流程
------------

``AgentRunner`` 负责创建 Channel 并驱动完整训练循环；开发者只需实现图中高亮的 ``AgentLoopWorker`` 与 ``ToolWorker``，其余组件（Rollout / Session Server、Reward / Actor）均由框架内置接管。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 940 426" width="100%" role="img" aria-label="AgentRunner 完整训练流程中的调用关系，突出 AgentLoopWorker 与 ToolWorker 的自定义接口和 Channel 交互" style="display:block;max-width:940px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="ar-pur-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
         <marker id="ar-grn-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#10b981"/>
         </marker>
         <marker id="ar-blu-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#3b82f6"/>
         </marker>
       </defs>

       <!-- Top legend -->
       <line x1="406" y1="14" x2="432" y2="14" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3"/>
       <text x="438" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">Runner 控制调用</text>
       <line x1="538" y1="14" x2="564" y2="14" stroke="#10b981" stroke-width="2"/>
       <text x="570" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">数据 Channel</text>
       <rect x="656" y="7" width="14" height="14" rx="3" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="676" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">需自定义实现</text>
       <rect x="762" y="7" width="14" height="14" rx="3" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="782" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">内置组件（细节略）</text>

       <!-- AgentRunner Control Plane Bar -->
       <rect x="16" y="30" width="908" height="56" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.10))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="34" y="51" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">AgentRunner（主控循环）</text>
       <text x="34" y="72" font-size="11" font-weight="500" fill="var(--pst-color-text-base, #0f172a)">① init_workers() 创建并分发 Channel   →   ② start_server()   →   ③ 每步循环：_put_batch() → _sync_weights() → run_agentloop_rollout() → _run_training()   →   ④ stop_server()</text>

       <!-- Dashed control arrows from AgentRunner -->
       <line x1="84" y1="86" x2="84" y2="128" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ar-pur-zh)"/>
       <text x="92" y="112" font-size="10.5" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">_put_batch()</text>

       <line x1="430" y1="86" x2="430" y2="122" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ar-pur-zh)"/>
       <text x="438" y="109" font-size="10.5" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">init_worker() · run_agentloop_rollout()</text>

       <path d="M 914 86 L 914 268 L 812 268 L 812 288" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ar-pur-zh)"/>
       <text x="812" y="261" text-anchor="middle" font-size="10" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">init_worker · start_server · stop_server</text>

       <!-- Left Column: Built-in Dataset & Reward/Actor -->
       <rect x="16" y="132" width="136" height="72" rx="12" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="84" y="160" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Dataloader</text>
       <text x="84" y="180" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">每步写入 Query 批次</text>

       <rect x="16" y="294" width="136" height="86" rx="12" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="84" y="324" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Reward / Actor</text>
       <text x="84" y="344" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">打分与参数更新</text>
       <text x="84" y="362" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">（细节略）</text>

       <!-- Channels: Left <-> Center -->
       <line x1="152" y1="168" x2="270" y2="168" stroke="#10b981" stroke-width="2" marker-end="url(#ar-grn-zh)"/>
       <text x="212" y="159" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="700" fill="#10b981">dataloader_channel</text>
       <text x="212" y="184" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">RolloutRequest</text>

       <line x1="274" y1="337" x2="156" y2="337" stroke="#10b981" stroke-width="2" marker-end="url(#ar-grn-zh)"/>
       <text x="214" y="328" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="700" fill="#10b981">rollout_channel</text>
       <text x="214" y="353" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">RolloutResult</text>

       <!-- Center Column: AgentLoopWorker (Highlighted Purple) -->
       <rect x="274" y="126" width="312" height="284" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2.2"/>
       <text x="292" y="150" font-size="14" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">AgentLoopWorker</text>
       <rect x="498" y="135" width="74" height="20" rx="10" fill="rgba(124,58,237,0.14)"/>
       <text x="535" y="149" text-anchor="middle" font-size="10" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">自定义实现</text>
       <text x="292" y="167" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">按组并发执行 run_one_query()</text>

       <!-- Step 1: pre_process -->
       <rect x="290" y="176" width="280" height="26" rx="7" fill="rgba(124,58,237,0.08)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1"/>
       <text x="430" y="193" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">1. pre_process(prompt_ids)</text>

       <!-- Turn Loop Frame -->
       <rect x="290" y="210" width="280" height="150" rx="10" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.2" stroke-dasharray="4 3"/>
       <text x="302" y="226" font-size="10.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">2. 多轮循环 run_one_query_turn()</text>

       <!-- Turn 2a: generate_llm_response -->
       <rect x="302" y="234" width="256" height="46" rx="8" fill="rgba(124,58,237,0.10)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.3"/>
       <text x="430" y="253" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">generate_llm_response()</text>
       <text x="430" y="271" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">调用 self.generate()，掩码置 1</text>

       <!-- Vertical arrow 2a -> 2b -->
       <line x1="430" y1="280" x2="430" y2="298" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" marker-end="url(#ar-pur-zh)"/>

       <!-- Turn 2b: generate_tool_response -->
       <rect x="302" y="302" width="256" height="48" rx="8" fill="rgba(59,130,246,0.10)" stroke="#3b82f6" stroke-width="1.3"/>
       <text x="430" y="321" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="700" fill="#2563eb">generate_tool_response()</text>
       <text x="430" y="339" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">toolcall_parser() + self.tool_call()，掩码置 0</text>

       <!-- Step 3: post_process -->
       <rect x="290" y="368" width="280" height="28" rx="7" fill="rgba(124,58,237,0.08)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1"/>
       <text x="430" y="386" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">3. post_process(ctx, output)</text>

       <!-- Right-Top: Rollout / Session Server (Built-in, Muted) -->
       <rect x="726" y="158" width="172" height="82" rx="12" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="812" y="186" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Rollout / Session Server</text>
       <text x="812" y="206" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">推理引擎生成 token</text>
       <text x="812" y="224" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">（细节略）</text>

       <!-- Channels: AgentLoopWorker <-> Rollout -->
       <line x1="586" y1="184" x2="722" y2="184" stroke="#10b981" stroke-width="1.8" marker-end="url(#ar-grn-zh)"/>
       <text x="656" y="176" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.5" font-weight="700" fill="#10b981">generate_input_channel</text>

       <line x1="726" y1="218" x2="590" y2="218" stroke="#10b981" stroke-width="1.8" marker-end="url(#ar-grn-zh)"/>
       <text x="656" y="210" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.5" font-weight="700" fill="#10b981">generate_output_channel</text>
       <text x="656" y="233" text-anchor="middle" font-size="9.5" fill="var(--pst-color-text-muted, #64748b)">或 Session Server HTTP</text>

       <!-- Right-Bottom: ToolWorker (Highlighted Blue) -->
       <rect x="726" y="292" width="172" height="118" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="2.2"/>
       <text x="742" y="316" font-size="13.5" font-weight="700" fill="#2563eb">ToolWorker</text>
       <rect x="826" y="302" width="60" height="18" rx="9" fill="rgba(59,130,246,0.14)"/>
       <text x="856" y="314.5" text-anchor="middle" font-size="9.5" font-weight="700" fill="#2563eb">自定义</text>
       <rect x="738" y="328" width="148" height="22" rx="6" fill="rgba(59,130,246,0.08)"/>
       <text x="812" y="343" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">start_server()</text>
       <rect x="738" y="355" width="148" height="22" rx="6" fill="rgba(59,130,246,0.08)"/>
       <text x="812" y="370" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">_process_requests()</text>
       <rect x="738" y="382" width="148" height="20" rx="6" fill="rgba(59,130,246,0.08)"/>
       <text x="812" y="395.5" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">stop_server()</text>

       <!-- Channels: AgentLoopWorker <-> ToolWorker -->
       <line x1="586" y1="322" x2="722" y2="322" stroke="#3b82f6" stroke-width="2" marker-end="url(#ar-blu-zh)"/>
       <text x="656" y="313" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.8" font-weight="700" fill="#2563eb">Tool-&lt;group&gt;</text>
       <text x="656" y="336" text-anchor="middle" font-size="9.5" fill="var(--pst-color-text-muted, #64748b)">ToolChannelRequest</text>

       <line x1="726" y1="368" x2="590" y2="368" stroke="#3b82f6" stroke-width="2" marker-end="url(#ar-blu-zh)"/>
       <text x="656" y="359" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.8" font-weight="700" fill="#2563eb">tool_output_channel</text>
       <text x="656" y="383" text-anchor="middle" font-size="9.5" fill="var(--pst-color-text-muted, #64748b)">ToolChannelResponse (key=session_id)</text>
     </svg>
   </div>

1. 解析工具调用
---------------

用 ``@register_toolcall_parser("calc")`` 注册解析器类，并实现异步 ``__call__(response_text)`` 返回 ``(普通文本, 工具请求列表)``。在 ``rlinf/agents/tool_call/parsers.py``\ （或外部扩展包）中添加：

.. code-block:: python

   import re

   from rlinf.agents.tool_call.schema import ToolRequest
   from rlinf.algorithms.registry import register_toolcall_parser


   @register_toolcall_parser("calc")
   class CalcToolCallParser:
       def __init__(self) -> None:
           self.pattern = re.compile(r"<calc>(.*?)</calc>", re.DOTALL)

       async def __call__(self, response_text: str) -> tuple[str, list[ToolRequest]]:
           matches = self.pattern.findall(response_text)
           if not matches:
               return response_text, []
           requests = [
               ToolRequest(name="calc", arguments={"expression": match.strip()})
               for match in matches
           ]
           content = self.pattern.sub("", response_text)
           return content, requests

.. dropdown:: 在本地 CPU 上快速验证解析器

   .. code-block:: bash

      python -c '
      import asyncio, re
      from rlinf.agents.tool_call.schema import ToolRequest
      from rlinf.algorithms.registry import get_toolcall_parser, register_toolcall_parser

      @register_toolcall_parser("calc")
      class CalcToolCallParser:
          async def __call__(self, response_text: str):
              pat = re.compile(r"<calc>(.*?)</calc>", re.DOTALL)
              matches = pat.findall(response_text)
              return pat.sub("", response_text), [
                  ToolRequest(name="calc", arguments={"expression": m.strip()}) for m in matches
              ]

      print(asyncio.run(get_toolcall_parser("calc")("Compute <calc>6 * 7</calc> please.")))
      '

2. 实现 ToolWorker
------------------

需要实现的接口
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - 方法
     - 必需
     - 说明
   * - ``start_server()``
     - 是
     - 启动后台异步循环，从输入通道读取工具请求、执行后按原会话 ID 写回输出通道。
   * - ``stop_server()``
     - 是
     - 训练结束时取消后台循环并释放外部连接或沙箱资源。
   * - ``init_worker()``
     - 可选
     - 基类已保存输入与输出通道，仅需初始化额外客户端时再覆写。

单次工具执行若发生异常，应捕获并返回失败响应，避免模型非法输出中断训练；需要跨轮保持状态时（``has_session=True``），再额外处理会话创建与销毁请求。

代码示例
~~~~~~~~

在 ``rlinf/agents/calc/calc_tool_worker.py`` 中实现计算器工具：

.. code-block:: python

   import ast
   import asyncio
   import operator

   from rlinf.agents.tool_call.schema import ToolChannelRequest, ToolChannelResponse
   from rlinf.workers.agent.tool_worker import ToolWorker

   _OPS = {
       ast.Add: operator.add,
       ast.Sub: operator.sub,
       ast.Mult: operator.mul,
       ast.Div: operator.truediv,
   }


   def _safe_eval(expr: str) -> float:
       def _eval(node: ast.AST) -> float:
           if isinstance(node, ast.Expression):
               return _eval(node.body)
           if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
               return float(node.value)
           if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
               return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
           raise ValueError(f"Unsupported expression: {expr}")

       return _eval(ast.parse(expr, mode="eval"))


   class CalcToolWorker(ToolWorker):
       def __init__(self) -> None:
           super().__init__()
           self._task: asyncio.Task | None = None

       def start_server(self) -> None:
           loop = asyncio.get_running_loop()
           self._task = loop.create_task(self._process_requests())

       def stop_server(self) -> None:
           if self._task and not self._task.done():
               self._task.cancel()

       async def _process_requests(self) -> None:
           while True:
               request: ToolChannelRequest = await self.input_channel.get(
                   async_op=True
               ).async_wait()
               try:
                   value = _safe_eval(request.tool_args["expression"])
                   response = ToolChannelResponse(success=True, result=str(value))
               except Exception as exc:
                   response = ToolChannelResponse(success=False, result=f"Error: {exc}")
               await self.output_channel.put(
                   response, key=request.session_id, async_op=True
               ).async_wait()

调用流程
~~~~~~~~

1. **注入通道：** 启动前，Runner 为每个工具组创建专属输入通道与共享输出通道，通过 ``init_worker()`` 交给 ``ToolWorker``。

2. **后台监听：** 进入训练循环前，Runner 调用 ``start_server()`` 启动请求消费协程。

3. **收发请求：** Rollout 期间，Agent 循环发送工具请求到输入通道；``ToolWorker`` 计算完毕后用相同的 ``session_id`` 写回输出通道，自动路由回发起方。

4. **退出清理：** 训练结束时，Runner 调用 ``stop_server()`` 停止后台任务。

HTTP 工具与带会话状态的工具可分别参考 ``rlinf/agents/searchr1/search_tool_worker.py`` 与 ``rlinf/agents/rstar2/rstar2_agent_loop.py``。

3. 实现 AgentLoopWorker
-----------------------

.. note::

   开启动态批次（``is_dynamic_rollout_batch: True``）或异步训练时，改继承 ``MultiAgentLoopWorker``\ （参考 ``rlinf/agents/searchr1/searchr1_agent_loop.py``）。

需要实现的接口
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - 方法
     - 必需
     - 说明
   * - ``generate_llm_response()``
     - 是
     - 调用基类 ``self.generate()`` 获取一轮模型输出，并把模型 token 掩码设为 ``1``。
   * - ``generate_tool_response()``
     - 是
     - 解析工具标签并调用 ``self.tool_call()``，编码结果后把工具 token 掩码设为 ``0``；无工具调用时返回 ``False`` 结束轨迹。
   * - ``pre_process()``
     - 可选
     - 初始化单条轨迹的跨轮上下文（如轮次计数、会话 ID）。
   * - ``post_process()``
     - 可选
     - 轨迹结束后做收尾（如释放会话或补充自定义字段）。

实现中可直接复用三个基类能力：``self.generate()``\ （请求推理）、``self.tool_call()``\ （调用工具）和 ``self.get_tool_response_ids()``\ （套用对话模板编码工具输出）。

代码示例
~~~~~~~~

在 ``rlinf/agents/calc/calc_agent_loop.py`` 中实现多轮循环：

.. code-block:: python

   import asyncio
   from typing import Any

   from omegaconf import DictConfig

   from rlinf.utils.placement import ModelParallelComponentPlacement
   from rlinf.workers.agent.agent_loop import AgentLoopWorker


   class CalcAgentLoopWorker(AgentLoopWorker):
       def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement) -> None:
           super().__init__(cfg, placement)
           self.max_prompt_len = int(cfg.data.max_prompt_length)
           self.max_resp_len = max(1, int(cfg.runner.seq_length) - self.max_prompt_len)
           self.max_turns = int(cfg.agentloop.get("max_turns", 3))
           assert self.toolcall_parser is not None, "Set agentloop.toolcall_parser."

       def pre_process(self, prompt_ids: list[int]) -> dict[str, Any]:
           return {"turns": 0}

       async def generate_llm_response(
           self, generate_context, trace_prints, problem_prompt_ids, turn_prompt_ids
       ):
           remaining = self.max_resp_len - (len(turn_prompt_ids) - len(problem_prompt_ids))
           if remaining <= 0 or generate_context["turns"] >= self.max_turns:
               return False, [], [], ([] if self.return_logprobs else None), ""

           result = await self.generate(
               turn_prompt_ids, sampling_params={"max_new_tokens": remaining}
           )
           response_ids = result["output_ids"][:remaining]
           response_mask = [1] * len(response_ids)
           response_logprobs = (
               result["logprobs"][: len(response_ids)] if self.return_logprobs else None
           )
           generate_context["turns"] += 1
           return (
               len(response_ids) < remaining,
               response_ids,
               response_mask,
               response_logprobs,
               self.tokenizer.decode(response_ids),
           )

       async def generate_tool_response(
           self,
           generate_context,
           trace_prints,
           problem_prompt_ids,
           turn_prompt_ids,
           llm_response_ids,
           llm_response_text,
       ):
           empty_logprobs = [] if self.return_logprobs else None
           _, tool_requests = await self.toolcall_parser(llm_response_text)
           if not tool_requests:
               return False, [], [], empty_logprobs

           tool_responses = await asyncio.gather(*(self.tool_call(req) for req in tool_requests))
           tool_ids = self.get_tool_response_ids(
               [{"role": "tool", "content": resp.text} for resp in tool_responses]
           )
           remaining = self.max_resp_len - (
               len(turn_prompt_ids) + len(llm_response_ids) - len(problem_prompt_ids)
           )
           if len(tool_ids) > remaining:
               return False, [], [], empty_logprobs

           tool_logprobs = [0.0] * len(tool_ids) if self.return_logprobs else None
           return True, tool_ids, [0] * len(tool_ids), tool_logprobs

调用流程
~~~~~~~~

1. **领取批次并按组并发：** 每步训练由 Runner 触发，基类从数据通道取出一个批次，并为每条 query 并发拉起整组轨迹。

2. **初始化轨迹：** 按最大 prompt 长度截断输入，调用 ``pre_process()`` 创建跨轮上下文。

3. **交替执行多轮：** 每轮依次调用 ``generate_llm_response()`` 与 ``generate_tool_response()``，把模型 token（掩码 ``1``）与工具 token（掩码 ``0``）拼接到上下文，直到任一阶段返回 ``False``。

4. **收尾并输出：** 调用 ``post_process()`` 完成收尾，整组汇总后写入输出通道，交给下游 Reward 与 Actor 更新参数。

4. 配置、启动与验证
-------------------

在 Hydra YAML 的 ``agentloop`` 下选择注册的解析器（可基于 ``examples/agent/searchr1/config/train_qwen2.5.yaml`` 修改）：

.. code-block:: yaml

   agentloop:
     group_name: "AgentLoopGroup"
     toolcall_parser: "calc"
     max_turns: 3
     print_outputs: False
     is_dynamic_rollout_batch: False

在入口脚本中拉起两个 Worker 组并传入 ``AgentRunner``\ （完整示例见 ``examples/agent/searchr1/train.py``）：

.. code-block:: python

   from rlinf.runners.agent_runner import AgentRunner
   from rlinf.scheduler import NodePlacementStrategy
   from rlinf.workers.agent.tool_worker import ToolWorkerInfo

   agentloop_group = CalcAgentLoopWorker.create_group(
       cfg, component_placement
   ).launch(
       cluster,
       name=cfg.agentloop.group_name,
       placement_strategy=agentloop_placement_strategy,
   )

   tool_workers = {
       CalcToolWorker.create_group().launch(
           cluster,
           name="calc",
           placement_strategy=NodePlacementStrategy([0]),
       ): ToolWorkerInfo(tool_names=["calc"], has_session=False),
   }

   runner = AgentRunner(
       cfg=cfg,
       placement=component_placement,
       train_dataset=train_ds,
       val_dataset=val_ds,
       rollout=rollout_group,
       inference=inference_group,
       actor=actor_group,
       reward=reward_group,
       agent_loop=agentloop_group,
       tool_workers=tool_workers,
   )
   runner.init_workers()
   runner.run()

解析器位于外部仓库时，启动前导出 ``RLINF_EXT_MODULE=my_pkg.rlinf_ext`` 即可在所有 Ray Worker 中自动注册（见 :ref:`out-of-tree-extensions`）。

下一步
------

- **已有 OpenAI API 或外部黑盒 Agent？** 参见 :doc:`黑盒 Agent <openai_agent>`，无需重写解析器与 ``ToolWorker``。
- **扩展到多机异步训练：** 开启 ``runner.async_mode: True``\ （见 :doc:`异步 Agent RL <../guides/async_agentic>`），并按需通过 :doc:`Session Server <../guides/session_server>` 将轨迹存储下沉到 rollout 节点。
