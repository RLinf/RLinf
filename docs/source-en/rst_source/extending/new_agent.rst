White-Box Agent
===============

A **white-box agent** implements its multi-turn interaction and tool execution directly inside RLinf, giving you full control over prompts, tool parsing, and token loss masks (for example Search-R1 or custom tag-based tool agents). Using a ``<calc>...</calc>`` calculator example, this guide walks through writing a tool parser, ``ToolWorker``, ``AgentLoopWorker``, and wiring them into training.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Component
     - Extension Point
     - Role
   * - Tool parser
     - ``@register_toolcall_parser``
     - Extracts tool names and arguments from model text; returns an empty list to end the trajectory.
   * - ``ToolWorker``
     - Subclass ``ToolWorker``
     - Executes tool requests in an isolated process so I/O or sandboxes never block rollout engines.
   * - ``AgentLoopWorker``
     - Subclass ``AgentLoopWorker``
     - Orchestrates multi-turn generation, tool calls, length limits, and token loss masks.

Overall Call Flow
-----------------

``AgentRunner`` creates all channels and drives the training loop. You only implement the highlighted ``AgentLoopWorker`` and ``ToolWorker``; Rollout / Session Server and Reward / Actor stay built-in.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 940 426" width="100%" role="img" aria-label="AgentRunner full training call flow highlighting AgentLoopWorker and ToolWorker interfaces and channels" style="display:block;max-width:940px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="ar-pur-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
         <marker id="ar-grn-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#10b981"/>
         </marker>
         <marker id="ar-blu-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#3b82f6"/>
         </marker>
       </defs>

       <!-- Top legend -->
       <line x1="364" y1="14" x2="390" y2="14" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3"/>
       <text x="396" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">Runner control call</text>
       <line x1="518" y1="14" x2="544" y2="14" stroke="#10b981" stroke-width="2"/>
       <text x="550" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">Data channel</text>
       <rect x="638" y="7" width="14" height="14" rx="3" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="658" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">Custom code</text>
       <rect x="746" y="7" width="14" height="14" rx="3" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="766" y="18" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">Built-in (details omitted)</text>

       <!-- AgentRunner Control Plane Bar -->
       <rect x="16" y="30" width="908" height="56" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.10))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="34" y="51" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">AgentRunner (Training Orchestrator)</text>
       <text x="34" y="72" font-size="10.8" font-weight="500" fill="var(--pst-color-text-base, #0f172a)">① init_workers() creates channels   →   ② start_server()   →   ③ Per step: _put_batch() → _sync_weights() → run_agentloop_rollout() → _run_training()   →   ④ stop_server()</text>

       <!-- Dashed control arrows from AgentRunner -->
       <line x1="84" y1="86" x2="84" y2="128" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ar-pur-en)"/>
       <text x="92" y="112" font-size="10.5" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">_put_batch()</text>

       <line x1="430" y1="86" x2="430" y2="122" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ar-pur-en)"/>
       <text x="438" y="109" font-size="10.5" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">init_worker() · run_agentloop_rollout()</text>

       <path d="M 914 86 L 914 268 L 812 268 L 812 288" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#ar-pur-en)"/>
       <text x="812" y="261" text-anchor="middle" font-size="10" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">init_worker · start_server · stop_server</text>

       <!-- Left Column: Built-in Dataset & Reward/Actor -->
       <rect x="16" y="132" width="136" height="72" rx="12" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="84" y="160" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Dataloader</text>
       <text x="84" y="180" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Feeds query batches</text>

       <rect x="16" y="294" width="136" height="86" rx="12" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="84" y="324" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Reward / Actor</text>
       <text x="84" y="344" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Scoring &amp; policy update</text>
       <text x="84" y="362" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">(details omitted)</text>

       <!-- Channels: Left <-> Center -->
       <line x1="152" y1="168" x2="270" y2="168" stroke="#10b981" stroke-width="2" marker-end="url(#ar-grn-en)"/>
       <text x="212" y="159" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="700" fill="#10b981">dataloader_channel</text>
       <text x="212" y="184" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">RolloutRequest</text>

       <line x1="274" y1="337" x2="156" y2="337" stroke="#10b981" stroke-width="2" marker-end="url(#ar-grn-en)"/>
       <text x="214" y="328" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="700" fill="#10b981">rollout_channel</text>
       <text x="214" y="353" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">RolloutResult</text>

       <!-- Center Column: AgentLoopWorker (Highlighted Purple) -->
       <rect x="274" y="126" width="312" height="284" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2.2"/>
       <text x="292" y="150" font-size="14" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">AgentLoopWorker</text>
       <rect x="508" y="135" width="64" height="20" rx="10" fill="rgba(124,58,237,0.14)"/>
       <text x="540" y="149" text-anchor="middle" font-size="10" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Custom</text>
       <text x="292" y="167" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Runs run_one_query() per trajectory</text>

       <!-- Step 1: pre_process -->
       <rect x="290" y="176" width="280" height="26" rx="7" fill="rgba(124,58,237,0.08)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1"/>
       <text x="430" y="193" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">1. pre_process(prompt_ids)</text>

       <!-- Turn Loop Frame -->
       <rect x="290" y="210" width="280" height="150" rx="10" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.2" stroke-dasharray="4 3"/>
       <text x="302" y="226" font-size="10.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">2. Turn loop: run_one_query_turn()</text>

       <!-- Turn 2a: generate_llm_response -->
       <rect x="302" y="234" width="256" height="46" rx="8" fill="rgba(124,58,237,0.10)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.3"/>
       <text x="430" y="253" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">generate_llm_response()</text>
       <text x="430" y="271" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Calls self.generate() · mask = 1</text>

       <!-- Vertical arrow 2a -> 2b -->
       <line x1="430" y1="280" x2="430" y2="298" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5" marker-end="url(#ar-pur-en)"/>

       <!-- Turn 2b: generate_tool_response -->
       <rect x="302" y="302" width="256" height="48" rx="8" fill="rgba(59,130,246,0.10)" stroke="#3b82f6" stroke-width="1.3"/>
       <text x="430" y="321" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="700" fill="#2563eb">generate_tool_response()</text>
       <text x="430" y="339" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">toolcall_parser() + self.tool_call() · mask = 0</text>

       <!-- Step 3: post_process -->
       <rect x="290" y="368" width="280" height="28" rx="7" fill="rgba(124,58,237,0.08)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1"/>
       <text x="430" y="386" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">3. post_process(ctx, output)</text>

       <!-- Right-Top: Rollout / Session Server (Built-in, Muted) -->
       <rect x="726" y="158" width="172" height="82" rx="12" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="812" y="186" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Rollout / Session Server</text>
       <text x="812" y="206" text-anchor="middle" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Generates tokens</text>
       <text x="812" y="224" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">(details omitted)</text>

       <!-- Channels: AgentLoopWorker <-> Rollout -->
       <line x1="586" y1="184" x2="722" y2="184" stroke="#10b981" stroke-width="1.8" marker-end="url(#ar-grn-en)"/>
       <text x="656" y="176" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.5" font-weight="700" fill="#10b981">generate_input_channel</text>

       <line x1="726" y1="218" x2="590" y2="218" stroke="#10b981" stroke-width="1.8" marker-end="url(#ar-grn-en)"/>
       <text x="656" y="210" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.5" font-weight="700" fill="#10b981">generate_output_channel</text>
       <text x="656" y="233" text-anchor="middle" font-size="9.5" fill="var(--pst-color-text-muted, #64748b)">or Session Server HTTP</text>

       <!-- Right-Bottom: ToolWorker (Highlighted Blue) -->
       <rect x="726" y="292" width="172" height="118" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="2.2"/>
       <text x="742" y="316" font-size="13.5" font-weight="700" fill="#2563eb">ToolWorker</text>
       <rect x="826" y="302" width="60" height="18" rx="9" fill="rgba(59,130,246,0.14)"/>
       <text x="856" y="314.5" text-anchor="middle" font-size="9.5" font-weight="700" fill="#2563eb">Custom</text>
       <rect x="738" y="328" width="148" height="22" rx="6" fill="rgba(59,130,246,0.08)"/>
       <text x="812" y="343" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">start_server()</text>
       <rect x="738" y="355" width="148" height="22" rx="6" fill="rgba(59,130,246,0.08)"/>
       <text x="812" y="370" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">_process_requests()</text>
       <rect x="738" y="382" width="148" height="20" rx="6" fill="rgba(59,130,246,0.08)"/>
       <text x="812" y="395.5" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="10" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">stop_server()</text>

       <!-- Channels: AgentLoopWorker <-> ToolWorker -->
       <line x1="586" y1="322" x2="722" y2="322" stroke="#3b82f6" stroke-width="2" marker-end="url(#ar-blu-en)"/>
       <text x="656" y="313" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.8" font-weight="700" fill="#2563eb">Tool-&lt;group&gt;</text>
       <text x="656" y="336" text-anchor="middle" font-size="9.5" fill="var(--pst-color-text-muted, #64748b)">ToolChannelRequest</text>

       <line x1="726" y1="368" x2="590" y2="368" stroke="#3b82f6" stroke-width="2" marker-end="url(#ar-blu-en)"/>
       <text x="656" y="359" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="9.8" font-weight="700" fill="#2563eb">tool_output_channel</text>
       <text x="656" y="383" text-anchor="middle" font-size="9.5" fill="var(--pst-color-text-muted, #64748b)">ToolChannelResponse (key=session_id)</text>
     </svg>
   </div>

1. Parse Tool Calls
-------------------

Register a parser class with ``@register_toolcall_parser("calc")`` and implement async ``__call__(response_text)`` returning ``(plain_text, tool_requests)``. Add it to ``rlinf/agents/tool_call/parsers.py`` (or your external extension package):

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

.. dropdown:: Verify the parser locally on CPU

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

2. Implement ToolWorker
-----------------------

Interfaces to Implement
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Method
     - Required
     - Role
   * - ``start_server()``
     - Yes
     - Launch a background async loop that reads requests from the input channel and writes replies to the output channel under the same session ID.
   * - ``stop_server()``
     - Yes
     - Cancel the background task and clean up external connections or sandboxes at shutdown.
   * - ``init_worker()``
     - Optional
     - The base class already saves both channels; override only if you open extra clients.

Catch execution errors into a failed response so malformed model output never crashes training. If your tool holds per-trajectory state (``has_session=True``), also handle session start and end requests.

Implementation Example
~~~~~~~~~~~~~~~~~~~~~~

Implement the calculator tool (for example in ``rlinf/agents/calc/calc_tool_worker.py``):

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

Call Flow
~~~~~~~~~

1. **Channel wiring:** Before training, the runner creates a dedicated input channel per tool group and one shared output channel, passing both via ``init_worker()``.

2. **Startup:** Right before the step loop, the runner calls ``start_server()`` to launch the background consumer.

3. **Request & reply:** During rollout, the agent loop pushes requests onto the input channel; ``ToolWorker`` evaluates them and writes replies to the output channel with the matching ``session_id`` so the caller wakes up.

4. **Shutdown:** When training finishes, the runner calls ``stop_server()`` to cancel the loop cleanly.

For HTTP and stateful-session examples, see ``rlinf/agents/searchr1/search_tool_worker.py`` and ``rlinf/agents/rstar2/rstar2_agent_loop.py``.

3. Implement AgentLoopWorker
----------------------------

.. note::

   For dynamic batches (``is_dynamic_rollout_batch: True``) or async training, subclass ``MultiAgentLoopWorker`` instead (see ``rlinf/agents/searchr1/searchr1_agent_loop.py``).

Interfaces to Implement
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Method
     - Required
     - Role
   * - ``generate_llm_response()``
     - Yes
     - Call ``self.generate()`` for one model turn and set model token masks to ``1``.
   * - ``generate_tool_response()``
     - Yes
     - Parse tool tags, call ``self.tool_call()``, encode tool text, and set tool token masks to ``0``; return ``False`` when there are no tool calls.
   * - ``pre_process()``
     - Optional
     - Initialize per-trajectory state shared across turns (such as turn counters or session IDs).
   * - ``post_process()``
     - Optional
     - Finalize a completed trajectory (such as releasing sessions or attaching extra fields).

Inside these methods, reuse three built-in helpers: ``self.generate()`` (model generation), ``self.tool_call()`` (tool dispatch), and ``self.get_tool_response_ids()`` (chat-template encoding of tool replies).

Implementation Example
~~~~~~~~~~~~~~~~~~~~~~

Implement the calculator loop (for example in ``rlinf/agents/calc/calc_agent_loop.py``):

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

Call Flow
~~~~~~~~~

1. **Batch dispatch:** On each step, the runner triggers rollout; the base worker pops a query batch from the input channel and spawns concurrent trajectory tasks per group.

2. **Per-query init:** Each trajectory truncates its prompt to the configured limit and initializes state via ``pre_process()``.

3. **Turn loop:** Each turn runs ``generate_llm_response()`` then ``generate_tool_response()``, appending LLM tokens (mask ``1``) and tool tokens (mask ``0``) until either hook returns ``False``.

4. **Hand-off:** After ``post_process()``, the completed group is packed and written to the output channel for Reward and Actor workers.

4. Configure, Launch, and Verify
--------------------------------

Select your registered parser under ``agentloop`` in YAML (starting from ``examples/agent/searchr1/config/train_qwen2.5.yaml``):

.. code-block:: yaml

   agentloop:
     group_name: "AgentLoopGroup"
     toolcall_parser: "calc"
     max_turns: 3
     print_outputs: False
     is_dynamic_rollout_batch: False

Launch both worker groups in your entry script and pass them to ``AgentRunner`` (see ``examples/agent/searchr1/train.py`` for the full script):

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

If the parser lives outside RLinf, export ``RLINF_EXT_MODULE=my_pkg.rlinf_ext`` before launching so every Ray worker registers it automatically (see :ref:`out-of-tree-extensions`).

Next Steps
----------

- **Already have an OpenAI API or external agent?** Follow :doc:`Black-Box Agent <openai_agent>` to connect without rewriting parsers or ``ToolWorker``.
- **Scale asynchronously across nodes:** Enable ``runner.async_mode: True`` (:doc:`Async Agentic RL <../guides/async_agentic>`) and offload rollout payloads with :doc:`Session Server <../guides/session_server>`.
