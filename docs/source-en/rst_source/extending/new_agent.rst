White-Box Agent
===============

Add a tool-call parser, a tool worker, and a multi-turn agent loop without changing RLinf's rollout engines or runners. By the end of this guide, a calculator agent parses ``<calc>...</calc>`` tags from model completions, evaluates arithmetic expressions inside a ``ToolWorker``, feeds results back into the conversation with tool tokens masked out of policy loss, and runs inside ``AgentRunner``.

The integration proceeds through four checkpoints. First, implement and verify the tool-call parser locally on CPU. Next, build a stateless ``ToolWorker`` that executes requests over RLinf channels. Then subclass ``AgentLoopWorker`` to orchestrate multi-turn LLM generation and tool calls. Finally, wire the YAML configuration and entry script. For an overview of how agentic runners schedule rollout and training, see :doc:`Execution Flow <../concepts/execution_flow>` and :doc:`Agentic RL Guide <../guides/agentic>`.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Aspect
     - Details
   * - Files to create or edit
     - ``rlinf/agents/tool_call/parsers.py`` (or an external module), ``rlinf/agents/<name>/`` for the tool worker and agent loop, plus your entry script and Hydra YAML config
   * - Extension points
     - ``@register_toolcall_parser``, ``ToolWorker``, ``ToolWorkerInfo``, and ``AgentLoopWorker`` (or ``MultiAgentLoopWorker`` when ``is_dynamic_rollout_batch`` is ``True``)
   * - Required methods
     - Parser ``async def __call__(response_text)``; tool worker ``start_server()`` and ``stop_server()``; agent loop ``generate_llm_response()`` and ``generate_tool_response()``
   * - Local verification
     - Direct CPU parser check before launching ``AgentRunner``

1. Parse Tool Calls
-------------------

During worker initialization, ``AgentLoopWorker`` reads ``cfg.agentloop.toolcall_parser`` and instantiates the matching parser through ``get_toolcall_parser()``. A parser is a class decorated with ``@register_toolcall_parser(name)`` whose async ``__call__`` method splits model output into plain content and a list of ``ToolRequest`` objects.

Create a calculator parser in ``rlinf/agents/tool_call/parsers.py`` (or in your own extension package):

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

Each call follows this contract:

- ``@register_toolcall_parser("calc")`` registers the class under its lowercase name in ``TOOLCALL_PARSER_REGISTRY``.
- ``get_toolcall_parser("calc")`` imports ``rlinf.agents.tool_call.parsers`` on first lookup and returns an instance of the registered class.
- ``__call__(response_text)`` returns ``(content, tool_requests)``. Return an empty list when the model produces no tool tag so the agent loop ends the trajectory. Each ``ToolRequest`` carries the tool ``name`` (matching a name declared in ``ToolWorkerInfo``) and an ``arguments`` dict.

Verify the parser directly on CPU before writing a worker:

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

   content, reqs = asyncio.run(get_toolcall_parser("calc")("Compute <calc>6 * 7</calc> please."))
   print(content.strip(), reqs)
   '

Expected output:

.. code-block:: text

   Compute  please. [ToolRequest(name='calc', arguments={'expression': '6 * 7'})]

2. Implement a Tool Worker
--------------------------

Tool execution runs in dedicated ``ToolWorker`` processes so blocking I/O or sandboxes stay isolated from rollout engines. When you pass ``tool_workers`` to ``AgentRunner`` as a mapping of ``{worker_group: ToolWorkerInfo(tool_names=[...], has_session=...)}``, the runner creates one input ``Channel`` per tool worker group (wrapped in ``ToolChannelInfo``) and one shared ``tool_output_channel``.

``AgentRunner`` initializes each tool worker by calling ``init_worker(input_channel, output_channel)``, starts its background request loop with ``start_server()``, and cancels it with ``stop_server()`` when rollout finishes. Each request arrives as a ``ToolChannelRequest`` carrying ``session_id``, ``request_type`` (``"execute"`` for stateless tools, plus ``"session_start"`` and ``"session_end"`` when ``has_session`` is ``True``), ``tool_name``, and ``tool_args``. The worker replies with a ``ToolChannelResponse`` written to ``output_channel`` under ``key=request.session_id``.

Implement the calculator tool worker (for example in ``rlinf/agents/calc/calc_tool_worker.py``):

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

Read the lifecycle in call order:

- ``ToolWorker.init_worker(input_channel, output_channel)`` stores ``self.input_channel`` and ``self.output_channel``; override it only if your worker opens extra clients.
- ``start_server()`` schedules the async receive loop on the running event loop.
- ``_process_requests()`` reads each ``ToolChannelRequest``, catches tool errors into ``ToolChannelResponse(success=False, result=...)`` so bad model input does not crash training, and puts the reply with ``key=request.session_id``.
- ``stop_server()`` cancels the loop cleanly at shutdown.

For an HTTP-backed tool example, see ``rlinf/agents/searchr1/search_tool_worker.py``. For stateful sessions using ``has_session=True``, see ``rlinf/agents/rstar2/rstar2_agent_loop.py``.

3. Build the Multi-Turn Agent Loop
----------------------------------

``AgentLoopWorker`` coordinates multi-turn rollouts for each query group. Its ``run_one_query()`` method calls ``pre_process()``, repeats ``generate_llm_response()`` and ``generate_tool_response()`` until either returns ``is_continue=False``, concatenates all generated and tool tokens into one ``AgentLoopOutput`` with ``response_mask`` (``1`` for LLM tokens, ``0`` for tool tokens), and finishes with ``post_process()``.

.. note::

   When ``agentloop.is_dynamic_rollout_batch`` is ``True``, subclass ``MultiAgentLoopWorker`` instead of ``AgentLoopWorker``. Each turn then produces its own ``AgentLoopOutput`` inside a ``MultiAgentLoopOutput`` (see ``rlinf/agents/searchr1/searchr1_agent_loop.py``).

Implement the calculator agent loop (for example in ``rlinf/agents/calc/calc_agent_loop.py``):

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

Every call builds on helpers inherited from ``AgentLoopWorker``:

- ``__init__`` sets ``self.max_prompt_len`` (read by ``run_one_query()``) and ``self.max_resp_len`` so the sequence stays within ``cfg.runner.seq_length``.
- ``pre_process(prompt_ids)`` returns a per-query context dict shared across turns. Override ``post_process(generate_context, output)`` if you need to release stateful sessions or attach ``extra_fields``.
- ``self.generate(turn_prompt_ids, sampling_params=...)`` sends a request to the rollout engine over ``generate_input_channel`` and returns ``output_ids`` (plus ``logprobs`` when ``self.return_logprobs`` is ``True``).
- ``self.tool_call(tool_request)`` looks up the tool's channel via ``tool_name_map`` and ``tool_channel_info_map``, sends a stateless ``ToolChannelRequest``, and returns a ``ToolResponse``.
- ``self.get_tool_response_ids(tool_messages)`` formats tool messages with the tokenizer's chat template and returns only the appended token IDs. Setting their ``response_mask`` entries to ``0`` excludes tool outputs from policy-gradient loss.

4. Configure, Register, and Test
--------------------------------

Select the registered parser and turn limit under ``agentloop`` in your Hydra YAML (starting from a working config such as ``examples/agent/searchr1/config/train_qwen2.5.yaml``):

.. code-block:: yaml

   agentloop:
     group_name: "AgentLoopGroup"
     toolcall_parser: "calc"
     max_turns: 3
     print_outputs: False
     is_dynamic_rollout_batch: False

Launch both worker groups in your entry script and pass them to ``AgentRunner`` (follow ``examples/agent/searchr1/train.py`` for dataset, actor, and rollout setup):

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

If your parser lives outside the RLinf tree, register it inside your package's ``register()`` function and export ``RLINF_EXT_MODULE`` before launching training so every Ray worker imports it (see :ref:`out-of-tree-extensions`):

.. code-block:: bash

   export RLINF_EXT_MODULE=my_pkg.rlinf_ext

Next Steps
----------

- **Already have an OpenAI API or external agent?** Follow :doc:`Black-Box Agent <openai_agent>` to connect without rewriting parsers or ``ToolWorker``.
- **Scale asynchronously across nodes:** Enable ``runner.async_mode: True`` (:doc:`Async Agentic RL <../guides/async_agentic>`) and offload rollout payloads with :doc:`Session Server <../guides/session_server>`.
