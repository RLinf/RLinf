白盒 Agent
==========

**白盒 Agent** 指把多轮交互与工具调用逻辑直接写在 RLinf 内，可精确控制每轮提示词、工具解析与 token 损失掩码（例如 Search-R1 或自定义标签工具 Agent）。本指南以计算器（``<calc>...</calc>``）为例，分四步完成工具解析器、``ToolWorker``、``AgentLoopWorker`` 与配置接入；底层调度流程见 :doc:`执行流程 <../concepts/execution_flow>`。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 项目
     - 说明
   * - 创建或修改的文件
     - ``rlinf/agents/tool_call/parsers.py``\ （或外部扩展模块）、``rlinf/agents/<name>/`` 下的工具 worker 与智能体循环、入口脚本及 Hydra YAML 配置
   * - 扩展点
     - ``@register_toolcall_parser``、``ToolWorker``、``ToolWorkerInfo``，以及 ``AgentLoopWorker``\ （开启 ``is_dynamic_rollout_batch`` 时改用 ``MultiAgentLoopWorker``\ ）
   * - 必须实现的方法
     - 解析器 ``async def __call__(response_text)``；工具 worker ``start_server()`` 与 ``stop_server()``；智能体循环 ``generate_llm_response()`` 与 ``generate_tool_response()``
   * - 本地验证
     - 启动 ``AgentRunner`` 前直接在本地 CPU 上验证解析器

1. 解析工具调用
---------------

``AgentLoopWorker`` 初始化时会读取 ``cfg.agentloop.toolcall_parser``，并通过 ``get_toolcall_parser()`` 实例化对应的解析器。解析器是用 ``@register_toolcall_parser(name)`` 装饰的类，其异步 ``__call__`` 方法负责把模型输出拆成普通文本与 ``ToolRequest`` 列表。

在 ``rlinf/agents/tool_call/parsers.py``\ （或外部扩展包）中添加计算器解析器：

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

解析器的调用约定如下：

- ``@register_toolcall_parser("calc")`` 按小写名称把解析器类写入 ``TOOLCALL_PARSER_REGISTRY``。
- ``get_toolcall_parser("calc")`` 在首次查找时自动导入 ``rlinf.agents.tool_call.parsers``，并返回已注册类的实例。
- ``__call__(response_text)`` 返回 ``(content, tool_requests)``。模型没有输出工具标签时返回空列表，智能体循环据此结束当前轨迹。每个 ``ToolRequest`` 包含工具名 ``name``\ （需与 ``ToolWorkerInfo`` 中声明的名称一致）以及参数字典 ``arguments``。

编写 worker 之前，先在本地 CPU 上验证解析逻辑：

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

预期输出：

.. code-block:: text

   Compute  please. [ToolRequest(name='calc', arguments={'expression': '6 * 7'})]

2. 实现工具 Worker
------------------

工具执行位于独立的 ``ToolWorker`` 进程中，避免阻塞 I/O 或沙箱环境影响推理引擎。入口脚本把 ``{worker_group: ToolWorkerInfo(tool_names=[...], has_session=...)}`` 映射传给 ``AgentRunner`` 后，runner 会为每个工具 worker 组创建一个输入 ``Channel``\ （封装在 ``ToolChannelInfo`` 中）以及一条共享的 ``tool_output_channel``。

``AgentRunner`` 依次调用 ``init_worker(input_channel, output_channel)``、``start_server()`` 与 ``stop_server()`` 来管理工具服务生命周期。rollout 期间，智能体循环向 ``input_channel`` 发送 ``ToolChannelRequest``\ （包含 ``session_id``、``request_type``、``tool_name`` 和 ``tool_args``），工具 worker 执行完成后把 ``ToolChannelResponse`` 以 ``key=request.session_id`` 写回 ``output_channel``。无状态工具只处理 ``"execute"`` 请求；若设置 ``has_session=True``，还需处理 ``"session_start"`` 与 ``"session_end"``。

在 ``rlinf/agents/calc/calc_tool_worker.py`` 等位置实现计算器工具 worker：

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

按生命周期顺序理解这些方法：

- ``ToolWorker.init_worker(input_channel, output_channel)`` 已自动保存 ``self.input_channel`` 和 ``self.output_channel``，仅在需要初始化额外客户端时再覆写。
- ``start_server()`` 在当前事件循环上启动后台协程。
- ``_process_requests()`` 循环读取 ``ToolChannelRequest``，将执行异常捕获为 ``ToolChannelResponse(success=False, result=...)``\ （避免非法模型输出拖垮整轮训练），再按 ``key=request.session_id`` 写回结果。
- ``stop_server()`` 在退出时取消后台任务。

HTTP 工具实现可参考 ``rlinf/agents/searchr1/search_tool_worker.py``，带会话状态（``has_session=True``）的交互可参考 ``rlinf/agents/rstar2/rstar2_agent_loop.py``。

3. 实现多轮智能体循环
---------------------

``AgentLoopWorker`` 负责单条 query 组内的多轮 rollout。``run_one_query()`` 先调用 ``pre_process()``，再交替执行 ``generate_llm_response()`` 与 ``generate_tool_response()``，直到任一阶段返回 ``is_continue=False``；最后把全部轮次拼接为单个 ``AgentLoopOutput``\ （``response_mask`` 中模型 token 为 ``1``、工具返回 token 为 ``0``），并调用 ``post_process()``。

.. note::

   若配置中 ``agentloop.is_dynamic_rollout_batch`` 为 ``True``，请改继承 ``MultiAgentLoopWorker``。此时每一轮都会产出独立的 ``AgentLoopOutput`` 并汇入 ``MultiAgentLoopOutput``\ （参见 ``rlinf/agents/searchr1/searchr1_agent_loop.py``）。

在 ``rlinf/agents/calc/calc_agent_loop.py`` 中实现计算器智能体循环：

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

上述实现按调用顺序复用 ``AgentLoopWorker`` 提供的基类能力：

- ``__init__`` 设置 ``self.max_prompt_len``\ （``run_one_query()`` 会直接读取该属性）与 ``self.max_resp_len``，确保拼接长度不超过 ``cfg.runner.seq_length``。
- ``pre_process(prompt_ids)`` 初始化单条 query 的上下文字典；如需释放会话或补充 ``extra_fields``，可一并覆写 ``post_process(generate_context, output)``。
- ``self.generate(turn_prompt_ids, sampling_params=...)`` 通过 ``generate_input_channel`` 请求推理引擎，返回 ``output_ids``\ （开启 ``self.return_logprobs`` 时同时返回 ``logprobs``）。
- ``self.tool_call(tool_request)`` 借助 ``tool_name_map`` 与 ``tool_channel_info_map`` 发送无状态 ``ToolChannelRequest``，并返回 ``ToolResponse``。
- ``self.get_tool_response_ids(tool_messages)`` 使用 tokenizer 的对话模板编码工具消息，仅返回新增 token ID；将对应 ``response_mask`` 置为 ``0`` 后，策略梯度更新会自动跳过工具输出。

4. 配置、注册与测试
-------------------

在 Hydra YAML 配置的 ``agentloop`` 节点下选择解析器与轮次上限（可从 ``examples/agent/searchr1/config/train_qwen2.5.yaml`` 复制修改）：

.. code-block:: yaml

   agentloop:
     group_name: "AgentLoopGroup"
     toolcall_parser: "calc"
     max_turns: 3
     print_outputs: False
     is_dynamic_rollout_batch: False

在入口脚本中分别拉起智能体循环 worker 组与工具 worker 组，并传入 ``AgentRunner``\ （数据集、actor 与 rollout 的完整构建方式见 ``examples/agent/searchr1/train.py``）：

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

如果解析器放在外部仓库中，将其注册逻辑写入包的 ``register()`` 函数，并在启动训练前导出 ``RLINF_EXT_MODULE``，确保每个 Ray worker 进程都能完成注册（见 :ref:`out-of-tree-extensions`\ ）：

.. code-block:: bash

   export RLINF_EXT_MODULE=my_pkg.rlinf_ext

下一步
------

- **已有 OpenAI API 或外部黑盒 Agent？** 参见 :doc:`黑盒 Agent <openai_agent>`，无需重写解析器与 ``ToolWorker``。
- **扩展到多机异步训练：** 开启 ``runner.async_mode: True``\ （见 :doc:`异步 Agent RL <../guides/async_agentic>`），并按需通过 :doc:`Session Server <../guides/session_server>` 将轨迹存储下沉到 rollout 节点。
