.. _session-server-openai-endpoint:

Black-Box Agent
===============

A **black-box agent** keeps its own orchestration outside RLinf and talks to the model only through the OpenAI Chat API (``/v1/chat/completions``)—for example an existing OpenAI SDK, AutoGen, or custom-framework agent. With :doc:`Session Server <../guides/session_server>` enabled, point your client at RLinf's OpenAI endpoint to collect training trajectories without rewriting tool or prompt logic.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Item
     - Summary
   * - Extension point
     - Subclass ``MultiAgentLoopWorker`` and override ``run_session_trajectory(self, binding, input_ids, answer)``
   * - Endpoint
     - ``/sessions/<session id>/v1/chat/completions`` (returned via ``self.session_server_client.session_base_url(...)``)
   * - Required config
     - ``session_server.enable: true``, ``agentloop.agentic_tokenizer``, and matching ``rollout.sglang.tool_call_parser``
   * - Prerequisites
     - Asynchronous agentic setup with Session Server enabled (see :doc:`Session Server <../guides/session_server>`)

1. Configure Tokenizer and Tool Parser
--------------------------------------

Enable the Session Server, select the model family in ``agentloop.agentic_tokenizer``, and pair it with the matching SGLang ``tool_call_parser`` (validated at startup):

.. code-block:: yaml

   session_server:
     enable: true
   agentloop:
     agentic_tokenizer: qwen     # glm47 for GLM-4.7; default for no tool parsing
   rollout:
     sglang:
       tool_call_parser: qwen25  # glm47 for GLM-4.7; omit for default

- Responses use OpenAI's ``chat.completion`` format, parsing reasoning into ``reasoning_content`` (for models with reasoning markers such as Qwen3 and GLM-4.7) and tool calls into ``tool_calls``.
- Set ``agentic_tokenizer: default`` for other models to render messages without tool-call parsing. Leaving ``agentloop.agentic_tokenizer`` unset causes chat requests to return HTTP 500.

2. Drive a Session with AsyncOpenAI
-----------------------------------

Subclass ``MultiAgentLoopWorker`` and override ``run_session_trajectory`` to run one trajectory against an already-open session. The base worker still opens and closes sessions, reports rewards, and discards failed groups:

.. code-block:: python

   from openai import APIStatusError, AsyncOpenAI, BadRequestError

   from rlinf.workers.agent.agent_loop import GroupDiscarded, MultiAgentLoopWorker


   class MyChatAgentLoopWorker(MultiAgentLoopWorker):
       async def run_session_trajectory(self, binding, input_ids, answer):
           base_url = self.session_server_client.session_base_url(
               binding.group_id, binding.session_id
           )
           prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
           messages = build_messages(prompt)  # extract task from templated prompt
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
                       result = await run_tool(call)  # your existing tool or agent step
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

Client contract:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Rule
     - Requirement
   * - Disable client retries
     - Set ``max_retries=0``; a retried request is sampled again, and only one answer is trained.
   * - Handle 410 and length limits
     - HTTP 410 means the Session Server already dropped the group (raise ``GroupDiscarded``); ``context_length_exceeded`` (HTTP 400) means the prompt no longer fits in ``runner.seq_length`` (score the last turn, or discard if turn 1 fails).
   * - Serialize per session
     - Sending a request or close while another turn is in flight returns HTTP 409.
   * - Prompt decoding & extra fields
     - ``input_ids`` already has the chat template applied, so ``build_messages`` extracts the task text first. Return ``idx_to_sub_traj`` when ``algorithm.loss_scales`` includes ``agent_level``.

3. Launch and Verify
--------------------

Launch your chat agent loop class in your entry script the same way ``examples/agent/searchr1/train.py`` launches its agent loop worker, and keep ``reward.reward_type: searchr1`` in the config to satisfy startup validation (even when your worker scores trajectories itself).

Monitor training under ``rollout/session_server/`` (see :ref:`Metrics to Watch <session-server-metrics>`):

- ``chat_reasoning_responses`` and ``chat_tool_call_responses`` confirm reasoning and tool calls are parsed as expected.
- ``chat_tool_call_parse_failed`` surfaces malformed tool-call blocks.
- ``prefix_rewritten`` counts turns where the client altered earlier history instead of appending new messages.

.. note::

   ``session_server.verify_assembly`` applies only to token-ID agent loops; keep it ``False`` when using ``/v1/chat/completions``.

.. dropdown:: Prefix token reuse & rejected request parameters

   To preserve exact engine tokens and KV-cache hits, the Session Server reuses stored tokens from previous turns (including ``reasoning_content`` even if the client strips it):

   - **Continue:** Appends only newly rendered user/tool messages to stored tokens (with ``clear_thinking: true``, a new user turn triggers a full re-render).
   - **Retry:** Repeating the previous request rolls back the abandoned turn (kept in payload, excluded from training).
   - **Rewrite:** Modifying earlier messages re-renders in full, increments ``prefix_rewritten``, and trains normally.

   Unsupported OpenAI request options return HTTP 400:

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Parameter
        - Rejection condition
      * - ``n``
        - Greater than 1
      * - ``temperature``, ``top_p``, ``top_k``
        - Different from configured values
      * - ``tool_choice``
        - Anything other than ``auto`` or ``none``
      * - ``response_format``
        - Any value (unsupported)
      * - ``chat_template_kwargs``
        - Different from configured values
