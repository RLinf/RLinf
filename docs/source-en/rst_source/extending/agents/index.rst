Add an Agent
============

Pick the path that matches how your agent is built: write a **white-box agent** inside RLinf for token-level control, or plug in an existing **black-box agent** over the OpenAI API without rewriting its logic.

Choose a Path
-------------

.. list-table::
   :header-rows: 1
   :widths: 26 48 26

   * - Starting point
     - Recommended approach & examples
     - Guide
   * - Build an agent inside RLinf (token-level control)
     - Register a tool-call parser, isolate tool execution in a ``ToolWorker``, and subclass ``AgentLoopWorker`` (for example Search-R1 or custom XML/tag-based tool agents).
     - :doc:`White-Box Agent <../new_agent>`
   * - Connect an existing external agent (message-level)
     - Keep your existing agent code (for example OpenAI SDK, AutoGen, or custom HTTP agents), enable Session Server, and point ``AsyncOpenAI`` at ``/sessions/<id>/v1/chat/completions``.
     - :doc:`Black-Box Agent <../openai_agent>`

Guides
------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. White-Box Agent
      :link: ../new_agent
      :link-type: doc

      Build a token-level multi-turn agent inside RLinf (for example Search-R1 or custom tool-tag parsers with ``ToolWorker`` and ``AgentLoopWorker``).

   .. grid-item-card:: 2. Black-Box Agent
      :link: ../openai_agent
      :link-type: doc

      Connect an existing OpenAI-SDK or external agent (for example AutoGen or custom tool frameworks) via Session Server's ``/v1/chat/completions`` endpoint.

.. toctree::
   :hidden:

   White-Box Agent <../new_agent>
   Black-Box Agent <../openai_agent>
