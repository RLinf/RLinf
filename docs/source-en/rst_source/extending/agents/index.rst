Add an Agent
============

RLinf supports two ways to bring an agent into RL training—pick the path based on whether you want to build the agent inside RLinf or connect an existing one over API.

Choose an Integration Path
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 56 24

   * - Agent Type
     - What It Is & Examples
     - Integration Path
   * - **White-Box Agent**
     - Multi-turn logic and tool execution live inside RLinf, giving direct control over prompts, tool parsing, and token masks (for example Search-R1 or a custom ``<calc>`` / ``<search>`` tool agent).
     - :doc:`White-Box Agent <../new_agent>`
   * - **Black-Box Agent**
     - An external agent that talks to the model only through the OpenAI Chat API (``/v1/chat/completions``), keeping its own orchestration unchanged (for example an OpenAI SDK, AutoGen, or existing production agent).
     - :doc:`Black-Box Agent <../openai_agent>`

Guides
------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. White-Box Agent
      :link: ../new_agent
      :link-type: doc

      Build multi-turn reasoning and tool execution inside RLinf with token-level control (for example Search-R1 or custom tag tools).

   .. grid-item-card:: 2. Black-Box Agent
      :link: ../openai_agent
      :link-type: doc

      Connect an existing external agent over RLinf's OpenAI API endpoint (for example OpenAI SDK or AutoGen) without rewriting its logic.

.. toctree::
   :hidden:

   White-Box Agent <../new_agent>
   Black-Box Agent <../openai_agent>
