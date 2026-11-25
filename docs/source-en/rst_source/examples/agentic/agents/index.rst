Agentic RL
==========

Reinforcement learning recipes for multi-turn tool use, multi-agent coordination, external agent frameworks, and online coding assistants. Pick a scenario below.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: AgentLightning
      :link: ../agentlightning_calc_x
      :link-type: doc

      Use Agent Lightning for agent workflow orchestration while delegating distributed RL training to RLinf on Calc-X.

   .. grid-item-card:: Coding Online RL
      :link: ../coding_online_rl
      :link-type: doc

      End-to-end online RL for IDE code completion combining RLinf with the Continue extension.

   .. grid-item-card:: rStar2
      :link: ../rstar2
      :link-type: doc

      Multi-turn mathematical reasoning and self-reflection using a local Python execution server.

   .. grid-item-card:: Search-R1
      :link: ../searchr1/index
      :link-type: doc

      Multi-turn search tool RL on Qwen2.5-3B and GLM-4.7-Flash (sync and async), with FAISS or Qdrant local wiki retrieval.

   .. grid-item-card:: WideSeek-R1
      :link: ../wideseek_r1/index
      :link-type: doc

      Lead-agent and parallel subagent MARL for broad information seeking on WideSearch, reaching 40.0% item F1 with a 4B model.

.. tip::

   **Integrating your own agent?** See :doc:`Add an Agent </rst_source/extending/agents/index>` to build a white-box agent inside RLinf or connect an existing black-box agent over the OpenAI API.

.. toctree::
   :hidden:
   :maxdepth: 1

   AgentLightning <../agentlightning_calc_x>
   Coding Online RL <../coding_online_rl>
   rStar2 <../rstar2>
   Search-R1 <../searchr1/index>
   WideSeek-R1 <../wideseek_r1/index>
