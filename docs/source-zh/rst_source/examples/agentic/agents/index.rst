Agent 训练
==========

面向多轮工具调用、多 Agent 协作、外部 Agent 框架集成与在线代码补全的强化学习配方。按场景选择对应的项目入口。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: AgentLightning
      :link: ../agentlightning_calc_x
      :link-type: doc

      使用 Agent Lightning 编排 Agent 工作流，并将分布式 RL 训练交给 RLinf 后端（Calc-X 示例）。

   .. grid-item-card:: Coding Online RL
      :link: ../coding_online_rl
      :link-type: doc

      结合 RLinf 与 Continue IDE 插件的端到端代码补全在线强化学习流程。

   .. grid-item-card:: rStar2
      :link: ../rstar2
      :link-type: doc

      结合本地 Python 代码执行服务，训练模型完成多轮数学推理与自我反思。

   .. grid-item-card:: Search-R1
      :link: ../searchr1/index
      :link-type: doc

      基于 Qwen2.5-3B 与 GLM-4.7-Flash 的多轮搜索调用 RL（支持同步与异步训练），配套 FAISS 或 Qdrant 本地检索服务。

   .. grid-item-card:: WideSeek-R1
      :link: ../wideseek_r1/index
      :link-type: doc

      面向广泛信息检索的主从多 Agent 强化学习（MARL），4B 模型在 WideSearch 上达到 40.0% item F1。

.. tip::

   **接入自己的 Agent？** 参见 :doc:`新增 Agent </rst_source/extending/agents/index>`，按需选择在 RLinf 内编写白盒 Agent 或通过 OpenAI API 接入外部黑盒 Agent。

.. toctree::
   :hidden:
   :maxdepth: 1

   AgentLightning <../agentlightning_calc_x>
   Coding Online RL <../coding_online_rl>
   rStar2 <../rstar2>
   Search-R1 <../searchr1/index>
   WideSeek-R1 <../wideseek_r1/index>
