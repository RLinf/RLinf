新增 Agent
==========

按现有 Agent 的形态选择接入路径：在 RLinf 内编写 **白盒 Agent** 获取 token 级精细控制，或将已有的外部 **黑盒 Agent** 通过 OpenAI API 免重写直接接入。

选择接入路径
------------

.. list-table::
   :header-rows: 1
   :widths: 26 48 26

   * - 起点
     - 推荐接入方式与示例
     - 对应文档
   * - 在 RLinf 内从零编写 Agent（Token 级控制）
     - 注册工具调用解析器、在 ``ToolWorker`` 中隔离执行工具，并继承 ``AgentLoopWorker``\ （例如 Search-R1 或自定义标签工具 Agent）。
     - :doc:`白盒 Agent <../new_agent>`
   * - 接入已有外部 Agent（消息级，免重写）
     - 保留原有 Agent 代码（例如基于 OpenAI SDK、AutoGen 或自有框架的外部 Agent），开启 Session Server，将 ``AsyncOpenAI`` 指向 ``/sessions/<id>/v1/chat/completions``。
     - :doc:`黑盒 Agent <../openai_agent>`

扩展指南
--------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. 白盒 Agent
      :link: ../new_agent
      :link-type: doc

      在 RLinf 内编写 token 级多轮 Agent（例如 Search-R1 或自定义标签解析器、``ToolWorker`` 与 ``AgentLoopWorker``）。

   .. grid-item-card:: 2. 黑盒 Agent
      :link: ../openai_agent
      :link-type: doc

      将已有的 OpenAI SDK 或外部 Agent（例如 AutoGen 等自有工作流）经 Session Server 的 ``/v1/chat/completions`` 免重写接入。

.. toctree::
   :hidden:

   白盒 Agent <../new_agent>
   黑盒 Agent <../openai_agent>
