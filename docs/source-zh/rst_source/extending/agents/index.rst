新增 Agent
==========

RLinf 支持两种 Agent 接入方式：在框架内编写白盒 Agent，或直接通过 API 接入已有的黑盒 Agent。按现有代码的形态选择对应路径即可。

选择接入路径
------------

.. list-table::
   :header-rows: 1
   :widths: 20 56 24

   * - Agent 类型
     - 说明与示例
     - 接入路径
   * - **白盒 Agent**
     - 在 RLinf 内部实现多轮交互与工具调用，直接控制每轮提示词、工具解析与 token 损失掩码（例如 Search-R1 或自定义 ``<calc>`` / ``<search>`` 标签工具 Agent）。
     - :doc:`白盒 Agent <../new_agent>`
   * - **黑盒 Agent**
     - 独立在外部编排、仅通过 OpenAI 兼容 API（``/v1/chat/completions``）调用模型的 Agent，无需改写原有流程（例如基于 OpenAI SDK、AutoGen 或自有框架的已有 Agent）。
     - :doc:`黑盒 Agent <../openai_agent>`

扩展指南
--------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. 白盒 Agent
      :link: ../new_agent
      :link-type: doc

      在 RLinf 内实现多轮推理与工具调用，获得 token 级精细控制（例如 Search-R1 或自定义标签工具）。

   .. grid-item-card:: 2. 黑盒 Agent
      :link: ../openai_agent
      :link-type: doc

      通过 OpenAI 兼容 API 直接接入已有外部 Agent（例如 OpenAI SDK 或 AutoGen），免重写原有编排逻辑。

.. toctree::
   :hidden:

   白盒 Agent <../new_agent>
   黑盒 Agent <../openai_agent>
