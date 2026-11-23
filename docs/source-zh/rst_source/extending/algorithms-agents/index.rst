新增算法、Reward 与 Agent
=========================

用于扩展 RL 训练算法、规则或学习型 Reward，以及自定义多轮 Agent 循环与工具调用。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 优势、损失与规则 Reward
      :link: ../new_algorithm
      :link-type: doc

      注册自定义优势估计函数、策略损失函数与基于规则的文本 Reward。

   .. grid-item-card:: 学习型 Reward 模型
      :link: ../reward_model
      :link-type: doc

      扩展 ``BaseRewardModel`` 或训练内置 ResNet / VLM 具身 Reward 模型。

   .. grid-item-card:: Agent 循环与工具调用
      :link: ../new_agent
      :link-type: doc

      实现自定义多轮 Agent 循环、工具 Worker 与工具调用解析器。

.. toctree::
   :hidden:

   优势、损失与规则 Reward <../new_algorithm>
   学习型 Reward 模型 <../reward_model>
   Agent 循环与工具调用 <../new_agent>
