新增 Reward
===========

按任务类型选择对应路径：推理与 Agent 任务用规则 Reward 打分，具身 RL 任务用学习型 Reward 模型。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 规则 Reward
      :link: ../rule_reward
      :link-type: doc

      通过 ``register_reward`` 为推理与 Agent 回复注册确定性 Python 打分类。

   .. grid-item-card:: 学习型 Reward 模型
      :link: ../reward_model
      :link-type: doc

      扩展 ``BaseRewardModel`` 或训练内置 ResNet / VLM 具身 Reward 模型。

.. toctree::
   :hidden:

   规则 Reward <../rule_reward>
   学习型 Reward 模型 <../reward_model>
