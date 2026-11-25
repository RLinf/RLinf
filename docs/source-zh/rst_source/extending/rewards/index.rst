新增 Reward
===========

按任务类型选择对应路径：推理与 Agent 任务使用基于规则的 Reward，具身 RL 任务使用基于模型的 Reward。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 基于规则的 Reward
      :link: ../rule_reward
      :link-type: doc

      通过 ``register_reward`` 为推理与 Agent 回复注册确定性 Python 打分类。

   .. grid-item-card:: 基于模型的 Reward
      :link: ../reward_model
      :link-type: doc

      扩展 ``BaseRewardModel`` 或训练内置 ResNet / VLM 具身 Reward 模型。

.. toctree::
   :hidden:

   基于规则的 Reward <../rule_reward>
   基于模型的 Reward <../reward_model>
