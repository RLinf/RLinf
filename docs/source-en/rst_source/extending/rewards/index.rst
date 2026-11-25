Add a Reward
============

Pick the reward path that fits your task: deterministic rule-based scorers for reasoning and agent tasks, or model-based reward predictors for embodied RL.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Rule-Based Reward
      :link: ../rule_reward
      :link-type: doc

      Register a deterministic Python scorer for reasoning and agent completions via ``register_reward``.

   .. grid-item-card:: Model-Based Reward
      :link: ../reward_model
      :link-type: doc

      Subclass ``BaseRewardModel`` or train built-in ResNet / VLM reward models for embodied RL.

.. toctree::
   :hidden:

   Rule-Based Reward <../rule_reward>
   Model-Based Reward <../reward_model>
