数学推理
========

使用 Megatron actor 与 SGLang 或 vLLM rollout，在 AIME24、AIME25 与 GPQA-diamond 等数学推理基准上用 GRPO 或 PPO 训练大语言模型。在下方选择模型与算法配方。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Qwen2.5 GRPO
      :link: reasoning
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/math_numbers_small.jpg
         :alt: Qwen2.5 数学推理 GRPO

      在 AReaL-boba 数学数据上使用 GRPO 训练 DeepSeek-R1-Distill-Qwen（1.5B 与 7B）。

   .. grid-item-card:: Qwen2.5 PPO
      :link: reasoning_ppo
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_vs_grpo.png
         :alt: Qwen2.5 数学推理 PPO

      基于 Qwen2.5-1.5B 的 PPO 数学推理训练，同时训练 actor 与 critic。

   .. grid-item-card:: Moonlight-16B GRPO
      :link: moonlight
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/moonlight/moonlight_grpo_reward.jpeg
         :alt: Moonlight-16B GRPO

      通过 Megatron-Bridge 使用 GRPO 训练 Moonlight-16B-A3B-Instruct（DeepSeek-V3 MLA + MoE）。

.. toctree::
   :hidden:
   :maxdepth: 1

   Qwen2.5 GRPO <reasoning>
   Qwen2.5 PPO <reasoning_ppo>
   Moonlight-16B GRPO <moonlight>
