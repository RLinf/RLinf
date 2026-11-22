Math Reasoning
==============

Train large language models on math reasoning benchmarks (AIME24, AIME25, GPQA-diamond) with GRPO and PPO using the Megatron actor and SGLang or vLLM rollout. Pick a model family and algorithm below.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Qwen2.5 GRPO
      :link: reasoning
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/math_numbers_small.jpg
         :alt: Qwen2.5 Math GRPO

      GRPO training for DeepSeek-R1-Distill-Qwen (1.5B and 7B) on AReaL-boba math data.

   .. grid-item-card:: Qwen2.5 PPO
      :link: reasoning_ppo
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_vs_grpo.png
         :alt: Qwen2.5 Math PPO

      PPO training with an actor and critic on Qwen2.5-1.5B math reasoning tasks.

   .. grid-item-card:: Moonlight-16B GRPO
      :link: moonlight
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/moonlight/moonlight_grpo_reward.jpeg
         :alt: Moonlight-16B GRPO

      GRPO training for Moonlight-16B-A3B-Instruct (DeepSeek-V3 MLA + MoE) via Megatron-Bridge.

.. toctree::
   :hidden:
   :maxdepth: 1

   Qwen2.5 GRPO <reasoning>
   Qwen2.5 PPO <reasoning_ppo>
   Moonlight-16B GRPO <moonlight>
