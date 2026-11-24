Reasoning
=========

Train language and vision-language models on single-turn math and multimodal reasoning benchmarks with GRPO or PPO using Megatron or FSDP actors and SGLang or vLLM rollout. Pick a model family and algorithm below.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Moonlight-16B (MoE GRPO)
      :link: moonlight
      :link-type: doc

      GRPO training for Moonlight-16B-A3B-Instruct (DeepSeek-V3 MLA + MoE) via Megatron-Bridge.

   .. grid-item-card:: Qwen2.5 Math (GRPO)
      :link: reasoning
      :link-type: doc

      GRPO training for DeepSeek-R1-Distill-Qwen (1.5B and 7B) on AReaL-boba math data, evaluated on AIME24, AIME25, and GPQA-diamond.

   .. grid-item-card:: Qwen2.5 Math (PPO)
      :link: reasoning_ppo
      :link-type: doc

      PPO training with an actor and critic on Qwen2.5-1.5B math reasoning tasks.

   .. grid-item-card:: Qwen3-VL Visual Reasoning (Geo3K)
      :link: ../qwen3_vl_geo3k
      :link-type: doc

      Vision-language GRPO training on geometric reasoning (Geo3K) with FSDP and SGLang.

.. toctree::
   :hidden:
   :maxdepth: 1

   Moonlight-16B (MoE GRPO) <moonlight>
   Qwen2.5 Math (GRPO) <reasoning>
   Qwen2.5 Math (PPO) <reasoning_ppo>
   Qwen3-VL Visual Reasoning (Geo3K) <../qwen3_vl_geo3k>
