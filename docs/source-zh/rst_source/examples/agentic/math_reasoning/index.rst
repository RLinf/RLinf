推理训练
========

使用 Megatron 或 FSDP 训练大语言模型与视觉语言模型的单轮推理能力（数学与多模态几何推理），配合 SGLang 或 vLLM rollout。在下方选择模型与算法配方。

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Moonlight-16B（MoE GRPO）
      :link: moonlight
      :link-type: doc

      通过 Megatron-Bridge 使用 GRPO 训练 Moonlight-16B-A3B-Instruct（DeepSeek-V3 MLA + MoE）。

   .. grid-item-card:: Qwen2.5 数学推理（GRPO）
      :link: reasoning
      :link-type: doc

      在 AReaL-boba 数学数据上使用 GRPO 训练 DeepSeek-R1-Distill-Qwen（1.5B 与 7B），覆盖 AIME24、AIME25 与 GPQA-diamond。

   .. grid-item-card:: Qwen2.5 数学推理（PPO）
      :link: reasoning_ppo
      :link-type: doc

      基于 Qwen2.5-1.5B 的 PPO 数学推理训练，同时训练 actor 与 critic。

   .. grid-item-card:: Qwen3-VL 视觉推理（Geo3K）
      :link: ../qwen3_vl_geo3k
      :link-type: doc

      基于 FSDP 与 SGLang 的多模态视觉语言模型 GRPO 训练，面向 Geo3K 几何推理任务。

.. toctree::
   :hidden:
   :maxdepth: 1

   Moonlight-16B（MoE GRPO） <moonlight>
   Qwen2.5 数学推理（GRPO） <reasoning>
   Qwen2.5 数学推理（PPO） <reasoning_ppo>
   Qwen3-VL 视觉推理（Geo3K） <../qwen3_vl_geo3k>
