新增模型
========

按模型类别与所用的训练 / Rollout 后端选择对应的接入指南。

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - 指南
     - 适用场景
     - 对应后端
   * - :doc:`具身 RL 模型（FSDP） <../new_model_fsdp>`
     - 接入新的 VLA 或具身 RL 策略
     - FSDP + Hugging Face
   * - :doc:`具身 SFT 模型（FSDP） <../new_model_sft>`
     - 接入自定义监督微调模型与数据格式
     - FSDP SFT Worker
   * - :doc:`具身 Rollout 模型（SGLang） <../sglang_embodied_model>`
     - 为具身或世界动作模型接入高速评测 Rollout
     - SGLang Embodied Worker
   * - :doc:`LLM 模型（Megatron-Bridge） <../mbridge>`
     - 推荐：免手写权重切分，接入 Hugging Face LLM
     - Megatron-Bridge
   * - :doc:`LLM 模型（手动 Megatron） <../new_model_megatron>`
     - Megatron-Bridge 暂未覆盖的自定义架构
     - 手动 Megatron + SGLang 权重重切分
   * - :doc:`自定义权重同步 <../weight_syncer>`
     - 定制或优化 Actor 到 Rollout 的模型权重传输
     - ``WeightSyncer``\ （FSDP + HF Rollout）

.. toctree::
   :hidden:

   具身 RL 模型（FSDP） <../new_model_fsdp>
   具身 SFT 模型（FSDP） <../new_model_sft>
   具身 Rollout 模型（SGLang） <../sglang_embodied_model>
   LLM 模型（Megatron-Bridge） <../mbridge>
   LLM 模型（手动 Megatron） <../new_model_megatron>
   自定义权重同步 <../weight_syncer>
