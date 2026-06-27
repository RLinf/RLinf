扩展
====

当你要向 RLinf 添加模型、环境或高级集成路径时，使用扩展页。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 扩展概览
      :link: overview
      :link-type: doc

      各扩展点位于何处，以及各部分如何协同。

   .. grid-item-card:: 新环境
      :link: new_env
      :link-type: doc

      添加一个新的 RL 环境并接入环境注册表。

   .. grid-item-card:: FSDP 新模型
      :link: new_model_fsdp
      :link-type: doc

      在 FSDP 后端上添加 HuggingFace 模型。

   .. grid-item-card:: Megatron 新模型
      :link: new_model_megatron
      :link-type: doc

      在 Megatron+SGLang 后端上添加 HuggingFace 模型。

   .. grid-item-card:: 新 SFT 模型
      :link: new_model_sft
      :link-type: doc

      将新模型接入 SFT 训练流程。

   .. grid-item-card:: 高级集成
      :link: advanced-integrations/index
      :link-type: doc

      添加 Megatron-Bridge、权重同步或奖励模型工作流。

.. toctree::
   :hidden:

   扩展概览 <overview>
   新环境 <new_env>
   FSDP 新模型 <new_model_fsdp>
   Megatron 新模型 <new_model_megatron>
   新 SFT 模型 <new_model_sft>
   高级集成 <advanced-integrations/index>
