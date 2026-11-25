LoRA 微调
=========

在 RLinf 中使用 LoRA（Low-Rank Adaptation，低秩适配），可以冻结基础模型权重、仅训练低秩适配矩阵，以更低显存微调大型具身 policy 和扩散模型。

支持的后端
----------

不同模型系列的 LoRA 配置项有所不同：

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - 模型系列
     - 配置项
     - 说明
   * - 具身 FSDP VLA（OpenVLA、OpenVLA-OFT 等）
     - ``actor.model.is_lora: True``
     - 支持 ``lora_rank``、``lora_path``、``lora_target_scope``，以及通过 ``merged_loras`` 预合并已有 adapter。
   * - 扩散模型（SD3.5 Medium、Wan2.2 TI2V 5B）
     - ``actor.model.<model>.use_lora: True``
     - 在 :doc:`Diffusion-NFT <../examples/diffusion_nft>` 与 :doc:`DiffusionART <../examples/diffusion_art>` 的 YAML 模型块下配置（例如 ``actor.model.wan22_ti2v_5b.use_lora: True``）。

配置具身 FSDP LoRA
------------------

在具身 FSDP 训练 YAML 的 ``actor.model`` 下开启 LoRA：

.. code-block:: yaml

   actor:
     model:
       is_lora: True
       lora_rank: 32
       lora_path: null            # 填写已有 adapter 路径即可继续训练
       lora_target_scope: null    # 可选：设为 "all_linear" 适配全部线性层

- ``is_lora``：是否启用 LoRA 微调；设为 ``False`` 时转为全参数微调。
- ``lora_rank``：低秩矩阵的秩（常用 ``8``–``64``）；``is_lora: False`` 时忽略。
- ``lora_path``：已有 LoRA 权重路径，留空（``null``）表示从头初始化新 adapter。
- ``lora_target_scope``：未设置时，RLinf 默认注入视觉、投影及 LLM 注意力与 MLP 层（``proj``、``qkv``、``fc1``–``fc3``、``out_proj``、``q_proj``/``k_proj``/``v_proj``/``o_proj``、``gate_proj``/``up_proj``/``down_proj``、``lm_head``）；设为 ``"all_linear"`` 可适配全部线性层。

预合并已有 LoRA Adapter
-----------------------

需要在加载基础模型时先合并一个或多个已有 LoRA、再在其上训练新 adapter 时，在 ``actor.model.merged_loras`` 中列出待合并项。每项包含 ``path`` 和可选的缩放系数 ``scale``\ （默认为 ``1.0``）：

.. code-block:: yaml

   actor:
     model:
       model_path: /path/to/base_model
       merged_loras:
         - path: /path/to/pretrained_lora
           scale: 0.5
       is_lora: True
       lora_rank: 32

RLinf 会在模型加载阶段按顺序将 ``merged_loras`` 合并进基础权重，再按 ``is_lora`` 注入新的可训练 adapter。完整用法可参考 ``examples/embodiment/config/maniskill_libero_grpo_openvlaoft.yaml`` 与 :doc:`异构环境训练 <hetero_envs>`。

合并与导出 LoRA Checkpoint
--------------------------

FSDP 训练结束后，按 :doc:`Checkpoint 转换 <convertor>` 将 ``full_weights.pt`` 导出为 Hugging Face ``.safetensors`` 格式。将 ``convertor.merge_lora_weighs`` 设为 ``True``\ （默认）可把 LoRA 权重合并进完整模型导出，设为 ``False`` 则只导出 adapter 权重。
