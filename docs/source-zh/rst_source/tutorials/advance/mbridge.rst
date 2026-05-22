Megatron-Bridge
===============

RLinf 通过 Megatron-LM 训练后端支持 `Megatron-Bridge`。该集成允许用户直接从
HuggingFace 格式的 checkpoint 启动 Megatron-LM 训练，享受到 Megatron-Mridge 所支持的特殊模型结构，
同时保持 RLinf 的训练流程、数据管线、日志记录和 checkpoint 管理方式不变。

当你有以下需求时，建议使用 Megatron-MBridge 替换原本的 Megatron-LM 后端:

- 需要在 actor 侧进行训练的模型特别大，使用 FSDP 或 FSDP2 运行该模型出现明显的性能瓶颈；
- 使用的模型结构特殊，当前 RLinf 内支持的基础 Megatron-LM 框架内并无此模型结构支持；

Megatron-Bridge 相关仓库：

- `Megatron-Bridge 原仓库 <https://github.com/NVIDIA/Megatron-Bridge>`__

- `当前 RLinf 使用的 Megatron-Bridge 版本 <https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.3.0>`__

- `对应 Megatron-LM 版本 <https://github.com/NVIDIA/Megatron-LM/tree/b0cc2706ddc60d2aefd5fff346445b5c013036a8>`__

使用介绍
--------

启用 MBridge 后，RLinf 会通过 ``Megatron-Bridge`` 导入和构建 Megatron-LM 模型，
而不是使用传统的 Megatron checkpoint 转换流程。

关键配置如下：
对于 reasoning 任务：

.. code:: yaml

   actor:
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: /path/to/huggingface_model
      
当 ``actor.megatron.mbridge`` 为 ``True`` 且 ``use_hf_ckpt`` 为 ``True`` 时，
RLinf 会读取 ``ckpt_convertor.hf_model_path`` 设定的模型路径，并交由 MBridge 构建 Megatron model provider。

对于 SFT 任务：

.. code:: yaml

   actor:
     training_backend: megatron
     model:
       model_path: /path/to/huggingface_model
     megatron:
       use_hf_ckpt: True
       mbridge: True

当 ``actor.megatron.mbridge`` 为 ``True`` 时，
RLinf 会读取 ``actor.model.model_path`` 设定的模型路径，并交由 MBridge 构建 Megatron model provider。

快速开始
--------

1. 将 Megatron-Bridge 和对应版本的 Megatron-LM 加入 ``PYTHONPATH``：

.. code:: bash

   export PYTHONPATH=/path/to/Megatron-Bridge/src:$PYTHONPATH
   export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
   export CUDA_DEVICE_MAX_CONNECTIONS=1

2. 准备 HuggingFace 模型目录，例如：

.. code:: text

   /path/to/Qwen2.5-VL-3B-Instruct

3. 在配置文件中更新模型和 tokenizer 路径：

路径差异
--------

在不同训练任务中，MBridge 读取 HuggingFace checkpoint 的配置入口略有不同：

- Reasoning / RL 任务：通常从 ``actor.megatron.ckpt_convertor.hf_model_path`` 读取 HuggingFace 模型路径；
- SFT 任务：通常从 ``actor.model.model_path`` 读取 HuggingFace 模型路径；
- tokenizer 路径仍由 ``actor.tokenizer.tokenizer_model`` 指定，建议与模型目录保持一致。

因此，配置时不要只复制 ``mbridge: True``，还需要确认模型路径配置在当前任务类型下是否生效。

Reasoning 任务示例：

.. code:: yaml

   actor:
     tokenizer:
       tokenizer_model: "/path/to/model/DeepSeek-R1-Distill-Qwen-1.5B"
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: /path/to/huggingface_model


SFT 示例：

.. code:: yaml

   actor:
     model:
       model_type: "qwen2.5_vl"
       model_path: "/path/to/Qwen2.5-VL-3B-Instruct"

     tokenizer:
       tokenizer_model: "/path/to/Qwen2.5-VL-3B-Instruct"

     megatron:
       use_hf_ckpt: True
       mbridge: True


4. 启动对应的训练脚本:

在仓库根目录启动 Reasoning 训练脚本:

.. code:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron

在仓库根目录启动 VLM SFT 训练脚本:

.. code:: bash

   bash examples/sft/run_vlm_sft.sh qwen2_5_vl_megatron_sft_vlm

Checkpoint 加载模式
-------------------

当前，RLinf 中使用 Megatron-MBridge 时，会同时保存两份模型的 checkpoint， 包含 HF checkpoint 和 Megatron checkpoint。
文件结构如下：

.. code:: text

   /path/to/logs/qwen2.5-1.5b-grpo-megatron/checkpoints/
   ├── global_step_10/
   │   └── actor/
   │       ├── hf_model/
   │       │   ├── model.safetensors
   │       │   └── tokenizer.json
   │       ├── iter_0000010/
   │       │   ├── mp_rank_00/
   │       │   │   ├── distrib_optim.pt
   │       │   │   └── model_optim_rng.pt
   │       │   └── mp_rank_01/
   │       │       ├── distrib_optim.pt
   │       │       └── model_optim_rng.pt
   │       └── latest_checkpointed_iteration.txt
   └── global_step_20/
       └── …

hf_model 目录下保存了 HF checkpoint 的模型权重和 tokenizer 文件。
iter_XXXXXXX 目录下保存了 Megatron checkpoint 的模型权重和 optimizer 文件。
latest_checkpointed_iteration.txt 文件保存了当前 checkpoint 的 step 信息。
如例子中 ``global_step_10/`` 和 ``global_step_20/`` 是两个不同的 checkpoint，分别对应 step 10 和 step 20 的 checkpoint。

如果只是想重新断点续训，可以只加载 Megatron checkpoint，无需加载 HF checkpoint。
加载方式：

.. code:: yaml

   runner:
     resume_dir: /path/to/logs/qwen2.5-1.5b-grpo-megatron/checkpoints/global_step_10

使用建议
--------

- 当 ``use_hf_ckpt: True`` 时，保持 ``actor.model.megatron_checkpoint: null``。
- 只有在加载已准备好的 Megatron checkpoint 时，才设置
  ``actor.megatron.use_hf_ckpt: False``。
- 对于 Qwen3-VL 模型，保持 ``actor.model.apply_rope_fusion: False``。
- 对于 Qwen2.5 模型，qkv_bias 会被强制打开以适配模型。
- 对于 Qwen3 模型，qk_layernorm 会被强制打开以适配模型。
- 确保 tokenizer 路径与 HuggingFace 模型保持匹配。

常见问题
--------

``model.megatron_checkpoint is required if use_hf_ckpt is False``
  当前关闭了 ``use_hf_ckpt``，但没有提供 Megatron checkpoint 路径。
  请设置 ``actor.megatron.use_hf_ckpt: True``，或者提供 ``resume_dir`` 路径。

``model.megatron_checkpoint should be None if use_hf_ckpt is True``
  HuggingFace 加载和 Megatron checkpoint 加载被同时启用了。

Qwen3-VL 报 ``deepstack_visual_indexes`` 相关断言
  模型的 visual deepstack 配置与当前 pipeline 切分不匹配。
  可以先尝试 ``pipeline_model_parallel_size: 1``。如果必须开启 pipeline parallel，
  需要确保第一段 language pipeline stage 的层数能够容纳所有
  ``deepstack_visual_indexes``。如果使用的是裁剪层数后的 checkpoint，还需要确认
  visual deepstack 配置与语言模型层数一致。