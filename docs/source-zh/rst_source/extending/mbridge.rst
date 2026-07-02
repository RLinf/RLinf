Megatron-Bridge
===============

RLinf 通过 Megatron-LM 训练后端支持 `Megatron-Bridge`。该集成允许你直接从
HuggingFace 格式的 checkpoint 启动 Megatron-LM 训练，使用 Megatron-Bridge 支持的模型结构，
同时保持 RLinf 的训练循环、数据管线、日志记录和 checkpoint 工作流不变。

当你有以下需求时，建议使用 Megatron-Bridge：

- actor 侧模型较大，FSDP 或 FSDP2 已经成为性能瓶颈；
- 模型架构尚未被 RLinf 原生 Megatron-LM 集成支持。

Megatron-Bridge 相关仓库：

- `Megatron-Bridge 原仓库 <https://github.com/NVIDIA/Megatron-Bridge>`__

- `当前 RLinf 使用的 Megatron-Bridge 版本 0.3.0 <https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.3.0>`__

- `对应 Megatron-LM 版本 b0cc2706ddc60d2aefd5fff346445b5c013036a8 <https://github.com/NVIDIA/Megatron-LM/tree/b0cc2706ddc60d2aefd5fff346445b5c013036a8>`__

安装环境
--------

MBridge 当前使用 RLinf 的 agentic 运行环境。先安装基础环境：

.. code:: bash

   bash requirements/install.sh agentic
   source .venv/bin/activate

安装 MBridge 路径所需的额外 Python 包：

.. code:: bash

   uv pip install transformers==4.57.1 bitsandbytes

Reasoning 镜像不包含 ``megatron.bridge`` 包。请下载
Megatron-Bridge 和匹配版本的 Megatron-LM，并把两个源码目录加入
``PYTHONPATH``：

.. code:: bash

   export MBRIDGE_ROOT=/workspace/dataset/Megatron-Bridge-0.3.0
   export MEGATRON_LM_ROOT=/workspace/dataset/Megatron-LM-b0cc2706ddc60d2aefd5fff346445b5c013036a8

   mkdir -p "$(dirname "${MBRIDGE_ROOT}")" "$(dirname "${MEGATRON_LM_ROOT}")"
   git clone --branch v0.3.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "${MBRIDGE_ROOT}"
   git clone https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM_ROOT}"
   git -C "${MEGATRON_LM_ROOT}" checkout b0cc2706ddc60d2aefd5fff346445b5c013036a8

   export PYTHONPATH="${MBRIDGE_ROOT}/src:${MEGATRON_LM_ROOT}:${PYTHONPATH}"
   export CUDA_DEVICE_MAX_CONNECTIONS=1
   python -c "from megatron.bridge import AutoBridge; print('Megatron-Bridge OK')"

如果集群镜像已经挂载了这两个仓库，保留相同的 ``PYTHONPATH`` 设置，
并跳过两个 ``git clone`` 命令。

下载 Reasoning 示例使用的模型和数据集：

.. code:: bash

   # 国内用户可设置 export HF_ENDPOINT=https://hf-mirror.com
   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
       --local-dir /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

   mkdir -p /dataset/boba
   hf download inclusionAI/AReaL-boba-Data AReaL-boba-106k.jsonl \
       --repo-type dataset \
       --local-dir /dataset/boba

下载 VLM SFT 示例使用的模型和数据集：

.. code:: bash

   # 国内用户可设置 export HF_ENDPOINT=https://hf-mirror.com
   hf download Qwen/Qwen2.5-VL-3B-Instruct \
       --local-dir /path/to/Qwen2.5-VL-3B-Instruct

   hf download keplerccc/Robo2VLM-1 \
       --repo-type dataset \
       --local-dir /path/to/Robo2VLM-1

.. warning::

   Robo2VLM 下载后会把训练和评估文件放在同一目录，例如
   ``train-00000-of-00262.parquet`` 和
   ``test-0000X-of-00003.parquet``。训练前请将它们移动到不同目录。
   否则，RLinf 会把整个数据集当作训练数据读取。

示例 SFT 配置默认读取 ``/path/to/Robo2VLM-1/data`` 和
``/path/to/Robo2VLM-1/eval_data``。如果数据集放在其他路径，请同步更新
``data.train_data_paths`` 和 ``data.eval_data_paths``。

使用介绍
--------

启用 MBridge 后，RLinf 会通过 ``Megatron-Bridge`` 导入并构建 Megatron-LM 模型，
而不是使用传统的 Megatron checkpoint 转换流程。

关键配置如下。

对于 Reasoning 任务：

.. code:: yaml

   actor:
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

当 ``actor.megatron.mbridge`` 为 ``True`` 且 ``use_hf_ckpt`` 为 ``True`` 时，
RLinf 会读取 ``ckpt_convertor.hf_model_path`` 设定的模型路径，并交由 MBridge 构建 Megatron model provider。

对于 SFT 任务：

.. code:: yaml

   actor:
     training_backend: megatron
     model:
       model_path: /path/to/Qwen2.5-VL-3B-Instruct
       megatron_checkpoint: null
     megatron:
       use_hf_ckpt: True
       mbridge: True

当 ``actor.megatron.mbridge`` 为 ``True`` 时，
RLinf 会读取 ``actor.model.model_path`` 设定的模型路径，并交由 MBridge 构建 Megatron model provider。

快速开始
--------

1. 启动训练前导出 MBridge 路径：

.. code:: bash

   export PYTHONPATH=/path/to/Megatron-Bridge-0.3.0/src:$PYTHONPATH
   export PYTHONPATH=/path/to/Megatron-LM-b0cc2706ddc60d2aefd5fff346445b5c013036a8:$PYTHONPATH
   export CUDA_DEVICE_MAX_CONNECTIONS=1

2. 准备 HuggingFace 模型和数据目录：

.. code:: text

   # Reasoning 任务需要：
   /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B
   /dataset/boba/AReaL-boba-106k.jsonl
   # SFT 任务需要：
   /path/to/Qwen2.5-VL-3B-Instruct
   /path/to/Robo2VLM-1/data
   /path/to/Robo2VLM-1/eval_data

3. 在配置文件中更新模型、tokenizer 和数据集路径：

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
         hf_model_path: /path/to/model/DeepSeek-R1-Distill-Qwen-1.5B

   data:
     train_data_paths: ["/dataset/boba/AReaL-boba-106k.jsonl"]
     val_data_paths: ["/dataset/boba/AReaL-boba-106k.jsonl"]

SFT 示例：

.. code:: yaml

   actor:
     model:
       model_type: "qwen2.5_vl"
       model_path: "/path/to/Qwen2.5-VL-3B-Instruct"
       megatron_checkpoint: null

     tokenizer:
       tokenizer_model: "/path/to/Qwen2.5-VL-3B-Instruct"

     megatron:
       use_hf_ckpt: True
       mbridge: True

   data:
     train_data_paths: "/path/to/Robo2VLM-1/data"
     eval_data_paths: "/path/to/Robo2VLM-1/eval_data"

4. 启动对应的训练脚本:

在仓库根目录启动 Reasoning 训练。这里直接使用 Python 入口，确保显式应用
MBridge 配置覆盖：

.. code:: bash

   python examples/reasoning/main_grpo.py \
       --config-path "$(pwd)/examples/reasoning/config/math" \
       --config-name qwen2.5-1.5b-grpo-megatron \
       actor.megatron.mbridge=True

在仓库根目录启动 VLM SFT 训练脚本:

.. code:: bash

   bash examples/sft/run_vlm_sft.sh qwen2_5_vl_megatron_sft_vlm

Checkpoint 加载模式
-------------------

当前，RLinf 中使用 Megatron-Bridge 时，会同时保存两份模型的 checkpoint，包含 HF checkpoint 和 Megatron checkpoint。
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