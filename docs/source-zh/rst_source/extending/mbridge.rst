LLM 模型（Megatron-Bridge）
===========================

RLinf 通过 Megatron-LM 训练后端支持 `Megatron-Bridge <https://github.com/NVIDIA/Megatron-Bridge>`__，可以直接从 HuggingFace 格式的 checkpoint 启动 Megatron-LM 训练，使用 Megatron-Bridge 支持的模型结构，同时保持 RLinf 的训练循环、数据管线、日志记录和 checkpoint 工作流不变。

满足以下场景时，建议使用 Megatron-Bridge：

- actor 侧模型较大，FSDP 或 FSDP2 已成为性能瓶颈；
- 模型架构尚未被 RLinf 原生 Megatron-LM 集成支持。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 项目
     - 说明
   * - 环境要求
     - Python 3.12，通过 ``bash requirements/install.sh agentic`` 安装（包含 ``megatron-bridge`` 0.5.0、``nvidia-modelopt`` 0.45.0 与 Megatron-LM ``core_r0.18.0``\ ）
   * - 关键 YAML 开关
     - ``actor.training_backend: megatron``、``actor.megatron.mbridge: True`` 与 ``actor.megatron.use_hf_ckpt: True``
   * - 支持任务
     - Reasoning RL 与 VLM SFT
   * - 验证命令
     - ``python -c "from megatron.bridge import AutoBridge; print('Megatron-Bridge OK')"``

上游仓库与版本参考：

- `Megatron-Bridge 原仓库 <https://github.com/NVIDIA/Megatron-Bridge>`__
- `RLinf 使用的 Megatron-Bridge 0.5.0 <https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.5.0>`__
- `对应的 Megatron-LM 分支 core_r0.18.0 <https://github.com/NVIDIA/Megatron-LM/tree/core_r0.18.0>`__

安装环境
--------

MBridge 使用 RLinf 的 agentic 运行环境。``install.sh`` 会一并安装 Megatron-Bridge 和匹配的 Megatron-LM，不需要另外下载源码或设置 ``PYTHONPATH``：

.. code-block:: bash

   bash requirements/install.sh agentic && source .venv/bin/activate
   python -c "from megatron.bridge import AutoBridge; print('Megatron-Bridge OK')"

该命令会创建 Python 3.12 虚拟环境，安装 ``megatron-bridge`` 0.5.0 与 ``nvidia-modelopt`` 0.45.0，并把 Megatron-LM ``core_r0.18.0`` 克隆到虚拟环境目录下，由 ``.venv/bin/activate`` 加入 ``PYTHONPATH``。如需复用已有的 Megatron-LM 目录，可在安装前设置 ``MEGATRON_PATH=<megatron_dir>``；``install.sh`` 会直接复用该目录且不切换分支，请自行确认其位于 ``core_r0.18.0``。

.. note::

   Megatron-Bridge 需要 Python 3.12。如果通过 ``--python`` 指定更低版本，``install.sh`` 会跳过 Megatron-Bridge 并打印 WARNING，上面的导入检查会失败。Reasoning 镜像中的虚拟环境使用 Python 3.11，不包含 ``megatron.bridge``；使用 MBridge 时请按上面的命令安装环境。

下载 Reasoning 示例使用的模型和数据集：

.. code-block:: bash

   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
       --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B

   mkdir -p ./dataset/boba
   hf download inclusionAI/AReaL-boba-Data AReaL-boba-106k.jsonl \
       --repo-type dataset \
       --local-dir ./dataset/boba

下载 VLM SFT 示例使用的模型和数据集：

.. code-block:: bash

   hf download Qwen/Qwen2.5-VL-3B-Instruct \
       --local-dir ./Qwen2.5-VL-3B-Instruct

   hf download keplerccc/Robo2VLM-1 \
       --repo-type dataset \
       --local-dir ./Robo2VLM-1

.. warning::

   Robo2VLM 下载后会把训练和评估文件放在同一目录，例如 ``train-00000-of-00262.parquet`` 和 ``test-0000X-of-00003.parquet``。训练前请将它们移动到不同目录，否则 RLinf 会把整个数据集当作训练数据读取。

示例 SFT 配置默认读取 ``./Robo2VLM-1/data`` 和 ``./Robo2VLM-1/eval_data``。如果数据集放在其他路径，请同步更新 ``data.train_data_paths`` 和 ``data.eval_data_paths``。

在支持的模型上启用 Megatron-Bridge
----------------------------------

启用 MBridge 后，RLinf 通过 Megatron-Bridge 直接导入并构建 Megatron-LM 模型，无需执行离线的 Megatron checkpoint 转换。

关键配置
~~~~~~~~

Reasoning RL 与 SFT 任务读取模型路径的配置字段不同。

对于 Reasoning 任务，将 ``actor.megatron.mbridge`` 与 ``use_hf_ckpt`` 均设为 ``True`` 后，RLinf 会读取 ``actor.megatron.ckpt_convertor.hf_model_path`` 设定的模型路径，并交由 MBridge 构建 Megatron model provider：

.. code-block:: yaml

   actor:
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: ./DeepSeek-R1-Distill-Qwen-1.5B

对于 SFT 任务，将 ``actor.megatron.mbridge`` 设为 ``True`` 后，RLinf 会读取 ``actor.model.model_path`` 设定的模型路径，并交由 MBridge 构建 Megatron model provider：

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_path: ./Qwen2.5-VL-3B-Instruct
       megatron_checkpoint: null
     megatron:
       use_hf_ckpt: True
       mbridge: True

快速开始
~~~~~~~~

1. 启动训练前激活环境并设置环境变量：

.. code-block:: bash

   source .venv/bin/activate
   export CUDA_DEVICE_MAX_CONNECTIONS=1

2. 准备 HuggingFace 模型和数据目录：

.. code-block:: text

   # Reasoning 任务需要：
   ./DeepSeek-R1-Distill-Qwen-1.5B
   ./dataset/boba/AReaL-boba-106k.jsonl
   # SFT 任务需要：
   ./Qwen2.5-VL-3B-Instruct
   ./Robo2VLM-1/data
   ./Robo2VLM-1/eval_data

3. 在配置文件中更新模型、tokenizer 和数据集路径（详见下方 `路径差异`_）。

4. 在仓库根目录启动对应的训练脚本。

启动 Reasoning 训练时直接使用 Python 入口，显式应用 MBridge 配置覆盖：

.. code-block:: bash

   python examples/reasoning/main_grpo.py \
       --config-path "$(pwd)/examples/reasoning/config/math" \
       --config-name qwen2.5-1.5b-grpo-megatron \
       +actor.megatron.mbridge=True

启动 VLM SFT 训练：

.. code-block:: bash

   bash examples/sft/run_vlm_sft.sh qwen2_5_vl_megatron_sft_vlm

路径差异
~~~~~~~~

不同训练任务读取 HuggingFace checkpoint 的配置入口不同：

- Reasoning / RL 任务从 ``actor.megatron.ckpt_convertor.hf_model_path`` 读取 HuggingFace 模型路径。
- SFT 任务从 ``actor.model.model_path`` 读取 HuggingFace 模型路径。
- 两类任务的 tokenizer 路径均由 ``actor.tokenizer.tokenizer_model`` 指定，建议与模型目录保持一致。

迁移配置时不要只复制 ``mbridge: True``，还应确认对应任务类型的模型路径字段已正确配置。

Reasoning 任务示例：

.. code-block:: yaml

   actor:
     tokenizer:
       tokenizer_model: "./DeepSeek-R1-Distill-Qwen-1.5B"
     training_backend: megatron
     megatron:
       mbridge: True
       use_hf_ckpt: True
       ckpt_convertor:
         hf_model_path: ./DeepSeek-R1-Distill-Qwen-1.5B

   data:
     train_data_paths: ["./dataset/boba/AReaL-boba-106k.jsonl"]
     val_data_paths: ["./dataset/boba/AReaL-boba-106k.jsonl"]

SFT 示例：

.. code-block:: yaml

   actor:
     model:
       model_type: "qwen2.5_vl"
       model_path: "./Qwen2.5-VL-3B-Instruct"
       megatron_checkpoint: null

     tokenizer:
       tokenizer_model: "./Qwen2.5-VL-3B-Instruct"

     megatron:
       use_hf_ckpt: True
       mbridge: True

   data:
     train_data_paths: "./Robo2VLM-1/data"
     eval_data_paths: "./Robo2VLM-1/eval_data"

Checkpoint 加载模式
~~~~~~~~~~~~~~~~~~~

在 RLinf 中启用 Megatron-Bridge 后，每个保存节点都会同时保存 HF checkpoint 和 Megatron checkpoint，目录结构如下：

.. code-block:: text

   ./logs/qwen2.5-1.5b-grpo-megatron/checkpoints/
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
       └── ...

``hf_model`` 目录保存 HF 格式的模型权重和 tokenizer 文件，``iter_XXXXXXX`` 目录保存 Megatron 模型权重与优化器状态，``latest_checkpointed_iteration.txt`` 记录最新保存的步数信息。

断点续训时只需通过 ``runner.resume_dir`` 加载 Megatron checkpoint，无需加载 HF checkpoint：

.. code-block:: yaml

   runner:
     resume_dir: ./logs/qwen2.5-1.5b-grpo-megatron/checkpoints/global_step_10

版本说明与常见问题
------------------

版本与模型注意事项
~~~~~~~~~~~~~~~~~~

- RLinf 锁定 ``megatron-bridge`` 0.5.0、``nvidia-modelopt`` 0.45.0 与 Megatron-LM ``core_r0.18.0``，并要求 Python 3.12。
- 启用 ``use_hf_ckpt: True`` 时，保持 ``actor.model.megatron_checkpoint: null``。
- 只有在加载已准备好的 Megatron checkpoint 时，才设置 ``actor.megatron.use_hf_ckpt: False``。
- Qwen3-VL 模型需要保持 ``actor.model.apply_rope_fusion: False``。
- Qwen2.5 模型会自动强制开启 ``qkv_bias`` 以保持兼容。
- Qwen3 模型会自动强制开启 ``qk_layernorm`` 以保持兼容。
- 确保 ``actor.tokenizer.tokenizer_model`` 与 HuggingFace 模型目录一致。

常见问题
~~~~~~~~

``model.megatron_checkpoint is required if use_hf_ckpt is False``
  已关闭 ``use_hf_ckpt``，但未提供 Megatron checkpoint 路径。请设置 ``actor.megatron.use_hf_ckpt: True``，或提供 ``runner.resume_dir`` 路径。

``model.megatron_checkpoint should be None if use_hf_ckpt is True``
  HuggingFace 加载与 Megatron checkpoint 加载被同时启用。请设置 ``actor.model.megatron_checkpoint: null``。

Qwen3-VL 报 ``deepstack_visual_indexes`` 相关断言
  模型的 visual deepstack 配置与当前 pipeline 切分不匹配。可以先尝试 ``pipeline_model_parallel_size: 1``；如果必须开启 pipeline parallel，需要确保第一段 language pipeline stage 的层数能够容纳全部 ``deepstack_visual_indexes``。使用裁剪层数后的 checkpoint 时，还需确认 visual deepstack 配置与语言模型层数一致。
