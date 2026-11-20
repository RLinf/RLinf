共享 Qwen3.5 图像预处理
=======================

共享预处理让 SGLang rollout 与 Megatron 训练复用相同的图像输入。本指南适用于已经接入图像观测、并在多轮对话中保留采样 Token ID 的 Qwen3.5 智能体工作流。

使用 SGLang 0.5.12.post1
------------------------

在 agentic 环境中使用 PyTorch 2.11.0、SGLang 0.5.12.post1、Megatron Core 0.18.0 和 Transformer Engine 2.17。Qwen3.5 的 Gated Delta 层还需要训练环境中的 ``fla-core==0.6.0``\ 。使用 SGLang 发布包时，在 rollout worker 的环境中设置以下变量，再启动服务。

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

从零安装训练环境时，可将 Megatron Core 固定到实测标签，再运行 agentic 安装脚本。

.. code-block:: bash

   git clone --depth 1 --branch core_v0.18.0 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM-0.18.0
   MEGATRON_PATH="$PWD/Megatron-LM-0.18.0" bash requirements/install.sh agentic --sglang 0.5.12.post1 --venv .venv-agentic

开启 ``rollout.sglang.enable_shared_multi_modal_processor`` 和 ``actor.expect_multi_modal_inputs``\ 。使用 Critic 时也为它开启图像输入。流水线并行和上下文并行规模均设为 1。每个 rollout worker 和本机的 tokenizer 进程需能访问同一个临时目录。

单卡验证两轮看图
~~~~~~~~~~~~~~~~

激活 agentic 环境，在仓库根目录运行以下命令。把卡号换成一张空闲的 A100 80 GB，模型路径换成本机的 Qwen3.5-4B 目录。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python examples/agent/qwen35/replay_images.py --model /path/to/Qwen3.5-4B

脚本启动本地 HTTP 服务，生成两张测试图片，逐轮核对采样历史、BF16 像素、图像网格和 M-RoPE 位置。出现 ``Turn 2 passed`` 表示两轮回放通过，随后服务会退出。模型训练需另行验证。

接入智能体
~~~~~~~~~~

将新输入的 token ID 追加到上一轮的实际 prompt 和采样输出之后。按图片在 prompt 中出现的顺序传入所有原图及对应区间。区间使用左闭右开的下标，仅包含图像占位 token。旧图沿用返回的展开区间，新图用一个占位 token。第一轮的 ``history`` 设为 ``None``\ ，后续使用上一轮的返回值：

.. code-block:: python

   result = await self.generate(
       prompt_ids, image_data=images, image_spans=spans, image_history=history,
       image_store_dir=shared_directory,
   )
   turn = AgentLoopOutput(
       prompt_ids=result["prompt_token_ids"],
       response_ids=result["output_ids"],
       response_logprobs=result["logprobs"],
       response_images=result["multimodal_input"],
   )
   history = result["image_history"]
   prefix = result["prompt_token_ids"] + result["output_ids"]
   spans = result["multimodal_input"]["image_spans"]

同一段 token 前缀可能对应不同图片。训练时保留各轮图像样本，只在送入 Bridge 前将当前微批补齐，再由 Bridge 嵌入图片、打包解码序列。

开启 rollout logprobs 后，按上例将每轮输出保存为一行训练样本。纯文本行保留 ``response_images=None``。训练端直接读取采样时保存的像素、网格和位置。共享目录为每个不同的像素张量保存一份文件，训练样本或 checkpoint 引用期间需保留。省略目录时，轨迹直接携带张量。后续轮次中，SGLang 会再次处理历史图片并与上轮记录核对。发布版无需开启实验性预处理缓存。

实验版 TokenSpace 接口
----------------------

沿用 ``prepare_multimodal()`` 预处理接口和 SGLang 原生预处理缓存的工作流需要固定源码版本及兼容 CUDA 13 的驱动。请使用独立环境并取消设置上述外部 Processor 环境变量：

.. code-block:: bash

   unset SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE
   bash requirements/install.sh agentic --sglang token-space --venv .venv-token-space

实验版可按下例配置共享缓存和每个训练进程的预取预算。

.. code-block:: yaml

   rollout:
     sglang:
       enable_shared_multi_modal_processor: true
       enable_token_space_preprocess_cache: true
       token_space_preprocess_cache_dir: /absolute/shared/image-cache
       token_space_preprocess_cache_trust_url: true
   actor:
     expect_multi_modal_inputs: true
     model:
       load_mtp: false
       lm_head_chunk_size: 256
     multimodal:
       cache_bytes: 268435456
       prefetch_batches: 2
       prefetch_bytes: 536870912
   critic:
     expect_multi_modal_inputs: true
     multimodal:
       cache_bytes: 268435456
       prefetch_batches: 2
       prefetch_bytes: 536870912

示例为每个 learner rank 分配 256 MiB 图像预处理缓存和 512 MiB 预取预算。两个 actor rank 和两个 critic rank 的缓存与预取预算合计为 3 GiB，模型状态、原图和 Processor 临时空间另计。预算为零时关闭相应功能。

分块评分需要显式关闭 MTP；可以计算熵指标，但熵奖励系数必须为零。训练与 rollout 两端都应冻结视觉编码器和投影层的参数。

检查输入一致性
--------------

加入第二张图后，检查旧图的区间、像素和第一轮的采样 token 是否保留。发布版会为每轮记录实际使用的 prompt ID 和 M-RoPE 位置。Megatron 按记录的真实图片区间计算视觉位置。模型生成的图片标记保留为普通文本 token。Megatron 在语言模型入口校验位置，并用保存的像素执行视觉编码器前向计算。要比较最终 logprob，需在相同权重和 token 上单独测量。

在 8 张 A100 80 GB 上，使用 PyTorch 2.11.0+cu128、SGLang 0.5.12.post1、Transformers 5.6.1 和 Bridge 0.5.0，四组共八轮看图均通过回放检查。用同一批八行样本，在 TP2、DP4 下重新分配样本后，普通输出层 Actor、分块输出层 Actor 和标量 Critic 各完成了一次优化器更新。另用冻结权重对照两端的 96 个 token，logprob 绝对差均值为 0.058638，最大值为 0.763876，其中 17 个超过 0.10。这组数据记录了输入回放和后端打分误差，策略效果需独立评估。

.. warning::

   可信缓存 URL 必须指向内容不变的文件。只要待训练轨迹或 checkpoint 仍引用图片，就应确保图片和缓存可访问。共享预处理支持图片，不支持视频和音频。

其余训练选项见 :doc:`../agentic_config`。
