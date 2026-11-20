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

.. list-table:: 支持范围
   :header-rows: 1

   * - 组件
     - 已覆盖的配置
   * - 采样
     - Qwen3.5-4B、SGLang 0.5.12.post1，输入为 token ID 和原始图片。在默认 rollout worker 启动前注册外部处理器。
   * - 训练
     - Megatron Bridge BF16 Actor（普通或分块输出层）和标量 Critic。要求 PP=1、CP=1。八卡更新采用 TP=2、DP=4。
   * - 媒体
     - 按对话顺序传入图片，回放像素、token 区间、网格和位置。训练侧仍执行视觉塔前向计算。

单卡验证两轮看图
~~~~~~~~~~~~~~~~

激活 agentic 环境，在仓库根目录运行以下命令。把卡号换成一张空闲的 A100 80 GB，模型路径换成本机的 Qwen3.5-4B 目录。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python examples/agent/qwen35/replay_images.py --model /path/to/Qwen3.5-4B

脚本启动本地 HTTP 服务，生成两张测试图片，逐轮核对采样历史、BF16 像素、图像网格和 M-RoPE 位置。出现 ``Turn 2 passed`` 表示两轮回放通过，随后服务会退出。模型训练需另行验证。

复现一次优化器更新
~~~~~~~~~~~~~~~~~~

在一台有 8 张空闲 A100 80 GB、没有运行 Ray 集群的机器上，从仓库根目录执行以下命令。脚本采集 4 组两轮对话，再将同一批 8 行样本依次交给普通输出层 Actor、分块输出层 Actor 和标量 Critic。每个角色输出各 rank 的 loss、梯度范数和发生变化的参数元素数。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/agent/qwen35/run_replay_update.sh /path/to/Qwen3.5-4B

若只使用两张卡，将上例的卡号换成两张空闲卡，例如 ``CUDA_VISIBLE_DEVICES=4,5``。这个配置使用 TP=2、DP=1。

此处用固定优势值和回报检验训练链路。策略效果需要单独评估。

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

开启 rollout logprobs 后，按上例将每轮输出保存为一行训练样本。纯文本行保留 ``response_images=None``。训练端直接读取采样时保存的像素、网格和位置。共享目录为每个不同的像素张量保存一份文件，训练样本或 checkpoint 引用期间需保留。省略目录时，轨迹直接携带张量。后续轮次中，SGLang 会再次处理历史图片并与上轮记录核对。

.. list-table:: 训练样本字段
   :header-rows: 1

   * - 字段
     - 类型与含义
   * - ``format``
     - 字符串，固定为 rlinf-qwen-image-inputs-v1。
   * - ``prompt_token_ids``
     - 整数列表，记录采样本轮回复时使用的完整 prompt，图片占位符已展开。
   * - ``image_spans``
     - 每张真实图片的半开区间 [start, end)，不含两端的视觉分隔符。
   * - ``image_grid_thw``
     - 每张图片对应一行，三列依次记录时间、高度和宽度方向的网格数。
   * - ``mrope_positions``
     - 形状为 [3, prompt 长度] 的整数张量，覆盖整个 prompt。
   * - ``mrope_position_delta``
     - 形状为 [1] 的整数张量，记录生成 token 所用的位置偏移。
   * - ``pixel_values``
     - 内联的原生处理器像素张量，训练侧读取时转为 BF16。
   * - ``pixel_refs``, ``pixel_values_bytes``
     - 启用共享存储时，每张图以路径和 SHA-256 引用代替内联像素。字节数用于预估训练侧 BF16 张量大小。

检查输入一致性
--------------

加入第二张图后，检查旧图的区间、像素和第一轮的采样 token 是否保留。发布版会为每轮记录实际使用的 prompt ID 和 M-RoPE 位置。Megatron 按记录的真实图片区间计算视觉位置。模型生成的图片标记保留为普通文本 token。Megatron 在语言模型入口校验位置，并用保存的像素执行视觉编码器前向计算。要比较最终 logprob，需在相同权重和 token 上单独测量。

在 8 张 A100 80 GB 上，使用 PyTorch 2.11.0+cu128、SGLang 0.5.12.post1、Transformers 5.6.1 和 Bridge 0.5.0，4 组、共 8 轮带图对话均通过输入回放校验。用同一批八行样本，在 TP2、DP4 下重新分配样本后，普通输出层 Actor、分块输出层 Actor 和标量 Critic 各完成了一次优化器更新。另用冻结权重对照两端的 96 个 token，logprob 绝对差均值为 0.058638，最大值为 0.763876，其中 17 个超过 0.10。这组数据记录了输入回放和后端打分误差，策略效果需独立评估。

其余训练选项见 :doc:`../agentic_config`。
