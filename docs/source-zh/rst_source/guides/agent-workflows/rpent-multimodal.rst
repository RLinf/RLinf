配置 RPent 多模态训练
============================

先完成 :doc:`单机 smoke <../../examples/agentic/rpent_episode_sync>`，再按需要调整重算、缓存和历史处理。

回放 Qwen3.5 图像输入
---------------------------

安装页的 SGLang 0.5.12.post1、Torch 2.11.0 和 Transformers 5.6.1 即可运行共享图像回放。Rollout 保存推理实际使用的 prompt ID、预处理像素、图片区间、网格和 M-RoPE。Megatron 的 Actor 和 Critic 读取这些像素，各自执行视觉塔前向。当前支持静态图片和 PP1/CP1。

训练和评测模板已注册 RLinf 图像处理器。要启用回放，请使用以下配置，并让 rollout 和 learner 节点通过同一绝对路径访问 ``tools.rpent.output_dir``。

.. code-block:: yaml

   rollout:
     env_vars:
       SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE: rlinf.hybrid_engines.sglang.multimodal
     sglang:
       enable_shared_multi_modal_processor: true
   tools:
     rpent:
       store_images_on_disk: true
       keep_episodes: all

有待训练的轨迹或需要恢复的 checkpoint 时，请保留引用的像素文件。发布版要求 ``agentloop.thinking_token_budget=null``，并关闭原生 TokenSpace 预处理缓存。模型生成的视觉标记保留原始 token ID。新套用模板的消息和工具定义仍会拒绝字面视觉标记。

恢复旧的原生 TokenSpace 训练时，沿用 ``requirements/agentic/sglang-token-space.txt`` 固定的 SGLang 构建，从配置和 rollout 环境中移除外部处理器，保留原缓存及图像文件。两种回放方式之间切换时，使用新的 rollout 状态。CPU-offload 优化器恢复所需的 Megatron 版本与补丁见安装页。

减少重复图像预处理
~~~~~~~~~~~~~~~~~~

下列 rollout 缓存参数仅用于固定源码版本的原生 TokenSpace。发布版回放可以单独使用 Actor 和 Critic 的预取参数。

此模式可以让 rollout、actor 和 critic 共用图像预处理缓存。缓存保存处理后的像素和网格信息。rollout 和训练端仍各自运行冻结的视觉编码器，不共享视觉 embedding。

共享预处理缓存和 CPU 预取默认都关闭。启用时，在旧版原生 TokenSpace 配置中加入以下内容。将 ``<cache-dir>`` 替换为本作业专用缓存目录的绝对路径。所有作业进程都必须能用同一路径访问缓存目录。

.. code-block:: yaml

   rollout:
     sglang:
       enable_token_space_preprocess_cache: true
       token_space_preprocess_cache_dir: <cache-dir>
       token_space_preprocess_cache_trust_url: true
       token_space_preprocess_cache_check_rate: 0.0
   actor:
     multimodal:
       cache_bytes: 0
       prefetch_batches: 2
       prefetch_bytes: 268435456
   critic:
     multimodal:
       cache_bytes: 0
       prefetch_batches: 2
       prefetch_bytes: 268435456

actor 和 critic 可以分别设置预算。同一配置项的新旧字段均未填写时，取值默认为 0，对应的缓存或预取功能关闭。``tools.rpent`` 下的旧字段 ``training_image_cache_bytes``、``training_image_prefetch_batches`` 和 ``training_image_prefetch_bytes`` 仍同时作用于两个训练角色。同一配置项的新旧字段同时存在时，取值必须一致，否则配置校验报错。独立评估无需配置 actor 或 critic。 示例模板已改用训练角色下的字段。通过命令行设置旧字段时，使用 Hydra 的 ``++`` 语法，例如 ``++tools.rpent.training_image_cache_bytes=268435456``。

缓存要求图片路径指向的内容保持不变。缓存按处理器指纹分目录保存预处理结果。抽查比例的取值为 0 到 1，提高该值会重算部分命中的内容，检查缓存是否一致。各进程须使用相同设置。

预取配置限制每个训练 rank 的微批次数和总字节数，预取队列和当前在用的微批次都计入上限。训练线程执行前向计算，单个 CPU 线程准备后续批次。示例中的 256 MiB 不包含处理器临时内存，也不包含单独配置的 ``multimodal.cache_bytes`` 缓存。规划主存时，将每个 rank 的预算乘以 actor 和 critic 的 rank 总数。提高预算前，先测量实际内存峰值。单个微批次超出预算会报错。预取要求 TokenSpace 轨迹记录处理后的像素字节数。重放缺少该字段的旧轨迹时，将 ``multimodal.prefetch_batches`` 设为 0。

.. warning::

   缓存读取会反序列化 Python 对象，目录只能由可信进程写入。原生磁盘缓存没有淘汰机制或容量上限，须使用作业专属目录，并配置文件系统配额或外部容量监控。所有作业进程退出后才能清理缓存；checkpoint 恢复依赖的图片文件须保留。新进程从保存的图片引用重建 CPU 预取队列。

CPU 测试覆盖缓存复用、图片校验、预取顺序和线程清理。GPU 上的 logprob、训练梯度和吞吐，以及在新进程中恢复训练的行为，仍需在部署环境验证。每项优化须与同一后端关闭该选项的结果比较；跨后端的 96 点误差单独报告，不能用算法校正后的结果代替。

可选运行设置
------------

训练和评测默认设置 ``rollout.sglang.disable_overlap_schedule=true``，供 presence penalty 和可选 Thinking 限制读取当前 CPU token 历史。修改前需在目标 SGLang 版本验证采样 logprob 与强制 token 对齐。


若要限制 chat template 已开启的首段 Thinking，设置 ``agentloop.thinking_token_budget=512`` 和 ``rollout.sglang.enable_custom_logit_processor=true``，并保持 ``enable_thinking=true`` 和 ``rollout.sglang.disable_overlap_schedule=true``。启用预算时，配置校验会拒绝重叠调度，避免 CPU token 历史落后于实际解码。预算默认是 ``null``。引擎记录实际强制插入的闭合 Token，该 Token 的训练 mask 为零；自然生成的闭合 Token 仍参与训练。此选项不限制后续重新开启的 Thinking 段。组装轨迹时必须保持返回的 Token ID、behavior logprob、强制标记和权重版本的对应关系。

设置 ``actor.model.lm_head_chunk_size=256``，按块计算参与 loss 的回复 token 的输出层结果；默认 ``0`` 不分块。反向传播时会重新计算这些结果，支持 PP1/CP1 的 MBridge Qwen3.5，包括 TP 和 sequence parallel。使用安装页记录的 Megatron 版本。Entropy 只作为 detached 诊断指标，不支持 entropy loss bonus；该模式也不支持 FP8、MTP、延迟 embedding 梯度、muP 和 autocast。这些约束界定受支持的执行方式，不代表已有吞吐或学习收益。

训练默认使用 SGLang Server，设置 ``rollout.sglang.backend_type=engine`` 可切换到进程内后端。评测默认使用 Engine。两个后端都保留图像、Thinking 预算标记和 policy 版本；Server 作为子进程运行，只绑定回环地址。

工具结果和图片保留在后续输入中。Policy loss 只作用于 policy 生成的 assistant token，user/tool/history token 不参与监督。逐 token 前缀兼容的连续 turn 合并，压缩或历史改写后分段。Actor 和 critic 使用相同的 response 位置映射；共享轨迹 packing 在任意一个计分 token 重叠时都会保留独立行，避免合并覆盖学习目标；这也适用于纯文本 SearchR1。回归测试比较 packing 前后的 loss 和梯度。response-only logits 要求 CP1。

RPent 通过 RPC 将仿真图像复制到 planner。轨迹或 checkpoint 引用这些图像期间，所有 RLinf rollout 和 learner worker 都应能读取 ``tools.rpent.output_dir`` 的同一绝对路径。仿真服务可以使用自己的存储。如果 RLinf worker 没有共享存储，将 ``tools.rpent.store_images_on_disk`` 设为 ``false``，让轨迹携带图像字节。

保留多轮图像的 token 边界
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

合并 Qwen3.5 图像 turn 时，在每台模型节点启动 Ray **之前** 启用保留 token 的 processor：

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

adaptive 训练和评测模板也通过 ``rollout.env_vars`` 传入此变量，使 worker 启动不只依赖终端环境。

该配置使用 SGLang 的外部 processor 注册入口。图像请求携带已展开的 prompt IDs 和原始图片；适配器核对图片数量及各图 pad 段与处理后 grid 的对应关系，保留原始文本和动作 IDs，再由 SGLang 根据这一序列计算 offset 和 M-RoPE。对已采样 token 解码后重新编码，即使文本相同，也可能改变 token 边界。字符串请求保持上游行为；该图像适配器会拒绝携带视频、音频或预计算图像输入的 token-ID 请求。

使用包含 config、tokenizer 和 processor 文件的本地模型目录运行回归：

.. code-block:: bash

   SGLANG_USE_CUDA_IPC_TRANSPORT=0 \
     RLINF_TEST_QWEN_VL_MODEL=/path/to/Qwen3.5-4B \
     PYTHONPATH=.:tests/unit_tests \
     python -m pytest tests/unit_tests/test_sglang.py -k qwen_image_processor

这些 CPU 检查覆盖输入 IDs、图像 tensor、offset，以及 HF 与 SGLang 的 M-RoPE 一致性。部署新版本前，还需要验证真实引擎的生成和 logprob、完整 learner 更新及 checkpoint 恢复。
