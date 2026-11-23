配置 Qwen3.5 权重同步
==============================

Megatron actor 可以向 SGLang rollout 同步稠密 Qwen3.5 的视觉编码器、merger、语言模型和可选的多 token 预测（MTP）分支。在已有任务中加入以下配置，即可加载并同步这些权重。本页不提供多模态 RL 训练任务。

准备环境
--------

模型配置中应声明 ``Qwen3_5ForConditionalGeneration`` 架构。判断架构应以配置为准，不能只看模型目录名。Actor 和 rollout 都需要读取完整 checkpoint，包括 tokenizer 和 processor 文件。

在仓库根目录安装 agentic SGLang 环境：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

该环境选择 SGLang 0.5.12.post1，并安装 Megatron Bridge 和 FLA。Qwen3.5 的门控 delta 网络使用 FLA 提供的计算内核。权重同步测试使用 NVIDIA A100 GPU，精度为 BF16，依赖版本为 Megatron Bridge 0.5.0、Transformers 5.6.1 和 fla-core 0.6.0。引擎版本的选择方法见 :doc:`version`\ 。

配置模型与 MTP
------------------

将以下设置加入 Megatron/SGLang 任务配置，把三处 ``<checkpoint>`` 都替换为同一份本地模型路径。这段配置只包含模型相关设置，数据、算法、placement 和显存配置沿用任务自身的设置。

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       precision: bf16
       load_mtp: false
       mtp_loss_scaling_factor: 0.1
     tokenizer:
       tokenizer_model: <checkpoint>
     megatron:
       mbridge: true
       use_hf_ckpt: true
       ckpt_convertor:
         hf_model_path: <checkpoint>
   rollout:
     rollout_backend: sglang
     model:
       model_type: qwen3_5
       model_path: <checkpoint>
       precision: bf16
     sglang:
       mm_enable_dp_encoder: true
       disable_radix_cache: true

``actor.model.load_mtp`` 默认为 ``false``\ 。此时只加载和同步语言模型、视觉编码器及 merger，不构建 MTP 层，也不启用投机解码。

设置 ``actor.model.load_mtp: true`` 后，actor 训练并同步 MTP 分支，SGLang 启用 NEXTN 投机解码。Draft 模型与 rollout 模型使用同一份 checkpoint。Megatron/SGLang 集成不支持外部 draft checkpoint。显式设置的 NEXTN、EAGLE 和解码参数优先于默认值，SGLang 将 NEXTN 按 EAGLE 处理。支持的 MTP 结构只有一个预测层，与主模型共享 embedding。

开启 MTP 后，RL actor 更新时会对回答 token 计算多 token 预测辅助 loss。用 ``actor.model.mtp_loss_scaling_factor`` 设置辅助 loss 的权重，默认值为 ``0.1``\ 。MTP 梯度也会更新语言模型主干。训练指标 ``actor/mtp_loss`` 记录未乘权重系数的辅助 loss。重算 logprob 和独立推理不会加入这项 loss。

开启 actor MTP 后，\ ``rollout.sglang.disable_radix_cache`` 的默认值取决于 Mamba 调度模式：\ ``no_buffer`` 对应 ``true``\ ，\ ``extra_buffer`` 对应 ``false``\ 。显式设置的值优先，由 SGLang 检查是否与调度模式兼容。关闭 actor MTP 后，这一配置沿用 SGLang 默认值，也可以显式设置。

如果主模型和 draft 的权重均由 actor 提供，且未开启 checkpoint 对照校验，SGLang 就用 dummy 权重初始化。SGLang 在 rollout 前接收 actor 权重。开启 ``rollout.validate_weight_first_sync`` 后，SGLang 先加载 checkpoint，首次同步后同时校验主模型和已启用的 MTP draft。内置校验比较带位置信息的范数并允许一定误差；逐 bit 验证需要另行比较参数字节。恢复训练会关闭首次同步的 checkpoint 对照，因为 actor 恢复的权重可能与初始 checkpoint 不同。

选择视觉分支布局
----------------

设置 ``rollout.sglang.mm_enable_dp_encoder: true`` 后，每个 rollout rank 保存完整的视觉编码器副本。设为 ``false`` 则对视觉编码器使用张量并行，默认值是 ``false``\ 。权重同步分别按语言注意力层、前馈层和视觉编码器的目标布局处理参数。

例如，rollout 引擎 TP 为 4、attention DP 为 2 时，attention TP 为 2。开启视觉 DP 后，每份视觉副本的 TP 为 1，语言前馈层仍使用引擎 TP 4。Actor TP 可以与这些 rollout 布局不同。

.. list-table:: 目标布局示例
   :header-rows: 1

   * - 组件
     - 张量并行大小
   * - 语言注意力层
     - 2
   * - 语言前馈层
     - 4
   * - 开启视觉 DP 的视觉编码器
     - 1

上述语言布局对应 ``rollout.tensor_parallel_size: 4``\ 、``rollout.sglang.enable_dp_attention: true`` 和 ``rollout.sglang.dp_size: 2``\ 。请设置 ``rollout.pipeline_parallel_size: 1``\ ：RLinf 的 SGLang 启动逻辑尚未将 rollout PP 传给引擎。这一限制不影响 actor 的流水线并行和虚拟流水线并行，两者均已通过完整模型的权重同步测试。

验证权重加载
------------

激活 agentic 环境后，在仓库根目录运行以下组件测试。保留环境已有的 Python 路径：

.. code-block:: bash

   export PYTHONPATH="$PWD:$PWD/tests/unit_tests:$PYTHONPATH"
   CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit_tests/test_ckpt_convertor.py -q
   CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit_tests/test_megatron.py -q

依赖完整时，测试应全部通过且没有跳过的用例。缺少 SGLang 或 Megatron 会跳过相关测试，应先补齐依赖再验证。这些 CPU 测试不能证明全量模型在 GPU 上的训练收敛性或投机解码性能。
