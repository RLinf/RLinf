配置 Qwen3.5 权重同步
==============================

Megatron actor 可以向 SGLang rollout 同步稠密 Qwen3.5 的视觉编码器、merger、语言模型和可选的多 token 预测（MTP）分支。本页说明任务接入时如何配置权重加载与同步，不提供多模态 RL 训练任务。

准备环境
--------

模型配置中应声明 ``Qwen3_5ForConditionalGeneration`` 架构。判断架构应以配置为准，不能只看模型目录名。Actor 和 rollout 都需要读取完整 checkpoint，包括 tokenizer 和 processor 文件。

在仓库根目录安装 agentic SGLang 环境：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

该环境选择 SGLang 0.5.12.post1，并安装 Megatron Bridge 和 FLA。Qwen3.5 的门控 delta 网络使用 FLA 提供的计算内核。权重同步测试在 NVIDIA A100 上使用 BF16、Megatron Bridge 0.5.0、Transformers 5.6.1 和 fla-core 0.6.0。引擎版本的选择方法见 :doc:`version`\ 。

配置模型与 MTP
------------------

将以下设置加入 Megatron/SGLang 任务配置，把三处 ``<checkpoint>`` 都替换为同一份本地模型路径。这段配置只包含模型相关设置，数据、算法、placement 和显存配置沿用任务自身的设置。

.. code-block:: yaml

   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       precision: bf16
       load_mtp: true
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

``actor.model.load_mtp`` 默认为 ``true``\ ：actor 加载 MTP，SGLang 启用 NEXTN 投机解码，并用同一份 checkpoint 构建 draft model。SGLang 内部会将 NEXTN 规范化为 EAGLE。由 ``actor.model.load_mtp`` 设置投机解码算法和 draft checkpoint。支持的 MTP 结构包含一个预测层，并与主模型共享 embedding。

设置 ``actor.model.load_mtp: false`` 后，只加载和同步视觉分支与语言分支。主模型仍然包含视觉编码器和 merger。Qwen3.5 使用 NEXTN 时，保留 ``rollout.sglang.disable_radix_cache: true``\ ，这也是该模型接入时的默认值。

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

上述语言布局对应 ``rollout.tensor_parallel_size: 4``\ 、``rollout.sglang.enable_dp_attention: true`` 和 ``rollout.sglang.dp_size: 2``\ 。请设置 ``rollout.pipeline_parallel_size: 1``\ ：RLinf 的 SGLang 启动逻辑尚未将 rollout PP 传给引擎。Actor 的流水线并行与虚拟流水线并行不受这一启动问题影响，已有全量模型权重同步验证。

验证权重加载
------------

在 agentic 环境中，从仓库根目录运行组件测试：

.. code-block:: bash

   PYTHONPATH=.:tests/unit_tests python -m pytest tests/unit_tests/test_ckpt_convertor.py -q

安装测试所用的依赖后，123 个用例应全部通过。使用其他环境时应检查跳过的用例；缺少 SGLang 或 Megatron 依赖时，即使其余测试通过，也没有验证这些库的权重加载流程。

全量模型验证会用 SGLang 的完整 checkpoint 加载流程生成独立参考值，再逐一比较目标参数的形状、数据类型和原始字节。测试覆盖初始权重、主动修改后的权重、同卡与分离部署、Actor/Rollout 不同 TP、Actor PP/VP、attention DP 和视觉 DP。4B 测试还覆盖共享输出 embedding 和关闭 MTP。这些检查证明权重加载的一致性，不代表生成输出一致、训练收敛或投机解码加速。

仍需单独开展集成测试，验证推理请求运行期间的异步权重更新、上下文并行和多节点同步。接入异步训练时，应沿用任务已有的暂停与恢复流程。
