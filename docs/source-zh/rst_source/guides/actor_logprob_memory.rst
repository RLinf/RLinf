降低 Actor 计算 logprob 的显存占用
==================================================

Megatron Actor 使用动态批次时，输出层投影（LM Head）和 softmax 的显存开销都随序列长度和词表大小增长。以下三项默认关闭，完整的 Actor 参数表见 :doc:`agentic_config`。

- 提示词或工具返回占大多数 token：设置 ``actor.response_only_logits: true``，只投影预测回复 token 的位置。上下文并行度须为 1。
- 显存峰值出现在 FP32 softmax：设置 ``actor.logprob_chunk_size: 256``，并设 ``algorithm.calculate_entropy: false`` 和 ``algorithm.entropy_bonus: 0``。
- 显存峰值出现在 Qwen3.5 MBridge 的输出层投影：设置 ``actor.model.lm_head_chunk_size: 256``。流水线和上下文并行度均须为 1，并设置 ``actor.model.load_mtp: false`` 和 ``algorithm.entropy_bonus: 0``。

输出层分块优先于另外两项，默认只投影预测回复的位置。记录熵时会投影全部位置，请将 ``actor.logprob_chunk_size`` 保持为 0。使用 Qwen3.5 图像输入时，按 :doc:`图像回放指南 <agent-workflows/qwen35-multimodal>` 配置。公开权重的加载方式见 :doc:`qwen35_weight_sync`。

**试跑 Search-R1**。按 :doc:`Qwen2.5 指南 <../examples/agentic/searchr1/qwen2.5>` 准备权重、数据和检索服务。单机八卡配置 ``train_qwen2.5_chunked.yaml`` 沿用基线配置，启用以下两项：

.. literalinclude:: ../../../../examples/agent/searchr1/config/train_qwen2.5_chunked.yaml
   :language: yaml
   :start-at: actor:

在仓库根目录运行 ``bash examples/agent/searchr1/run_train.sh train_qwen2.5_chunked``。跑对照组时，把最后一个参数换成 ``train_qwen2.5``。

**测量输出层的峰值显存**。在 CUDA 环境中用合成样本跑一次 BF16 前向和反向传播::

   python -m toolkits.bench_lm_head --tokens 4096 --chunk-size 256

.. csv-table::
   :header: "测试配置", "常规投影峰值", "分块投影峰值"

   "单张 A100 80 GB，4,096 个 token", "1.586 GiB", "0.283 GiB"

短序列开启分块可能更占显存。请用同一批训练样本对比峰值，整步训练的历史数据见 `#1886 <https://github.com/RLinf/RLinf/pull/1886>`_。
