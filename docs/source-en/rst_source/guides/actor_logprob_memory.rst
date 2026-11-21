Reduce Memory When Computing Logprobs
=====================================

With dynamic Megatron batches, the output projection and softmax can allocate tensors proportional to sequence length × vocabulary size. All three options default to off. See :doc:`agentic_config` for the Actor parameters.

- If prompts or tools supply most tokens, set ``actor.response_only_logits: true`` (context parallel size 1).
- If GPU memory peaks during the FP32 softmax, set ``actor.logprob_chunk_size: 256``, ``algorithm.calculate_entropy: false`` and ``algorithm.entropy_bonus: 0``.
- If GPU memory peaks during the Qwen3.5 MBridge output projection, set ``actor.model.lm_head_chunk_size: 256``. Use pipeline and context parallel sizes of 1, ``actor.model.load_mtp: false`` and ``algorithm.entropy_bonus: 0``.

LM Head chunking takes precedence over the other two options. It projects response positions by default and all positions when logging entropy. To log entropy, leave ``actor.logprob_chunk_size`` at zero.
For Qwen3.5 image inputs, follow :doc:`the image replay guide <agent-workflows/qwen35-multimodal>`. See :doc:`qwen35_weight_sync` for loading public weights.

**Try Search-R1.** Follow the :doc:`Qwen2.5 setup <../examples/agentic/searchr1/qwen2.5>` for weights, data and the retrieval server. On one eight-GPU node, ``train_qwen2.5_chunked.yaml`` inherits the base configuration and adds:

.. literalinclude:: ../../../../examples/agent/searchr1/config/train_qwen2.5_chunked.yaml
   :language: yaml
   :start-at: actor:

From the repository root, run ``bash examples/agent/searchr1/run_train.sh train_qwen2.5_chunked``. For the baseline, replace the last argument with ``train_qwen2.5``.

**Measure output-layer memory.** This CUDA benchmark runs a BF16 forward and backward pass on synthetic inputs::

   python -m toolkits.bench_lm_head --tokens 4096 --chunk-size 256

.. csv-table::
   :header: "Benchmark setup", "Dense peak", "Chunked peak"

   "1 × A100 80 GB, 4,096 tokens", "1.586 GiB", "0.283 GiB"

Short sequences can use more memory with chunking. Compare peak memory on the same training batch. See the `Actor measurements in #1886 <https://github.com/RLinf/RLinf/pull/1886>`_.
