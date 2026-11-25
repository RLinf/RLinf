Configure RPent Multimodal Training
===================================

Use these options after completing :doc:`the single-node smoke <../../examples/agentic/rpent_episode_sync>`.

Replay Qwen3.5 Image Inputs
---------------------------

Use SGLang 0.5.12.post1 with Torch 2.11.0 and Transformers 5.6.1 from the paired training environment. The rollout worker captures the actual prompt IDs, processed pixels, image spans, grid and M-RoPE for each image request. The Megatron actor and critic read those pixels and run their own vision encoders. Static images with PP1/CP1 are supported.

The train and eval templates already register the RLinf image processor. To opt in, set these values and point ``tools.rpent.output_dir`` to the same absolute shared path on rollout and learner nodes:

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

Keep the referenced files while pending episodes or checkpoints use them. The release path requires ``agentloop.thinking_token_budget=null`` and the native TokenSpace preprocess cache disabled. Sampled media markers retain their text token IDs. Literal vision markers in newly templated messages and tool schemas are still rejected.

To resume an older native TokenSpace run, keep its pinned SGLang build from ``requirements/agentic/sglang-token-space.txt``, remove the external processor from the config and rollout environment, and preserve the old cache and image files. Start with fresh rollout state when switching between native TokenSpace and release replay. CPU-offloaded optimizer recovery has separate Megatron requirements of the bundled training environment.

Reduce Repeated Image Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This mode can share preprocessed images between rollout, actor and critic. The cache contains processed pixels and grid metadata. Each backend still runs its own frozen vision encoder; visual embeddings are not shared.

Shared preprocessing caches and CPU prefetching are disabled by default. The rollout cache options below apply only to native TokenSpace. The learner prefetch options also work with release replay. Replace ``<cache-dir>`` with the absolute path to a directory reserved for this job. Every process in the job must be able to access the directory at that path.

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

Actor and critic budgets are independent. If neither a learner field nor its legacy setting is present, the value defaults to zero, disabling the corresponding cache or prefetch feature. The legacy fields under ``tools.rpent`` (``training_image_cache_bytes``, ``training_image_prefetch_batches`` and ``training_image_prefetch_bytes``) still apply to both learners. If a legacy setting and its learner setting are both present, their values must match; conflicting values fail validation. Standalone evaluation needs no actor or critic configuration. The example template uses the learner fields. To override a legacy field from the command line, use Hydra’s ``++`` syntax, for example ``++tools.rpent.training_image_cache_bytes=268435456``.

Cached image paths must always refer to the same image contents. The cache stores preprocessed outputs in a separate directory for each processor fingerprint. Set the check rate between 0 and 1 to recompute a sample of cache hits and verify their contents. Keep this setting identical across processes.

The prefetch settings limit the number and total bytes of prepared microbatches per learner rank, including the current microbatch. One CPU thread prepares the following batch while the training thread runs the forward pass for the current microbatch. The 256 MiB example excludes processor scratch space and the separate ``multimodal.cache_bytes`` cache. Multiply the per-rank budgets by the number of actor and critic ranks, and measure peak host memory before increasing them. A microbatch larger than the budget fails with an error. Prefetching requires token-space trajectories that record the size of the preprocessed pixel data in bytes. Set ``multimodal.prefetch_batches: 0`` when replaying older trajectories.

.. warning::

   Only trusted processes may write the cache directory: cache reads deserialize Python objects. The native disk cache has no eviction or disk-size limit. Use a job-owned directory with a filesystem quota or external capacity monitoring. Remove it only after all job processes have exited; image files must remain available for checkpoint recovery. A new process rebuilds its CPU prefetch queue from the saved image references.

CPU tests cover cache reuse, image integrity, prefetch ordering and thread cleanup. Validate GPU logprobs, training gradients, throughput and training recovery in a new process in the deployment environment. Compare each optimization against the same backend with the option disabled; report the separate 96-point cross-backend diagnostic alongside those results.

Optional Runtime Settings
-------------------------

Training and evaluation set ``rollout.sglang.disable_overlap_schedule=true`` so presence penalties and optional thinking limits read the current CPU token history. Validate sampling logprobs and forced-token alignment on the deployed SGLang version before changing it.


Set ``agentloop.chat_template_kwargs.enable_thinking=false`` to disable thinking.

To cap the first thinking span already opened by the chat template, set ``agentloop.thinking_token_budget=512`` and ``rollout.sglang.enable_custom_logit_processor=true``, keeping ``enable_thinking=true`` and ``rollout.sglang.disable_overlap_schedule=true``. Configuration validation rejects a budget with overlap enabled because its CPU token history can lag decoding. The default budget is ``null``. The engine records which closing token it forces; that token has zero training mask, while a naturally generated closing token remains trainable. A later reopened thinking span is not capped by this option. Keep the returned IDs, behavior logprobs, forced-token flags and weight versions together through trajectory assembly.

Set ``actor.model.lm_head_chunk_size=256`` to calculate the output layer in blocks for response tokens that contribute to the loss. The default ``0`` disables chunking. Results are recomputed during the backward pass. It supports MBridge Qwen3.5 with PP1/CP1, including TP and sequence parallel. Use the Megatron version listed on the setup page. Entropy is a detached diagnostic only, so an entropy loss bonus is unsupported. FP8, MTP, deferred embedding gradients, muP and autocast are also rejected for this mode. These are support constraints, not evidence of higher throughput or learning gains.

Training uses the SGLang Server backend by default; set ``rollout.sglang.backend_type=engine`` for the in-process backend. Evaluation uses Engine by default. Both backends preserve images, thinking-budget metadata and policy versions. The server binds to loopback and runs as a child process.

Tool results and images remain in subsequent prompts. Only policy-generated assistant tokens receive policy loss; user/tool/history tokens do not. Prefix-compatible turns merge, while context compression/history rewriting starts a new row. The critic and actor use the same response-position mapping. Shared trajectory packing keeps rows separate if even one scored token overlaps: merging would overwrite a learning target. This applies to text-only SearchR1 as well as RPent; packing regressions compare both losses and gradients. Response-only logits require CP1.

RPent copies simulator images to the planner over RPC. Keep
``tools.rpent.output_dir`` at an absolute path that all RLinf rollout and
learner workers can read while a trajectory or checkpoint references it.
Simulator hosts can use their own storage. Set
``tools.rpent.store_images_on_disk=false`` if the RLinf workers have no shared
storage. In that configuration, trajectories carry the image bytes.

Preserve Image-Turn Token Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For merged Qwen3.5 image turns, enable the token-preserving processor on each model node **before starting Ray**:

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

The adaptive train/eval templates also pass this variable through ``rollout.env_vars``, so worker launch does not depend solely on the terminal environment. This uses SGLang's external processor registry. Image requests carry already expanded prompt IDs and raw images. The adapter validates the number of images and each image's pad span against its processed grid, retains the exact text/action IDs, and lets SGLang derive offsets and M-RoPE from that sequence. Decoding and re-encoding sampled tokens can otherwise change their boundaries, even when the decoded text is identical. String requests retain the upstream behavior. Token-ID requests with video, audio or precomputed image inputs are rejected by this image-only adapter.

Run the processor regressions with a local model directory containing its config, tokenizer and processor files:

.. code-block:: bash

   SGLANG_USE_CUDA_IPC_TRANSPORT=0 \
     RLINF_TEST_QWEN_VL_MODEL=/path/to/Qwen3.5-4B \
     PYTHONPATH=.:tests/unit_tests \
     python -m pytest tests/unit_tests/test_sglang.py -k qwen_image_processor

These CPU checks cover input IDs, image tensors, offsets and HF/SGLang M-RoPE agreement. They do not replace real-engine generation/logprob checks or a complete learner update and checkpoint recovery before deploying a new processor version.
