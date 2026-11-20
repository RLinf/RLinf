Share Qwen3.5 Image Preprocessing
=================================

Use shared preprocessing to replay the same image inputs in SGLang rollout and
Megatron training. This guide assumes a Qwen3.5 agent workflow that sends image
observations and preserves the sampled token IDs across turns.

SGLang 0.5.12.post1
-------------------

Use the agentic environment with PyTorch 2.11.0, SGLang 0.5.12.post1,
Megatron Core 0.18.0 and Transformer Engine 2.17. For the Qwen3.5 gated
delta layers, the learner also needs the agentic stack's ``fla-core==0.6.0``.
This path uses the released SGLang wheel. Set the following in the rollout
worker environment before the server starts:

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

For a fresh learner environment matching the tested Core release, point the
agentic installer at the tagged checkout:

.. code-block:: bash

   git clone --depth 1 --branch core_v0.18.0 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM-0.18.0
   MEGATRON_PATH="$PWD/Megatron-LM-0.18.0" bash requirements/install.sh agentic --sglang 0.5.12.post1 --venv .venv-agentic

Set ``rollout.sglang.enable_shared_multi_modal_processor=true`` and
``actor.expect_multi_modal_inputs=true`` (also set the latter on any critic).
Use pipeline and context parallel sizes of one. Each rollout worker and its
local tokenizer process must share the same temporary directory.

Two-turn smoke test
~~~~~~~~~~~~~~~~~~~

With the agentic environment active and one free A100 80 GB, run this from the
repository root. Set the model path and GPU to match your installation.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python examples/agent/qwen35/replay_images.py --model /path/to/Qwen3.5-4B

The script starts a loopback HTTP server, generates two images locally, checks
both turns against the learner inputs and shuts down its server. Look for
``Turn 2 passed``. The checks cover sampled history, BF16 pixels, image grids
and M-RoPE positions. Model training is a separate step.

Use replay in an agent
~~~~~~~~~~~~~~~~~~~~~~

For each image turn, append new input to the previous served prompt and sampled
output IDs. Supply all raw images in prompt order and their half-open pad spans
(excluding the vision delimiters). Keep the previous expanded spans and use one
pad for each new image. Pass ``None`` on the first call and the returned history on subsequent calls:

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

Enable rollout logprobs and return one training row per turn.
Text-only rows use ``response_images=None``. The learner reads the captured
pixels, grid and positions. The shared directory contains one file per distinct
pixel tensor. Keep it accessible until all pending rows and checkpoints have
expired. Keep image turns as separate learner rows: equal token prefixes can
refer to different images. The learner pads only the current image microbatch
before Bridge embeds the images and packs the decoder sequence. Omit ``image_store_dir`` to carry tensors inline. Old images are
processed again by SGLang on later turns and checked against the recorded
history. This mode does not require the experimental preprocessing cache.

Experimental TokenSpace API
---------------------------

Workflows using the separate ``prepare_multimodal()`` preflight and SGLang's
native preprocessing cache require the pinned build and a CUDA 13 driver.
Install it in a separate environment and unset the external processor variable:

.. code-block:: bash

   unset SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE
   bash requirements/install.sh agentic --sglang token-space --venv .venv-token-space

Configure the shared storage and per-rank budgets on the experimental path:

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

The example gives each learner rank 256 MiB of processed-image cache and a
512 MiB prefetch budget. With two actor and two critic ranks, these budgets sum
to 3 GiB of host memory; model state, raw images and processor scratch space are
additional. Cache and prefetch are disabled when their budgets are zero.

Chunked scoring requires MTP to be disabled explicitly. It supports entropy
metrics with zero entropy bonus. Keep the vision encoder and projection frozen
in both training and rollout.

Check the Input Contract
------------------------

When adding a second image, check that the previous image span, pixels and
sampled IDs remain intact. The release processor records the served prompt IDs
and M-RoPE positions for each turn. Only recorded image spans contribute
visual positions. Sampled image markers retain their token IDs. Megatron verifies
the positions at the language model input and runs its vision encoder on the saved pixels. Compare
final logprobs separately using the same weights and tokens.

On eight A100 80 GB GPUs (PyTorch 2.11.0+cu128, SGLang 0.5.12.post1,
Transformers 5.6.1 and Bridge 0.5.0), four pairs of image turns passed the
replay checks. The same eight rows were redistributed across four data-parallel
replicas with tensor parallel size two. Both dense and chunked actors and a
scalar critic completed an optimizer update. A separate 96-token, frozen-weight
SGLang/Megatron comparison measured a mean absolute logprob difference of
0.058638 and a maximum of 0.763876. Seventeen differences exceeded 0.10.
These measurements describe input replay and backend scoring on this panel.
They do not measure policy improvement.

.. warning::

   Trusted cache URLs must refer to immutable files. Keep image files and the
   cache accessible for as long as pending trajectories or checkpoints use them.
   Shared preprocessing accepts images; video and audio are unsupported.

See :doc:`../agentic_config` for the other agent training options.
