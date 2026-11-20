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

.. list-table:: Supported configuration
   :header-rows: 1

   * - Component
     - Coverage
   * - Rollout
     - Qwen3.5-4B, SGLang 0.5.12.post1, raw images and token-ID prompts. Register the external processor before the default rollout worker starts.
   * - Learner
     - Megatron Bridge BF16 actor (dense or chunked projection) and scalar critic. Use PP=1 and CP=1. The eight-GPU update uses TP=2 and DP=4.
   * - Media
     - Ordered images. Replay covers pixels, token spans, grids and positions. The learner executes its own vision forward pass.

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

Reproduce the optimizer updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a single node with eight available A100 80 GB GPUs and no active Ray cluster,
run the following command from the repository root. The script captures four
two-turn episodes and sends the same eight rows to a dense actor, a chunked
actor, and a scalar critic. Each role reports its loss, gradient norm and
changed parameter count for every rank.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/agent/qwen35/run_replay_update.sh /path/to/Qwen3.5-4B

The check uses fixed advantages and returns to exercise the optimizer. The
result is a training-path check. Policy quality requires a separate evaluation.

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
history.

.. list-table:: Captured row fields
   :header-rows: 1

   * - Field
     - Type and meaning
   * - ``format``
     - String, fixed to rlinf-qwen-image-inputs-v1.
   * - ``prompt_token_ids``
     - List of integers. Full expanded prompt used to sample this turn.
   * - ``image_spans``
     - One [start, end) pair per real image in the expanded prompt, excluding delimiters.
   * - ``image_grid_thw``
     - Integer tensor [images, 3]. Temporal, height and width grid dimensions in prompt order.
   * - ``mrope_positions``
     - Integer tensor [3, prompt length]. Positions for the complete served prompt.
   * - ``mrope_position_delta``
     - Integer tensor [1]. Offset for positions after the prompt.
   * - ``pixel_values``
     - Inline native processor tensor. The learner casts it to BF16.
   * - ``pixel_refs``, ``pixel_values_bytes``
     - With shared storage, a list of path/SHA-256 pairs replaces inline pixels. The byte count budgets the BF16 learner copy.

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

See :doc:`../agentic_config` for the other agent training options.
