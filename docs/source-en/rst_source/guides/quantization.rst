Quantize an Embodied Policy During Training
===========================================

Use the CALVIN example's default NVFP4 W4A4 recipe for OpenPI pi0.5 QAT, training-time validation, and checkpoint resume, or explicitly select INT8 or disable quantization. The actor retains floating-point parameters while rollout runs native low-bit GEMMs. This workflow uses synchronous FSDP1 and HF rollout.

The actor recomputes fake-quantized weights on every forward, including frozen VLM weights. Rollout packs synchronized weights into low-bit caches and reuses them until the next weight update; activations are quantized dynamically on each forward.

Prepare Dependencies and Assets
-------------------------------

Run these commands from the RLinf checkout containing the QAT configs. Prepare the simulator and SFT checkpoint with the :doc:`CALVIN guide <../examples/embodied/calvin>`. Use either the public image or a source installation. Both paths install the current RLinf checkout rather than relying on an older editable installation.

Use the Prebuilt Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With an NVIDIA driver and Docker available, mount your checkout and model directory into the public CALVIN image. The command below exposes one GPU for the short check. The reference GPU is SM120 with 72 GB; memory use depends on batches and placement. The short check's fixed budget supports 1, 2, 4, or 8 visible GPUs. On an eight-GPU host, use --gpus all; other topologies require changes to environment counts, batches, and placement to satisfy divisibility constraints.

.. code-block:: bash

   docker run -it --name rlinf-qat --gpus device=0 --cpus=64 --shm-size=32g \
     --ulimit memlock=-1 --ulimit stack=67108864 \
     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
     -v "$PWD:/workspace/RLinf" -v "/path/to/model:/workspace/model" \
     -w /workspace/RLinf \
     infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-calvin

   source switch_env openpi
   # Select a public index for subsequent uv commands when needed in China.
   export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
   export HF_ENDPOINT=https://hf-mirror.com
   uv pip install --no-deps -e .
   export QAT_MODEL_PATH=/workspace/model

The tested image index digest is ``sha256:4640269ba481cad1e6f11a9db566d07eece209465685027f5ecc3d2588d1a1d7``. Append @ and that digest to the image name to pin the environment. It provides Torch 2.11.0+cu128, Triton 3.6.0, OpenPI, and CALVIN; add only the selected quantization backend's dependencies below.

Build the Environment from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without the prebuilt environment, use the existing installer on NVIDIA Linux with a CUDA 12.8 toolchain. Current pi0.5 uses Torch operations, so the short check does not require a flash-attn build. The --install-rlinf option also installs your checkout.

.. code-block:: bash

   # Add --use-mirror when needed in China.
   UV_TORCH_BACKEND=cu128 bash requirements/install.sh --torch 2.11.0 \
     --no-flash-attn --install-rlinf embodied --model openpi --env calvin
   source .venv/bin/activate
   export QAT_MODEL_PATH=/path/to/RLinf-Pi05-CALVIN-ABC-D-SFT

Check Inputs and Backend Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After either environment setup, the default NVFP4 recipe also needs FlashInfer. An explicit INT8 selection uses the installed Torch/Triton. Check weights and normalization files before confirming the interpreter and imported checkout.

.. code-block:: bash

   # Use these public mirrors when needed in China.
   export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
   export HF_ENDPOINT=https://hf-mirror.com
   uv pip install 'flashinfer-python==0.7.1'  # NVFP4 only.
   test -s "$QAT_MODEL_PATH/model.safetensors"
   test -s "$QAT_MODEL_PATH/InternRobotics/InternData-Calvin_ABC/norm_stats.json"
   python -c 'import rlinf, torch; print(rlinf.__file__, torch.__version__, torch.version.cuda)'
   export EMBODIED_PATH="$PWD/examples/embodiment"

If the model is not available yet, run ``hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT --revision ffe18938a37d3855c9ff50d6220d39bcd6434f15 --local-dir "$QAT_MODEL_PATH"`` in the activated environment, then check its files. Time environment setup, model download, and first-use JIT separately; predownloaded inputs do not establish a successful cold download.

INT8 requires BF16-capable SM80 or newer. NVFP4 GEMM requires SM100/103/120/121 and compiles its native backend on first use. Backend initialization checks GPU architecture and dependency versions; numerical correctness is covered by the unit tests. Record GPU, driver, dependency, and source versions before training.

Choose a Configuration Group
----------------------------

Quantization rules and behavior correction live in ``examples/embodiment/config/qat/`` and modify only the top-level ``qat`` node. The main example defaults to nvfp4; short checks and resume use the same config. Select ``qat=int8`` or ``qat=off`` to switch. The experiment config independently controls PPO, FSDP, workers, and training budgets; switching quantization groups preserves those settings.

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - qat option
     - Weights
     - Activations
     - Actor forward
   * - off
     - Original model
     - Original model
     - Original model
   * - int8
     - Input block128
     - Input block128
     - Fake quant
   * - nvfp4
     - Input block16
     - Input block16
     - Fake quant

Use ``qat=off`` for a floating-point control with the same experiment settings. For the full official BF16 setup, select the existing calvin_d_d_ppo_openpi_pi05 config. The QAT experiment uses synchronous FSDP1/HF with compile, AMP, and offload disabled. CUDA Graph is opt-in for NVFP4 validation. These are this implementation's operating conditions, not general QAT requirements.

Both recipes use input-axis blocks for weights and activations, dynamic scales, and nearest_even rounding: halfway values round to the nearest even code. INT8 computes each scale in FP32 as max(amax × (1/127), 1e-12). The actor always uses fake quantization, while rollout selects native low-bit GEMM from the format. The runtime derives internal numerical recipe IDs and versions and includes them in the hash used for synchronization checks.

Select Quantization Targets
---------------------------

``qat.targets`` is an inclusion list of projection paths; multiple entries form a union. The following values work with OpenPI pi0.5. Other models use the target paths supplied by their adapters.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - targets value
     - Quantized scope
   * - ["*"]
     - All supported projections: vision, both Gemma experts, and action/time input and output projections.
   * - ["img.*"]
     - Vision projections, including the patch convolution, vision attention, and FFN projections.
   * - ["llm.*"]
     - Projections in both the Gemma language expert and action expert.
   * - ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
     - Only the separate action/time input and output projections, excluding the Gemma action expert.

Matching uses case-sensitive fnmatch patterns. ``*`` can span dots in a path, and ``**`` has no additional recursive meaning. Exact paths are accepted; every pattern must match at least one candidate. To quantize only vision, add ``qat.targets='["img.*"]'`` to the command below. Embeddings, norms, softmax, attention-score matrix multiplications, and the value head are outside the candidate scope, including when using the default wildcard.

Inspect Before Launching
------------------------

Expand the configuration first to check both checkpoint paths, precision, and budget. This command prints configuration without starting a model.

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-path="$EMBODIED_PATH/config" \
     --config-name=calvin_d_d_ppo_openpi_pi05_qat \
     qat=nvfp4 \
     actor.model.model_path="$QAT_MODEL_PATH" \
     rollout.model.model_path="$QAT_MODEL_PATH" \
     --cfg job --resolve

Remove ``--cfg job --resolve`` to launch. The example retains the official D-to-D budget of 64 environments, 8 rollout epochs, and 4 PPO epochs. The model defaults freeze the VLM and train action/value parameters; quantization covers visual, language, and action projections. Embeddings, norms, softmax, and attention-score GEMMs keep their original implementations. NVFP4 global scales span a logical projection matrix, so keep batch composition and placement fixed across comparisons.

Enable CUDA Graph for Validation
--------------------------------

To capture NVFP4 validation inference, append these overrides to the training command. Use the same overrides with ``qat=off`` for a BF16 comparison.

.. code-block:: text

   rollout.enable_cuda_graph=true runner.val_check_interval=10

Validation captures the GPU observation preprocessing, vision/language prefix, and full action denoising loop. Environment input/output transforms remain outside the graph. Actor updates and Pi0RL training rollout sampling keep their existing execution paths. The BF16/NVFP4 comparisons here keep compile and offload disabled for both precisions. Disabling compile remains a current QAT requirement, not a general requirement of the BF16 CUDA Graph interface.

The first validation batch prepares the graph; measure steady-state inference after capture. New observation values and sampling noise are copied into static input buffers on each call. Layout or sampling-step changes require another capture. Weight synchronization updates existing buffers in place, so the graph uses the updated values. Release the graph before moving the model to another device, changing its dtype, or replacing parameters, then capture again after the change. The rollout worker follows this order for offload and reload.

NVFP4 selects CUTLASS GEMM tactics before capturing a new graph. FlashInfer reuses tactics in the worker process and tunes shapes not covered by its cache. Tuning finishes before capture; graph replay reuses the selected tactics. Exclude tuning, warmup, and capture from steady-state inference timing.

.. note::

   Captured graphs retain GPU buffers and increase validation memory use. INT8 validation graphs and actor graphs are outside the supported QAT configuration.

Check One Iteration First
-------------------------

Reuse the main config with explicit CLI overrides for a smaller training budget: 8 environments, 10 steps, one rollout epoch, one PPO epoch, a checkpoint, and short validation. It checks the training data path, not policy accuracy. Adjust placement and batches for your memory budget and retain the resolved configuration.

.. code-block:: bash

   export QAT_OUTPUT="$PWD/results/qat-smoke"
   python examples/embodiment/train_embodied_agent.py \
     --config-path="$EMBODIED_PATH/config" \
     --config-name=calvin_d_d_ppo_openpi_pi05_qat \
     actor.model.model_path="$QAT_MODEL_PATH" rollout.model.model_path="$QAT_MODEL_PATH" \
     runner.logger.log_path="$QAT_OUTPUT" runner.logger.experiment_name=pi05_qat_smoke \
     runner.max_epochs=1 runner.save_interval=1 runner.val_check_interval=1 \
     env.train.total_num_envs=8 env.train.rollout_epoch=1 \
     env.train.max_episode_steps=10 env.train.max_steps_per_rollout_epoch=10 \
     env.eval.total_num_envs=8 env.eval.rollout_epoch=1 \
     env.eval.max_episode_steps=10 env.eval.max_steps_per_rollout_epoch=10 \
     actor.micro_batch_size=1 actor.global_batch_size=8 algorithm.update_epoch=1

The checkpoint is written to ``$QAT_OUTPUT/pi05_qat_smoke/checkpoints/global_step_1/actor``. Default TensorBoard metrics and resolved config.yaml are written to ``$QAT_OUTPUT/tensorboard/``. Action/value gradients should be finite after backward, frozen VLM parameters should remain unchanged, and native rollout caches should refresh after weight synchronization. Check these before restoring the full budget.

Resume and Compare
------------------

Keep the short-check command's configuration, topology, and batches fixed. Change runner.max_epochs from 1 to 2 and add the following override to continue from iteration 1. Quantized checkpoint packing is automatic.

.. code-block:: text

   runner.resume_dir="$QAT_OUTPUT/pi05_qat_smoke/checkpoints/global_step_1"

Standard checkpoints save actor parameters, optimizer, and scheduler state. Derived quantization caches are rebuilt on resume. Environment and rollout processes restart, so resume does not promise a bitwise-continuous simulator trajectory.

The example uses the patch weight syncer with init_sync enabled, which synchronizes all actor parameters at startup before sending incremental updates. If you select the bucket syncer, actor and rollout must start with identical frozen parameters: only trainable parameters and persistent buffers are synchronized. When resuming a checkpoint whose frozen VLM differs from the rollout checkpoint, use patch init_sync or load matching frozen weights into rollout.

For BF16/PTQ/QAT comparisons, fix the SFT checkpoint, normalization, task sequences, initial states, noise, batches, step limits, and training budget. Include an ordinary-RL-then-PTQ control. Training-time validation uses the current policy; QAT presets reject standalone only_eval. Historical 1000-sequence evaluations and 200-iteration results used the original evaluation harness and are not results reproduced by this new entry point. Validate the smoke before running a fixed scientific protocol.

Configure Behavior Correction
-----------------------------

Rollout retains original behavior log-probs. Before PPO updates, the actor scores each batch once and freezes its old-policy anchor. ``qat.correction.mode=importance_sampling`` weights PPO contributions by exp(old-actor log-prob minus rollout log-prob), capped by ``qat.correction.max_weight`` (2 by default). Set ``qat.correction.mode=off`` to use a behavior weight of 1 while retaining quantized actor and rollout execution.

The experiment's algorithm.recompute_logprobs=false disables the older path that overwrites behavior scores after recomputing them. QAT still recomputes actor scores, saving them separately as proximal_logprobs for the PPO anchor and behavior correction.

OpenPI scores a selected Gaussian transition, not the full final-action policy density.

INT8 uses [-128,127], nearest-even rounding, and a 1e-12 scale floor. NVFP4 uses E2M1 values, E4M3FN block scales, and an FP32 global scale. Fake and native GEMM accumulation can differ; correction does not make their outputs identical.

Extend to Another Model
-----------------------

Follow :doc:`the quantization model extension guide <../extending/quantization>` to register another model once. OpenPI pi0.5 is implemented; custom policies must also provide replayable sampling and scoring events.

Current QAT gates reject named FSDP precision rules, standalone only_eval, async/RTC, LoRA, compile, actor CUDA Graph, and trainable sigma. QAT also rejects ``openpi.short_memory=true`` and ``openpi.pytorch_numerics=true``; both options default to false, so the default CALVIN configuration is unaffected. NVFP4 supports CUDA Graph for periodic validation with a positive runner.val_check_interval. Recipes define RLinf arithmetic, not ApxInf or ModelOpt checkpoint compatibility. Keep resolved config, Git SHA, dependencies, checkpoint/normalization fingerprints, and raw results to distinguish environment, training, and quantization effects.
