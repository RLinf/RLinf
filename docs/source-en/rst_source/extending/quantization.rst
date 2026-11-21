Add Quantization to a Model
===========================

Extend a numerical quantization format or connect a model to the shared QAT runtime. Backends implement numerical operations; model adapters describe projection targets and sampling constraints. OpenPI pi0.5 provides a concrete model reference.

Keep the Existing Policy Interface
----------------------------------

First follow :doc:`the FSDP model guide <new_model_fsdp>` so the floating-point policy trains normally. Reuse its model_type for quantization without changing workers, runners, or checkpoints. Standard Linear projections can reuse FakeQuantLinear and QuantizedLinear; handle functional projections explicitly in the model adapter. Actor and rollout projections receive a QuantizationBackend object. The factory selects the configured format once; model forwards call backend methods without inspecting the format. The example configuration still defaults to NVFP4.

Implement a Numerical Backend
------------------------------

Inherit QuantizationBackend from ``rlinf/quantization/backends/base.py``. Implement quantize, dequantize, gemm, and probe; the latter certifies actual hardware and numerics. Register the format in create_quantization_backend and add its supported configuration recipe. Model adapters need no format allowlist or new branches.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Contract
   * - ``quantize(values, role="weight", native=False)``
     - Return QuantizedTensor with data, scales, logical_shape, format, storage, and optional global_scale/padded_shape. role identifies activations or weights.
   * - ``dequantize(value)``
     - Return logical FP32 values, excluding padding. logical/native storage identifies unpacked codes versus native packed layouts explicitly.
   * - ``gemm(activation, weight, bias=None, output_dtype=...)``
     - Compute with compatible native low-bit matrices on supported hardware, preserving the recipe's output rounding and bias order.
   * - ``autotune()``
     - Return an eager tuning context, with no extra work by default. NVFP4 selects CUTLASS tactics through FlashInfer. Exit this context before graph capture.
   * - ``fake_quantize(values, role="weight")``
     - Apply QDQ and STE. Override when the format needs another STE forward or a fused numerical path.
   * - ``fake_linear(input, weight, bias=None, weight_dtype=None)``
     - Run the actor's fake-quantized projection. weight_dtype preserves a model's optional floating parameter cast. Fixed-precision recipes define their own compute boundary.

The public quantize and dequantize methods dispatch logical and native layouts to separate private methods. Each format shares its numerical formula between those methods. Both layouts use QuantizedTensor: native NVFP4 decoding separates nibble unpacking and scale-layout restoration before the shared dequantization. Native GEMM consumes this same container directly.

The kernels directory retains only execution paths used by actor or rollout. Logical codes use the PyTorch reference on CPU or CUDA; there is no separate codes-output kernel. Native GEMM is inference-only; actor gradients use the fake-quantized path.

NVFP4 overrides both fake methods: it retains fused BF16 QDQ, Passthrough STE, and BF16 GEMM followed by FP32 bias and the input dtype. Gate and up weights retain separate global scales. Separate numerical operations from parameter ownership so a new format can reuse these four components:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Responsibility
   * - QuantizationBackend
     - Implement quantization, dequantization, native GEMM, and fake forwards; hold the numerical recipe.
   * - QuantizedWeight
     - Own derived low-bit data, scales, global_scale, and shapes, without floating Parameters or forward computation.
   * - QuantizedLinear
     - Preserve floating Parameters and checkpoint keys, own QuantizedWeight, and organize dynamic activation quantization, native GEMM, and output reshaping.
   * - FakeQuantLinear
     - Preserve floating Parameters and call backend.fake_linear on each forward, passing gradients to master weights through STE.

QuantizedTensor is the value returned by one numerical operation. QuantizedWeight is the Module that retains those derived buffers. Call QuantizedWeight.load_weights(master_weight) to refresh them from current floating weights; value constructs a QuantizedTensor view from current buffers. clear() makes that value unavailable while retaining allocations for the next refresh. Device/dtype transforms and checkpoint loads also withdraw the value until the next synchronization rebuilds it.

For a standard Linear, call QuantizedLinear.load_weights() to refresh derived weights, then compute through forward. Functional FFN, QKV, and patch adapters hold QuantizedWeight directly and reuse quantized_linear in ``rlinf/quantization/layers/linear.py`` to organize computation.

Pi0 selects eager sampling or graph replay and manages the lifecycle through capture_cuda_graph() and release_cuda_graph(). Only when needs_capture() requests a new graph does Pi0 clear the old graph, prepare the policy, and call capture(). For quantized policies, this preparation runs _sample_actions_eager() with the actual observations and explicit noise inside runtime.backend.autotune(). Each eligible call uses replay(). EvalCudaGraph manages static inputs, capture, and replay; Pi0 owns quantization preparation.

Before moving the model, changing its dtype, or replacing parameters with load_state_dict(assign=True), call release_cuda_graph(); call capture_cuda_graph() after the change. The rollout worker follows this order for offload and reload.

FakeQuantLinear and QuantizedLinear share initialization and from_linear conversion through BaseQuantizedLinear, preserving the original Parameters, checkpoint keys, training state, and FSDP wrapping markers. Each subclass implements its own forward computation.

``kernels/int8.py`` and ``kernels/nvfp4.py`` expose GPU entry points; ``kernels/int8_triton.py`` and ``kernels/nvfp4_triton.py`` implement the corresponding Triton kernels. Entry points import Triton implementations on demand, keeping configuration validation lightweight.

The directory responsibilities are:

.. code-block:: text

   quantization/
     backends/       # Abstract interface, format implementations, compatibility checks
     kernels/        # CUDA/FlashInfer wrappers and Triton kernels
     layers/         # Shared native/fake Linear and derived weight buffers
     algorithms/     # Sampling events, frozen anchors, PPO correction
     models/openpi/  # Model targets, parameter views, and projection replacements

Register Both Responsibilities Once
-----------------------------------

Config validation runs in the CPU driver. Validation callbacks and registration do not load models or inference backends; the RLinf base package itself depends on PyTorch. A lazy factory constructs execution callbacks inside each worker. Put this registration in an extension module such as my_policy_qat. Call its register() explicitly in the driver before validate_cfg, and set ``RLINF_EXT_MODULE=my_policy_qat`` so each worker loads the same registration. The standard entry point does not load that environment variable before config validation.

.. code-block:: python

   from functools import cache

   from rlinf.quantization.registry import register_quantization_model
   from my_policy_qat.config import validate_model_config

   def build_adapter():
       from rlinf.quantization.registry import QuantizationAdapter
       from my_policy_qat.adapter import (
           discover_targets, apply_actor, apply_rollout, load_rollout_weights,
       )
       return QuantizationAdapter(
           discover=discover_targets,
           apply_actor=apply_actor,
           apply_rollout=apply_rollout,
           load_rollout_weights=load_rollout_weights,
       )

   @cache
   def register():
       register_quantization_model("my_policy", validate_model_config, build_adapter)

The standard-library cache decorator makes register() run once per process, avoiding duplicate registration when workers initialize more than once. In a custom training entry point, register before calling the existing entry point.

.. code-block:: python

   from my_policy_qat import register
   from examples.embodiment.train_embodied_agent import main

   if __name__ == "__main__":
       register()
       main()

Both hooks belong to one entry. Duplicate names and missing hooks are errors. The validator accepts a model mapping, CorrectionSpec, and actor/rollout scope. It checks supported settings and returns comparable effective sampling/preprocessing configuration. Actor and rollout contracts must match; validation does not construct execution adapters.

Implement Model Operations
--------------------------

The factory returns four callbacks through QuantizationAdapter. Shared runtime discovers targets once, resolves the patterns in qat.targets, and passes the selected TargetManifestEntry sequence to both install callbacks. Adapters consume this manifest without rediscovering targets or matching selectors.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Callback
     - Contract
   * - ``discover(model)``
     - Return TargetManifestEntry items sorted by path, including the original parameter's state_dict key, actual shape, and model-specific kind.
   * - ``apply_actor(model, targets, *, backend)``
     - Install actor fake-quantized projections from the resolved manifest and return the same targets. Preserve Parameter identity, requires_grad, and checkpoint keys.
   * - ``apply_rollout(model, targets, *, backend)``
     - Install true low-bit rollout projections with QuantizedWeight; derived buffers stay out of checkpoints. Preserve the same parameter contract.
   * - ``load_rollout_weights(model)``
     - Call load_weights() on each projection to derive quantized weights from current floating master parameters. Shared runtime publishes the model version.

Runtime constructs the backend from validated block_size and quantization_format, then passes backend to both install callbacks. validate_qat_config checks the full recipe and actor/rollout configuration agreement; backends check block-size invariants on creation. Discovery and projection factories check supported model structure, and installation consumes the resolved manifest synchronously. Refresh model-specific FSDP wrapping markers after replacement in the adapter.

QuantizedPolicyRuntime owns the model version and profile hash. Each load_weights(version) withdraws the published version and clears availability on every QuantizedWeight before calling the model loader. Runtime publishes the new version only after all weights load and their backend recipes match. Each QuantizedWeight exposes its derived value after loading.

Public numerical entry points validate actual tensor devices, shapes, and layouts before private kernel launches. NVFP4 GEMM owns input/output argument checks; its private epilogue reuses that contract. Backend initialization checks GPU architecture and dependency versions. Validate quantization, packing, and GEMM numerics in tests.

OpenPI keeps actor fake quantization in ``models/openpi/actor.py`` and true low-bit rollout projections in ``models/openpi/rollout.py``. The adapter uses apply_pi05_actor_quantization, apply_pi05_rollout_quantization, and load_pi05_rollout_weights. All formats share model projections and ``rlinf/quantization/layers/weights.py``. Format-specific numerics live in ``rlinf/quantization/backends/``; native GPU calls live in ``rlinf/quantization/kernels/``.

Replay Stored Choices
---------------------

Store the sampled inputs and original rollout log-probs so the actor can score the same choices before its first PPO update. RLinf freezes those actor scores once per rollout batch as proximal_logprobs and retains prev_logprobs as behavior scores. OpenPI replays its existing action chain and selected transition indices.

Use matching actor and rollout sampling and preprocessing settings. Shared configuration validation compares those settings through the model validator. For a different distribution, implement log-probability scoring for the stored choices.

Numerical dispatch selects an explicit format and granularity implementation and raises for unsupported choices.

Validate the Adapter
--------------------

Check that disabled QAT retains the original policy. Then verify parameter identity, state_dict keys, frozen scope, gradients, derived-weight availability, and published model versions. Check real backend calls and fixed-input/noise numerics before model-specific training and task evaluation. Registration success alone does not establish full RL support.

Existing tests register a separate small model once and reach config validation and QAT execution. Fresh-process tests initialize RLinf, then reject additional Torch/model imports on the validation path. Each new adapter still needs real-model, sampling, replay, and training coverage. See :doc:`the QAT user guide <../guides/quantization>` for launch commands.
