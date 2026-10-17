.. Copyright 2026 The RLinf Authors.
   Licensed under the Apache License, Version 2.0.

From Compute to Throughput: Optimizing Embodied Models on Ascend
================================================================

RLinf improves complete-call throughput on Ascend by reusing attention metadata, overlapping sharded parameter communication, and reorganizing host and device work during action generation. These designs address a common gap: an accelerator may have substantial matrix throughput while the model waits for input preparation, control values, or the next layer's parameters. Understanding how those dependencies enter the critical path explains why an optimization works, where it applies, and which conditions invalidate it.

An embodied prediction usually combines image preprocessing, vision and language encoding, and action generation. Denoising models execute the action network repeatedly. Reinforcement learning adds logprob recomputation, backward computation, and sharded parameter communication. Matrix performance, complete action-call performance, and full RL iteration speed therefore measure different things. The following sections trace those dependencies, derive the main designs, and connect them to measurements on matched inputs.

From Matrix Throughput to Complete Calls
----------------------------------------

The first step is to identify what delays the next piece of useful device work. In the four-device servers measured here, an 8192 × 8192 BF16 dense matrix multiplication (GEMM) achieved 272.8 TFLOPS on Ascend 910B3 and 293.2 TFLOPS on A100-SXM4-80GB, a ratio of 93.04%. Device copy bandwidth reached 72.24% of the A100 result, while four-rank all-gather bus bandwidth reached 30.06%. The hosts differed too: the same Python loop took 193.0 ms on Kunpeng-920 and 98.0 ms on Xeon Platinum 8358. Matrix utilization is one part of the system; communication and host dependencies also determine when the next computation can start.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/0eb11a375a8a73130527211d079d6c21ccca8c7f/pic/ascend_opt/call_critical_path_en.png
   :alt: Dependencies across action inference, terminal-value inference and training
   :align: center

   Calls cross host and device boundaries. Arrows show dependencies rather than measured durations; training also introduces dependencies between parameter communication and computation.

During an action call, the host prepares images and text, the device computes vision and language features, and the action head produces controls. If Python reads an integer from a device tensor, subsequent host work must wait until that integer is available. The integer may occupy only a few bytes, but earlier work in the device queue may still be unfinished. Dependency length, rather than payload size, determines the waiting cost. Repeating that synchronization around short operations can cost more than the operations themselves.

Bootstrap has a different consumer: it needs the next observation's value to estimate future return. OpenVLA with a value head can use the prompt hidden state without completing autoregressive action-token decoding. Action and bootstrap FLOPs must therefore be counted separately. Training adds backward and FSDP communication, so an inference speedup does not automatically carry over to a training call.

These consumers establish the design boundaries. Reuse control metadata for its actual lifetime, overlap necessary work when dependencies permit it, and trim unused outputs only after checking the call contract. Each change then needs measurement through the complete interface to establish whether the local saving shortens the critical path.

The discussion distinguishes repository implementations, existing configuration combinations, and measured local prototypes. A performance result remains tied to the source and configuration used for that measurement.

Move Attention Metadata off the Per-Layer Critical Path
-------------------------------------------------------

The largest attention improvement comes from organizing its inputs, rather than replacing Q/K/V computation. Text lengths can differ within a batch, and visual token counts depend on image grids. Attention needs valid-token indices, sequence boundaries, and masks in addition to features. Those control values often remain unchanged across dozens of layers in one call.

Attention matches queries (Q) against keys (K), normalizes the resulting scores, and uses those weights to aggregate values (V). Each layer computes Q/K/V from features; lengths and masks describe their organization and allowed interactions. Metadata reuse exploits the different rates at which these quantities change.

Sequence Boundaries Can Introduce Host Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://raw.githubusercontent.com/RLinf/misc/0eb11a375a8a73130527211d079d6c21ccca8c7f/pic/ascend_opt/metadata_lifetime_en.png
   :alt: Version checks, metadata reuse and recomputation boundaries
   :align: center

   Tensor metadata caches and call-local vision-grid reuse have different lifetimes. Every layer continues to compute outputs from its current features.

Consider two sequences with valid lengths 5 and 3. After removing padding, their tokens can be packed contiguously with cumulative boundaries ``[0, 5, 8]``. The checked legacy Transformers NPU FlashAttention path expects the sequence endpoints as a CPU tuple, ``(5, 8)``. Converting device boundaries into that tuple requires a host transfer. Repeating it in each layer makes unchanged control information a recurring synchronization point between device work and host submission.

``install_npu_flash_attention_cache`` in ``rlinf/models/embodiment/gr00t/npu_common.py`` reuses three kinds of objects on the recognized legacy interface: unpadding indices and lengths, CPU endpoints derived from cumulative lengths, and a causal mask on each device. The first use computes these objects through the original path. Later layers reuse them while the input remains valid. Their Q/K/V tensors, attention outputs, and gradients are still computed afresh.

Those objects have different lifetimes. Padding and sequence boundaries belong to the current input and must be invalidated when it changes. A fixed-size causal mask describes positional ordering and can be reused per device. A single cache indexed only by shape would conceal that distinction: two padding masks with identical shapes can mark different valid tokens.

Invalidation Is Part of the Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tensor cache checks object identity and version. Ordinary PyTorch tensors update their version counter after an in-place write, so changing the same object triggers recomputation. A weak reference confirms that the original object is still alive and prevents an address reused by another Python object from appearing to be a hit. Bounded entries constrain the state retained during a long-running job.

Inference tensors do not offer the same version-tracking conditions. Graph capture also requires computations to follow the current dynamic contents written into captured inputs. These paths recompute metadata. Reuse is conditional on proving that it remains valid; callers should not inherit an undocumented obligation to keep a cache correct.

Vision grids follow a related but shorter scope. ``grid_thw`` describes patch counts along time, height, and width; ``[1, 3, 2]`` describes one frame with a 3 × 2 patch grid. Position encoding and window indexing both consume those integers. The measured prototype prepares one CPU description inside a visual forward, shares it between consumers, and clears the scope when the call exits. A later image with the same grid still requires fresh feature extraction. Sharing grid descriptions must not turn into sharing image results.

The call scope gives cleanup an explicit boundary. Entry stores the device grid and its CPU copy; position-processing methods confirm that they received the current grid before using the copy. A ``finally`` block restores the previous scope on exit. The measured inference and training prototypes apply this design to the examined visual paths. The inference prototype covers single-frame images; other temporal layouts must retain their original position-encoding rules.

At B32, StarVLA complete-action latency falls from 0.764 s to 0.654 s, improving throughput by 16.96%; the same-batch GR00T N1.7 comparison yields 10.83%. Adding position and grid processing to the StarVLA M16 baseline with FSDP prefetch improves per-rank training-call throughput by 8.28%–29.52%. The first two comparisons isolate grid processing. The training result combines position and grid changes and cannot be attributed entirely to one CPU copy.

Mask Conversion Must Preserve Numerical Meaning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenVLA also uses attention through PyTorch SDPA. A floating mask is added to attention logits; a boolean mask selects participating positions. On the measured torch-npu version, an eligible boolean representation permits a fused kernel, whereas a floating representation may use a decomposed path. ``rlinf/scheduler/hardware/accelerators/ascend_npu.py`` installs a conversion wrapper during NPU worker initialization.

This conversion is not a truthiness test. Allowed entries must be exactly zero, and blocked entries must meet the checked large-negative condition. A value of ``+0.2`` changes an attention score, as does ``-0.2``; neither represents a binary choice. Differentiable masks also retain the floating path because conversion would change their gradient behavior.

Fully blocked rows need a separate check. Adding the same finite negative value to all entries can still leave a nonzero softmax distribution, whereas an all-false boolean row has different output semantics. The implementation checks these rows before choosing a boolean representation. It caches the verdict by tensor version across layers and recomputes it after mutation.

Scope and Measured Contribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GR00T N1.5/N1.6 comparisons keep batch size fixed and toggle the metadata cache. GR00T N1.6 B16 action latency falls from 4.249 s to 1.122 s, improving sample throughput by 278.62%. GR00T N1.5 falls from 1.067 s to 0.361 s, a 195.52% improvement. Their four-rank training-call gains are 37.01%–39.04% and 41.76%–42.32%, respectively. Repeated metadata preparation accounted for a substantial part of the original complete-call cost.

OpenVLA's reference already contains its earlier mask cache. The measured view-reuse prototype additionally checks storage, shape, stride, offset, dtype, device, and version before reusing a verdict for the same underlying data. Its layout conditions are more specific than the repository's tensor-identity cache. Adding this prototype improves B64 action throughput by 3.51% and bootstrap by 1.15%. The available saving depends on the repeated work left in each baseline; the GR00T gains are not predictions for OpenVLA or another Transformers version.

Overlap Sharded Parameter Communication with Compute
----------------------------------------------------

After removing repeated control work, training still waits for parameters. Fully Sharded Data Parallel (FSDP) distributes parameters, gradients, and optimizer state across ranks to reduce resident memory. Here, each rank is the training process responsible for one accelerator. Before a sharded module computes, all-gather reconstructs its full parameters. Waiting until the current layer finishes before starting the next gather serializes communication and computation on the critical path.

Prefetch Changes When Communication Starts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://raw.githubusercontent.com/RLinf/misc/0eb11a375a8a73130527211d079d6c21ccca8c7f/pic/ascend_opt/fsdp_overlap_en.png
   :alt: Dependency timelines for on-demand gathering and parameter prefetch
   :align: center

   Prefetch starts the next layer's all-gather during current-layer computation. Block widths illustrate dependencies rather than measured durations.

If current compute takes ``C`` and the next gather takes ``G``, a serialized local segment takes approximately ``C + G``. Ideal overlap reduces that segment to approximately ``max(C, G)``. This relation describes a local dependency, not the whole training step: startup, tails, backward work, resource contention, and other synchronization remain.

Prefetch does not raise physical interconnect bandwidth or reduce the parameter bytes that must be transferred. When ``G`` exceeds ``C``, some communication remains exposed. Contention may also slow down the overlapped work. The useful overlap must be measured with the actual FSDP wrapping granularity and device topology.

The measured OpenVLA-OFT configuration uses existing RLinf options:

.. code-block:: yaml

   actor:
     fsdp_config:
       forward_prefetch: true
       backward_prefetch: pre
       limit_all_gathers: true

``forward_prefetch`` prepares parameters for later forward work. ``backward_prefetch: pre`` starts communication needed by subsequent backward computation early. Earlier arrival also means that more than one module's full parameters may reside in memory; ``limit_all_gathers`` constrains in-flight gathers. Prefetch depth, wrapping granularity, and memory budget must be considered together.

Batch Size Affects Both Overlap and Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A larger per-rank micro batch usually provides more forward and backward work. A module's parameter gather does not grow proportionally with sample count, which can create a longer overlap window and amortize fixed launch costs. Larger activations also consume memory that prefetch needs. Batch size and prefetch are therefore coupled decisions.

Global batch has a separate training meaning: it sets the number of samples per optimizer update. Moving from micro batch 16 to 32 and four-rank global batch 64 to 128 changes the relation between samples and update frequency. A throughput experiment can establish samples processed per second; a learning experiment must establish whether the resulting training remains effective.

The batch comparisons use ``batch / time``. For π₀, M16/G64 to M32/G128 doubles sample count while per-rank time grows from roughly 0.86 s to 1.32 s, yielding 29.67%–31.08% higher sample throughput. The corresponding gains are 27.62%–28.08% for π₀.₅ and 14.57%–14.72% for Dexbotic-π₀. These experiments change global batch and do not establish acceleration along an identical update trajectory.

At fixed M8/G32, forward and backward prefetch improve OpenVLA-OFT per-rank training-call throughput by 39.90%–40.76%. StarVLA gains 186.25%–240.03% at fixed M16/G64. Timing covers forward, loss, backward, and communication inside the call, excluding optimizer time; StarVLA restores a fixed checkpoint outside each measured call to retain valid gradients. These are call-level gains for those configurations, rather than full RL iteration speedups.

The final GR00T N1.6/N1.7 training combinations also use prefetch, but the available experiments do not isolate its contribution. Their combination results retain that attribution boundary.

Reduce Host Gaps During Multi-Step Action Generation
----------------------------------------------------

Communication overlap addresses dependencies between devices. Action generation also contains fine-grained dependencies between host and device. Models such as π₀, π₀.₅, and Evo-1 denoise actions over several steps. Actions change at each step, while observation context, predetermined timesteps, and some branch choices remain stable within one sampling call. Their different rates of change suggest different execution scopes.

Read Predetermined Control Values Once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the sampler has generated its timestep sequence, reading each value with ``.item()`` inside the loop adds avoidable synchronization. The measured π₀/π₀.₅ prototype first calls the original sampler, transfers the resulting sequence to a CPU list once, and uses it in the original order:

.. code-block:: python

   timesteps = rl_sampler.get_timesteps(num_steps, device)
   timestep_values = timesteps.detach().cpu().tolist()
   for idx in range(num_steps):
       t_val = float(timestep_values[idx])
       # The original step uses t_val and the current action tensor.

The sampler's values stay unchanged; only the synchronization point moves. Independently reconstructing apparently equivalent values on the CPU can change results through dtype and operation ordering. A control value that depends on later network outputs cannot be moved out of the loop this way.

Evo-1's random-step selection has a different source but follows the same principle. Python's random-number generator (RNG) already selects the step that receives noise on the host. The implementation uses that selection directly in the loop and separately stores the indices needed for replay, avoiding a device round trip. Random calls retain their count and order; noise and the full denoising trajectory still participate in later logprob replay.

Host configuration can amplify or offset these savings. More CPU threads may help large preprocessing work while adding coordination cost for small operations. The π₀/π₀.₅ comparisons also use eight CPU threads, so their complete-action throughput gains of 64.48% and 8.78% belong to those combination configurations. They cannot be split into independent control-handling and thread-count gains.

Multi-socket hosts divide cores and memory into NUMA nodes, where nearby memory and device access can cost less than remote access. NUMA affinity addresses that host-side cost. Keeping the threads responsible for an NPU near its CPU topology can reduce remote access and migration effects. ABot-M0 gains 5.61%–6.66% per rank with the same training configuration. This experiment constrains CPU affinity without explicitly binding memory allocation. Another server needs its own device/NUMA topology rather than copied CPU indices.

Submit Stable Compute Regions as One Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moving control reads out of the loop can still leave gaps between short kernel launches. A graph captures a region's operations and dependencies, then replays it through one submission. Suitable regions have stable input structure and explicit storage lifetimes, such as a language MLP or the deterministic network part of an action head. Sampling noise and dynamic host logic stay outside that region.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/0eb11a375a8a73130527211d079d6c21ccca8c7f/pic/ascend_opt/graph_replay_ownership_en.png
   :alt: Static input buffers, parameter storage and graph output ownership
   :align: center

   Replay reuses the execution plan and reads current inputs. Static inputs keep stable addresses; cloned outputs belong to callers. The caller releases old graphs before parameter storage moves.

``CUDAGraphManager`` in ``rlinf/utils/cuda_graph.py`` handles capture, input updates, replay, and release for CUDA and NPU. ``graph_signature`` distinguishes shape, dtype, specific device, container structure, and scalar types. The signature describes execution conditions rather than image or hidden-state contents, which are copied into static buffers before each replay.

The captured ``GraphSpec`` holds the graph and its static inputs and outputs. The next replay overwrites those output buffers. An asynchronous channel may still be sending the previous result, or a caller may need it in later work. The model wrapper therefore clones outputs before returning them, separating the graph's storage from caller-owned results:

.. code-block:: python

   copy_tensor_tree(graph.inputs, current_inputs)
   graph.graph.replay()
   owned_outputs = clone_tensor_tree(graph.outputs)

``copy_tensor_tree`` validates shapes, dtypes, and static metadata before updating captured buffers in place. ``clone_tensor_tree`` creates independent tensor storage. Complete-call timing includes both copies; timing replay alone would omit the cost of preserving the interface contract.

Parameter storage has its own conditions. An optimizer can update weights in place and replay reads the new values. Offload, module replacement, or reallocation can invalidate the storage referenced by an old graph. The model caller owns that lifecycle: release before moving weights and recapture when needed. The measured language-MLP prototype also checks parameter storage explicitly and falls back to eager execution after a change. The graph manager does not infer a model's lifecycle for its caller.

The current NPU graph backend lacks the corresponding CUDA generator-registration interface, so these designs keep sampling outside captured regions. They reduce launch overhead in deterministic computation while retaining the sampling order and RNG state. Capturing a larger region can instead introduce more constraints around dynamic inputs, randomness, and memory.

Language-MLP replay improves complete-action throughput by 4.24% for GR00T N1.6 and 1.40% for ABot-M0, including input/output copies. Their baselines already contain the corresponding position handling and action-head configuration. Amdahl's law bounds the result: even a large reduction in a small launch-overhead component yields a limited whole-call gain. Evo-1 also exposes an optional action-head graph path, but the available isolated measurements do not establish the same gain for it.

Choose the Compute Scope from Its Consumers
-------------------------------------------

After reducing waiting, check which computations the task actually consumes. Metadata reuse and prefetch mostly change scheduling. Consumer-directed execution may reduce actual FLOPs, so sample throughput and TFLOPS need separate interpretation: processing the same samples faster does not require a proportional rise in floating-point operations per second.

Keep Features and Trim Unused Vocabulary Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A language backbone first produces hidden states, then projects them into vocabulary logits. With batch B, length L, hidden size D, and vocabulary size V, the full projection costs approximately ``2BLDV`` operations and produces ``[B, L, V]``. An action head may consume all ``[B, L, D]`` features while ignoring those complete logits.

The measured GR00T N1.5 action-training prototype checks that there are no language labels and that complete hidden states are returned, then uses the language model's existing ``logits_to_keep=1`` option. The action head retains the full feature sequence, while vocabulary projection handles one position at approximately ``2BDV`` operations. Trimming the projection is different from trimming the features the action head needs.

At M128/G512, this improves per-rank training-call throughput by 9.34%–9.50%. FLOP accounting follows the reduced execution scope. Retaining the original full-projection count would inflate the optimized TFLOPS. Calls with language labels or consumers of complete vocabulary scores retain the full path.

Request the Value Needed for Bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenVLA bootstrap can also follow its consumer's requirements. A value head reads the prompt's final-position hidden state, available before the first action token is generated. ``predict_value_batch(env_obs)`` shares image and prompt preprocessing with action inference and returns a ``[B, 1]`` value tensor after one feature extraction.

Two independent HuggingFace rollout-worker options control request organization and compute scope:

.. code-block:: yaml

   rollout:
     batch_bootstrap_inference: true
     value_only_bootstrap: true

The first batches the currently received terminal-value requests and splits results back by original request sizes. A terminal observation belongs to the previous episode; a reset observation starts the next one. Their values cannot be interchanged, and each result must retain its trajectory ownership.

The second requires ``model_type: openvla`` and directly requests values when a value head and the algorithm path permit it. RLT, DAgger, and models without a value head retain the original path. Both options default to false and can be verified separately.

Skipping unused action generation also consumes fewer random numbers, potentially changing later actions under the same seed. That is distinct from whether the current value is numerically consistent. Value comparisons check the interface result; learning validation checks the resulting training process. No independently unmeasured percentage gain is assigned to these switches here.

Reuse Stable Context Projections within One Sampling Call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During Evo-1 denoising, action queries change while the same call's image/text context stays fixed. In a cross-attention block, ``K = X_context W_K`` and ``V = X_context W_V`` remain unchanged while both context and weights are unchanged. They can be projected on first use and reused in later steps. Current Q projections and attention still run at each step.

The measured native-operator design limits this reuse to a single sampling call in eval mode with gradients disabled. A new observation starts fresh projections; gradient-bearing training retains its graph construction. Those boundaries prevent stale cross-observation features and detached inference results from entering training.

This differs from caching attention lengths or masks. Those caches reuse sequence organization; K/V reuse retains actual intermediate features under an invariance condition. Both need explicit lifetimes, but neither can use shape alone as proof of validity. The final Evo-1 combination contains this reuse without an isolated toggle comparison, so only its overall performance is reported.

Batch Inputs without Changing Preprocessing Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-frame object construction, normalization, and transfer repeatedly pay Python and small-copy costs. Grouping equal-shaped images before transfer and processing can amortize them. OpenVLA/OFT, StarVLA, GR00T, Evo-1, and Dexbotic paths organize this work around their individual processors; a generic resize cannot replace every model's preprocessing.

Preprocessing defines the model's input. Crop/resize order, interpolation, padding color, floating-image ranges, and clipping before uint8 conversion can all alter pixels. Already matching the target size does not prove that the original crop can be skipped. Device batching therefore needs comparison with the native processor on ordinary and boundary inputs, with original-path fallback for unsupported formats.

Actual dtype must also agree across parameters, projection inputs, and continuous state. Dexbotic's model factory passes configured ``torch_dtype`` into loading and aligns continuous inputs at projection boundaries; unspecified precision retains the original loading behavior. Matching a BF16 YAML label alone does not establish matching execution precision.

The independent GR00T N1.6 prototype retains the original collator resize and text flow, groups transfers for supported images, and batches normalization through the original image processor. At B16 with eight CPU threads, complete-action latency falls from 0.609 s to 0.586 s, improving throughput by 3.95%. This prototype has not been integrated into the main implementation. The other input paths listed here do not have separately attributable complete-call gains in this comparison. Their value depends on the original processor's host cost, input batch, transfers, and extra temporary-buffer memory.

Explain Gains with Auditable Measurements
-----------------------------------------

These designs change metadata lifetime, work scheduling, or the amount of computation. Interpreting their results requires matching the compared workloads and stating the measured scope. Matrix benchmarks, individual kernels, complete model calls, and full RL training support conclusions at different levels.

Compare Action, Bootstrap and Four-Rank Training on Fixed Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance comparison uses PyTorch 2.6.0+cpu, torch_npu 2.6.0, and CANN 9.1.1 on Ascend, and PyTorch 2.11.0+cu128 with CUDA 12.8 on A100. Models use separate environments. The two platforms match checkpoints, configurations, actual dtypes, and input shapes; differences in the underlying software versions are part of this system comparison.

Inference replays saved real observations and checks processed pixels, tokens, masks, and sequence lengths across platforms. Complete-interface timing includes host processing, device work, and decoding. Five warmup calls precede 20 measured calls, summarized by synchronized median latency. Larger batches are assembled from saved observations and describe model-interface capacity, not simulator concurrency.

Training uses saved micro batches from four real actor ranks and executes forward, loss, backward, FSDP communication, and optimizer updates. TFLOPS timing covers forward/loss/backward and in-call communication, excluding optimizer time. Each rank has 25 calls, with the first five used for warmup. Training ratios pair corresponding rank inputs; the fastest rank does not stand for whole-system training efficiency.

FLOPs follow the actual matched execution path, including backward and recomputation. Custom attention counts use real sequence lengths and head structure. ``TFLOPS = actual FLOPs / call time / 10¹²``. Trimming vocabulary projection or frozen-module backward graph construction changes the numerator; removing waiting alone does not.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/0eb11a375a8a73130527211d079d6c21ccca8c7f/pic/ascend_opt/model_performance_en.png
   :alt: Matched-workload action, bootstrap and training TFLOPS ratios for eleven models
   :align: center

   A100 is the per-model reference. Training ranges pair corresponding ranks, and Evo-1 is highlighted. These fixed-input measurements were completed through October 10, 2026.

Download the `chart data, TFLOPS values and batch configuration as CSV <https://raw.githubusercontent.com/RLinf/misc/536580ae126e13fc4ab4e43a85c372534bbd63f3/pic/ascend_opt/model_performance.csv>`__. Training ratios pair corresponding ranks; dividing arbitrary endpoints of the two TFLOPS ranges does not reproduce those ratios.

Ten models reach 75% for action, bootstrap, and every training rank. Evo-1's native-operator design reaches 37.98%, 38.10%, and 39.55%–41.19%, respectively, showing that these shared techniques do not eliminate every model's remaining performance gap. Model architectures, batches, and execution paths differ; cross-model TFLOPS do not rank model quality. The results describe the model and software versions used in these experiments; deployments on other revisions require fresh measurements.

Ablations and combinations retain separate attribution. GR00T metadata caching and OpenVLA-OFT prefetch have toggle comparisons; π₀ control handling also changes thread count. Percentages from different baselines cannot be added. The final combination needs its own measurement.

Validate Outputs, State and Lifecycles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A matching action alone does not establish training correctness. Validation includes rollout actions/logprobs, stored replay inputs, training gradients, and parameter/optimizer state after real updates. Reuse and graph paths also need changed inputs, in-place weight updates, batch changes, offload/reload, release, and reuse. Bugs that emerge on a second call or after an update require those actual lifecycle transitions.

Random state is an execution result too. Comparing Python, CPU, and device RNG states identifies changes in random-call ordering. Value-only bootstrap deliberately skips stochastic action generation, so that difference needs to be stated and its learning effects checked separately rather than applying one unsuitable bitwise requirement to every optimization.

A correct micro-batch update proves the training path can execute. A full learning curve addresses whether training remains effective over time. Resumes require model, optimizer, learning-rate scheduler, and per-rank RNG checks; reinitialized simulator and rollout state belong in the experiment conditions. Training-rollout success is also different from success on an independent evaluation set. Reporting both complete-call performance and the associated learning evidence gives each conclusion a clear scope.

Observe Learning over Sustained Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numerical comparisons check whether an update preserves expected behavior. Sustained training observes whether those updates accumulate into learning progress. For OpenVLA-OFT, we started from the same initial checkpoint on A100 and Ascend 910B3 and completed 45 epochs per platform under the same LIBERO-10 GRPO protocol. Both experiments were complete by October 17, 2026.

Each platform used four accelerators, 32 training environments, BF16, micro batch 16, and global batch 8192. Each epoch contained 16 rollout rounds and four optimizer updates. The environment seed was 0, the actor seed was 1234, and the training sampling temperature was 1.6. The model had no value head. These batches and the experiment's purpose differ from the fixed-input performance measurements above.

The experiment retained the original GRPO configuration: FSDP ``forward_prefetch`` and ``backward_prefetch`` were disabled on both platforms, and ``limit_all_gathers`` was ``False``. These curves describe sustained training under that configuration. Matched-input, four-rank numerical comparisons separately verified gradients and optimizer updates for the prefetch configuration.

Both runs recorded all 45 success points with finite, nonzero gradient norms and exited successfully. Integrity checks covered every final checkpoint shard, optimizer state and progress, scheduler progress, and per-rank RNG state. Each optimizer reached 180 updates, and the scheduler reached epoch 45. The training-rollout metric ``env/success_once`` is the fraction of episodes that succeed at least once; the table uses the raw observations.

.. list-table:: OpenVLA-OFT training-rollout success
   :header-rows: 1
   :widths: 28 24 24 24

   * - Platform
     - First 5 epochs, mean
     - Epochs 41–45, mean
     - Epoch 45
   * - A100
     - 11.37%
     - 24.73%
     - 24.22%
   * - Ascend 910B3
     - 13.01%
     - 29.22%
     - 30.47%

.. figure:: https://raw.githubusercontent.com/RLinf/misc/4e86cb054172e447dcca68cc8e6d6b4b450331bf/pic/ascend_opt/learning_curves_en.png
   :alt: Raw training success for 45 epochs per platform and observed historical references, with vertical dotted resume boundaries
   :align: center

   Training-rollout success by epoch. Historical curves end at A100 epoch 35 and Ascend epoch 15; vertical dotted lines mark resumes in the current runs.

Both observed curves trend upward over training. Historical records cover only the first 35 A100 epochs and 15 Ascend epochs. Over each common interval, the current mean success was lower by 2.33 and 0.44 percentage points, respectively. These differences need to be considered alongside single-run variation, platform execution differences, and the resume conditions; the available data does not identify their cause.

A100 resumed at the start of epochs 28 and 42; Ascend resumed at the start of epoch 13. Model, optimizer, scheduler, and RNG state were restored from checkpoints, while simulator and rollout state were reinitialized. The marked boundaries preserve this experiment condition, and each historical curve ends at its last observed record.

Download the `raw epoch metrics, gradient norms and historical records as CSV <https://raw.githubusercontent.com/RLinf/misc/536580ae126e13fc4ab4e43a85c372534bbd63f3/pic/ascend_opt/learning_curves.csv>`__. Empty cells mean the historical record does not cover that epoch. There is one observed run per platform, and success is measured on training rollouts. This case shows sustained updates and learning progress for the stated OpenVLA-OFT configuration. Statistical equivalence and generalization performance require repeated runs across seeds and independent evaluation.
