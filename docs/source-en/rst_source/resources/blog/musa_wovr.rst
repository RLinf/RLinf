RLinf × MUSA: Reproducing WoVR on Moore Threads S5000
============================================================

This article explains the engineering changes used to reproduce WoVR's
world-model RL training loop on Moore Threads S5000 accelerators with RLinf.
It follows the port from device discovery and model correctness through weight
synchronization, memory management, and performance tuning on eight and sixteen
GPUs. MUSA is Moore Threads' software and runtime platform; S5000 is the
accelerator used in these experiments.

The reported S5000 and A100 training success-rate curves have a correlation of
0.976 over 191 steps. At step 190, external LIBERO evaluation reaches 74.6% on
S5000 and 73.4% on A100. In separate performance experiments, an eight-GPU
configuration reduces step time by 10.6%, while two-node hybrid sharding reduces
step time by 15.3% and weight synchronization time by about 66%.

.. note::

   This is an engineering report of an experimental adaptation referenced to
   upstream commit ``4677808a``. The implementation details and measurements
   below describe that adaptation. Its ``WOVR_*`` switches, ``perfA`` configuration,
   and ``wan_spatial_official_s5000.yaml`` are experiment-specific artifacts,
   rather than configuration options shipped with this documentation revision.
   Use the :doc:`Wan example <../../examples/embodied/wan>` for the maintained
   setup instructions and documented hardware coverage.

The Training Loop and GPU Layout
--------------------------------

The port has to preserve a closed loop: a policy chooses actions, a world model
predicts their consequences, and the resulting reward guides the next policy
update. In :doc:`WoVR <../publications/wovr>`, OpenVLA-OFT emits an eight-action
chunk, Wan generates future frames, and a reward model judges task completion.
RLinf then updates OpenVLA-OFT with GRPO.

RLinf workers run environment interaction, policy inference, and parameter
updates. The scheduler places them on GPUs, and channels carry rollout data
between workers. Weight synchronization and FSDP collectives handle parameter
transfer and gradient communication. The simplified data flow is:

.. code-block:: text

   OpenVLA-OFT (rollout) -- eight actions --> Wan + reward model (env)
            ^                                      |
            |                         frames, rewards, trajectories
            |                                      v
            +--- updated weights --- GRPO / OpenVLA-OFT (actor)

The environment, rollout, and actor are logical roles. Their GPU placement
determines when models occupy memory and when data must move between devices.

.. list-table:: Execution layouts
   :header-rows: 1
   :widths: 18 42 40

   * - Layout
     - GPU use
     - Main trade-off
   * - Collocated
     - Roles take turns on the same GPUs, with offloading between stages.
     - Simple placement; a slow rollout delays the whole step.
   * - Disaggregated
     - Roles occupy separate GPU groups.
     - Less model swapping; data dependencies can leave GPUs idle.
   * - Hybrid
     - Environment and rollout groups form a pipeline; the actor uses the full GPU group for updates.
     - Queues decouple production and consumption, with additional scheduling requirements.

The reproduction uses collocated execution. Starting on one eight-GPU node made
it possible to validate device assignment, memory residency, and communication
through the same end-to-end workload. See :doc:`execution modes
<../../concepts/execution_modes>` for the broader placement model.

Making the Training Loop Run on MUSA
------------------------------------

Before tuning throughput, the adaptation had to discover the devices, assign
each process correctly, and preserve the model's behavior. The work covered four
areas, with performance changes enabled separately from compatibility fixes.

.. list-table:: Adaptation areas
   :header-rows: 1
   :widths: 22 38 40

   * - Area
     - Components in the experimental implementation
     - Problem addressed
   * - Device and process-group interfaces
     - ``musa_gpu.py``, ``base_world_env.py``, FSDP initialization
     - Device counts, rank-specific device indices, and backend initialization
   * - Communication compatibility
     - ``fsdp.py``, ``multi_channel_pg.py``, weight-sync configuration
     - MCCL broadcast hangs, missing CPU backend, and unstable cross-process device transfers
   * - Environment compatibility
     - ``envs/utils.py``, LIBERO subprocess setup, finite-value checks
     - Non-Gym environment attributes, CPU simulator processes, and occasional invalid batches
   * - Performance
     - ``world_model_wan_env.py``, hybrid-shard mesh, external DiffSynth patch
     - Repeated frame copies and cross-node parameter gathering

Device Discovery and Assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Device discovery failed before any model could load. The target container lacked
the expected ``pymtml`` Python binding, and a same-named PyPI package made the
dependency ambiguous. A failed management-library query was interpreted as zero
GPUs. The adaptation retained that query and added a fallback to
``torch.musa.device_count()``.

Finding the devices did not establish which process should use each one. In the
tested runtime, rewriting ``MUSA_VISIBLE_DEVICES`` to ``0`` inside a process
could select physical GPU 0 instead of the intended local device. The experiment
restricted visibility at the container boundary with
``MTHREADS_VISIBLE_DEVICES`` and preserved RLinf's rank bookkeeping inside it.

Environment Attributes and Distributed Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once device assignment was correct, initialization exposed an environment
interface mismatch. Attribute lookup tried ``get_wrapper_attr`` first, but the
Wan environment was not a standard Gym wrapper. An ``AttributeError`` from that
path was mistaken for a missing attribute. Falling back to ordinary ``getattr``
preserved attribute delegation through wrappers. The current ``get_env_attr``
also walks the wrapper chain when this lookup fails.

Distributed initialization needed both a device and a CPU backend. MCCL handles
MUSA collectives, while the sharded checkpoint state-dict path also needs Gloo.
The experimental implementation explicitly initialized both before creating the
device mesh:

.. code-block:: python

   dist.init_process_group(backend="cpu:gloo,musa:mccl")

This selects Gloo for CPU tensors and MCCL for MUSA tensors. Establishing those
backends was an initialization requirement; checkpoint resume still requires
separate validation.

Matching OpenVLA-OFT Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A running process could still produce the wrong policy. OpenVLA-OFT requires
bidirectional attention, but upstream Transformers 4.40.1 without the OFT patches
applied causal attention. Under the same training configuration, the first-step
S5000 success rate was zero before the fix. With ``is_causal=False`` and a
padding-only mask, it reached 0.516, compared with 0.469 for the A100 control.

That mismatch changed the policy distribution enough to produce nearly random
actions and deprive GRPO of a useful training signal. Matching attention semantics
therefore came before comparing numerical precision or performance.

Synchronizing Weights Through CPU Memory
----------------------------------------

With model behavior aligned, the next obstacle was moving roughly 15 GB of new
weights from the actor group to the rollout group after each training step.
Isolated same-GPU IPC and cross-GPU ``isend``/``irecv`` probes passed, but direct
device transfers under eight-GPU concurrency still produced
``launch timed out`` failures. The full workload combined shared GPUs, MCCL, and
process isolation in a way those probes did not exercise.

The stable experimental path staged weights through CPU memory:

.. code-block:: text

   actor GPU -> CPU buffer -> CPU/Gloo -> rollout CPU -> rollout GPU

The weight-sync configuration selected ``transport_device: cpu``. This adds
copies, but synchronization happens once per step and weight shards can travel
in parallel. A smoke test measured about 41 seconds, while synchronization
accounted for roughly 2% of the full reference training step. World-model
generation and VLA inference still dominated the collocated workload.

Two accompanying changes kept communication predictable. Each rank loaded the
same Hugging Face checkpoint independently, allowing the adaptation to disable
``sync_module_states`` and avoid a hanging coalesced broadcast of the 7.5B model.
Weight transfer used CPU/Gloo, while device collectives continued to use MCCL.
That path also avoided a dependency on an unreleased additional communication
library.

Controlling Memory Residency and Invalid Batches
------------------------------------------------

Stable communication made it possible to reach the next constraint: keeping the
training, rollout, and world models resident together required more than 80 GB.
The experiment controlled peak memory by changing residency across stages.

The world model remained in the environment process at about 24.9 GB. The rollout
model moved to CPU when it was not performing inference. Actor parameters and
optimizer state were offloaded while the environment generated frames, then
reloaded for training. This supported the eight-GPU collocated run; micro-batch
size still had to be checked against the node's available memory.

Concurrent execution also exposed occasional NaN or Inf values in batch rows.
The adaptation checked VLA and Wan forward passes and recomputed an affected
batch, with at most three retries. Failures that did not recur in isolated replay
were treated as a runtime workaround, rather than evidence of algorithmic
stability. Before spawning LIBERO's CPU simulator processes, the adaptation also
cleared MUSA device visibility to avoid unnecessary device contexts aggravating
initialization races.

Reducing World-Model Copies While Preserving Values
------------------------------------------------------------

Once memory use was under control, repeated transfers between host and device
became a performance target. Input frames passed through PIL and NumPy, and VAE
outputs returned to the host frame by frame. The experiment grouped changes to
these paths under ``WOVR_FAST_IO``:

- Keep conditioning frames on the device and remove per-frame PIL round trips.
- Consolidate finite-value checks from several synchronizations per denoising
  iteration to one per denoising pass, while retaining NaN/Inf retries.
- Convert Python scalars on DiffSynth's frequently executed paths to
  zero-dimensional device tensors.

A small numerical error still matters because frames are quantized back to
``uint8`` and fed into later predictions. A difference of one unit in the
last place can grow through that feedback. To reproduce the original NumPy
normalization, the adaptation precomputed a 256-entry lookup table for the
``uint8`` to ``[-1, 1]`` conversion on the host, then indexed it on the device.
The reported bitwise checks found zero differences across 20.44 million elements
on both A100 and S5000.

In this eight-GPU comparison, environment interaction time fell from about
2050 to 975 seconds. Full training-step time fell from about 3100 seconds with
the performance switches disabled to about 1900 seconds with them enabled.
These changes targeted data movement and synchronization while preserving the
training computation. This comparison is separate from the configuration A/B
experiment below.

Single-Node Accuracy and Performance
------------------------------------

The numerical checks covered specific transformations. The policy still needed
to be evaluated after training, and the performance settings needed a comparison
on the same machine and data.

Training Curves and Simulator Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reported training success-rate curves have a correlation of 0.976 across
191 steps. Final policy quality was measured in the LIBERO simulator, separately
from reward predicted inside the world model.

.. list-table:: External LIBERO success rate
   :header-rows: 1
   :widths: 40 30 30

   * - Evaluation point
     - S5000
     - A100
   * - Before RL training
     - 55.0%
     - 59.7%
   * - Step 190
     - 74.6%
     - 73.4%

The reported difference test gives ``p = 0.76``. The observed difference was not
statistically significant in that experiment window; this result alone does not
establish equivalence across other configurations or training runs.

Configuration A/B on Eight GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The experiment's ``perfA`` configuration used ``actor.micro_batch_size: 32``,
``actor.global_batch_size: 8192``, and
``actor.fsdp_config.enable_gradient_accumulation: False``. The source report
identifies that last setting as the traceable difference from its
``wan_spatial_official_s5000.yaml`` configuration. Those settings describe the
reported experiment; they do not reconstruct historical configurations that
were not retained.

In FSDP1, this flag controls whether intermediate micro-batches defer gradient
synchronization through ``no_sync()``. Setting it to ``False`` allows
synchronization on every micro-batch while retaining the configured global batch
size. The measured comparison used the same eight-GPU machine and data, averaged
over three steps.

.. list-table:: Eight-GPU configuration A/B
   :header-rows: 1
   :widths: 34 22 22 22

   * - Metric
     - Baseline
     - perfA
     - Change
   * - Full step
     - 2110.7 s
     - 1887.6 s
     - −10.6%
   * - Actor / GRPO training
     - 832.0 s
     - 607.4 s
     - −27.0%
   * - World-model environment
     - 1004.6 s
     - 1005.1 s
     - Approximately unchanged
   * - Weight synchronization
     - 39.8 s
     - 37.2 s
     - Within reported noise

The reduction is concentrated in actor training, consistent with lower
micro-step overhead. World-model generation still occupies about half the step.
This is a configuration experiment with a reversible switch, not a general
recommendation to change gradient synchronization for every model or network.

Scaling to Two Nodes and Sixteen GPUs
-------------------------------------

Moving to two eight-GPU nodes introduced initialization and network constraints
beyond the single-node run. Sixteen processes first touching MUSA concurrently
could trigger a lazy driver-initialization race and hang without logs. Missing
RDMA userspace components could also make MCCL fall back to sockets, turning a
network setup problem into an apparent training stall.

The experimental ``WOVR_DEV_INIT_STAGGER=3`` setting staggered initialization by
rank. ``WOVR_FSDP_SHARD_SIZE=8`` created a two-dimensional device mesh with axes
for node count and GPUs per node. Hybrid sharding then kept parameter all-gathers
within a node and used cross-node communication for the required gradient
reductions. FSDP1's gradient-norm calculation also had to reduce over the
within-node shard group.

.. list-table:: Two-node, sixteen-GPU sharding comparison
   :header-rows: 1
   :widths: 34 22 22 22

   * - Metric
     - Full sharding
     - Hybrid sharding
     - Change
   * - Full step
     - 824.8 s
     - 698.6 s
     - −15.3%
   * - Weight synchronization
     - 278.9 s
     - 93.7 s
     - About −66%

Keeping large parameter gathers within each node reduced pressure on the
roughly 4.7 GB/s inter-node link. Staggered initialization, container process
limits, RDMA and network configuration, and a clean Ray job state were all part
of the launch conditions. The two-node measurements use a separate configuration
and measurement window from the eight-GPU experiments, so the tables do not
establish an eight-to-sixteen-GPU scaling factor.

Remaining Validation
--------------------

The adaptation established a working training loop and measured improvements at
three different levels: policy quality, single-node step time, and two-node
communication. Keep those comparisons separate when reproducing the results.
Checkpoint resume and some driver-initialization cases still need independent
validation. NaN retries and CPU-staged weight synchronization remain platform
workarounds in the reported implementation.

Further work should target world-model generation overhead and re-evaluate
direct device transfers after driver and communication-library fixes. For the
maintained training workflow, follow the :doc:`Wan example
<../../examples/embodied/wan>`; for the method, see the :doc:`WoVR publication
<../publications/wovr>` and `WoVR paper <https://arxiv.org/abs/2602.13977>`_.
