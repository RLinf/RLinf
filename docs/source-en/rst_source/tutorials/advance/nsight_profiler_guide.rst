Nsight Systems Profiling Guide
==============================

This tutorial shows how to use **Nsight Systems** to profile RLinf training
pipelines and analyze performance from captured timelines.  We use a
*multi-node async dummy realworld SAC* experiment as the running example.

Standard single-process profiling (CUDA kernels, forward/backward passes,
optimizer steps) is well-documented elsewhere and straightforward to read
in the Nsight GUI.  This guide focuses on what is **unique to RLinf's
distributed architecture**: the multi-process Channel communication model,
the ChannelWorker threading internals, cross-node data routing, and how
to correlate NVTX annotations across heterogeneous worker groups.

By the end of this guide you will be able to:

* Enable per-step profiling with ``nsight_profiler`` in YAML
* Read the multi-process nsys timeline
* Trace data transfers across workers and nodes through Channel's 2-hop architecture
* Understand ChannelWorker threading model and cross-node transport selection


Prerequisites
-------------

* NVIDIA Nsight Systems CLI (``nsys``) in your ``$PATH``
* Python ``nvtx`` package installed: ``pip install nvtx``
* A Ray cluster running inside an NVIDIA container (nsys is pre-installed)
* Nsight Systems GUI on your local machine for viewing ``.nsys-rep`` files


Quick Start
-----------

Add the ``nsight_profiler`` block to your training YAML:

.. code-block:: yaml

   nsight_profiler:
     steps: [5, 6, 7]
     discrete: false
     nsight_options:
       t: "cuda,cudnn,cublas,nvtx,osrt"
       cuda-memory-usage: "true"
       capture-range: "cudaProfilerApi"
       capture-range-end: "repeat-shutdown:10"
     actor:
       enable: true
       all_ranks: true
     rollout:
       enable: true
       all_ranks: true
     env:
       enable: true
       all_ranks: true

Run training as usual.  ``.nsys-rep`` files appear under
``/tmp/ray/session_latest/logs/nsight/`` on each node.

Output files follow this naming convention:

* **Business workers** (``capture-range: cudaProfilerApi``):
  ``{GroupName}_rank{R}_pid{PID}.{N}.nsys-rep``
  where ``N`` is the segment index (1-based, mapping to ``steps[0]``, ``steps[1]``, ...).

* **ChannelWorkers** (``capture-range: none``, always recording):
  ``Channel_{Name}_rank{R}_pid{PID}.nsys-rep``


Experiment Configuration
------------------------

Payload sizes and latencies in this tutorial come from the config
``realworld_dummy_turtle2_dsrl_pi05_2node_4gpu_async.yaml``.
Key parameters that affect profiler numbers:

.. code-block:: text

   Cluster:  2 nodes (H20 train + L20 infer), 4 GPU/node, distributed_channel=true
   Network:  TCP 1 Gbps (GLOO) + ConnectX-7 400 Gbps InfiniBand (NCCL)
   Model:    Pi0.5 (pi05_turtle), bfloat16, DSRL 10 Q-heads
   Env:      4 envs, 3 cameras @ 224x224 RGB, state/action dim=6, dummy mode
   Rollout:  50 action chunks, 3 diffusion steps, huggingface backend
   Training: micro_bs=32, global_bs=128, update_epoch=32, FSDP no_shard, no RTC(Real Time Chunking)
   Profiler: steps=[6,7,8], discrete=false

Cross-node traffic uses two distinct fabrics: **GLOO over TCP (1 Gbps)**
for channel-based data transfers (trajectories), and **NCCL over
InfiniBand (ConnectX-7, 400 Gbps)** for weight synchronization.  The
~400x bandwidth gap explains why the ~8.5 GB weight sync completes in
under 500 ms while a ~2.8 MB trajectory transfer takes ~45 ms --- the
former is bandwidth-bound on IB, the latter is latency-bound on TCP.

These determine the payload sizes on each data path:

* **Obs**: 3 x 224 x 224 x 3 (uint8) + state = **~452 KB** pickle blob
* **Actions**: 11 tensors = **~491 KB**
* **Trajectory**: 22 tensors = **~2.79 MB**
* **Weight sync**: Pi0.5 state dict = **~8.5 GB**


1. Overall Workflow
-------------------

The async SAC pipeline consists of long-running worker processes that
interact through Channels.  All communication passes through a
**ChannelWorker** (Ray actor) --- data always takes a 2-hop path:
producer -> ChannelWorker -> consumer.

.. code-block:: text

   ┌──────────────────────────────────────────────────┐
   │              AsyncEmbodiedRunner                  │
   │   (orchestrator, starts all components)           │
   └──────────────────────────────────────────────────┘

   H20 Training Node                        L20 Inference Node
   ═════════════════                        ══════════════════

   ┌──────────────┐  weight sync (NCCL)  ┌──────────────┐
   │  ActorGroup  │ ──────────────────→  │ RolloutGroup │
   │  (GPU 0-3)   │                      │  (GPU 0-3)   │
   │ run_training │                      │  predict()   │
   └──────────────┘                      └──────────────┘
         ↑                                 ↑ obs  │ action
         │                                 │      ↓
         │                        ┌────────────┐ ┌─────────────────┐
         │                        │ Channel_Env│ │ Channel_Rollout │
         │                        └────────────┘ └─────────────────┘
         │                                 ↑      │
         │                                 │      ↓
         │  trajectory  ┌───────────────┐ ┌──────────┐
         └──────────────│ Channel_RB    │←│ EnvGroup │
         (GLOO TCP)     │  (H20)        │ │ (no GPU) │
                        └───────────────┘ └──────────┘

   Cross-node traffic: trajectory (GLOO TCP) + weight sync (NCCL IB)
   Intra-node traffic: obs / action (GLOO loopback, via Channel)

Process inventory
~~~~~~~~~~~~~~~~~

With ``distributed_channel: true``, each distributed channel spawns one
ChannelWorker **per node**.  The full set of processes for the 2-node
4-GPU experiment is:

.. list-table::
   :header-rows: 1
   :widths: 12 30 18 18 22

   * - Node
     - Process
     - Ranks
     - GPU
     - nsys-rep files
   * - H20
     - ActorGroup
     - 0, 1, 2, 3
     - GPU 0-3
     - ``H20/ActorGroup_rank{0-3}_pid*.nsys-rep``
   * - H20
     - Channel_Actor (rank 0)
     - 0
     - ---
     - ``H20/Channel_Actor_rank0_pid*.nsys-rep``
   * - H20
     - Channel_Env (rank 0)
     - 0
     - ---
     - ``H20/Channel_Env_rank0_pid*.nsys-rep``
   * - H20
     - Channel_Rollout (rank 0)
     - 0
     - ---
     - ``H20/Channel_Rollout_rank0_pid*.nsys-rep``
   * - H20
     - Channel_ReplayBuffer (rank 0)
     - 0
     - ---
     - ``H20/Channel_ReplayBuffer_rank0_pid*.nsys-rep``
   * - L20
     - RolloutGroup
     - 0, 1, 2, 3
     - GPU 0-3
     - ``L20/RolloutGroup_rank{0-3}_pid*.nsys-rep``
   * - L20
     - EnvGroup
     - 0, 1, 2, 3
     - no GPU
     - ``L20/EnvGroup_rank{0-3}_pid*.nsys-rep``
   * - L20
     - Channel_Actor (rank 1)
     - 1
     - ---
     - ``L20/Channel_Actor_rank1_pid*.nsys-rep``
   * - L20
     - Channel_Env (rank 1)
     - 1
     - ---
     - ``L20/Channel_Env_rank1_pid*.nsys-rep``
   * - L20
     - Channel_Rollout (rank 1)
     - 1
     - ---
     - ``L20/Channel_Rollout_rank1_pid*.nsys-rep``

**Key observations**:

* Channel_ReplayBuffer is **only on H20** (non-distributed) --- trajectory
  data flows Env@L20 -> Channel_RB@H20 -> Actor@H20, so the cross-node
  hop is Env -> Channel_RB.
* Channel_Env and Channel_Rollout have replicas on **both** nodes.
  Since Env and Rollout are both on L20, their traffic routes through the
  **L20 rank 1** replicas (intra-node).  The H20 rank 0 replicas are idle
  for obs/action but exist due to ``distributed_channel: true``.
* Business workers (Actor, Rollout, Env) use ``capture-range: cudaProfilerApi``
  --- they only record during profiled steps.  ChannelWorkers use
  ``capture-range: none`` --- they record the **entire** training run.

Data routing
~~~~~~~~~~~~

With ``total_num_envs: 4``, 4 Env ranks, and 4 Rollout ranks, CommMapper
produces a **1-to-1** rank pairing.  Distributed channel ranks indicate
which node hosts the ChannelWorker replica (rank 0 = H20, rank 1 = L20).

.. code-block:: text

   Observations (Env -> Rollout, intra-node L20)
   Channel: Env  |  key: {env_rank}_{rollout_rank}_train_obs
   ---------------------------------------------------------
     EnvGroup_rank0 --> Channel_Env_rank1 --> RolloutGroup_rank0
     EnvGroup_rank1 --> Channel_Env_rank1 --> RolloutGroup_rank1
     EnvGroup_rank2 --> Channel_Env_rank1 --> RolloutGroup_rank2
     EnvGroup_rank3 --> Channel_Env_rank1 --> RolloutGroup_rank3

   Actions / rollout results (Rollout -> Env, intra-node L20)
   Channel: Rollout  |  key: {rollout_rank}_{env_rank}_train_rollout_results
   ---------------------------------------------------------
     RolloutGroup_rank0 --> Channel_Rollout_rank1 --> EnvGroup_rank0
     RolloutGroup_rank1 --> Channel_Rollout_rank1 --> EnvGroup_rank1
     RolloutGroup_rank2 --> Channel_Rollout_rank1 --> EnvGroup_rank2
     RolloutGroup_rank3 --> Channel_Rollout_rank1 --> EnvGroup_rank3

   Trajectories (Env -> Actor, cross-node L20 -> H20)
   Channel: ReplayBuffer (non-distributed, H20 only)  |  key: default_queue
   ---------------------------------------------------------
     EnvGroup_rank{0-3} --GLOO TCP--> Channel_RB_rank0 --> ActorGroup_rank{0-3}
     (L20)                cross-node   (H20, shared FIFO)   (H20)

   Weight sync (P2P NCCL, no channel, cross-node H20 -> L20)
   ---------------------------------------------------------
     ActorGroup_rank0 ===NCCL IB=== RolloutGroup_rank0
     ActorGroup_rank1 ===NCCL IB=== RolloutGroup_rank1
     ActorGroup_rank2 ===NCCL IB=== RolloutGroup_rank2
     ActorGroup_rank3 ===NCCL IB=== RolloutGroup_rank3
     (H20)                           (L20)

Obs and action traffic stays on L20 (both Env and Rollout are co-located).
Channel_Env_rank0 and Channel_Rollout_rank0 on H20 exist but are **idle**
--- ``distributed_channel: true`` creates replicas on every node even when
no local producers or consumers use them.

Trajectory transfer is the only cross-node data path through channels.
Because Channel_ReplayBuffer is **non-distributed** (single replica on
H20), hop 1 (Env -> Channel_RB) is the cross-node segment.  All four Env
ranks feed into a single FIFO ``default_queue``; any Actor rank dequeues
the next available trajectory.


**Which files to load** for common analyses:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Analysis
     - Files to open
   * - Actor training pipeline
     - ``H20/ActorGroup_rank0`` + ``H20/Channel_ReplayBuffer_rank0``
   * - Rollout inference pipeline
     - ``L20/RolloutGroup_rank0`` + ``L20/Channel_Env_rank1`` + ``L20/Channel_Rollout_rank1``
   * - Cross-node trajectory transfer
     - ``L20/EnvGroup_rank0`` + ``H20/Channel_ReplayBuffer_rank0`` + ``H20/ActorGroup_rank0``
   * - Full end-to-end view
     - All of the above (6 files)

When you open these ``.nsys-rep`` files in the Nsight Systems GUI, each
process appears as a separate row.  Expand a process to see:

* **NVTX** rows --- custom annotations from ``@NsightProfiler.annotate``
* **CUDA** rows --- kernel launches, memcpy, streams
* **OS Runtime** rows --- system calls (useful for GLOO socket transfers)

The screenshot below shows **all compute-worker ranks** loaded together.
From top to bottom: ActorGroup rank 0--3 (H20) with
``actor/run_training`` blocks and CUDA kernel bursts, RolloutGroup rank
0--3 (L20) with ``rollout/generate_epoch`` / ``rollout/recv_obs``, and
EnvGroup rank 0--3 (L20) with ``env/interact_once`` / ``env/step``.
All three worker groups run **asynchronously** in parallel.

.. image:: /_static/images/nsight_profiler/all_overall.png
   :alt: Overall view showing all ActorGroup, RolloutGroup, and EnvGroup ranks
   :width: 100%

Step boundaries
~~~~~~~~~~~~~~~

The runner calls ``start_profile(step)`` / ``stop_profile()`` on each
worker group.  For business workers using ``capture-range: cudaProfilerApi``,
this translates to ``torch.cuda.profiler.start()`` / ``stop()`` calls.

With ``steps: [6, 7, 8]`` and ``discrete: false``, all three steps are
captured in a single ``.nsys-rep`` file (continuous capture).

ChannelWorkers always record, providing a complete view of data transit
throughout the entire training.


2. Data Transfer Paths
----------------------

Every Channel transfer follows a **2-hop** path: producer -> ChannelWorker
-> consumer.  There are four data paths in the async SAC pipeline; the
first three use Channels, the fourth is a direct P2P NCCL transfer.

Reading the timeline rows
~~~~~~~~~~~~~~~~~~~~~~~~~

In the Nsight Systems GUI each OS process is a top-level row; threads
belonging to that process appear as indented sub-rows.  RLinf workers
are Ray actors, so the row labels follow a fixed pattern:

.. code-block:: text

   [PID] python            ← process row  (one per Ray actor)
     [All Streams]         ← aggregated GPU activity
     [TID] ray::IDLE       ← thread row   (Ray's asyncio event-loop thread)
       Start & End         ← NVTX ranges emitted by that thread
       NCCL                ← NCCL kernel row (if applicable)
     [TID] ray::XXXX       ← another thread in the same process (send / recv)
       Start & End

* **``[PID] python``** -- the process.  The number is the OS PID.
  Every Ray actor (ActorGroup_rank0, Channel_Env_rank1, ...) is a
  separate process.
* **``[TID] ray::IDLE``** -- a thread inside the process.  The number
  is the OS thread ID (TID).  Ray labels its worker threads
  ``ray::IDLE``; the name does not mean the thread is idle.
* **``Start & End``** -- NVTX annotation sub-rows showing the
  ``nvtx.start_range`` / ``nvtx.end_range`` pairs emitted by that
  thread.

A single Ray actor typically has multiple threads:

* **Event-loop thread** -- runs the async Ray methods; shows high-level
  NVTX annotations such as ``rollout/generate_epoch`` or
  ``env/interact_once``.
* **Send thread(s)** -- one per outbound collective connection;
  shows ``collective/send`` and ``pg/send`` ranges.
* **Recv thread(s)** -- one per inbound collective connection;
  shows ``collective/recv`` and ``pg/recv`` ranges.

In the screenshots that follow, we reference these rows as
``[PID]`` for the process and ``[TID]`` for specific threads.  For
example, "``[637728]``, ``[638135]``" means process 637728, thread
638135.

Reading NVTX labels
~~~~~~~~~~~~~~~~~~~

The other piece of vocabulary is the NVTX label format emitted by the
collective layer:

.. code-block:: text

   collective/send type=DATACLASS_WITH_TENSORS transport=GLOO peer=0
                  group=cg-Rollout:1-RolloutGroup:0

* ``type`` -- serialization format.  ``DATACLASS_WITH_TENSORS`` means a
  Python dataclass whose tensor fields are extracted and sent individually;
  ``TENSOR_DICT`` is a plain ``dict[str, Tensor]``; ``OBJECT`` is a single
  pickled blob.
* ``transport`` -- ``GLOO`` (CPU / TCP / loopback) or ``NCCL`` (GPU / IB).
  Note: ``transport=NCCL`` only means the tensor resides on GPU; the
  underlying transfer may still fall back to GLOO if the two endpoints have
  heterogeneous accelerators.
* ``peer`` -- the remote rank *within the 2-process collective group*
  (always 0 or 1).
* ``group=cg-{A}:{rA}-{B}:{rB}`` -- identifies the point-to-point process
  group.  ``A``/``B`` are component names (e.g., ``Rollout``,
  ``RolloutGroup``), ``rA``/``rB`` are their respective ranks.  In the
  example above the group connects Channel_Rollout rank 1 to
  RolloutGroup rank 0.

Lower-level ranges inside a ``collective/send`` include
``collective/send_tensor_list/cpu_payload n=... bytes=...`` (per-batch
metadata + tensor data) and ``pg/send backend=GLOO bytes=...``
(one call per tensor or metadata chunk).

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Path
     - Route
     - Payload
     - Transport
   * - Actions
     - Rollout -> Channel_Rollout -> Env (intra-node)
     - RolloutResult (~491 KB, 11 tensors)
     - GLOO loopback
   * - Observations
     - Env -> Channel_Env -> Rollout (intra-node)
     - Images + states (~452 KB pickle blob)
     - GLOO loopback
   * - Trajectories
     - Env -> Channel_RB -> Actor (cross-node)
     - Trajectory (~2.79 MB, 22 tensors)
     - GLOO TCP
   * - Weight Sync
     - Actor -> Rollout (P2P, no channel)
     - Full state dict
     - NCCL IB


Actions: Rollout -> Channel_Rollout -> Env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start with the action path because it best illustrates the ChannelWorker
threading model.

The overview screenshot below shows Channel_Rollout_rank1 receiving from
all four RolloutGroup ranks.  Notice that the **recv thread** is a single
thread shared across all producers --- it processes incoming payloads
sequentially.  At large scale or with large payloads this single recv
thread could become a serialization point.

.. image:: /_static/images/nsight_profiler/RolloutWorker0_RolloutChannel1_EnvWorker0_overview.png
   :alt: Channel_Rollout overview showing single recv thread handling all producers
   :width: 100%

The zoomed screenshot below traces a single rank-0 action transfer.
From top to bottom:

* **RolloutGroup_rank0** (``[637728]``, ``[638135]``) ---
  after ``rollout/generate_epoch`` completes, the send thread ``[640937]``
  fires ``collective/send type=DATACLASS_WITH_TENSORS
  group=cg-Rollout:1-RolloutGroup:0 [19.9 ms]`` containing
  ``n=11 bytes=491036``.  Each of the 11 tensors is sent as a separate
  ``pg/send backend=GLOO`` call.
* **Channel_Rollout_rank1** (``[638798]``, ``[639466]``) ---
  the recv thread receives the 11 tensors
  (``collective/recv type=DATACLASS_WITH_TENSORS
  group=cg-Rollout:1-RolloutGroup:0 [15 ms]``).
  The yellow ``cw/Rollout/dequeue key=0_0_train_rollout_results`` bar on
  the event-loop thread shows the wait for the internal queue.
  A dedicated **send thread** ``[640995]`` then forwards the payload to
  EnvGroup: ``collective/send group=cg-EnvGroup:0-Rollout:1 [11.2 ms]``.
* **EnvGroup_rank0** (``[638322]``, recv thread ``[640678]``) ---
  ``channel/Rollout/get key=0_0_train_rollout_results [244.9 ms]``
  blocks until the data arrives, then ``env/interact_once`` resumes.

.. image:: /_static/images/nsight_profiler/RolloutWorker0_RolloutChannel1_EnvWorker0.png
   :alt: Action transfer from RolloutGroup_rank0 to EnvGroup_rank0 via Channel_Rollout_rank1
   :width: 100%


Observations: Env -> Channel_Env -> Rollout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The obs path follows the same 2-hop pattern.  The screenshot below shows
EnvGroup_rank0 -> Channel_Env_rank1 -> RolloutGroup_rank0.

Key difference from the action path: observations are serialized as a
single pickle blob (``type=OBJECT``, one ``pg/send``) rather than a tensor
list, because the obs dataclass mixes images and non-tensor fields.

* **EnvGroup_rank0** (``[638322]``, send thread ``[640603]``) ---
  ``channel/Env/put key=0_0_train_obs dst=1 bytes=451608 [15.6 ms]``.
* **Channel_Env_rank1** (``[639247]``) ---
  recv thread ``[639888]`` (shared, single-threaded),
  send threads ``[640712]``/``[640786]`` (one per consumer rank).
* **RolloutGroup_rank0** (event-loop thread ``[638135]``, recv thread
  ``[640613]``; process row ``[637728]`` is not pinned in this screenshot) ---
  ``rollout/recv_obs [6.582 s]`` blocks until the obs arrives.

.. image:: /_static/images/nsight_profiler/EnvWorker0_EnvChannel1_RolloutWorker0.png
   :alt: Observation transfer from EnvGroup_rank0 to RolloutGroup_rank0 via Channel_Env_rank1
   :width: 100%

Each NVTX range label includes the payload size and queue depth, e.g.:

.. code-block:: text

   channel/Env/put key=0_0_train_obs dst=1 bytes=451608 cpu=451608 gpu=0

Note: ``dst=1`` refers to channel rank 1 (L20's ChannelWorker replica),
not rollout rank 1.


Trajectories: Env -> Channel_ReplayBuffer -> Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the only **cross-node** channel path.  Channel_ReplayBuffer is
non-distributed (single replica on H20), so hop 1 (Env@L20 ->
Channel_RB@H20) crosses the network.

* **EnvGroup_rank0** (``[638322]``, send thread ``[641308]``) ---
  ``collective/send type=DATACLASS_WITH_TENSORS
  group=cg-EnvGroup:0-ReplayBuffer:0 [44.5 ms]``, sending
  ``n=22 bytes=2794154`` via GLOO TCP.
* **Channel_ReplayBuffer_rank0** (``[154377]``, ``[155053]``) ---
  recv thread receives the trajectory across the network; the yellow
  ``cw/ReplayBuffer/dequeue key=default_queue`` bars show the event-loop
  thread waiting.  Send thread ``[158332]`` forwards to the actor
  (``n=22 bytes=2794154 [25.0 ms]``, intra-node on H20).
* **ActorGroup_rank0** (``[152797]``, recv thread ``[157946]``) ---
  ``actor/recv_traj_batch idx=11 n=1 [27.9 s]`` blocks in a background
  thread; received trajectories are drained into the replay buffer at
  the start of each ``run_training`` step.

.. image:: /_static/images/nsight_profiler/EnvWorker0_ReplayBuffer0_ActorWorker0.png
   :alt: Trajectory transfer from EnvGroup_rank0 to ActorGroup_rank0 via Channel_ReplayBuffer_rank0
   :width: 100%


Weight Sync: Actor -> Rollout (P2P NCCL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weight sync does **not** go through a Channel.  Each ActorGroup rank
pushes the full state dict directly to the corresponding RolloutGroup rank
using a **point-to-point NCCL** process group over InfiniBand.  Because
H20 and L20 have different GPU topologies (PIX vs NODE, see the topology
matrix in the experiment notes), the NCCL transport is **heterogeneous**
--- each side negotiates IB resources independently.

The screenshot below shows ActorGroup_rank0 (``[152797]``) calling
``actor/sync_model_to_rollout [492 ms]``.  The send thread ``[157830]``
issues ``collective/send type=TENSOR_DICT
group=cg-ActorGroup:0-RolloutGroup:0``.  On the receiving side,
RolloutGroup_rank0 (``[637728]``) shows the corresponding
``collective/recv type=TENSOR_DICT [398 ms]``.

Note the **NCCL kernel** row visible on both processes
(``ncclDevKernel_Broadcast_RING_LL``) --- this confirms the transfer runs
on GPU via NCCL, not via CPU-side GLOO.

.. image:: /_static/images/nsight_profiler/ActorWorker_RolloutWorker.png
   :alt: Weight sync from ActorGroup_rank0 to RolloutGroup_rank0 via P2P NCCL
   :width: 100%


Appendix: Troubleshooting
--------------------------

nsys not found
~~~~~~~~~~~~~~

Ensure ``nsys`` is in ``$PATH`` before starting Ray:

.. code-block:: bash

   export PATH="/usr/local/bin:$PATH"
   which nsys
   ray stop && ray start --head

nsys-rep version mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~

The nsys GUI on your local machine must be **>= the version** used to
capture the ``.nsys-rep`` file.  Container nsys version can be checked with
``nsys --version``.

No NVTX annotations visible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Verify ``nvtx`` is installed: ``python -c "import nvtx; print('ok')"``
* Check that ``nsight_options.t`` includes ``nvtx``
* Ensure the profiled steps actually run (``max_epochs`` must be larger
  than the highest step in ``steps``)
