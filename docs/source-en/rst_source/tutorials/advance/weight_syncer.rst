Weight Synchronization
==============================

This document introduces the ``weight_syncer`` mechanism in RLinf for
**embodied training**. It is designed to optimize weight synchronization from
the actor side to the rollout-side policy model, reducing the communication and
loading overhead after each parameter update.

At the moment, this capability is mainly intended for the
**FSDP actor + HuggingFace rollout** path used by
``examples/embodiment/train_embodied_agent.py`` and
``examples/embodiment/train_async.py``.


Why Weight Syncer Exists
------------------------------

In embodied RL, every actor update usually needs to be synchronized to the
rollout workers. For large models such as OpenPI, OpenVLA, OpenVLA-OFT, and
GR00T, full-weight synchronization can become expensive:

- The model is large, so full sync can easily become a major part of step time.
- Repeatedly loading a full ``state_dict`` on the rollout side also adds GPU and
  CPU overhead.
- In async settings, blocking full sync directly hurts rollout throughput and
  policy freshness.

To address this, RLinf abstracts the logic into a unified ``WeightSyncer``
interface so different synchronization strategies can share the same sender /
receiver workflow.


Core Interface
------------------------------

``WeightSyncer`` has four main responsibilities:

- ``init_sender(...)``: one-time sender-side initialization
- ``init_receiver(...)``: one-time receiver-side initialization
- ``sync(...)``: send the current version of model weights
- ``apply(...)``: receive and apply weights, then return the applied ``version``

This means rollout code does not need to care whether the underlying mechanism
is patch-based sync or bucket-based sync. After initialization, it only needs to
call ``apply(...)`` through the common interface.

The implementation lives in ``rlinf/hybrid_engines/weight_syncer/``, while the
YAML entry point remains the same independent ``weight_syncer`` Hydra config
group.


Supported Sync Strategies
------------------------------

RLinf currently provides two strategies:

``patch``
  Incremental synchronization. The sender maintains a snapshot and only sends
  the changed positions and values relative to that snapshot.

``bucket``
  Full synchronization. The complete ``state_dict`` is split into buckets and
  sent sequentially.


State Dict Device Requirements
------------------------------

Different ``weight_syncer`` implementations have different requirements for
the sender-side ``state_dict`` device:

``bucket``
  There is no special device requirement for the sender-side ``state_dict``
  passed to ``sync(...)``. Parameters can live on either CPU or GPU, and the
  bucket syncer stages them according to ``bucket_device`` and ``bucket_dtype``
  before sending. On the receiver side, ``apply(...)`` uses ``load_state_dict``;
  PyTorch copies input tensors to the target model parameter device and casts
  them to the target parameter dtype.

``patch``
  The sender-side ``state_dict`` passed to ``init_sender(...)`` and ``sync(...)``
  is expected to be on GPU. Even when ``snapshot_device: cpu`` is used, only the
  sender-side snapshot stays on CPU; difference comparison, ``nonzero``, and
  new-value gathering still run on GPU. Providing a CPU sender ``state_dict``
  would turn patch construction into CPU scanning, which cannot use the current
  optimized path and is not the intended patch-mode design.

On the receiver side, ``apply(...)`` moves patch payload tensors to the target
model parameter device before writing them. The receiver model must still match
the metadata and initial-weight assumptions required by patch mode.


Recommendation
------------------------------

For the mainstream embodied VLA configurations in RLinf, ``patch`` is the
recommended default because:

- Weight updates after each actor step are often highly sparse.
- Actor and rollout usually start from the same checkpoint or model path.
- Patch mode often sends far less data than full sync.

But there is one critical caveat:

.. warning::

   The current ``patch`` mode is **not an independent full weight snapshot**.
   It sends deltas relative to the sender-side snapshot, which means actor and
   rollout are expected to start from the **same initial weights** before patch
   synchronization begins.

   If you cannot guarantee that, you should first perform a full sync or use
   ``bucket`` directly.


How To Enable It In YAML
------------------------------

``weight_syncer`` is exposed as an independent Hydra config group.

In embodied YAMLs, the recommended usage looks like this:

.. code-block:: yaml

   defaults:
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer

The corresponding config files are:

- ``examples/embodiment/config/weight_syncer/patch_syncer.yaml``
- ``examples/embodiment/config/weight_syncer/bucket_syncer.yaml``


Patch Mode
------------------------------

A typical patch configuration looks like this:

.. code-block:: yaml

   weight_syncer:
     type: patch
     patch:
       snapshot_device: cpu
       transport_device: cpu
       delta_encoding: true
       compression: none

The fields mean:

``type``
  Fixed to ``patch`` to enable incremental synchronization.

``patch.snapshot_device``
  Device where the snapshot is stored. It can be either ``cpu`` or ``cuda``.
  ``cpu`` is currently recommended as the default: it avoids keeping an
  additional model-sized snapshot in GPU memory, and after GPU-side comparison,
  asynchronous prefetching, and background snapshot flushing optimizations, its
  synchronization latency is already close to ``snapshot_device: cuda``. If GPU
  memory is very abundant, ``cuda`` remains the most direct low-latency path.

``patch.transport_device``
  Device used before sending the patch. The default can be ``cpu``. If you want
  GPU-side compression or GPU transport, this is typically ``cuda``.

``patch.delta_encoding``
  Whether to delta-encode COO coordinates. Enabled by default and recommended.

``patch.compression``
  Compression algorithm. Supported values currently include:

  - ``none``: no compression
  - ``nvcomp_lz4``: GPU-side lossless compression via nvCOMP


How Patch Mode Works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patch mode is roughly split into two stages:

1. One-time initialization

   - The receiver sends local model metadata in ``init_receiver(...)``.
   - The sender receives the metadata in ``init_sender(...)`` and creates its
     snapshot from it.

   The metadata currently includes:

   - a fixed parameter order ``ordered_keys``
   - the original shape of each tensor in ``original_shapes``
   - the dtype of each receiver-side tensor

   The receiver **does not store the sender-side snapshot**. It only stores the
   structural information needed to apply patches correctly to its local model.
   The sender-side snapshot uses the same dtype as the corresponding
   receiver-side weight, so mixed-precision models with both ``bfloat16`` and
   ``float32`` weights are handled correctly.

2. Per-sync update

   - The sender compares the current ``state_dict`` with the snapshot.
   - The changed entries are packed into a patch and sent.
   - The receiver applies those changes directly to local model parameters.


CPU Snapshot Optimization Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``snapshot_device: cpu``, the sender-side snapshot stays on CPU while the
current ``state_dict`` remains on GPU. To avoid moving the patch-building hot
path back to CPU, RLinf applies several optimizations for this case:

- The CPU snapshot is stored in pinned memory to enable asynchronous CPU-GPU copies.
- Before comparing each tensor, the corresponding CPU snapshot tensor is
  asynchronously prefetched to the GPU where the state tensor lives.
- Snapshot prefetch uses a dedicated CUDA copy stream so it can overlap as much
  as possible with GPU-side comparison of other tensors.
- Difference comparison, ``nonzero``, and new-value gathering all run on GPU,
  avoiding CPU-side element scanning.
- The ``rows``, ``cols``, and ``values`` needed by the patch are asynchronously
  copied into pinned CPU staging buffers, and ``torch.cuda.Event`` is used to
  mark when those copies complete.
- After patch construction finishes, the sender can return immediately and
  continue with the following transfer steps; CPU snapshot flushing is handled
  by a background thread.
- Before the next patch construction starts, RLinf waits for the previous
  background flush to finish, which preserves snapshot consistency.

Therefore, ``snapshot_device: cpu`` no longer means "compare on CPU". The
effective path is:

.. code-block:: text

   CPU snapshot -> GPU prefetch -> GPU compare/nonzero/gather
   -> pinned CPU staging -> background snapshot flush

This trades a small amount of extra asynchronous copy and background flushing
for much lower GPU memory usage. In current embodied VLA training
configurations, CPU snapshot synchronization latency can already be close to GPU
snapshot latency. When GPU memory is tight, ``snapshot_device: cpu`` is usually
the safer default.


Patch Data Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current patch representation is based on flattened tensor index information.
The main fields are:

- ``ordinals``: which tensor changed
- ``nnz_per_tensor``: number of changed entries in that tensor
- ``rows`` / ``cols``: 2D coordinates of changed positions
- ``values``: the new values at those positions
- ``version``: the sync version carried by this patch

These 2D coordinates come from an internal 2D COO-style view. Tensors are
interpreted as:

- scalars: ``(1, 1)``
- 1D tensors: ``(1, N)``
- 2D tensors: unchanged
- 3D and higher: ``(shape[0], prod(shape[1:]))``

This makes it possible to express tensors of different ranks with one uniform
patch format.


Delta Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``delta_encoding=true``, ``rows`` and ``cols`` do not send absolute
coordinates directly. Instead, they send delta-encoded coordinates:

- ``rows`` stores increments between adjacent row coordinates
- if two adjacent entries stay on the same row, ``cols`` stores column deltas
- when switching to a new row, ``cols`` stores the absolute starting column of
  that row

This helps because:

- index values usually become smaller
- they can often be downscaled to tighter dtypes such as ``uint8`` or ``int32``
- downstream compression becomes more effective


Compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patch compression only applies to the patch payload itself, not the full model
weights.

RLinf currently provides these patch compressors:

- ``none``: send patch tensors directly
- ``nvcomp_lz4``: apply GPU-side lossless compression separately to
  ``rows``, ``cols``, and ``values``

If you enable ``nvcomp_lz4``, you need:

- ``transport_device: cuda``
- ``nvidia-nvcomp-cu12`` installed in the runtime environment

If you install embodied environments through
``bash requirements/install.sh embodied ...``, this dependency is installed as
part of the common embodied requirements.


When Patch Mode Is Not A Good Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patch mode is not a good default in the following cases:

- actor and rollout do not start from the same initial weights
- you need an explicit bootstrap or full sync
- updates are not sparse enough for patching to pay off
- you want the most conservative synchronization strategy first when debugging
  correctness issues


Bucket Mode
------------------------------

A typical bucket configuration looks like this:

.. code-block:: yaml

   weight_syncer:
     type: bucket
     bucket:
       bucket_size: 536870912
       bucket_dtype: null
       bucket_device: cuda
       is_agent: false
       load_instant: true

The fields mean:

``type``
  Fixed to ``bucket`` to enable full bucket-based synchronization.

``bucket.bucket_size``
  Maximum size in bytes of each bucket.

``bucket.bucket_dtype``
  Dtype used when sending bucket payloads. If set to ``null``, each tensor keeps
  its original dtype. If set to ``bfloat16``, ``float16``, or ``float32``, only
  floating-point tensors are converted; non-floating buffers such as ``int`` and
  ``bool`` keep their original dtype to avoid corrupting model state.

``bucket.bucket_device``
  Device where bucket tensors are staged, typically ``cuda``.

``bucket.is_agent``
  A compatibility switch for some agent-side naming behavior. For embodied
  training, this is usually kept as ``false``.

``bucket.load_instant``
  Whether to call ``load_state_dict`` immediately after each bucket is received.


Characteristics Of Bucket Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bucket mode splits the full ``state_dict`` into multiple chunks and sends them
in order. Its main characteristics are:

- Advantage: simple semantics, suitable for full sync and bootstrapping
- Advantage: does not depend on a sender-side snapshot and does not assume sparse updates
- Disadvantage: typically much more data is transferred than in patch mode

If ``load_instant=true``, each bucket is loaded immediately after it arrives.
If ``load_instant=false``, the receiver buffers buckets first and loads them at
the end.


Behavior In Async Training
------------------------------

In async embodied training, if ``actor.sync_weight_no_wait=true`` is enabled,
rollout-side weight receiving and applying are handled in a background
``asyncio`` task.

This means:

- rollout does not necessarily block immediately when actor requests a sync
- new weights only become effective after the background task completes
- there may be a small delay between "sync requested" and "sync applied"

In this async path, version propagation matters more. ``WeightSyncer.apply(...)``
returns the version that was actually applied on rollout, and rollout updates
its internal version state from that result.


Performance Suggestions
------------------------------

If your priority is to reduce synchronization overhead, a good tuning order is:

1. Start with ``patch`` and confirm the initial weights are identical.
2. Prefer ``snapshot_device: cpu`` by default, which avoids an extra
   model-sized GPU-memory snapshot while providing synchronization latency close
   to GPU snapshot.
3. Keep ``delta_encoding: true``.
4. First get the workflow stable with ``compression: none``, then evaluate
   whether ``nvcomp_lz4`` is worth enabling.
5. If GPU memory is very abundant and you are pursuing the lowest possible sync
   latency, evaluate ``snapshot_device: cuda``.
6. If you truly need full sync or are debugging correctness, switch back to
   ``bucket``.

Patch mode keeps an extra sender-side snapshot. When ``snapshot_device: cuda``,
that snapshot consumes GPU memory roughly equal to the number of model
parameters multiplied by the byte size of the corresponding receiver-side
weight dtype. For large models or memory-tight setups, reserve enough GPU memory
for this snapshot to avoid OOM during training or synchronization.

When ``snapshot_device: cpu``, this snapshot does not consume GPU memory, but it
does consume one model-sized CPU pinned-memory copy. Its size is also roughly the
number of model parameters multiplied by the byte size of the corresponding
receiver-side weight dtype.
In this mode, patch comparison still runs on GPU, and CPU snapshot overhead is
reduced through prefetching, event synchronization, and background flushing. For
memory-tight training jobs, this is the currently recommended configuration. In
addition, ``nvcomp_lz4`` requires ``transport_device`` to be ``cuda``.


Limitations And Caveats
------------------------------

The current implementation has several constraints to keep in mind:

- ``patch`` assumes actor and rollout start from the same initial weights
- ``patch`` is currently designed primarily for the embodied HuggingFace rollout path
- high-rank tensors are converted to a 2D view internally; if trailing dimensions
  cannot be flattened as a view, patch mode will raise an error
- compression settings in this document refer to patch payload compression, not
  compression of the model weights themselves
- if your immediate goal is "make weight sync correct first", use ``bucket``;
  if your goal is "make weight sync fast after correctness is verified", use
  ``patch``


Recommended Usage Pattern
------------------------------

A simple rule of thumb is:

- default training: start with ``patch``
- bootstrap or debugging: start with ``bucket``
- high sparsity and aggressive optimization: ``patch + delta_encoding + optional nvcomp``

If you are not fully sure the patch assumptions hold for your pipeline, the
safest approach is:

- first ensure actor and rollout load the same model path or checkpoint
- verify correctness with ``bucket`` first
- then switch to ``patch`` for performance optimization
