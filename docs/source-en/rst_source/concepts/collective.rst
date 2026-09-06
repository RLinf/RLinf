Adaptive P2P Communication
===================================

This component provides point-to-point (P2P) data transfer between workers with **strict ordering** and **async handles**, on top of PyTorch ``torch.distributed``.
It consists of two public-facing classes:

- **Collective**: per-worker singleton that creates/caches communication groups.
- **CollectiveGroup**: a two-rank communication group that implements P2P send/recv for tensors, lists/dicts of tensors, and picklable Python objects.


Group Creation & Caching
----------------------------------------

The ``Collective`` class is instantiated on each worker (as a singleton per worker) and is responsible for creating and caching ``CollectiveGroup`` instances.
When two workers or a set of workers need to communicate, a collective group must be established that includes all participants.
The typical usage in this framework is to form groups for point-to-point communication by
``Collective.create_collective_group(worker_addresses, group_name=None)``,
which either retrieves an existing ``CollectiveGroup`` for the given set of worker addresses or creates a new one.


.. _collectivegroup_p2p:

P2P Communication
-------------------------------------

A ``CollectiveGroup`` is the core abstraction in RLinf for managing point-to-point communication between two workers.
It determines the local rank (0 or 1) from ``group_info`` and **lazily initializes** communication process groups on first use.
Internally, separate **send** and **receive** process groups are created for both GPU (NCCL) and CPU (Gloo), forming dedicated one-way channels; in a two-worker setup, a carefully configured broadcast is equivalent to a send/receive.
Initialization uses a TCP rendezvous to coordinate port allocation and synchronization, ensuring both sides are ready.
Each direction maintains a work queue backed by a dedicated CUDA stream, strictly preserving the order of send/recv operations and preventing message interleaving.

With the process groups in place, ``CollectiveGroup`` can perform communications. The main APIs are:

- **Send**: ``send(obj, async_op=False)`` sends an object (tensor, list of tensors, dict of tensors, or arbitrary picklable object) to the single other peer in the group.
  This method first sends a small **header** indicating the object type so that the receiver can interpret the payload.

- **Recv**: ``recv(async_op=False)`` receives an object from the peer.
  It first receives the type code (CPU/Gloo), then dispatches to the appropriate receiver to reconstruct the object.

- **Direct Tensor Send/Recv**: ``send_tensor(tensor, async_op=False)`` and ``recv_tensor(tensor, async_op=False)`` are optimized for the case where only one tensor is being transferred and the receiver already has an allocated tensor buffer.
  These avoid the extra round-trip of sending metadata.

.. note::
   All tensors must be contiguous; non-contiguity raises a helpful error. CPU
   and accelerator tensors may coexist in one supported tensor container and
   are partitioned onto their respective communication paths.

.. warning::
   ``send_tensor`` **must** be paired with ``recv_tensor`` (and vice versa). Do not mix them with the generic ``send``/``recv`` for the same message.


Tensor Compression
---------------------------------

Enable lossless CPU tensor compression when network transfer time outweighs the
extra CPU work. Compression is optional and job-wide: the driver validates one
configuration and propagates it to every Worker.

Scope
~~~~~

Compression applies to CPU tensors carried by the optimized generic
``Worker.send``/``Worker.recv`` paths: a tensor, a tensor list or tuple, a
tensor-valued dictionary, or tensor fields extracted from a dataclass. For a
mixed CPU/accelerator container, only its CPU tensors are candidates. A Channel
that communicates through Worker send/recv inherits the same behavior.

Arbitrary pickled Python objects, accelerator tensors, ``broadcast``, and direct
``send_tensor``/``recv_tensor`` calls are not compressed. These boundaries are
deliberate:

- The arbitrary-object path is intended for control data and metadata. Its
  serialized payloads are usually small, so another codec pass, buffer, and
  compression header would normally cost more than the bytes saved. Large
  tensor-bearing data should use the tensor list, dictionary, or dataclass paths
  so tensor storage does not pass through pickle and eligible CPU tensors can be
  compressed directly.
- Accelerator tensors normally use high-throughput device communication such as
  NCCL-like collectives or same-device IPC. CPU compression would require device
  synchronization and device-to-host and host-to-device copies, disrupting the
  accelerator communication pipeline; common floating-point model tensors may
  also compress poorly. The extra work can therefore produce a net slowdown.
- Direct ``send_tensor``/``recv_tensor`` deliberately avoids metadata exchange
  because the receiver already owns the destination. Compression would require
  adding a variable wire size and framing metadata, which would break that fast
  path's contract.
- ``broadcast`` has separate multi-receiver and topology-aware transfer paths.
  This change is limited to point-to-point payload ownership and fallback, so it
  does not alter broadcast scheduling or buffer lifetime.

Tensor-list elements also remain separate wire payloads; this feature reduces
their byte counts but does not coalesce them.

Choose a Codec
~~~~~~~~~~~~~~

Both codecs are lossless, operate on the raw bytes of a contiguous CPU tensor,
and restore the original dtype and shape byte-for-byte. They write directly
between tensors and preallocated ``torch.uint8`` buffers without creating an
intermediate Python ``bytes`` object.

.. list-table:: Codec trade-offs
   :header-rows: 1
   :widths: 14 30 24 32

   * - Codec
     - Characteristics
     - Codec parameters
     - Choose it when
   * - ``lz4``
     - Prioritizes compression and decompression speed with relatively low CPU
       cost, usually at a lower compression ratio than Zstd.
     - ``acceleration`` controls LZ4 fast compression. Higher values favor speed
       and may reduce the compression ratio.
     - CPU time matters or the link is only moderately bandwidth-bound. Start
       here.
   * - ``zstd``
     - Usually reduces wire bytes more than LZ4, with higher compression and
       decompression cost.
     - ``level`` controls the compression ratio/CPU trade-off;
       ``max_inflight`` bounds concurrent reusable contexts.
     - The network is slow enough that reducing bytes dominates codec time.

Measure with representative payloads before choosing. Already compressed or
high-entropy tensors may not shrink, while dense tensors containing many zeros
or repeated values can benefit substantially. LZ4 also has a per-tensor input
limit; unsupported tensor sizes automatically follow the raw path.

Configure Compression
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   cluster:
     collective:
       tensor_buffer_pool:
         max_bytes: 2147483648
       tensor_compression:
         enabled: true
         codec: lz4
         min_bytes: 16384
         excluded_dtypes: [float32]
         acceleration: 1

For Zstd, select it with ``codec`` and pass only Zstd parameters:

.. code-block:: yaml

   cluster:
     collective:
       tensor_compression:
         enabled: true
         codec: zstd
         min_bytes: 16384
         excluded_dtypes: [float32]
         level: 1
         max_inflight: 4

Omit ``tensor_compression`` or set ``enabled: false`` to use the original wire
path. The compression options and defaults are:

.. list-table:: Compression options
   :header-rows: 1
   :widths: 20 16 64

   * - Option
     - Default
     - Meaning
   * - ``enabled``
     - ``true``
     - Enables compression when the ``tensor_compression`` block is present.
   * - ``codec``
     - required
     - Selects the codec config that owns the rest of the block: ``lz4`` or
       ``zstd``.
   * - ``min_bytes``
     - ``16384``
     - Skips tensors smaller than this raw byte count.
   * - ``excluded_dtypes``
     - ``[float32]``
     - Skips codec attempts for tensors whose dtype is listed. Set it to ``[]``
       to make every dtype eligible.
   * - codec parameters
     - codec-specific
     - Declared by the selected codec and given in the same block. LZ4 accepts
       ``acceleration`` (default ``1``). Zstd accepts ``level`` (default ``1``)
       and ``max_inflight`` (default ``4``).

``codec`` selects a codec configuration class, and only that class's parameters
are accepted alongside it. The contract is exact:

.. list-table:: Codec parameters
   :header-rows: 1
   :widths: 16 20 14 50

   * - Codec
     - Parameter
     - Default
     - Validation and behavior
   * - ``lz4``
     - ``acceleration``
     - ``1``
     - Must be at least ``1``. Higher values ask LZ4 to spend less time finding
       matches, usually trading compression ratio for speed. LZ4 is stateless,
       so the Worker shares one codec without an acquisition limit.
   * - ``zstd``
     - ``level``
     - ``1``
     - Must be at least ``1``. Higher levels generally spend more CPU time to
       seek a better compression ratio.
   * - ``zstd``
     - ``max_inflight``
     - ``4``
     - Must be at least ``1``. It bounds the reusable Zstd codecs shared by
       compression and decompression within each Worker. A sender falls back to
       raw transfer rather than waiting for a busy codec; a receiver waits for a
       codec after an already-compressed payload arrives.

A codec only accepts its own parameters: LZ4 rejects ``max_inflight`` while
Zstd rejects ``acceleration``, and both are reported like any other unknown
``cluster`` key. The driver validates the block once as part of the cluster
configuration, which then reaches every Worker; wire metadata therefore only
has to identify the codec, since each receiver already holds the same job-wide
parameters.

``tensor_buffer_pool`` is independent of ``tensor_compression``. Its
``max_bytes`` option limits the combined active and cached CPU buffer capacity
per Worker and defaults to 2 GiB. Configuring compression without this block
automatically supplies the default pool.

Runtime and Fallback
~~~~~~~~~~~~~~~~~~~~

Each Worker probes the configured system codec library while loading its
job-wide configuration, so a missing dependency fails during Worker startup.
Native codec contexts and buffers remain lazy: each Worker creates an empty
``TensorBufferPool`` while loading its configuration, then creates one
``TensorCodecProvider`` when an eligible tensor first needs compression. Both
resources are shared across all of the Worker's ``CollectiveGroup`` instances.
LZ4 shares one stateless codec without a lock or slot limit. Zstd leases an
exclusive codec from one bounded LIFO queue shared by compression and
decompression. A send uses them as follows:

1. It obtains the provider's codec. LZ4 access cannot be saturated. Zstd does
   not wait for a busy codec queue; a saturated queue keeps the transfer raw.
2. It visits CPU tensors in their original order, filters them by ``min_bytes``
   and ``excluded_dtypes``, and tries to lease a buffer for each eligible tensor.
   A tensor stays raw when its codec bound is unsupported or no buffer fits
   within the budget.
3. It keeps the compressed result only when it is smaller than the original.
   Otherwise, the tensor stays raw and that buffer is discarded rather than
   cached.
4. It sends per-tensor compression sizes in the existing metadata. The receiver
   restores compressed tensors directly into their preallocated destinations.
5. It releases the codec immediately after compression. Compressed payload
   buffers remain leased until their synchronous payload sends finish, then
   return to the Worker buffer pool. The receiver acquires a decoder only after
   the compressed wire payload has arrived and releases it after decompression.

When ``cluster.net_emulation`` is also enabled, bandwidth is charged against the
actual compressed CPU tensor bytes while preserving the original payload's
metadata estimate. Raw and ineligible tensors keep their original accounting.

The buffer pool indexes idle buffers by capacity and finds the smallest cached
size that fits. It reuses that buffer when its capacity is at most twice the
requested size, or when allocating an exact-size buffer would exceed the budget;
otherwise it allocates an exact-size buffer. It maintains separate lists for
repeated sizes and tracks active plus cached capacity against ``max_bytes``. When
a new allocation needs room, it evicts buffers starting with the largest idle
size bucket. Buffer acquisition never waits; an unavailable buffer therefore
preserves baseline behavior for that tensor.

The common dependency installation installs the LZ4 and Zstandard system
libraries required by these codecs. Ensure the same RLinf version and its
compression dependencies are available on all worker nodes.


Asynchronous API 
---------------------------------

All P2P APIs support asynchronous operation and return awaitable **work handles** when ``async_op=True``. Internally, we expose a small hierarchy:

- ``AsyncWork``: abstract base with ``wait()``, ``async_wait()``, ``then(func, *args, **kwargs)``, ``done()``, and chaining helpers (``get_next_work()``, ``get_last_work()``).
- ``AsyncFuncWork``: executes a Python callback when its predecessor completes, records a CUDA event, and can be chained via ``then``. If the callback returns another ``AsyncWork``, completion is deferred until the **last** work in that chain finishes.
- ``AsyncCollWork``: wraps a ``torch.distributed`` work (e.g., broadcast) into our awaitable interface. It also supports ``then`` (single underlying work).
- ``AsyncChannelWork``: wraps a ``ray.ObjectRef`` as an awaitable (for channel RPCs).

Key properties:

* **Waiting:** ``wait()`` is blocking ; ``async_wait()`` is ``asyncio``-friendly. Both ensure the recorded CUDA event has completed before returning.
* **Chaining:** ``then`` schedules a follow-up callback.
* **Completion:** ``done()`` is a non-blocking query to check whether the underlying work finished.

Minimal examples:

.. code-block:: python

   # Async object send/recv with await
   send_work = group.send(obj, async_op=True)      # AsyncWork
   await send_work.async_wait()                    # non-blocking await

   recv_work = group.recv(async_op=True)           # AsyncWork
   obj = recv_work.wait()                          # blocking wait; returns received object

.. code-block:: python

   # Chaining a post-processing step
   def postprocess(buf):
       # e.g., move to CPU, cast, or notify another subsystem
       return None

   w = group.recv_tensor(tensor, async_op=True)    # receiver-side preallocated tensor
   w2 = w.then(postprocess)                        # AsyncFuncWork
   w2.wait()                                       # ensure postprocess finished

Summary
--------------

In summary, the **collective** component provides the engine for P2P data transfer between workers. It abstracts away the details of using PyTorch's distributed backends, managing multiple process groups to simulate send/receive, and optimizing for GPU transfers. 
Users of the framework typically invoke these via the `Worker.send/recv` or channel operations, rather than calling `CollectiveGroup` directly.
