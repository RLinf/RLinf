Powerful P2P Communication
===================================

This component provides point-to-point (P2P) data transfer between workers with **strict ordering** and **async handles**, on top of PyTorch ``torch.distributed``.
It consists of two public-facing classes:

- **Collective**: per-worker singleton that creates/caches communication groups.
- **CollectiveGroup**: a two-rank communication group that implements P2P send/recv for tensors, lists/dicts of tensors, and picklable Python objects.


Group Creation & Caching
----------------------------------------

The ``Collective`` class is instantiated on each task (as a singleton per task) and is responsible for creating and caching ``CollectiveGroup`` instances.
When two tasks or a set of tasks need to communicate, a collective group must be established that includes all participants.
The typical usage in this framework is to form groups for point-to-point communication by
``Collective.create_collective_group(task_addresses, group_name=None)``,
which either retrieves an existing ``CollectiveGroup`` for the given set of task addresses or creates a new one.


.. _collectivegroup_p2p:

CollectiveGroup and P2P Communication
-------------------------------------

A ``CollectiveGroup`` is the core abstraction in RLinf for managing point-to-point communication between two tasks.
It determines the local rank (0 or 1) from ``group_info`` and **lazily initializes** communication process groups on first use.
Internally, separate **send** and **receive** process groups are created for both GPU (NCCL) and CPU (Gloo), forming dedicated one-way channels; in a two-task setup, a carefully configured broadcast is equivalent to a send/receive.
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
   All **CUDA tensors must be contiguous**; non-contiguity raises a helpful error.
   Mixing CPU and CUDA tensors in a single list/dict is disallowed.

.. warning::
   ``send_tensor`` **must** be paired with ``recv_tensor`` (and vice versa). Do not mix them with the generic ``send``/``recv`` for the same message.


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

In summary, the **collective** component provides the engine for P2P data transfer between tasks. It abstracts away the details of using PyTorch's distributed backends, managing multiple process groups to simulate send/receive, and optimizing for GPU transfers. 
Users of the framework typically invoke these via the `Worker.send/recv` or channel operations, rather than calling `CollectiveGroup` directly.

