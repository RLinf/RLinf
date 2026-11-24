Worker and WorkerGroup
======================

``Worker`` and ``WorkerGroup`` are RLinf's core execution abstractions. Every RL role—actor, rollout, environment, reward, critic, or agent—subclasses ``Worker`` so it can be placed onto any node or accelerator and orchestrated uniformly as a group. For full API signatures, see :doc:`../reference/api/worker`.

Worker
------

A ``Worker`` represents a single remote process or execution unit. By inheriting from ``Worker``, a component gains:

- Remote execution on any node and accelerator assigned by ``Cluster``.
- Peer-to-peer and collective communication with other workers across groups.
- Automatic distributed environment setup (``MASTER_ADDR``, ``MASTER_PORT``, ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``).

WorkerTimer
~~~~~~~~~~~

RLinf includes a built-in timer to profile worker methods without external instrumentation.

**Context manager usage**:

.. code-block:: python

   class MyWorker(Worker):
       def run_training(self, batch):
           with self.worker_timer("run_training"):
               return self.train_step(batch)

       def train_step(self, batch): ...

**Decorator usage**:

.. code-block:: python

   class MyWorker(Worker):
       @Worker.timer("run_training")
       def run_training(self, batch):
           return self.train_step(batch)

       @Worker.timer("train_step")
       def train_step(self, batch): ...

**Retrieving timing metrics**:

.. code-block:: python

   handle = worker_group.run_training(...)
   result = handle.wait()
   timing = handle.consume_durations()  # dict of tag -> duration

Timer tags are aggregated across ranks in the worker group (reducing with ``max`` by default), and both outer and nested tags appear in the returned dictionary.

WorkerInfo
~~~~~~~~~~

``WorkerInfo`` captures runtime metadata for a worker so peers can exchange placement details without extra remote lookups:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``address``
     - Hierarchical ``WorkerAddress`` of the worker.
   * - ``rank``
     - Rank of the worker within its group.
   * - ``node_id``
     - Identifier of the node hosting the worker.
   * - ``gpu_id``
     - Accelerator ID assigned to the worker.
   * - ``node_ip``
     - IP address of the hosting node.
   * - ``available_gpus``
     - List of accelerator device IDs visible to the worker.

WorkerAddress
~~~~~~~~~~~~~

``WorkerAddress`` gives every worker a hierarchical name made of its root group name and rank path—for example, ``"ActorGroup:0"`` and ``"ActorGroup:1"``, or ``"ActorGroup:0:0"`` when a worker spawns child sub-workers. Helpers such as ``get_name()``, ``get_parent_rank()``, ``get_parent_address()``, and ``get_child_address(rank)`` let any worker address peers across groups.

Communication Methods
~~~~~~~~~~~~~~~~~~~~~

A ``Worker`` exposes two communication patterns:

- **Point-to-point messaging**: ``send(object, dst_group_name, dst_rank, async_op=False)`` and ``recv(src_group_name, src_rank, async_op=False)`` transfer arbitrary Python objects or tensors, while ``send_tensor`` and ``recv_tensor`` transfer pre-allocated tensors without extra metadata overhead. Actual transport is delegated to ``CollectiveGroup`` (see :ref:`collectivegroup_p2p`).
- **Channels**: ``create_channel(name, node_id=0, maxsize=0)`` and ``connect_channel(name)`` connect workers to FIFO queues for producer-consumer data flow using ``put()``, ``get()``, and ``get_batch()`` (see :doc:`channel`).

WorkerGroup
-----------

``WorkerGroup`` launches and manages a collection of workers of the same class:

- **Group Creation**: ``MyWorker.create_group(...).launch(cluster, placement_strategy=placement)`` resolves how many workers to spawn and where each runs (see :doc:`placement`), injects distributed environment variables, and delegates actor creation to ``Cluster``.
- **Collective Method Calls**: ``WorkerGroup`` mirrors every public method of the underlying ``Worker`` class. Calling a method on the group invokes it in parallel across all ranks and returns a handle whose ``.wait()`` gathers results. Use ``execute_on(ranks)`` when only a subset of ranks should run the next call.

Next Steps
----------

- :doc:`../reference/api/worker`: full ``Worker`` and ``WorkerGroup`` API reference.
- :doc:`placement`: placement strategies mapping worker groups to hardware.
- :doc:`channel`: queue-based communication between worker groups.
