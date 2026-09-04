Use Channel for Communication
=============================

The Channel module provides a distributed producer-consumer queue for moving
data between workers. One or more producers can ``put`` items into a named
``Channel``, while one or more consumers ``get`` those items independently of
the producers' execution schedule.

The default shared queue supports synchronous and asynchronous operations,
logical queues selected by key, bounded capacity, and weighted batches. Most
applications can use that behavior directly. Add a ``Collector`` only when data
must be transformed before enqueueing, and a ``Dispatcher`` only when outputs
must be assigned to specific consumers.

Create and Share a Channel
--------------------------

Create a channel outside the workers, then pass it to WorkerGroup methods that
produce or consume data:

.. code-block:: python

   from rlinf.scheduler import Channel

   sample_channel = Channel.create(name="Samples", maxsize=64)
   producer_group.produce(sample_channel)
   consumer_group.consume(sample_channel)

``produce`` and ``consume`` are application-defined worker methods in this
example. Passing the channel into a worker binds communication to that worker.
Code already running inside a worker may instead call
``self.create_channel(...)`` or ``self.connect_channel("Samples")``.

Put, Get, and Batch Items
-------------------------

``put`` sends one serializable item. ``get`` removes the next item from the
same logical queue:

.. code-block:: python

   sample_channel.put(sample, key="train", weight=num_tokens)
   sample = sample_channel.get(key="train")

Each key identifies an independent FIFO queue. The optional weight records an
item's cost or size; it does not change queue order. Use ``get_batch`` to remove
as many leading items as fit within a target total weight:

.. code-block:: python

   batch = sample_channel.get_batch(
       target_weight=global_batch_size,
       key="train",
   )

This is useful when samples have different lengths or processing costs. For
ordinary single-item consumption, leave the weight at its default value.

Use Asynchronous Operations and Backpressure
--------------------------------------------

``put`` and ``get`` block by default. Set ``async_op=True`` to receive a work
handle and overlap communication with other work:

.. code-block:: python

   put_work = sample_channel.put(sample, async_op=True)
   # Do independent work here.
   put_work.wait()

   get_work = sample_channel.get(async_op=True)
   sample = get_work.wait()

In an asyncio function, use ``await work.async_wait()`` instead. A positive
``maxsize`` applies backpressure by blocking ``put`` while its destination queue
is full. ``put_nowait`` and ``get_nowait`` provide non-blocking queue semantics.

Hook Execution Model
--------------------

Every ``put`` follows one ordered path on the ``ChannelWorker``:

.. code-block:: text

   producer.put(item, key)
            |
            v
   Collector.collect(item, key)
            |
            |  zero or more (output_key, output_item) pairs
            v
   Dispatcher.route(output_item, output_key)
            |
            v
   shared queue or one consumer's private queue

The two hooks solve different problems:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Hook
     - Input
     - Decision
   * - ``Collector``
     - One item passed to ``put``
     - Drop, transform, merge, split, or re-key it.
   * - ``Dispatcher``
     - Each item emitted by the Collector
     - Keep it shared or assign it to one consumer.

Both hooks are synchronous and stateful. ``setup(ctx)`` runs once before
traffic starts. ``collect()`` and ``route()`` then run serially on the channel
worker, so their state does not need locks.

Register a Collector and Dispatcher
-----------------------------------

Define hooks in an importable module, for example
``my_project/channel_hooks.py``:

.. code-block:: python

   from rlinf.scheduler import (
       Collector,
       Dispatcher,
       register_collector,
       register_dispatcher,
   )


   @register_collector("flatten_batch")
   class FlattenBatchCollector(Collector):
       """Emit each sample in an incoming batch as one queue item."""

       def collect(self, item, key):
           for sample in item:
               yield key, sample


   @register_dispatcher("even")
   class EvenDispatcher(Dispatcher):
       """Assign items to consumers in a fixed rotation."""

       def setup(self, ctx):
           self.consumers = sorted(ctx.consumers)
           self.next_consumer = 0

       def route(self, item, key):
           if not self.consumers:
               return None
           consumer = self.consumers[
               self.next_consumer % len(self.consumers)
           ]
           self.next_consumer += 1
           return consumer

The decorators register each class under a lowercase name. ``Channel.create``
also accepts a hook instance, a hook class, or a ``"module:ClassName"`` path.
Passing the class directly is the simplest choice for application-local hooks:

.. code-block:: python

   from my_project.channel_hooks import EvenDispatcher, FlattenBatchCollector
   from rlinf.scheduler import Channel

   sample_channel = Channel.create(
       "Samples",
       collector=FlattenBatchCollector,
       dispatcher=EvenDispatcher,
       producers=[rollout_group],
       consumers=[actor_group],
   )

``rollout_group`` and ``actor_group`` are launched WorkerGroups. Channel setup
expands ``actor_group`` into stable consumer ids such as ``actor:0``,
``actor:1``, and ``actor:2`` and exposes them through ``ctx.consumers``.

If the registration module is imported in the ChannelWorker process, select a
hook by its registered name instead:

.. code-block:: python

   sample_channel = Channel.create(
       "Samples",
       collector="flatten_batch",
       dispatcher="even",
       producers=[rollout_group],
       consumers=[actor_group],
   )

Framework-integrated hooks normally use registered names. External modules can
always use ``"my_project.channel_hooks:FlattenBatchCollector"`` or pass the
class directly.

Follow One Batch Through the Hooks
----------------------------------

Assume one producer sends six samples:

.. code-block:: python

   sample_channel.put(["s0", "s1", "s2", "s3", "s4", "s5"])

The Collector converts one input batch into six queue items. The Dispatcher
then rotates over three consumers:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Consumer
     - First Assignment
     - Second Assignment
   * - ``actor:0``
     - ``s0``
     - ``s3``
   * - ``actor:1``
     - ``s1``
     - ``s4``
   * - ``actor:2``
     - ``s2``
     - ``s5``

Each Actor calls ``sample_channel.get()`` with the same queue key. The Channel
identifies the calling worker and reads its private queue. Assignment counts
differ by at most one when the number of outputs is not divisible by the
consumer count.

Choose the Queue Behavior
-------------------------

Use the defaults until data transformation or consumer ownership is required:

.. list-table::
   :header-rows: 1
   :widths: 29 31 40

   * - Dispatcher
     - Queue Ownership
     - Use It When
   * - ``None`` or ``"shared"``
     - One shared queue per key
     - Any consumer may process any item.
   * - ``"round_robin"``
     - One private queue per consumer
     - Assignment must be deterministic and even.
   * - ``"least_loaded"``
     - One private queue per consumer
     - Producers should balance total assignments.
   * - ``"least_loaded_stealing"``
     - Private queues with stealing
     - Slow consumers must not hold idle peers back.

A Dispatcher that returns ``None`` leaves an item in the shared queue. Returning
a consumer id assigns the item before any consumer calls ``get``. Override
``rebalance()`` only when a blocking consumer may steal from a peer.

Combine Hooks with Keys and Weights
-----------------------------------

The core key and weight behavior remains unchanged when hooks are enabled. A
Collector may change the input key, and a Dispatcher routes independently
within each output key. Routing happens before weighted items enter the shared
or private queue used by ``get`` and ``get_batch``.

For exact method signatures, see :doc:`../reference/api/channel`. For a
production Collector that joins embodied rollout data, continue with
:doc:`trajectory_collector`.
