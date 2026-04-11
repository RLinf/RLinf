Channel Queuing for Pipelining
===============================

The channel module provides a high-level **distributed producer–consumer queue** abstraction for workers to exchange data asynchronously.  
A ``Channel`` allows one or more producer workers to ``put`` items into a named queue and one or more consumer workers to ``get`` them, optionally accumulating **batches** based on per-item weights.

Channel Creation and Connection
--------------------------------

A new channel can be created using::

    Worker.create_channel(
        channel_name,
        node_id=0,
        maxsize=0
    )

This method:

- **Determines placement** — If ``group_affinity`` or ``group_rank_affinity`` are not specified, the channel is hosted in the current worker’s **group** and **rank** (same node and GPU).
- **Launches a dedicated channel actor** — Uses ``PackedPlacementStrategy`` to start a ``ChannelWorker`` (that actually holds the queue) with ``num_processes=1`` on the selected node/GPU.
- **Returns** a ``Channel`` object that wraps the actor. The channel actor’s address is ``channel_name:0``.

To connect to an existing channel from another worker, use::

    Worker.connect_channel(channel_name)

This looks up the channel actor in the Ray namespace and returns a ``Channel`` object bound to both the actor and the current worker.



Putting Items into the Channel
--------------------------------

Use ``channel.put(item, weight=0, key="default", async_op=False)`` to send data.

- The sending worker first transmits the ``item`` to the ``ChannelWorker`` that actually owns the target queue.  
- The ``ChannelWorker`` receives the data, wraps it as a ``WeightedItem`` (with the given ``weight``), and enqueues it into the specified queue.  
  If the queue has a size limit (``maxsize`` > 0) and is full, the enqueue will block until space becomes available.

Getting Items from the Channel
--------------------------------

Use ``channel.get(key="default", async_op=False)`` to retrieve data which is essentially the reverse of ``put``.  

- The ``ChannelWorker`` first dequeues an item from the specified queue.  
- It then sends this item to the worker that requested it, where it is returned to the caller.

Batch Retrieval
--------------------------------

Use ``channel.get_batch(batch_weight, key="default", async_op=False)`` to retrieve multiple items at once.

- The ``ChannelWorker`` repeatedly dequeues items from the queue, summing their weight values.  
- Once the accumulated weight reaches or exceeds ``batch_weight``, it stops.  
- All dequeued items are combined into a list and sent to the requesting worker in one message.

This feature is useful for dynamically forming batches of experiences or workers to process, where each item has a cost or size (the weight) and you want to process roughly uniform batch sizes. 

Load Balancing
--------------

During the Rollout stage, trajectories often vary significantly in length. If these are distributed to each data parallel (DP) training group without any design, it can result in severe load imbalance.

To address this issue, we implement a channel-based load balancing mechanism. Specifically, all generators in the generation stage sequentially ``put`` complete rollout trajectories into a shared ``rollout_output_queue``. 
Since the trajectories are inserted in temporal order, the sequence lengths in the ``rollout_output_queue`` tend to grow over time.

Using a round-robin strategy, we continuously ``get`` trajectories from the ``rollout_output_queue`` and assign them to each DP training group in turn. This method helps approximate balanced workload distribution across all training DP groups, ensuring better utilization and efficiency during training.



Example
--------

.. autoclass:: rlinf.scheduler.Channel
   :no-members:
   :no-index:
   :no-inherited-members:
   :exclude-members: __init__, __new__

Summary
--------------------------------

The `Channel` component offers a distributed producer-consumer queue for worker communication. 
It wraps the collective send/recv mechanism with an intuitive interface supporting priority and batching, 
enabling decoupled, asynchronous data flow—ideal for reinforcement learning scenarios with parallel data collection and batched consumption.






