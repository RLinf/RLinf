Worker-Based Programming Interface
===================================

In this section, we introduce the most fundamental components of the RLinf framework — **Worker** and **WorkerGroup** — 
the building blocks upon which the entire framework is constructed.


.. The **Worker** module defines the primary abstractions for distributed workers and their identification within a hierarchy of worker groups. 
.. It provides classes to represent worker information, address workers in a group structure, and manage the execution context of Workers.
.. This module also includes **WorkerGroup** for launching and managing groups of Workers collectively.


Worker 
------------

A **Worker** represents a single remote process or computational unit.  
By inheriting from :class:`Worker`, a worker or processor class gains the ability to:

- Run remotely across nodes in a distributed environment.

- Communicate with other workers in the cluster.

- Automatically receive essential environment variables such as ``MASTER_ADDR``, ``MASTER_PORT``, ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``.

These features enable the seamless creation of process groups and simplify distributed training setup.  
A Worker encapsulates the logic for an individual execution unit, making it easy to scale tasks across multiple GPUs and nodes.

WorkerTimer
~~~~~~~~~~~~~

RLinf provides a lightweight timer utility to profile worker functions.

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

The timer tags are aggregated across ranks in the worker group. By default,
``consume_durations()`` reduces with ``max``.

The returned timing map contains both ``run_training`` and ``train_step`` so you
can profile nested/child functions explicitly.


.. The `Worker` class encapsulates a remote or local unit of computation. In a Ray-based setup, each `Worker` typically runs as a Ray actor on a specific node and GPU. 

.. **Initialization**  
.. - **Environment Variables and Context**: When a `Worker` actor is created, Ray injects environment variables such as `RANK`, `WORLD_SIZE`, `NODE_ID`, `GPU_ID`, etc. The `Worker`’s constructor uses `_env_setup_before_init()` (called in `__new__`) to read these and initialize internal fields like `_rank` (the Worker’s index in its group), `_world_size` (total Workers in the group), `_Worker_name` (string form of its address), and `_Worker_address` (the `WorkerAddress` object). If the worker is not running under Ray (e.g. spawned as a subprocess), these variables might not be set by Ray, so in such cases the code handles initialization differently (passing explicit parent address and rank to the constructor). 

.. - **Ray Actor vs. Non-Actor Mode**: The `Worker` class can represent both Ray actors and regular processes. It uses an `_is_ray_actor` flag to differentiate. If running as a Ray actor, certain setup steps are performed: for example, registering signal handlers in the main thread (`_register_signal_handlers`) to log stack traces on crashes, and isolating the GPU visibility if required. The method `_setup_local_rank_world_size()` uses the provided `NODE_LOCAL_RANK` and `NODE_LOCAL_WORLD_SIZE` (or sets them) to configure local rank (which GPU index the worker should consider as device 0) and how many Workers share the node. `_setup_gpu_info()` determines which CUDA devices are available to this worker process (it queries `torch.cuda.device_count()` and collects each device’s UUID if accessible). This helps detect if two Workers share the same physical GPU, which is used later to optimize peer-to-peer communication.

.. - **Manager Proxy and Collective Initialization**: Each worker needs to register itself and participate in collective operations. `_init_ray_and_proxies()` is responsible for connecting to the global coordination services. It ensures that Ray is initialized (in case the worker process was forked outside of Ray’s direct control) in the correct **namespace** (the cluster’s namespace), and obtains a proxy to the `WorkerManager` (a global manager actor). The worker then calls `WorkerManager.register_Worker` via this proxy to record its existence and `WorkerInfo` in a central registry. It also creates a `Collective` instance (`self._collective = Collective(self)`) for orchestrating distributed communications involving this worker. After this point, the worker is ready to send and receive data to/from other Workers using collective groups.

.. - **Logging Setup**: The worker configures a logger with a name corresponding to its worker address (making it easier to trace messages per worker). The logging format includes the worker name, timestamps, and code location, which is helpful for debugging in a distributed context.

WorkerInfo 
~~~~~~~~~~~

The `WorkerInfo` dataclass **captures key properties of a worker** at runtime. 

.. It includes attributes like the Worker’s `address` (a `WorkerAddress`), its `rank` in the group, the `node_id` and `gpu_id` where it runs, the node’s IP (`node_ip`), and a list of `available_gpus` (identifiers of GPUs visible to that worker). 
.. This structure allows convenient local access to a Worker’s metadata without needing remote calls. For example, when setting up communication, Workers can share their `WorkerInfo` so peers know each other’s locations and GPU availability.

+---------------------+-----------------------------------------------+
| Attribute           | Description                                   |
+=====================+===============================================+
| ``address``         | WorkerAddress of the worker                   |
+---------------------+-----------------------------------------------+
| ``rank``            | Rank of the worker within its group           |
+---------------------+-----------------------------------------------+
| ``node_id``         | Identifier of the node hosting the worker     |
+---------------------+-----------------------------------------------+
| ``gpu_id``          | Identifier of the GPU assigned to the worker  |
+---------------------+-----------------------------------------------+
| ``node_ip``         | IP address of the node hosting the worker     |
+---------------------+-----------------------------------------------+
| ``available_gpus``  | List of CUDA device IDs available to worker   |
+---------------------+-----------------------------------------------+


WorkerAddress
~~~~~~~~~~~~~

The `WorkerAddress` class **provides a hierarchical naming scheme** for Workers. 
It combines a root group name with an ordered path of ranks to uniquely identify a worker in a worker group structure. 

For instance, a root worker group might be named `"Worker_group_MyWorker"`, and Workers within it have addresses like `"Worker_group_MyWorker:0"`, `"Worker_group_MyWorker:1"`, etc. 
If those Workers spawn their own sub-Workers, additional ranks are appended (e.g. `"Worker_group_MyWorker:0:0"` for a child of rank 0). 
The `WorkerAddress` supports operations to navigate this hierarchy: one can get a string name via `get_name()`, retrieve the parent’s rank or address (`get_parent_rank()`, `get_parent_address()`), or derive a child’s address (`get_child_address(rank)`). 

This address system is crucial for identifying Workers across the cluster in a nested scenario—any worker can refer to another by its address, even across different groups, enabling flexible communication patterns.


Communication Methods
~~~~~~~~~~~~~~~~~~~~~~

Once initialized, a `Worker` exposes high-level methods to communicate with other Workers:

- `send(object, dst_group_name, dst_rank, async_op=False)` and the counterpart `recv(src_group_name, src_rank, async_op=False)` allow transferring arbitrary Python objects or tensors between Workers. 
  Under the hood, these calls construct a `WorkerAddress` for the peer and use an appropriate collective group to perform point-to-point communication. 

- Optimized tensor operations: `send_tensor(tensor, dst_group_name, dst_rank, async_op=False)` and `recv_tensor(tensor, src_group_name, src_rank, async_op=False)` are specialized for sending a single tensor efficiently. 
  They avoid sending extra metadata about tensor shapes and types by assuming the receiver is already prepared with a correctly sized tensor buffer. 

The `Worker` does not handle communication directly; instead, it delegates the actual communication to a `CollectiveGroup`.  
See :ref:`collectivegroup_p2p` for more details.

.. These should not be mixed with the generic send/recv in the same pairing, as the protocols differ (the generic send transmits type information first, whereas `send_tensor` does not).



In addition to pairwise communications, the `Worker` also provides an interface for **Channels**, which are FIFO queues for exchanging data between Workers:

- `create_channel(name, node_id=0, maxsize=0)` sets up a new channel. 
  `connect_channel(name)` allows other Workers to connect to an existing channel by name. 
  
- Once a channel is connected, data can be stored and retrieved through it using methods such as `put()`, `get()`, and `get_batch()`.

These channel methods show how Workers coordinate higher-level workflows: 
the actual data transfer in channels still relies on the Worker’s `send` and `recv` methods, 
while the channel abstraction takes care of queuing data and controlling the flow (see :doc:`../communication/channel` for details).



WorkerGroup
------------

`WorkerGroup` is a utility for creating and managing a collection of Workers of the same type. 
It simplifies the process of launching multiple Workers across the cluster and executing methods on them in parallel. Key aspects of `WorkerGroup` include:

- **Group Creation**: Calling `MyWorker.create_group().launch(cluster, placement)` creates a group of `MyWorker` instances on the cluster's resources.  
  The placement strategy defines how many workers are launched and the specific node/GPU each will occupy (see :doc:`placement` for details).  
  During this process, the environment variables required for distributed execution are set automatically, 
  and `Cluster.allocate(...)` is invoked to start each Ray actor on the designated node and GPU with those variables.

- **Collective Execution of Methods**: One powerful feature of `WorkerGroup` is the ability to call a method on all Workers as if it were a single call. 
  After creating the group, the `WorkerGroup` instance dynamically attaches all the methods of the underlying `Worker` class onto itself. 
  When you call one of these methods on the `WorkerGroup`, it will internally invoke that method on each worker in parallel (via Ray remote calls). 

.. - **Selective Execution**: By default, the proxy methods execute on all Workers in the group. However, you can restrict execution to a subset of worker ranks by using `WorkerGroup.execute_on(ranks)`. Calling this will make the next method invocation apply only to the specified ranks, after which the WorkerGroup resets to broadcasting to all ranks. This is useful for scenarios where only one worker (e.g., rank 0) should perform a certain operation or when splitting work among different subsets of Workers.

Example
--------

.. autoclass:: rlinf.scheduler.worker.worker.Worker
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__
   :noindex:

Summary
--------

In summary, the **Worker** module provides the foundation for distributed execution. 
`WorkerAddress` gives each worker a unique identity in a potentially nested group structure, 
`WorkerInfo` holds runtime metadata, 
and the `Worker` class manages the lifecycle of each distributed worker. On top of this, 
`WorkerGroup` groups multiple Workers, handling their placement and collective method execution. 
These abstractions hide much of the Ray-specific details and low-level environment setup, allowing users to focus on the higher-level logic of their distributed reinforcement learning algorithm.
