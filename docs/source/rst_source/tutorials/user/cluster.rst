Ray-Based Cluster Launching
===============================

The **Cluster** class manages the connection to the Ray cluster and the launching of management actors and worker actors. 
It serves as a singleton representing the entire cluster’s resources and provides methods to allocate workers on specific nodes and GPUs. 
By encapsulating Ray initialization and node information, it simplifies distributed setup for the rest of the framework.

.. note::

   **Ray version requirement**: the framework requires ``ray>=2.47.0`` (enforced at import time).  
   Do **not** call ``ray.init`` before creating a :class:`Cluster`; premature initialization can interfere with the namespace and logging configuration.
   
Initialization and Ray Setup
----------------------------

When a `Cluster` object is created, it performs the following steps in its initialization:

- **Ray Initialization** : If Ray is not already started, it calls `ray.init()` with the namespace `Cluster.NAMESPACE`. 

- **Waiting for Nodes** : After Ray initialization, `Cluster` waits until the expected number of nodes (`num_nodes`) have registered with Ray. 

- **Gathering Node Information** : Once the nodes are ready, `Cluster` constructs a list of `NodeInfo` objects (Ray ID, IP, CPU and GPU counts).
  The 'master' node is placed first; remaining nodes are sorted by IP.

- **Master Address and Port** : The master node's IP is stored and a free TCP port is chosen for collective communications. 

- **Global Manager Actors** : A key part of initialization is launching three singleton manager actors:

  * `WorkerManager` : tracks every worker's metadata.  
  * `CollectiveManager` : stores collective-group information, including
    rendezvous ports.  
  * `NodeManager` : provides node layout (IP, GPU count, master port) to workers.


Using Cluster to Allocate Workers
-----------------------------------

``Cluster.allocate()`` starts a Ray actor of class ``cls`` on a **specific node** with a controlled runtime environment:

.. code-block:: python

   handle = Cluster.allocate(
       cls,            # The actor class to launch
       worker_name,    # A unique, human-readable name for the actor
       node_id,        # Index into Cluster's node list (0 is master)
       gpu_id,         # Local GPU index on that node (used for env isolation)
       env_vars,       # Dict of environment variables for this actor (e.g., CUDA_VISIBLE_DEVICES)
       cls_args=[],    # Positional args to the actor's constructor
       cls_kwargs={},  # Keyword args to the actor's constructor
   )

What it does:

- Validates ``node_id`` and ``gpu_id`` against discovered topology.

- Wraps ``cls`` via ``ray.remote(cls)``.

- Applies options:

  - ``runtime_env={"env_vars": env_vars}`` (propagates variables like ``CUDA_VISIBLE_DEVICES``, ``rank``... )

  - ``name=worker_name`` (makes the actor discoverable by name)

  - ``scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=<Ray NodeID>, soft=False)`` (pins the actor to the requested **physical node**)

Finally, it invokes ``.remote(*cls_args, **cls_kwargs)`` to launch the actor asynchronously and returns the actor handle.


Attaching to an Existing Cluster
--------------------------------

When a ``Cluster`` is created without arguments, it attaches to the running cluster:

- Ensures Ray is initialized with ``address="auto"`` and the known namespace.
- Retrieves the existing managers and shared state from ``NodeManager``:

  .. code-block:: python

     self._node_manager     = NodeManager.get_proxy()
     self._nodes            = self._node_manager.get_nodes()
     self._num_nodes        = len(self._nodes)
     self._master_ip        = self._node_manager.get_master_ip()
     self._master_port      = self._node_manager.get_master_port()
     self._num_gpus_per_node= self._node_manager.get_num_gpus_per_node()

This guarantees that every process using the same namespace observes the same cluster view.


Summary
-------

The `Cluster` singleton centralizes Ray initialization, node discovery, and manager-actor lifecycle under a stable namespace.  
Driver code initializes once (launching managers and selecting a master), while subsequent processes simply attach and retrieve the shared view.  
With ``allocate()``, users can reliably place actors on specific nodes and set the environment variables accordingly, 
keeping distributed orchestration predictable and consistent across the framework.
