Cluster
=======

The ``Cluster`` singleton manages the connection to Ray, discovers cluster hardware topology, and launches both global management actors and training workers. By centralizing Ray initialization, node metadata, and actor placement, it keeps distributed setup consistent across single-node and multi-node runs.

.. note::

   **Ray version requirement**: RLinf requires ``ray>=2.47.0`` (enforced at import time). Do **not** call ``ray.init`` before creating a :class:`Cluster`; premature initialization can interfere with the namespace and logging configuration.

Initialization and Ray Setup
----------------------------

When the driver creates a ``Cluster``, initialization proceeds in five steps:

- **Ray Initialization**: if Ray is not already initialized, ``Cluster`` connects to or starts Ray under the shared namespace ``Cluster.NAMESPACE``.
- **Waiting for Nodes**: ``Cluster`` waits until the configured number of nodes (``num_nodes``) have registered with Ray.
- **Gathering Node Information**: once nodes are ready, ``Cluster`` builds a list of ``NodeInfo`` records (Ray node ID, IP, CPU count, accelerator inventory, and optional hardware node groups). Node rank ``0`` acts as the head node.
- **Master Address and Port**: ``Cluster`` records the head node IP and selects a free TCP port for collective rendezvous.
- **Global Manager Actors**: ``Cluster`` launches three singleton management actors on the cluster:

  - ``WorkerManager``: tracks every launched worker's metadata and lifecycle.
  - ``CollectiveManager``: coordinates collective group rendezvous information.
  - ``NodeManager``: exposes cluster node layout (IP, accelerator counts, master port) to workers.

Node Groups and Heterogeneous Hardware
--------------------------------------

Beyond homogeneous GPU nodes, ``Cluster`` supports grouping nodes by role or attached hardware through ``cluster.node_groups``. Each node group selects one or more ``node_ranks`` and can declare:

- **Accelerator or CPU-only roles**: separate GPU training nodes from CPU simulator or controller nodes.
- **Physical robot hardware**: attach ``hardware`` entries (such as Franka arms, SO-101 arms, or cameras) to the specific node ranks that own those cables and SDKs. During cluster boot, node probes enumerate those devices and hand typed ``RobotInfo`` records to real-world environments.
- **Node-group-scoped placement**: placement strategies can target a named node group rather than hardcoding raw node indices.

Allocating Workers on Nodes and Accelerators
--------------------------------------------

Rather than launching raw Ray actors directly, ``WorkerGroup`` delegates actor creation to ``Cluster.allocate()``. Given a worker class, worker name, target ``node_rank``, runtime environment variables, and resource requirements, ``Cluster``:

- validates the target node rank and accelerator assignment against the discovered cluster topology,
- prepares the worker's Ray ``runtime_env`` (propagating distributed variables such as ``RANK``, ``WORLD_SIZE``, device isolation variables, and optional profiling context),
- pins the actor to the target physical node using Ray's ``NodeAffinitySchedulingStrategy``, and
- launches the Ray actor asynchronously and returns its handle to the owning ``WorkerGroup``.

For the full ``Cluster`` class methods and parameters, see :doc:`../reference/api/cluster`.

Multi-Node Launch and Attaching to an Existing Cluster
------------------------------------------------------

On multi-node clusters, start Ray on every node before launching RLinf, and export a unique ``RLINF_NODE_RANK`` on each node prior to ``ray start`` so Ray captures the node rank. Run the training entry script only on the head node.

When a worker or helper process instantiates ``Cluster()`` without initialization arguments, it attaches to the already running cluster instead of creating a new one:

- Connects to Ray using ``address="auto"`` and the shared RLinf namespace.
- Retrieves the existing ``NodeManager`` proxy and reads the shared node list, head IP, and rendezvous port.

.. code-block:: python

   from rlinf.scheduler import Cluster

   # Driver initializes once; workers attach with Cluster() using no arguments.
   cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
   node_info = cluster.get_node_info(node_rank=0)

Every process in the job therefore observes the same cluster topology and manager actors.

Next Steps
----------

- :doc:`../reference/api/cluster`: full ``Cluster`` API reference.
- :doc:`placement`: how placement strategies map worker groups onto cluster nodes and accelerators.
- :doc:`worker`: how ``WorkerGroup`` launches actors through ``Cluster``.
