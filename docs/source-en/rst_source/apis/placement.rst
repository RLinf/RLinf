Placement Interface
================================

This section introduces the GPU and node placement strategies in RLinf.
Whether in **collocated mode**, **disaggregated mode**, or **hybrid mode**, ``ComponentPlacement`` is the user-facing interface for generating the placements of different component workers (e.g., actor, env, rollout, inference), while placement strategies are the underlying mechanisms for obtaining precise allocation of each node and each GPU resource.  
The generated **placement metadata** is later used for remote launching with Ray.



Component Placement
-----------------------------------


In the **embodied intelligence** and **MATH reasoning** settings, 
``HybridComponentPlacement`` and ``ModelParallelComponentPlacement`` are used to generate the worker placements, respectively.
Both placements accepts the ``DictConfig`` parsed by OmegaConf and translate the ``cluster.component_placement`` field to precise GPU and node allocations.

The component placement accepts dictionary syntax of ``cluster.component_placement``:

- The key is the names of components, e.g., ``rollout``, or ``rollout,inference,actor``
- The value is the global GPU IDs allocated to the components, which can be:
   - "all": use all GPUs in the cluster
   - A single integer, e.g., "3": use GPU 3
   - A list of integers separated by comma, e.g., "0,2,3": use GPU 0, 2, and 3
   - A range of integers separated by hyphen, e.g., "0-3": use GPU 0, 1, 2, and 3
   - A combination of the above two, e.g., "0-3,5,7": use GPU 0, 1, 2, 3, 5, and 7

HybridComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.utils.placement.HybridComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

ModelParallelComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.utils.placement.ModelParallelComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

Placement Strategies
-----------------------------------

Placement strategies are the underlying mechanisms for obtaining precise allocation of each node and each GPU resource used by component placement.
If you wish to customize placements, you can refer to the following built-in strategies, namely ``FlexiblePlacementStrategy``, ``PackedPlacementStrategy`` and ``NodePlacementStrategy``.
Specifically, ``FlexiblePlacementStrategy`` and ``PackedPlacementStrategy`` are used for placing worker processes on top of accelerators/GPUs, while ``NodePlacementStrategy`` is used for placing worker processes on specific nodes without considering the underlying accelerator resources and thus useful for CPU-only workers.


FlexiblePlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.flexible.FlexiblePlacementStrategy
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

PackedPlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.packed.PackedPlacementStrategy
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

NodePlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.node.NodePlacementStrategy
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class  

Placement Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.placement.Placement
   :members:
   :member-order: bysource
   :exclude-members: __init__
