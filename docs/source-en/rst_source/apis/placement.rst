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
Normally, users do not need to directly use placement strategies.
But if you wish to implement more customized placements, you can refer to the following two built-in strategies, namely ``PackedPlacementStrategy`` and ``FlexiblePlacementStrategy``.


FlexiblePlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.packed.FlexiblePlacementStrategy
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

Example Usage
---------

.. autoclass:: rlinf.scheduler.placement.PlacementStrategy
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__

Placement Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.placement.Placement
   :members:
   :member-order: bysource
   :exclude-members: __init__
