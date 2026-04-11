Placement Interface
================================

This section introduces the GPU and node placement strategies in RLinf.
Whether in **collocated mode**, **disaggregated mode**, or **hybrid mode**, ``ComponentPlacement`` is the user-facing interface for generating the placements of different component workers (e.g., actor, env, rollout, inference), while placement strategies are the underlying mechanisms for obtaining precise allocation of each node and each GPU resource.  
The generated **placement metadata** is later used for remote launching with Ray.



Component Placement
-----------------------------------

The ``ComponentPlacement`` interface is responsible for parsing the ``cluster.component_placement`` field in the configuration file and generating precise placements for different component workers.

Notably, ``ComponentPlacement`` also supports configuration of heterogeneous clusters through the ``node_group`` field in ``cluster.node_groups``.

The detailed explanation of the syntax can be found in the docs below.

.. autoclass:: rlinf.utils.placement.ComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

In the **embodied intelligence** and **MATH reasoning** settings, 
``HybridComponentPlacement`` and ``ModelParallelComponentPlacement`` are used to generate the worker placements, respectively.
``HybridComponentPlacement`` is a direct inheritance of ``ComponentPlacement``, while ``ModelParallelComponentPlacement`` extends the placement logic to support model parallelism of inference engines across multiple GPUs.


HybridComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.utils.placement.HybridComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

ModelParallelComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
