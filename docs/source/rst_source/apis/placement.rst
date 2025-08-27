Placement Interface
================================

This section introduces the GPU placement strategies in RLinf.  
Whether in **collocated mode**, **disaggregated mode**, or **hybrid mode**, placement strategies are used to obtain precise allocation of each node and each GPU resource.  
They hold the **resource metadata**, which is later used for remote launching with Ray.



ComponentPlacement
-----------------------------------

In both **embodied intelligence** and **MATH reasoning** settings, the following 
``ComponentPlacement`` serves as the base class.  
It extracts placement metadata from the configuration, processes it, 
invokes the ``PackedPlacementStrategy``, and returns the final placement result.

.. autoclass:: rlinf.utils.placement.ComponentPlacement
   :members:


PackedPlacementStrategy
------------------------

.. autoclass:: rlinf.scheduler.placement.packed.PackedPlacementStrategy
   :show-inheritance:
   :members:
   :class-doc-from: init
   :exclude-members: __init__,

Placement Metadata
------------------

.. autoclass:: rlinf.scheduler.placement.placement.Placement
   :members:
