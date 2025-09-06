Cluster Interface
=============================

This section introduces the **Cluster** class in RLinf, which is responsible for launching remote nodes and GPUs.  
Based on the metadata obtained from the placement strategy, it leverages **Ray** to precisely schedule all training resources for distributed training.

Cluster
------------------------

.. autoclass:: rlinf.scheduler.cluster.Cluster
   :members:
   :exclude-members: __new__,

NodeInfo
-----------

.. autoclass:: rlinf.scheduler.cluster.NodeInfo
   :members:
