Cluster 接口
=============================

本节介绍 RLinf 中的 **Cluster** 类，它负责启动远程节点和 GPU。  
该类基于从 Placement 策略获取的元数据，利用 **Ray** 来精确调度所有训练资源，以支持分布式训练。  

Cluster
------------------------

.. autoclass:: rlinf.scheduler.cluster.Cluster
   :members:
   :exclude-members: __new__,

NodeInfo
-----------

.. autoclass:: rlinf.scheduler.cluster.NodeInfo
   :members:
