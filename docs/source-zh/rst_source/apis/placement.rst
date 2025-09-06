Placement 接口
================================

本节介绍 RLinf 中的 GPU 资源放置策略。  
无论是在 **共享式模式**、**分离式模式** 还是 **混合式模式** 下，放置策略都用于精确分配每个节点和每块 GPU 资源。  
它们会保存 **资源元数据**，这些信息稍后会被 Ray 用于远程启动。  


ComponentPlacement
-----------------------------------

在 **具身智能** 和 **数学推理** 两种场景下，``ComponentPlacement`` 都作为基类使用。  
它从配置文件中提取放置信息，进行处理，调用 ``PackedPlacementStrategy``，并返回最终的放置结果。  

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
