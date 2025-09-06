Data 接口
======================

本节介绍 RLinf 中在 **Megatron + SGLang 后端** 组合下，不同 Worker 之间进行数据传输所使用的关键 **数据结构**。  
其中包含两个基本结构：`RolloutRequest` 和 `RolloutResult`。  


RolloutRequest
---------------

.. autoclass:: rlinf.data.io_struct.RolloutRequest
   :members: 

RolloutResult
-----------------------

.. autoclass:: rlinf.data.io_struct.RolloutResult
   :members: 
   


