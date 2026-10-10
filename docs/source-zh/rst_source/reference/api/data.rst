Data 接口
======================

本节介绍 RLinf 中在 **Megatron + SGLang 后端** 组合下，不同 Worker 之间进行数据传输所使用的关键 **数据结构**。  
其中包含两个基本结构：`RolloutRequest` 和 `RolloutResult`，以及将 actor batch 切分为微批的辅助函数。


RolloutRequest
---------------

.. autoclass:: rlinf.data.schema.reasoning_requests.RolloutRequest
   :members: 
   :member-order: bysource

RolloutResult
-----------------------

.. autoclass:: rlinf.data.schema.reasoning_results.RolloutResult
   :members: 
   :member-order: bysource

切分 actor batch
----------------

``get_iterator_k_split`` 沿 batch 维度切分字典中的 tensor 和逐样本列表。默认要求样本数能被微批数整除；不能整除时，传入 ``enforce_divisible_batch=False``，即可保留所有样本，并保持 tensor 行与列表元素的对应关系：

.. code-block:: python

   import torch
   from rlinf.utils.data_iter_utils import get_iterator_k_split

   batch = {
       "input_ids": torch.arange(18).reshape(9, 2),
       "sample_ids": list(range(9)),
   }
   microbatches = list(
       get_iterator_k_split(batch, num_splits=4, enforce_divisible_batch=False)
   )
   assert [len(item["sample_ids"]) for item in microbatches] == [3, 2, 2, 2]
   assert torch.equal(
       torch.cat([item["input_ids"] for item in microbatches]), batch["input_ids"]
   )

迭代器按输入顺序生成四个微批，第一个微批多一个样本。如果微批数超过样本数，末尾会出现空微批；模型不支持空 batch 时，调用者需要减少切分数量。此选项不会填充或丢弃样本，也不会启用按 token 数进行的动态批处理。
   


