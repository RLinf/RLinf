Data 接口
======================

使用 ``RolloutRequest`` 和 ``RolloutResult`` 在 **Megatron + SGLang 后端** 组合的 Worker 之间传输数据。分发请求前，将逐样本字段切分为相互对应的块。

切分请求字段
------------

使用 ``split_list`` 将逐样本请求字段切分为指定数量的块，块数必须为正整数。默认要求样本数能被块数整除。例如，六个样本可以均分为三块：

.. code-block:: python

   from rlinf.utils.data_iter_utils import split_list

   sample_ids = list(range(6))
   assert split_list(sample_ids, 3) == [[0, 1], [2, 3], [4, 5]]

将这六个样本均分为四块会抛出 ``AssertionError``。需要不等长切分时，设置 ``enforce_divisible_batch=False``：

.. code-block:: python

   assert split_list(sample_ids, 4, enforce_divisible_batch=False) == [
       [0, 1], [2, 3], [4], [5]
   ]

切分结果保留全部样本和原始顺序，较大的块排在前面。各字段使用相同的块数切分，以保持样本对应关系。空输入会产生指定数量的空块；不等长切分的块数超过样本数时，末尾也会出现空块。下游要求非空 batch 时，应减少块数。


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
   


