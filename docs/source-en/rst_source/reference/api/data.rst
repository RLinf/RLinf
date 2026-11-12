Data Interface
======================

Use ``RolloutRequest`` and ``RolloutResult`` to exchange data between workers
with the **Megatron + SGLang backend** combination. Split per-sample request
fields into matching chunks before distributing them to workers.

Splitting Request Fields
------------------------

Use ``split_list`` to divide per-sample request fields into a positive number
of chunks. Its default mode requires the sample count to be divisible by the
chunk count. For example, six samples form three equal chunks:

.. code-block:: python

   from rlinf.utils.data_iter_utils import split_list

   sample_ids = list(range(6))
   assert split_list(sample_ids, 3) == [[0, 1], [2, 3], [4, 5]]

Requesting four equal chunks for these six samples raises ``AssertionError``.
For uneven splits, set ``enforce_divisible_batch=False``:

.. code-block:: python

   assert split_list(sample_ids, 4, enforce_divisible_batch=False) == [
       [0, 1], [2, 3], [4], [5]
   ]

This keeps every sample in input order, with larger chunks first. Apply the
same split count to each field to preserve sample alignment. An empty input
produces the requested number of empty chunks; uneven splits can also end in
empty chunks when the chunk count exceeds the sample count. Choose a smaller
count when the consumer requires nonempty batches.


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
   


