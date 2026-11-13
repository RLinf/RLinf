Data Interface
======================

This section provides the key **data structures** used for data transmission between different workers  
under the **Megatron + SGLang backend** combination in RLinf.  
It includes two fundamental structures: `RolloutRequest` and `RolloutResult`,
followed by the helper for splitting actor batches into microbatches.


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

Splitting actor batches
-----------------------

Use ``get_iterator_k_split`` to divide a dictionary of batched tensors and
per-sample lists along the batch dimension. By default, the batch size must be
divisible by the requested number of microbatches. For an uneven batch, pass
``enforce_divisible_batch=False`` to retain every sample and keep tensor rows
aligned with their list entries:

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

The iterator yields exactly four microbatches in input order; the first gets the
extra sample. If the number of microbatches exceeds the number of samples, the
trailing microbatches are empty. Callers whose model cannot accept an empty batch
must choose a smaller split count. This option does not pad or drop samples and
does not enable token-based dynamic batching.
   


