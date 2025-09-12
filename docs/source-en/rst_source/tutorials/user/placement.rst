GPU Resource Placement Strategy
========================================

The placement module defines how workers are distributed across the hardware resources (nodes and GPUs) of the cluster. 
A **PlacementStrategy** encapsulates a particular policy for assigning workers to GPUs and nodes. 
This allows flexibility in how a set of workers (e.g., a `WorkerGroup`) utilizes the available GPUs. 
For each strategy, a list of `Placement` objects is returned by the `get_placement(num_gpus_per_node, isolate_gpu)` method. Each `Placement` includes:

+---------------------------+-----------------------------------------------+
| Property                  | Description                                   |
+===========================+===============================================+
| `rank`                    | Unique global worker index                    | 
+---------------------------+-----------------------------------------------+
| `node_id`                 | Identifier of the node where the worker runs  |
+---------------------------+-----------------------------------------------+
| `node_rank`               | Index of the node within the cluster          |
+---------------------------+-----------------------------------------------+
| `local_gpu_id`            | GPU index assigned to the worker on that node |
+---------------------------+-----------------------------------------------+
| `local_rank`              | Worker’s index among workers on the same node |
+---------------------------+-----------------------------------------------+
| `local_world_size`        | Total number of workers on the same node      |
+---------------------------+-----------------------------------------------+
| `cuda_visible_devices`    | List of GPU IDs visible to the worker         |
+---------------------------+-----------------------------------------------+
| `isolate_gpu`             | Whether the worker is restricted to those GPUs|
+---------------------------+-----------------------------------------------+

PackedPlacementStrategy
-----------------------

This placement strategy can:

* **Pack GPUs contiguously** (`stride = 1`) – the classic “close-packed”
  behaviour; or
* **Assign GPUs in a fixed-stride pattern** (`stride > 1`) – the former
  “strided” mode, e.g. `0, 2, 4` for `stride = 2`.

Required inputs
~~~~~~~~~~~~~~~~~

* ``start_gpu_id`` – first *global* GPU index to consider.  
* ``end_gpu_id`` – last *global* GPU index (inclusive).  
* ``num_gpus_per_process`` – number of GPUs given to **each** process.  
* ``stride`` – distance between successive GPUs *inside one process*
  (``1`` = contiguous; ``>1`` = strided).  
* ``isolate_gpu`` – whether to set ``CUDA_VISIBLE_DEVICES`` so the
  process only “sees” its assigned GPUs (defaults to ``True``).  

Placement principle
~~~~~~~~~~~~~~~~~~~~~

Starting at ``start_gpu_id`` the scheduler walks forward through the
GPU IDs:

1. **Allocate a block** of
   ``num_gpus_per_process × stride`` consecutive global IDs.  
2. **Select every stride-th** ID inside that block; those become the
   GPUs for the current rank  
   (e.g. ``[0, 1, 2, 3]`` → stride 2 → ``[0, 2]``).  
3. Repeat until **all** IDs up to ``end_gpu_id`` are consumed, wrapping
   to the next node when a node’s GPU count is exceeded.

The constructor enforces  
``total_GPUs % (num_gpus_per_process × stride) == 0`` so every process
obtains a full GPU set without spilling across node boundaries.

When ``isolate_gpu=True`` the generated placement also sets
``CUDA_VISIBLE_DEVICES`` to the local (node-relative) GPU list, ensuring
library calls see only those devices.

Purpose
~~~~~~~~~~~~~

* **Contiguous mode** (`stride = 1`) remains the default for
  data-parallel or per-rank model-parallel jobs that expect sequential
  device IDs.
* **Strided mode** (`stride > 1`) is useful to colocation placement of rollout and training models that places model parallel source 
  ranks on the same GPUs, enabling fast zero-copy-cudaIPC-based weight synchronization

FlexiblePlacementStrategy
-----------------------

This placement strategy allows arbitrary GPU IDs to be assigned to each process, specified as a list of GPU ID lists, each GPU ID list contains the global GPU IDs assigned to a worker process.

Required inputs
~~~~~~~~~~~~~~~~~

* ``gpu_id_lists`` – a list of lists of global GPU IDs, each inner list specifies the GPUs assigned to a worker process.

Placement principle
~~~~~~~~~~~~~~~~~~~~~
The scheduler iterates through the provided ``gpu_id_lists``, and for each inner list, it assigns the specified global GPU IDs to a worker process. The node ID is determined by the first GPU ID in the inner list, and the local GPU IDs are calculated relative to that node.

Purpose
~~~~~~~~~~~~~
This strategy provides maximum flexibility, allowing users to define exactly which GPUs each worker should use, regardless of contiguity or stride. It is particularly useful in scenarios where specific GPU assignments are required due to hardware topology or other constraints.


Example
---------

.. autoclass:: rlinf.scheduler.placement.PlacementStrategy
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__

Summary
--------

In summary, the **placement** component ensures that workers are deployed in a way that matches the desired parallel execution pattern and resource usage policy, which is crucial for performance and correctness in distributed training.

