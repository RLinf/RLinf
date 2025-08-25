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

This unified placement strategy that can:

* **Pack GPUs contiguously** (`stride = 1`) – the classic “close-packed”
  behaviour; or
* **Assign GPUs in a fixed-stride pattern** (`stride > 1`) – the former
  “strided” mode, e.g. `0, 2, 4` for `stride = 2`.

By tuning a **single** parameter (`stride`) you can therefore express
both placement styles while keeping one coherent implementation.

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




.. PackedPlacementStrategy
.. -----------------------

.. This strategy places processes onto GPUs in a close-packed, contiguous fashion.

.. **Required inputs**

.. - ``master_node`` — start node
.. - One of ``num_nodes`` or ``num_processes`` (mutually exclusive)
.. - ``master_gpu`` — start GPU on the master node
.. - ``num_gpus_per_process`` — contiguous GPUs per process
.. - ``isolate_gpu`` — Whether the worker is restricted to those GPUs

.. **Placement principle**

.. Starting at ``(master_node, master_gpu)``, the scheduler assigns each process a contiguous block of size ``num_gpus_per_process`` on the current node, 
.. moving linearly across GPU indices. When a node’s GPUs are exhausted, 
.. it advances to the next node. If ``num_nodes`` is provided, the total ``num_processes`` is derived from available GPUs and the per-process width. 
.. Optionally, ``CUDA_VISIBLE_DEVICES`` is set so each process only “sees” its assigned GPUs.

.. **Purpose**

.. A straightforward default placement strategy that aligns with frameworks expecting contiguous device IDs per process, 
.. while also allowing execution from an offset subset specified by ``(master_node, master_gpu)`` without occupying the entire cluster.


.. StridedPlacementStrategy
.. ------------------------

.. This strategy assigns multiple GPUs per process in a fixed-stride pattern (e.g., ``0, 2, 4`` for stride 2), spreading a process’s devices across the node’s GPU index space.

.. **Required inputs**

.. - ``master_node`` — start node
.. - ``num_nodes`` — span of nodes starting at master
.. - ``stride`` — GPU index stride
.. - ``num_gpus_per_process`` — GPUs per process
.. - ``isolate_gpu`` — must be ``True`` (enforced)

.. **Placement principle**

.. The total GPU pool over the selected nodes is conceptually partitioned into equal “groups” of size ``stride × num_gpus_per_process``. Within each group, ``stride`` processes are interleaved, and each process receives GPUs ``[g, g+stride, …]``. Groups are then tightly filled across nodes, subject to divisibility and fit constraints to ensure every process’s last GPU index is in range.

.. **Purpose**

.. Useful to colocation placement of rollout and training models that places model parallel source ranks on the same GPUs, enabling fast zero-copy-cudaIPC-based weight synchronization

Example
---------

.. autoclass:: rlinf.scheduler.placement.PlacementStrategy
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__

Summary
--------

In summary, the **placement** component ensures that workers are deployed in a way that matches the desired parallel execution pattern and resource usage policy, which is crucial for performance and correctness in distributed training.

