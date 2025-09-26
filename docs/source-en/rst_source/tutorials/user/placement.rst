GPU Resource Placement Strategy
========================================

The placement module defines how workers are distributed across the hardware resources (nodes and GPUs) of the cluster. 
A **PlacementStrategy** encapsulates a particular policy for assigning workers to GPUs and nodes. 
This allows flexibility in how a set of workers (e.g., a `WorkerGroup`) utilizes the available GPUs. 
For each strategy, a list of `Placement` objects is returned by the `get_placement(cluster, isolate_accelerator)` method. Each `Placement` includes:

+---------------------------+--------------------------------------------------------+
| Property                  | Description                                            |
+===========================+========================================================+
| `rank`                    | Unique global worker index                             |
+---------------------------+--------------------------------------------------------+
| `node_id`                 | Identifier of the node where the worker runs           |
+---------------------------+--------------------------------------------------------+
| `node_rank`               | Index of the node within the cluster                   |
+---------------------------+--------------------------------------------------------+
| `local_accelerator_id`    | Accelerator index of the worker on that node           |
+---------------------------+--------------------------------------------------------+
| `local_rank`              | Worker’s index among workers on the same node          |
+---------------------------+--------------------------------------------------------+
| `local_world_size`        | Total number of workers on the same node               |
+---------------------------+--------------------------------------------------------+
| `visible_accelerators`    | List of accelerator IDs visible to the worker          |
+---------------------------+--------------------------------------------------------+
| `isolate_accelerator`     | Whether worker is restricted to assigned accelerators  |
+---------------------------+--------------------------------------------------------+

Placement Strategies
---------

See :doc:`../../apis/placement` for detailed description of the built-in placement strategies.

