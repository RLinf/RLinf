GPU 资源部署策略
========================================

Placement 模块定义了 Worker 在集群的硬件资源（节点和 GPU）上的分布方式。  
一个 **PlacementStrategy** 表示一种将 Worker 分配到 GPU 和节点的策略。  
这样，像 `WorkerGroup` 这样的 Worker 集合就可以灵活地使用可用的 GPU。  
每种策略通过 `get_placement(num_gpus_per_node, isolate_gpu)` 方法返回一个 `Placement` 对象列表。每个 `Placement` 包含：

.. list-table:: Placement 属性说明
   :header-rows: 1
   :widths: 25 75

   * - 属性
     - 描述
   * - ``rank``
     - Worker 的全局唯一编号（global index）
   * - ``node_id``
     - 运行该 Worker 的节点标识符
   * - ``node_rank``
     - 节点在整个集群中的编号
   * - ``local_gpu_id``
     - 分配给该 Worker 的节点内 GPU 编号
   * - ``local_rank``
     - 该 Worker 在当前节点中所属的编号
   * - ``local_world_size``
     - 当前节点上的总 Worker 数量
   * - ``cuda_visible_devices``
     - 该 Worker 可见的 GPU ID 列表
   * - ``isolate_gpu``
     - 是否限制该 Worker 只使用指定的 GPU

PackedPlacementStrategy
-----------------------

这是一种统一的部署策略，支持：

* **连续分配 GPU** （`stride = 1`）—— 即传统的“紧密打包”方式；或
* **固定步长分配 GPU** （`stride > 1`）—— 类似早期的“跨步分配”模式，例如当 `stride = 2` 时为 `0, 2, 4`

通过调整一个参数（`stride`），即可在两种部署风格之间切换，同时保持代码实现的一致性。

必要输入参数
~~~~~~~~~~~~~~~~~

* ``start_gpu_id`` – 起始的 *全局* GPU 编号  
* ``end_gpu_id`` – 结束的 *全局* GPU 编号（包含）  
* ``num_gpus_per_process`` – 每个进程分配的 GPU 数量  
* ``stride`` – 同一进程内部，连续 GPU 编号之间的间隔（`1` 表示连续分配，`>1` 表示跨步分配）  
* ``isolate_gpu`` – 是否通过设置 ``CUDA_VISIBLE_DEVICES`` 来限制进程只看到分配给它的 GPU（默认为 ``True``）  

部署原则
~~~~~~~~~~~~~~~~~~~~~

调度器从 ``start_gpu_id`` 开始，按以下规则向前遍历 GPU：

1. **分配一个块**：包含 ``num_gpus_per_process × stride`` 个连续的全局 GPU ID  
2. **每隔 stride 个取一个 GPU**：作为当前 Worker 使用的设备  
   （例如 ``[0, 1, 2, 3]`` 配合 `stride = 2` 得到 ``[0, 2]``）  
3. 重复直到所有 ID 被分配完，如遇某节点 GPU 不足则切换到下一个节点

构造函数强制要求  
``total_GPUs % (num_gpus_per_process × stride) == 0``  
这样可以确保每个进程获得一整块完整的 GPU 集合，避免跨节点分配。

当 ``isolate_gpu=True`` 时，会设置 ``CUDA_VISIBLE_DEVICES`` 变量为节点内对应的 GPU 编号，确保库调用只会“看到”这些设备。

用途
~~~~~~~~~~~~~

* **连续模式** （`stride = 1`）是默认选项，适用于数据并行或模型并行等依赖连续设备 ID 的任务  
* **跨步模式** （`stride > 1`）适合用于 **共享式部署 rollout 与训练模型**，可以将模型并行源 rank 放在同一个 GPU 上，  
  从而启用 **zero-copy、CUDA IPC 权重同步**

示例
---------

.. autoclass:: rlinf.scheduler.placement.PlacementStrategy
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__

总结
--------

总之，**Placement** 模块用于确保 Worker 被合理地部署到 GPU 和节点上，  
以匹配所需的并行执行模式和资源使用策略。这对于分布式训练的性能和正确性至关重要。
