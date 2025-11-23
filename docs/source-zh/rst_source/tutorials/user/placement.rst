GPU 资源部署策略
========================================

Placement 模块定义了 Worker 在集群的硬件资源（节点和 GPU）上的分布方式。  
一个 **PlacementStrategy** 表示一种将 Worker 分配到 GPU 和节点的策略。  
这样，像 `WorkerGroup` 这样的 Worker 集合就可以灵活地使用可用的 GPU。  
每种策略通过 `get_placement(cluster, isolate_gpu)` 方法返回一个 `Placement` 对象列表。每个 `Placement` 包含：

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
   * - ``local_accelerator_id``
     - 分配给该 Worker 的节点内 GPU 编号
   * - ``local_rank``
     - 该 Worker 在当前节点中所属的编号
   * - ``local_world_size``
     - 当前节点上的总 Worker 数量
   * - ``visible_accelerators``
     - 该 Worker 可见的 GPU ID 列表
   * - ``isolate_accelerator``
     - 是否限制该 Worker 只使用指定的 GPU

Placement 策略
---------------------------

见:doc:`../../apis/placement`中对Placement策略的详细描述。
