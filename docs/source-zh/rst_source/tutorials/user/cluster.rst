基于 Ray 的集群启动
===============================

**Cluster** 类负责连接 Ray 集群，并启动管理 actor 与 worker actor。  
它作为整个集群资源的单例入口，提供了在特定节点和 GPU 上分配 worker 的方法。  
通过封装 Ray 的初始化与节点信息，它简化了整个框架的分布式部署流程。

.. note::

   **Ray 版本要求**：RLinf 需要 ``ray>=2.47.0`` （在导入时强制检查）。  
   请 **不要** 在创建 :class:`Cluster` 之前调用 ``ray.init`` ，  
   否则可能会破坏命名空间和日志配置。

初始化与 Ray 设置
----------------------------

当创建一个 `Cluster` 实例时，其初始化过程包括以下步骤：

- **Ray 初始化**：如果 Ray 尚未启动，将使用命名空间 `Cluster.NAMESPACE` 调用 `ray.init()`。

- **等待节点就绪**：Ray 初始化后，`Cluster` 会等待 `num_nodes` 个节点在 Ray 中注册完成。

- **收集节点信息**：节点就绪后，`Cluster` 会构造一个 `NodeInfo` 列表（包含 Ray ID、IP、CPU 和 GPU 数量）。  
  “主节点”排在列表首位，其余节点按 IP 排序。

- **主地址与端口**：记录主节点的 IP，并选择一个空闲的 TCP 端口用于集体通信。

- **全局管理器 actor**：初始化阶段还会启动三个全局管理器 actor：

  * `WorkerManager`：追踪每个 worker 的元信息  
  * `CollectiveManager`：存储集体组信息，包括通信端口等  
  * `NodeManager`：为 workers 提供节点布局（IP、GPU 数、主端口）

使用 Cluster 分配 Worker
-----------------------------------

``Cluster.allocate()`` 会在 **指定节点** 上以受控环境启动一个 Ray actor：

.. code-block:: python

   handle = Cluster.allocate(
       cls,            # 要启动的 actor 类
       worker_name,    # 该 actor 的唯一可读名称
       node_id,        # 节点在 Cluster 列表中的索引（0 表示主节点）
       gpu_id,         # 节点内的本地 GPU 编号（用于隔离）
       env_vars,       # 该 actor 的环境变量（如 CUDA_VISIBLE_DEVICES）
       cls_args=[],    # 传给构造函数的位置参数
       cls_kwargs={},  # 传给构造函数的关键字参数
   )

该方法执行以下操作：

- 验证 ``node_id`` 与 ``gpu_id`` 是否存在于已发现的集群拓扑中。
- 使用 ``ray.remote(cls)`` 包装传入的类。
- 应用如下运行配置：

  - ``runtime_env={"env_vars": env_vars}`` （传递变量如 ``CUDA_VISIBLE_DEVICES``、``rank`` 等）
  - ``name=worker_name`` （使 actor 可通过名称发现）
  - ``scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=<Ray NodeID>, soft=False)``  
    （将 actor 固定调度到指定的 **物理节点**）

最后，调用 ``.remote(*cls_args, **cls_kwargs)`` 异步启动 actor，并返回该 actor 的句柄。

连接已有集群
----------------------------

当不带参数创建 ``Cluster`` 时，它会自动附着到当前运行的集群：

- 确保 Ray 以 ``address="auto"`` 和指定命名空间初始化；
- 从 ``NodeManager`` 获取现有管理器与共享状态：

  .. code-block:: python

     self._node_manager     = NodeManager.get_proxy()
     self._nodes            = self._node_manager.get_nodes()
     self._num_nodes        = len(self._nodes)
     self._master_ip        = self._node_manager.get_master_ip()
     self._master_port      = self._node_manager.get_master_port()
     self._num_gpus_per_node= self._node_manager.get_num_gpus_per_node()

这可以确保所有使用相同命名空间的进程观察到相同的集群视图。

总结
-------

`Cluster` 单例统一管理 Ray 初始化、节点发现、以及管理器 actor 生命周期，并维持在一个稳定命名空间下。  
主驱动程序只需初始化一次（启动管理器、选定主节点），  
后续所有进程只需附着连接，并共享该集群视图。  
通过 ``allocate()`` 方法，用户可以可靠地将 actor 安置在指定节点，并设置其环境变量，  
从而在整个框架中保持分布式调度的稳定性与一致性。
