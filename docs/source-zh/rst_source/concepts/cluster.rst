Cluster
=======

``Cluster`` 单例负责连接 Ray 集群、探测节点与硬件拓扑，并启动全局管理 actor 与各类训练 worker。它将 Ray 初始化、节点元信息与 actor 分配集中在一处，使单机和多机运行保持一致的分布式视图。

.. note::

   **Ray 版本要求**：RLinf 要求 ``ray>=2.47.0``\ （导入时自动检查）。创建 :class:`Cluster` 前请 **不要** 手动调用 ``ray.init``，以免干扰命名空间与日志配置。

初始化与 Ray 设置
-----------------

主控进程首次创建 ``Cluster`` 时，按以下五个步骤完成初始化：

- **Ray 初始化**：若当前进程尚未连接 Ray，``Cluster`` 会在共享命名空间 ``Cluster.NAMESPACE`` 下连接或启动 Ray。
- **等待节点就绪**：等待配置中的 ``num_nodes`` 个节点全部注册到 Ray 集群。
- **收集节点信息**：为每个节点构造 ``NodeInfo`` 记录（Ray 节点 ID、IP、CPU 数量、加速器信息与节点组配置），节点 rank ``0`` 作为主节点。
- **主地址与端口**：记录主节点 IP，并分配一个空闲 TCP 端口用于集合通信握手。
- **全局管理器 actor**：在集群上启动三个单例管理 actor：

  - ``WorkerManager``：记录每个已启动 worker 的元信息与状态。
  - ``CollectiveManager``：协调集合通信组的地址与端口信息。
  - ``NodeManager``：向各 worker 提供节点拓扑、加速器数量与主端口信息。

节点组与异构硬件
----------------

除同构 GPU 集群外，``Cluster`` 还通过 ``cluster.node_groups`` 按角色或所连硬件对节点分组。每个节点组指定一个或多个 ``node_ranks``，并支持：

- **区分加速卡与纯 CPU 节点**：将 GPU 训练节点与 CPU 仿真或真机控制节点分开管理。
- **声明真机硬件**：把 Franka 机械臂、SO-101 机械臂或相机等 ``hardware`` 条目绑定到实际接线的节点 rank；集群启动时，节点探针自动枚举设备并将类型化的 ``RobotInfo`` 交给真机环境。
- **按节点组调度**：placement 策略可直接面向具名节点组分配 worker，无需在各处硬编码节点编号。

在节点与加速器上分配 Worker
---------------------------

``WorkerGroup`` 不直接手动创建底层 Ray actor，而是统一通过 ``Cluster.allocate()`` 分配计算单元。收到 worker 类、worker 名称、目标 ``node_rank``、环境变量和资源需求后，``Cluster`` 会完成以下工作：

- 对照已发现的集群拓扑校验目标节点与加速器分配；
- 组装该 worker 的 Ray ``runtime_env``\ （注入 ``RANK``、``WORLD_SIZE``、设备隔离变量以及可选的 profiling 上下文）；
- 使用 Ray 的 ``NodeAffinitySchedulingStrategy`` 将 actor 绑定到指定的物理节点；
- 异步启动 Ray actor，并将句柄返回给对应的 ``WorkerGroup``。

``Cluster`` 的完整类方法与参数参见 :doc:`../reference/api/cluster`。

多节点启动与连接已有集群
------------------------

多节点运行时，需先在每个节点导出唯一的 ``RLINF_NODE_RANK``，再执行 ``ray start``，确保 Ray 在启动时捕获节点编号；随后仅在头节点运行训练入口脚本。

Worker 或辅助进程在不传初始化参数的情况下创建 ``Cluster()`` 时，会自动附着到正在运行的集群，而不会重新初始化：

- 通过 ``address="auto"`` 与共享命名空间连接到现有 Ray 集群；
- 获取 ``NodeManager`` 代理并读取共享的节点列表、主节点 IP 与通信端口。

.. code-block:: python

   from rlinf.scheduler import Cluster

   # 主控进程初始化一次；其余进程通过无参 Cluster() 附着到集群。
   cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
   node_info = cluster.get_node_info(node_rank=0)

同一作业中的所有进程因此共享一致的集群视图与全局管理器。

延伸阅读
--------

- :doc:`../reference/api/cluster`：``Cluster`` 完整 API 文档。
- :doc:`placement`：placement 策略如何将 worker 组映射到节点与加速器。
- :doc:`worker`：``WorkerGroup`` 如何通过 ``Cluster`` 启动远程 actor。
