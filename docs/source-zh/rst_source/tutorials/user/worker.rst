基于 Worker 的编程接口
===================================

本节介绍 RLinf 框架中最基本的组件 —— **Worker** 和 **WorkerGroup**，它们是构建整个框架的基石。

Worker 
------------

一个 **Worker** 表示一个远程进程或计算单元。  
通过继承 :class:`Worker`，一个 Worker 或处理器类将具备以下能力：

- 在分布式环境中的多个节点上远程运行。
- 与集群中的其他 Workers 通信。
- 自动接收如 ``MASTER_ADDR``、``MASTER_PORT``、``RANK``、``LOCAL_RANK`` 和 ``WORLD_SIZE`` 等必要的环境变量。

这些功能使得进程组的创建变得简单，并简化了分布式训练的设置流程。  
一个 Worker 封装了单个执行单元的逻辑，使得在多个 GPU 和节点间扩展任务变得容易。

WorkerTimer
~~~~~~~~~~~~~

RLinf 提供轻量级计时工具，用于统计 Worker 函数的耗时。

**上下文管理器用法**：

.. code-block:: python

   class MyWorker(Worker):
       def run_training(self, batch):
           with self.worker_timer("run_training"):
               return self.train_step(batch)

       def train_step(self, batch): ...

**装饰器用法**：

.. code-block:: python

   class MyWorker(Worker):
       @Worker.timer("run_training")
       def run_training(self, batch):
           return self.train_step(batch)

       @Worker.timer("train_step")
       def train_step(self, batch): ...

**获取耗时指标**：

.. code-block:: python

   handle = worker_group.run_training(...)
   result = handle.wait()
   timing = handle.consume_durations()  # dict: tag -> duration

计时 tag 会在 worker group 内进行聚合，默认使用 ``max`` 进行归约。

返回的 timing 会同时包含 ``run_training`` 和 ``train_step``，
便于统计子函数耗时。

WorkerInfo 
~~~~~~~~~~~

`WorkerInfo` 数据类 **在运行时捕获 Worker 的关键属性**。

.. list-table:: WorkerInfo 属性说明
   :header-rows: 1
   :widths: 25 75

   * - 属性
     - 描述
   * - ``address``
     - Worker 的 WorkerAddress
   * - ``rank``
     - Worker 在其所属组内的编号（rank）
   * - ``node_id``
     - 承载该 Worker 的节点标识符
   * - ``gpu_id``
     - 分配给该 Worker 的 GPU 编号
   * - ``node_ip``
     - 承载该 Worker 的节点的 IP 地址
   * - ``available_gpus``
     - 该 Worker 可用的 CUDA 设备编号列表



WorkerAddress
~~~~~~~~~~~~~

`WorkerAddress` 类 **为 Workers 提供了层级化的命名机制**。  
它通过将根组名与一系列 rank 路径组合在一起，为 Worker 群体中的每一个 Worker 提供唯一标识。

例如，根 WorkerGroup 可能命名为 `"Worker_group_MyWorker"`，其中的 Worker 地址如 `"Worker_group_MyWorker:0"`、`"Worker_group_MyWorker:1"` 等。  
若这些 Worker 创建了子 worker，则地址会继续追加 rank（例如 `"Worker_group_MyWorker:0:0"` 表示 rank 0 的子 worker）。  
`WorkerAddress` 提供了一些函数用于在层级结构中导航：可通过 `get_name()` 获取字符串形式的地址，通过 `get_parent_rank()` 或 `get_parent_address()` 获取上级信息，或通过 `get_child_address(rank)` 获取某个 rank 的子地址。

这种地址系统在嵌套结构的集群中尤为关键 —— 任意 Worker 可通过地址引用其他 worker，即使它们不在同一个组内，从而实现灵活的通信模式。

通信方法
~~~~~~~~~~~~~~~~~~~~~~

一旦初始化完成，`Worker` 提供了多个高级方法用于与其他 Worker 通信：

- `send(object, dst_group_name, dst_rank, async_op=False)` 和 `recv(src_group_name, src_rank, async_op=False)` 可实现任意 Python 对象或张量的传输。  
  底层通过构造 `WorkerAddress` 并使用合适的 collective group 执行点对点通信。

- 针对张量传输进行了优化：`send_tensor(tensor, dst_group_name, dst_rank, async_op=False)` 和 `recv_tensor(tensor, src_group_name, src_rank, async_op=False)` 提供了高效传输单个张量的能力。  
  由于假设接收端已准备好合适尺寸的缓冲区，因此可避免发送额外的张量形状和类型信息。

`Worker` 本身并不直接处理通信，而是将通信委托给 `CollectiveGroup`。  
详细内容请见 :ref:`collectivegroup_p2p`。

除了点对点通信外，`Worker` 还支持用于 Worker 间数据交换的 **通道（Channel）** 接口，这是一种先进先出（FIFO）的队列机制：

- 使用 `create_channel(name, node_id=0, maxsize=0)` 创建新通道，  
  使用 `connect_channel(name)` 允许其他 Worker 按名称连接到已存在的通道。

- 一旦连接成功，可通过 `put()`、`get()` 和 `get_batch()` 等方法存入或提取数据。

这些通道方法展示了 Worker 如何协调更高级别的工作流程：  
通道的数据传输仍然基于 Worker 的 `send` 和 `recv` 方法完成，  
而通道的抽象则管理队列行为和流量控制（详细内容见 :doc:`../communication/channel`）。



WorkerGroup
------------

`WorkerGroup` 是一个用于创建和管理一组同类 Worker 的工具类。  
它简化了在集群中启动多个 Worker 并在其上并行执行方法的流程。其核心特性包括：

- **组创建**：通过 `MyWorker.create_group().launch(cluster, placement)` 可在集群资源上创建一组 `MyWorker` 实例。  
  placement 策略定义了启动的 Worker 数量以及它们分配到的具体节点/GPU（详见 :doc:`placement`）。  
  在这一过程中，所需的环境变量将自动设置，  
  并调用 `Cluster.allocate(...)` 以使用这些变量在指定节点和 GPU 上启动每个 Ray actor。

- **方法的并行执行**：`WorkerGroup` 的强大之处在于可以像调用一个函数那样一次性在所有 workers 上调用同一个方法。  
  创建 group 后，`WorkerGroup` 会自动绑定底层 `Worker` 类中的所有方法。  
  当你调用其中一个方法时，它将在所有 Worker 上并行执行该方法（通过 Ray 的远程调用实现）。


示例
--------

.. autoclass:: rlinf.scheduler.worker.worker.Worker
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__
   :noindex:

总结
--------

总而言之，**Worker** 模块是分布式执行的基础。  
`WorkerAddress` 为每个 Worker 提供唯一标识，支持嵌套结构；  
`WorkerInfo` 保存运行时的元信息；  
`Worker` 类管理每个分布式 Worker 的生命周期。  
在此之上，`WorkerGroup` 可创建并管理多个 worker，负责其 placement 及方法的并行执行。  
这些抽象隐藏了大量与 Ray 和底层环境设置相关的细节，使用户可以更专注于构建高层的分布式强化学习算法逻辑。
