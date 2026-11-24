Worker 与 WorkerGroup
=====================

``Worker`` 与 ``WorkerGroup`` 是 RLinf 的核心执行抽象。Actor、rollout、env、reward、critic 和 agent 等角色都继承自 ``Worker``，从而可以灵活部署到任意节点或加速器，并按组统一调度。完整 API 签名见 :doc:`../reference/api/worker`。

Worker
------

一个 ``Worker`` 对应一个远程进程或计算单元。继承 ``Worker`` 后，组件自动获得以下能力：

- 按 ``Cluster`` 分配运行在指定节点与加速器上。
- 与同组或跨组的其他 worker 执行点对点与集合通信。
- 自动注入分布式运行所需的环境变量（``MASTER_ADDR``、``MASTER_PORT``、``RANK``、``LOCAL_RANK`` 与 ``WORLD_SIZE``）。

WorkerTimer
~~~~~~~~~~~

RLinf 内置轻量级计时工具，可直接统计 worker 方法与子步骤耗时。

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

计时 tag 会在 worker group 内跨 rank 聚合（默认按 ``max`` 归约），外层与嵌套子函数的 tag 都会一并返回。

WorkerInfo
~~~~~~~~~~

``WorkerInfo`` 保存单个 worker 的运行时元信息，便于节点间交换位置与设备信息而无需额外远程查询：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 属性
     - 说明
   * - ``address``
     - 该 worker 的层级化 ``WorkerAddress``。
   * - ``rank``
     - 该 worker 在所属组内的 rank 编号。
   * - ``node_id``
     - 承载该 worker 的节点标识。
   * - ``gpu_id``
     - 分配给该 worker 的加速器编号。
   * - ``node_ip``
     - 承载该 worker 的节点 IP 地址。
   * - ``available_gpus``
     - 该 worker 可见的加速器设备列表。

WorkerAddress
~~~~~~~~~~~~~

``WorkerAddress`` 用“根组名 + rank 路径”为每个 worker 提供唯一层级地址，例如 ``"ActorGroup:0"``、``"ActorGroup:1"``；若 worker 创建子 worker，则继续追加 rank（如 ``"ActorGroup:0:0"``）。通过 ``get_name()``、``get_parent_rank()``、``get_parent_address()`` 与 ``get_child_address(rank)`` 可以跨组寻址。

通信方法
~~~~~~~~

``Worker`` 提供两种通信方式：

- **点对点通信**：``send(object, dst_group_name, dst_rank, async_op=False)`` 与 ``recv(src_group_name, src_rank, async_op=False)`` 用于传输任意 Python 对象或张量；``send_tensor`` 与 ``recv_tensor`` 面向已预分配缓冲区的单张量高效传输。底层传输由 ``CollectiveGroup`` 完成（见 :ref:`collectivegroup_p2p`）。
- **数据通道（Channel）**：``create_channel(name, node_id=0, maxsize=0)`` 与 ``connect_channel(name)`` 连接到 FIFO 队列，供 worker 通过 ``put()``、``get()`` 与 ``get_batch()`` 交换数据（见 :doc:`channel`）。

WorkerGroup
-----------

``WorkerGroup`` 用于统一启动并管理同一类 worker：

- **组创建**：``MyWorker.create_group(...).launch(cluster, placement_strategy=placement)`` 根据 placement 策略确定 worker 数量与物理位置（见 :doc:`placement`），自动注入环境变量并交由 ``Cluster`` 创建各 Ray actor。
- **并行方法调用**：``WorkerGroup`` 会暴露底层 ``Worker`` 类的全部公有方法。调用该组方法时会在全部 rank 上并行执行，并返回句柄，调用 ``.wait()`` 收集结果；只需在部分 rank 执行时，可配合 ``execute_on(ranks)``。

延伸阅读
--------

- :doc:`../reference/api/worker`：``Worker`` 与 ``WorkerGroup`` 完整 API 文档。
- :doc:`placement`：将 worker 组映射到硬件的 placement 策略。
- :doc:`channel`：基于队列的跨组数据通信。
