使用 Channel 进行通信
=====================

Channel 模块提供分布式生产者—消费者队列，用于在 Worker 之间传输数据。一个或多个
生产者可以向命名 ``Channel`` 中 ``put`` 数据项，一个或多个消费者则可以独立于
生产者的执行进度，通过 ``get`` 获取数据。

默认共享队列支持同步与异步操作、按 key 划分逻辑队列、容量限制和加权 batch。大多数
应用可以直接使用这些默认行为。只有在数据入队前需要转换时才添加 ``Collector``，
只有在输出必须分配给指定消费者时才添加 ``Dispatcher``。

创建并共享 Channel
-------------------

在 Worker 外创建 channel，然后将其传给负责生产或消费数据的 WorkerGroup 方法：

.. code-block:: python

   from rlinf.scheduler import Channel

   sample_channel = Channel.create(name="Samples", maxsize=64)
   producer_group.produce(sample_channel)
   consumer_group.consume(sample_channel)

本例中的 ``produce`` 和 ``consume`` 是应用自定义的 Worker 方法。将 channel 传入
Worker 后，通信会自动绑定到该 Worker。已经在 Worker 内运行的代码也可以调用
``self.create_channel(...)`` 或 ``self.connect_channel("Samples")``。

放入、获取和批量获取数据
--------------------------

``put`` 发送一个可序列化的数据项，``get`` 从同一逻辑队列中移除下一个数据项：

.. code-block:: python

   sample_channel.put(sample, key="train", weight=num_tokens)
   sample = sample_channel.get(key="train")

每个 key 对应一个独立的 FIFO 队列。可选的 weight 表示数据项的开销或大小，但不会
改变队列顺序。使用 ``get_batch`` 可以取出队首连续数据项，使其总 weight 不超过
目标值：

.. code-block:: python

   batch = sample_channel.get_batch(
       target_weight=global_batch_size,
       key="train",
   )

该功能适合长度或处理开销不同的 sample。普通的单项消费可以保留默认 weight。

使用异步操作和背压
--------------------

``put`` 和 ``get`` 默认阻塞。设置 ``async_op=True`` 会返回 work handle，从而将通信
与其他工作重叠：

.. code-block:: python

   put_work = sample_channel.put(sample, async_op=True)
   # Do independent work here.
   put_work.wait()

   get_work = sample_channel.get(async_op=True)
   sample = get_work.wait()

在 asyncio 函数中，应使用 ``await work.async_wait()``。正数 ``maxsize`` 会在目标
队列已满时阻塞 ``put``，从而施加背压。``put_nowait`` 和 ``get_nowait`` 提供非阻塞
队列语义。

Hook 执行模型
-------------

每次 ``put`` 都会在 ``ChannelWorker`` 上按固定顺序执行：

.. code-block:: text

   producer.put(item, key)
            |
            v
   Collector.collect(item, key)
            |
            |  零个或多个 (output_key, output_item)
            v
   Dispatcher.route(output_item, output_key)
            |
            v
   共享队列或某个消费者的私有队列

两个 hook 解决不同问题：

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Hook
     - 输入
     - 决策
   * - ``Collector``
     - 传给 ``put`` 的一个数据项
     - 丢弃、转换、合并、拆分或修改 key。
   * - ``Dispatcher``
     - Collector 产生的每个数据项
     - 保持共享，或分配给一个消费者。

两个 hook 都是同步且有状态的。流量开始前，``setup(ctx)`` 只执行一次。之后
``collect()`` 和 ``route()`` 在 channel worker 上串行执行，因此内部状态不需要锁。

注册 Collector 和 Dispatcher
-----------------------------

在可导入模块中定义 hook，例如 ``my_project/channel_hooks.py``：

.. code-block:: python

   from rlinf.scheduler import (
       Collector,
       Dispatcher,
       register_collector,
       register_dispatcher,
   )


   @register_collector("flatten_batch")
   class FlattenBatchCollector(Collector):
       """Emit each sample in an incoming batch as one queue item."""

       def collect(self, item, key):
           for sample in item:
               yield key, sample


   @register_dispatcher("even")
   class EvenDispatcher(Dispatcher):
       """Assign items to consumers in a fixed rotation."""

       def setup(self, ctx):
           self.consumers = sorted(ctx.consumers)
           self.next_consumer = 0

       def route(self, item, key):
           if not self.consumers:
               return None
           consumer = self.consumers[
               self.next_consumer % len(self.consumers)
           ]
           self.next_consumer += 1
           return consumer

装饰器使用小写名称注册每个类。``Channel.create`` 也接受 hook 实例、hook 类或
``"module:ClassName"`` 路径。对于应用内部 hook，直接传类最简单：

.. code-block:: python

   from my_project.channel_hooks import EvenDispatcher, FlattenBatchCollector
   from rlinf.scheduler import Channel

   sample_channel = Channel.create(
       "Samples",
       collector=FlattenBatchCollector,
       dispatcher=EvenDispatcher,
       producers=[rollout_group],
       consumers=[actor_group],
   )

``rollout_group`` 和 ``actor_group`` 是已启动的 WorkerGroup。Channel setup 将
``actor_group`` 展开为 ``actor:0``、``actor:1`` 和 ``actor:2`` 等稳定消费者 id，
并通过 ``ctx.consumers`` 提供给 hook。

如果 ChannelWorker 进程已经导入注册模块，也可以用注册名称选择 hook：

.. code-block:: python

   sample_channel = Channel.create(
       "Samples",
       collector="flatten_batch",
       dispatcher="even",
       producers=[rollout_group],
       consumers=[actor_group],
   )

框架内置 hook 通常使用注册名称。外部模块始终可以使用
``"my_project.channel_hooks:FlattenBatchCollector"``，或直接传入类。

跟踪一个 Batch 的 Hook 流程
----------------------------

假设一个生产者发送六个 sample：

.. code-block:: python

   sample_channel.put(["s0", "s1", "s2", "s3", "s4", "s5"])

Collector 将一个输入 batch 转换为六个队列项。Dispatcher 随后在三个消费者之间
轮转：

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 消费者
     - 第一次分配
     - 第二次分配
   * - ``actor:0``
     - ``s0``
     - ``s3``
   * - ``actor:1``
     - ``s1``
     - ``s4``
   * - ``actor:2``
     - ``s2``
     - ``s5``

每个 Actor 使用相同 queue key 调用 ``sample_channel.get()``。Channel 自动识别
调用方，并读取其私有队列。当输出数量不能整除消费者数量时，各消费者的分配数量
最多相差一。

选择队列行为
------------

在确实需要数据转换或消费者所有权前，保留默认行为：

.. list-table::
   :header-rows: 1
   :widths: 29 31 40

   * - Dispatcher
     - 队列所有权
     - 适用场景
   * - ``None`` 或 ``"shared"``
     - 每个 key 一个共享队列
     - 任意消费者都可以处理任意数据项。
   * - ``"round_robin"``
     - 每个消费者一个私有队列
     - 分配必须确定且均匀。
   * - ``"least_loaded"``
     - 每个消费者一个私有队列
     - 生产端需要平衡累计分配数量。
   * - ``"least_loaded_stealing"``
     - 支持 stealing 的私有队列
     - 慢消费者不能阻塞空闲 peer。

Dispatcher 返回 ``None`` 时，数据项留在共享队列；返回消费者 id 时，数据项会在
任何消费者调用 ``get`` 之前完成分配。只有阻塞消费者允许从 peer 取任务时才需要
重写 ``rebalance()``。

组合使用 Hook、Key 和 Weight
----------------------------

启用 hook 后，Channel 的 key 和 weight 行为保持不变。Collector 可以修改输入 key，
Dispatcher 则在每个输出 key 内独立路由。加权数据项进入 ``get`` 和 ``get_batch``
使用的共享或私有队列之前，会先完成路由。

完整方法签名参见 :doc:`../reference/api/channel`。要了解一个用于拼接具身 rollout
数据的生产级 Collector，请继续阅读 :doc:`trajectory_collector`。
