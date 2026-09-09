使用 Channel 进行通信
=======================

channel 模块为 Worker 之间的异步数据交换提供了一个高层次的 **分布式生产者–消费者队列** 抽象。  
一个 ``Channel`` 允许一个或多个生产者 Worker 向命名队列中 ``put`` 数据项，  
并允许一个或多个消费者 Worker ``get`` 这些数据项，  
同时可以选择基于每个数据项的权重来累积 **批次**。

Channel 的创建与连接
--------------------------------

可以通过如下方式创建一个新的 channel::

    Worker.create_channel(
        channel_name,
        maxsize=0,
        distributed=False,
        node_rank=0,
        local=False,
    )

该方法：

- **决定队列放在哪里** — ``local=True`` 时队列留在调用进程内部，其他 Worker 无法连接。否则由 ``ChannelWorker`` actor 持有：放在 ``node_rank`` 指定的节点上；``distributed=True`` 时则每个节点各放一个。
- **启动 actor** — 用 ``NodePlacementStrategy`` 在选定的一个或多个节点上启动持有队列的 ``ChannelWorker``。如果同名 channel 已经存在，会直接连接到它，而不是报错。
- **返回** 一个封装该 actor 的 ``Channel`` 对象。

分布式 channel 会在某个 ``key`` 第一次被使用时把它绑定到一个副本上，选的是调用方所在节点的那个副本，数据因此留在产生它的地方。只有当同一个 key 总是从同一个节点产生时这才划算；如果 key 来自任意节点，就只剩下路由开销而没有局部性，此时 ``distributed`` 保持 False 即可。

若要从其他 Worker 连接到已存在的 channel，请使用::

    Worker.connect_channel(channel_name)

该方法会在 Ray 命名空间中查找对应的 channel actor，并返回一个与该 actor 和当前 Worker 绑定的 ``Channel`` 对象。  


向 Channel 中放入数据
--------------------------------

使用 ``channel.put(item, weight=0, key="default", async_op=False)`` 发送数据。

- 发送 Worker 首先将 ``item`` 传输给实际拥有目标队列的 ``ChannelWorker``。  
- ``ChannelWorker`` 接收数据后，将其封装为一个带有指定 ``weight`` 的 ``WeightedItem``，并放入指定队列。  
  如果队列设置了大小限制（``maxsize`` > 0）且已满，则入队会阻塞，直到队列有空间可用。  


从 Channel 中获取数据
--------------------------------

使用 ``channel.get(key="default", async_op=False)`` 获取数据，这实际上是 ``put`` 的逆过程。  

- ``ChannelWorker`` 会先从指定队列中取出一个数据项。  
- 然后将该数据项发送给请求的 Worker，并最终返回给调用者。  


批量获取
--------------------------------

使用 ``channel.get_batch(batch_weight, key="default", async_op=False)`` 一次获取多个数据。

- ``ChannelWorker`` 会不断从队列中取出数据项，并累加其权重值。  
- 当累计权重达到或超过 ``batch_weight`` 时，停止取数。  
- 所有取出的数据项会组合成一个列表，并通过一次消息发送给请求的 Worker。  

该功能适合在处理体验或任务时动态形成批次，  
当每个数据项有不同的开销或大小（权重）时，可以保证批次大致均匀。  


负载均衡
--------------

在 Rollout 阶段，轨迹长度往往差异较大。  
如果不加设计地直接分配到各个数据并行（DP）训练组，会导致严重的负载不均。

为了解决这一问题，我们实现了基于 channel 的负载均衡机制。  
具体来说，生成阶段的所有生成器会依次将完整的 rollout 轨迹 ``put`` 到共享的 ``rollout_output_queue`` 中。  
由于轨迹按时间顺序插入，``rollout_output_queue`` 中的序列长度会随时间逐渐增长。

然后使用轮询策略，我们不断从 ``rollout_output_queue`` 中 ``get`` 轨迹，  
并依次分配给每个 DP 训练组。  
这种方式能够近似实现各个 DP 训练组之间的工作量均衡，  
从而确保训练过程中的更好利用率和效率。  


示例
--------

.. autoclass:: rlinf.scheduler.Channel
   :no-members:
   :no-index:
   :no-inherited-members:
   :exclude-members: __init__, __new__


总结
--------------------------------

`Channel` 组件为 Worker 通信提供了一个分布式生产者–消费者队列。  
它在集体通信 send/recv 机制的基础上进行了封装，提供了直观的接口，支持优先级和批处理，  
实现了解耦的、异步的数据流，非常适合在并行数据采集与批量消费的强化学习场景中使用。  
