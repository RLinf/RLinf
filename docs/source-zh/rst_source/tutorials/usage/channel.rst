流水线中的 Channel 队列
===============================

channel 模块为 Worker 之间的异步数据交换提供了一个高层次的 **分布式生产者–消费者队列** 抽象。  
一个 ``Channel`` 允许一个或多个生产者 Worker 向命名队列中 ``put`` 数据项，  
并允许一个或多个消费者 Worker ``get`` 这些数据项，  
同时可以选择基于每个数据项的权重来累积 **批次**。

Channel 的创建与连接
--------------------------------

可以通过如下方式创建一个新的 channel::

    Worker.create_channel(
        channel_name,
        node_id=0,
        maxsize=0
    )

该方法：

- **确定放置位置** — 如果未指定 ``group_affinity`` 或 ``group_rank_affinity``，则 channel 会托管在当前 Worker 的 **group** 和 **rank** 上（即相同节点和 GPU）。  
- **启动专用的 channel actor** — 使用 ``PackedPlacementStrategy`` 在所选节点/GPU 上启动一个 ``ChannelWorker`` （实际持有队列），并设置 ``num_processes=1``。  
- **返回** 一个 ``Channel`` 对象，用于封装该 actor。channel actor 的地址为 ``channel_name:0``。  

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
