自适应点对点通信
===================================

该组件在 PyTorch ``torch.distributed`` 之上为 Worker 之间提供 **严格顺序** 和 **异步句柄** 的点对点 (P2P) 数据传输。  
它包含两个对外的类：

- **Collective**：每个 Worker 的单例，用于创建/缓存通信组。  
- **CollectiveGroup**：一个两节点的通信组，实现了张量、张量列表/字典以及可序列化 Python 对象的 P2P send/recv。  


组的创建与缓存
----------------------------------------

``Collective`` 类在每个 Worker 上实例化（每个 Worker 一个单例），  
负责创建并缓存 ``CollectiveGroup`` 实例。  
当两个 Worker 或一组 Worker 需要通信时，必须建立一个包含所有参与者的 collective group。  
在本框架中的典型用法是通过  
``Collective.create_collective_group(worker_addresses, group_name=None)``  
来形成点对点通信组，该方法会返回给定 Worker 地址集合的现有 ``CollectiveGroup``，或者新建一个。  


.. _collectivegroup_p2p:

点对点通信
-------------------------------------

``CollectiveGroup`` 是 RLinf 中管理两个 Worker 间点对点通信的核心抽象。  
它会根据 ``group_info`` 确定本地 rank（0 或 1），并在首次使用时 **延迟初始化** 通信进程组。  

在内部，会分别为 GPU (NCCL) 和 CPU (Gloo) 创建独立的 **发送** 和 **接收** 进程组，形成专用的单向通道；  
在双 Worker 设置中，精心配置的广播等价于 send/recv。  
初始化过程使用 TCP rendezvous 协调端口分配与同步，确保双方准备就绪。  
每个方向都有一个基于专用 CUDA stream 的工作队列，严格保证 send/recv 操作的顺序，避免消息交错。  

建立进程组后，``CollectiveGroup`` 可以执行通信。主要 API 有：

- **Send**: ``send(obj, async_op=False)``  
  向组内的另一方发送一个对象（张量、张量列表、张量字典或任意可序列化对象）。  
  此方法会先发送一个小的 **header**，指明对象类型，以便接收端正确解析负载。  

- **Recv**: ``recv(async_op=False)``  
  从对端接收一个对象。它首先接收类型码（CPU/Gloo），然后调用相应的接收器重建对象。  

- **Direct Tensor Send/Recv**: ``send_tensor(tensor, async_op=False)`` 与 ``recv_tensor(tensor, async_op=False)``  
  针对仅传输单个张量且接收端已分配好张量缓冲区的情况进行了优化，避免了额外的元数据往返。  

.. note::
   所有 tensor 都必须连续；非连续 tensor 会触发错误提示。同一个受支持的 tensor 容器中
   可以同时包含 CPU 与 accelerator tensor，系统会将它们划分到对应的通信路径。

.. warning::
   ``send_tensor`` **必须** 与 ``recv_tensor`` 配对使用（反之亦然）。  
   不要在同一消息中将它们与通用的 ``send``/``recv`` 混用。  


Tensor 压缩
---------------------------------

当网络传输时间高于额外的 CPU 开销时，可以启用无损 CPU tensor 压缩。压缩是可选的
作业级能力：driver 校验一份配置，并将其下发到所有 Worker。

适用范围
~~~~~~~~

压缩仅作用于通用 ``Worker.send``/``Worker.recv`` 优化路径携带的 CPU tensor：单个
tensor、tensor list 或 tuple、值全为 tensor 的 dictionary，以及从 dataclass 中提取的
tensor 字段。对于同时包含 CPU 和 accelerator tensor 的容器，只有 CPU tensor 会成为
候选。通过 Worker send/recv 通信的 Channel 会继承相同行为。

任意 pickled Python object、accelerator tensor、``broadcast``，以及直接调用
``send_tensor``/``recv_tensor`` 的路径不会压缩。这些边界是有意保留的：

- 任意 object 路径主要承载控制数据和 metadata，其序列化 payload 通常很小；再执行一次
  codec、申请 buffer 并发送压缩 header，开销通常会高于节省的字节数。包含大量 tensor 的
  数据应使用 tensor list、dictionary 或 dataclass 路径，避免 tensor storage 经由 pickle，
  并让符合条件的 CPU tensor 直接参与压缩。
- accelerator tensor 通常使用 NCCL 类 collective 或同设备 IPC 等高吞吐设备通信。CPU
  压缩会引入设备同步、device-to-host 与 host-to-device copy，破坏 accelerator 通信流水线；
  常见的浮点模型 tensor 也可能难以压缩，因此额外工作可能带来负收益。
- 直接 ``send_tensor``/``recv_tensor`` 的接收端已经持有目标 tensor，因此该路径会刻意
  省略 metadata exchange。压缩会引入可变 wire size 和 framing metadata，从而破坏该快速
  路径的契约。
- ``broadcast`` 使用独立的多接收端、拓扑感知传输路径。本次改动只处理点对点 payload
  ownership 与 fallback，不改变 broadcast scheduling 或 buffer lifetime。

tensor list 的元素仍然是独立的 wire payload；该功能只减少每个 payload 的字节数，不会
将它们拼凑起来。

选择 Codec
~~~~~~~~~~

两种 codec 都是无损的。它们处理连续 CPU tensor 的原始字节，并逐字节恢复原始 dtype 和
shape。压缩和解压缩直接在 tensor 与预分配的 ``torch.uint8`` buffer 之间进行，不会创建
中间 Python ``bytes`` object。

.. list-table:: Codec 取舍
   :header-rows: 1
   :widths: 14 30 24 32

   * - Codec
     - 特征
     - Codec 参数
     - 适用情况
   * - ``lz4``
     - 优先保证压缩和解压缩速度，CPU 开销相对较低，但压缩率通常低于 Zstd。
     - ``acceleration`` 控制 LZ4 fast compression。值越高越偏向速度，并可能降低压缩率。
     - CPU 时间敏感，或链路仅中度受限。可以从它开始。
   * - ``zstd``
     - 通常比 LZ4 减少更多 wire bytes，但压缩和解压缩开销更高。
     - ``level`` 控制压缩率与 CPU 的取舍；``max_inflight`` 限制并发复用的 context 数量。
     - 链路足够慢，减少 wire bytes 的收益高于 codec 开销。

请使用有代表性的 payload 实测后再选择。已经压缩或高熵的 tensor 可能无法缩小，而包含
大量零值或重复值的 dense tensor 可能有明显收益。LZ4 还存在单 tensor 输入大小上限；
不支持的大小会自动走 raw 路径。

配置压缩
~~~~~~~~

.. code-block:: yaml

   cluster:
     collective:
       tensor_buffer_pool:
         max_bytes: 2147483648
       tensor_compression:
         enabled: true
         codec: lz4
         min_bytes: 16384
         excluded_dtypes: [float32]
         acceleration: 1

使用 Zstd 时，用 ``codec`` 选择它，并且只传入 Zstd 参数：

.. code-block:: yaml

   cluster:
     collective:
       tensor_compression:
         enabled: true
         codec: zstd
         min_bytes: 16384
         excluded_dtypes: [float32]
         level: 1
         max_inflight: 4

省略 ``tensor_compression``，或设置 ``enabled: false``，即可使用原始 wire 路径。压缩
选项及默认值如下：

.. list-table:: 压缩选项
   :header-rows: 1
   :widths: 20 16 64

   * - 选项
     - 默认值
     - 含义
   * - ``enabled``
     - ``true``
     - 存在 ``tensor_compression`` 配置段时启用压缩。
   * - ``codec``
     - 必填
     - 选择承载本段其余参数的 codec 配置：``lz4`` 或 ``zstd``。
   * - ``min_bytes``
     - ``16384``
     - 跳过 raw byte count 小于该值的 tensor。
   * - ``excluded_dtypes``
     - ``[float32]``
     - 跳过列表中 dtype 对应 tensor 的 codec attempt。设为 ``[]`` 可让所有 dtype
       都参与判断。
   * - codec 参数
     - 由 codec 决定
     - 由所选 codec 声明，并写在同一段中。LZ4 接受 ``acceleration``\ （默认
       ``1``）；Zstd 接受 ``level``\ （默认 ``1``）与 ``max_inflight``\ （默认
       ``4``）。

``codec`` 选择一个 codec 配置类，与之并列的参数也只接受该类声明的字段。契约如下：

.. list-table:: Codec 参数
   :header-rows: 1
   :widths: 16 20 14 50

   * - Codec
     - 参数
     - 默认值
     - 校验与行为
   * - ``lz4``
     - ``acceleration``
     - ``1``
     - 必须不小于 ``1``。值越高，LZ4 用于寻找 match 的时间通常越少，以压缩率换取速度。
       LZ4 是无状态的，因此 Worker 共享一个 codec，不设 acquisition limit。
   * - ``zstd``
     - ``level``
     - ``1``
     - 必须不小于 ``1``。level 越高通常会使用更多 CPU 时间寻找更好的压缩率。
   * - ``zstd``
     - ``max_inflight``
     - ``4``
     - 必须不小于 ``1``。它限制每个 Worker 内由 compression 和 decompression 共享的
       可复用 Zstd codec 数量。codec 繁忙时 sender 不会等待，而是保持 raw transfer；已经
       收到 compressed payload 后，receiver 会等待 codec。

每种 codec 只接受自己的参数：LZ4 会拒绝 ``max_inflight``，Zstd 会拒绝
``acceleration``，二者都与其他 ``cluster`` 配置项的未知键一样报错。driver 在解析
cluster 配置时统一校验该配置段，并随之下发到所有 Worker；因此 wire metadata 只需标识
codec，receiver 已经持有相同的作业级参数。

``tensor_buffer_pool`` 独立于 ``tensor_compression``。它的 ``max_bytes`` 限制单个
Worker 内 active 与 cached CPU buffer 的总容量，默认值为 2 GiB。配置压缩但省略该段时，
会自动提供默认 pool。

运行与回退
~~~~~~~~~~

每个 Worker 在加载作业级配置时会 probe 已配置的系统 codec library，因此缺少依赖会在
Worker 启动期间失败。native codec context 和 buffer 仍然延迟创建：每个 Worker 在加载配置
时创建一个空的 ``TensorBufferPool``，并在首个符合压缩条件的 tensor 出现时创建一个
``TensorCodecProvider``。两者都由该 Worker 的所有 ``CollectiveGroup`` 共享。LZ4 无锁、
无 slot 限制地共享一个无状态 codec；Zstd 则从由 compression 和 decompression 共享的单个
有界 LIFO queue 中独占 codec。发送流程如下：

1. 获取 provider 的 codec。LZ4 不会耗尽；Zstd codec queue 饱和时不会等待，本次传输保持
   raw。
2. 按原始顺序遍历 CPU tensor，根据 ``min_bytes`` 和 ``excluded_dtypes`` 筛选，并为每个
   符合条件的 tensor 尝试获取 buffer。当 codec 不支持该 tensor 的大小，或者预算内没有
   可用 buffer 时，该 tensor 保持 raw。
3. 只有压缩结果小于原始 tensor 时才使用它；否则保持 raw，并直接丢弃该 buffer，而不是
   将其放入 cache。
4. 在现有 metadata 中发送每个 tensor 的压缩大小。接收端直接将压缩 tensor 恢复到预分配
   的目标 tensor。
5. 压缩结束后立即释放 codec。仅保持压缩 payload 的 buffer lease，直到同步 payload send
   完成，再将 buffer 返回 Worker buffer pool。接收端在 compressed wire payload 到达后才
   获取 decoder，并在解压完成后立即释放。

同时启用 ``cluster.net_emulation`` 时，bandwidth 按实际压缩后的 CPU tensor bytes 计费，
并保留原 payload 的 metadata 估算；raw 或不符合压缩条件的 tensor 仍使用原计费方式。

buffer pool 按 capacity 索引 idle buffer，并查找能够容纳请求的最小 cached size。当其
capacity 不超过请求大小的两倍，或分配精确大小的 buffer 会超出预算时，pool 会复用该
buffer；否则会分配精确大小的新 buffer。相同 size 使用独立 list，active 与 cached 容量
共同受 ``max_bytes`` 限制。当新分配需要空间时，pool 会从最大的 idle size bucket 开始
淘汰 buffer。buffer acquisition 不会等待；因此 buffer 不可用时，该 tensor 会保持
baseline 行为。

公共依赖安装会安装这两种 codec 所需的 LZ4 和 Zstandard 系统库。请确保所有 Worker 节点
使用相同版本的 RLinf，并安装相应的 compression 依赖。


异步 API
---------------------------------

所有 P2P API 都支持异步操作，并在 ``async_op=True`` 时返回可等待的 **work handles**。  
内部实现中，提供了一个小型的层次结构：

- ``AsyncWork``：抽象基类，包含 ``wait()``、``async_wait()``、``then(func, *args, **kwargs)``、``done()``，以及链式操作辅助函数（``get_next_work()``、``get_last_work()``）。  
- ``AsyncFuncWork``：在前序任务完成时执行 Python 回调，记录一个 CUDA 事件，并可通过 ``then`` 进行链式调用。若回调返回另一个 ``AsyncWork``，则完成会延迟到链中最后的任务完成。  
- ``AsyncCollWork``：将一个 ``torch.distributed`` 的工作（如 broadcast）封装为可等待接口。它也支持 ``then`` （单一底层任务）。  
- ``AsyncChannelWork``：将 ``ray.ObjectRef`` 封装为可等待对象（用于 channel RPC）。  

关键特性：

* **等待：** ``wait()`` 为阻塞式；``async_wait()`` 适合 ``asyncio``，两者都会确保记录的 CUDA 事件完成后返回。  
* **链式调用：** ``then`` 可调度后续回调。  
* **完成检测：** ``done()`` 为非阻塞查询，用于检测底层任务是否完成。  

最小示例：

.. code-block:: python

   # 使用 await 的异步对象 send/recv
   send_work = group.send(obj, async_op=True)      # AsyncWork
   await send_work.async_wait()                    # 非阻塞等待

   recv_work = group.recv(async_op=True)           # AsyncWork
   obj = recv_work.wait()                          # 阻塞等待；返回接收到的对象

.. code-block:: python

   # 链式调用后处理步骤
   def postprocess(buf):
       # 例如：转移到 CPU、类型转换或通知其他子系统
       return None

   w = group.recv_tensor(tensor, async_op=True)    # 接收端预分配的张量
   w2 = w.then(postprocess)                        # AsyncFuncWork
   w2.wait()                                       # 确保 postprocess 完成


总结
--------------

总之，**collective** 组件为 Worker 之间的点对点数据传输提供了引擎。  
它屏蔽了 PyTorch 分布式后端的复杂细节，通过管理多个进程组来模拟 send/recv，并对 GPU 传输进行了优化。  
框架用户通常通过 `Worker.send/recv` 或 channel 操作来调用这些功能，而不是直接调用 `CollectiveGroup`。  
