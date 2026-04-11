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
   所有 **CUDA 张量必须是连续的**；非连续张量会触发错误提示。  
   不允许在同一列表/字典中混合 CPU 与 CUDA 张量。  

.. warning::
   ``send_tensor`` **必须** 与 ``recv_tensor`` 配对使用（反之亦然）。  
   不要在同一消息中将它们与通用的 ``send``/``recv`` 混用。  


异步 API
---------------------------------

所有 P2P API 都支持异步操作，并在 ``async_op=True`` 时返回可等待的 **work handles**。  
内部实现中，提供了一个小型的层次结构：

- ``AsyncWork``：抽象基类，包含 ``wait()``、``async_wait()``、``then(func, *args, **kwargs)``、``done()``，以及链式操作辅助函数（``get_next_work()``、``get_last_work()``）。  
- ``AsyncFuncWork``：在前序任务完成时执行 Python 回调，记录一个 CUDA 事件，并可通过 ``then`` 进行链式调用。若回调返回另一个 ``AsyncWork``，则完成会延迟到链中**最后**的任务完成。  
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
