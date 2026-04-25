权重同步
==============================

本文介绍 RLinf 在 具身训练中的 ``weight_syncer`` 机制。它主要用于优化
actor 侧训练权重向 rollout 侧策略模型的同步过程，减少每次参数更新后的通信与加载开销。

当前这套能力主要面向具身训练中的 **FSDP actor + HuggingFace rollout**
链路，也就是 ``examples/embodiment/train_embodied_agent.py`` 与
``examples/embodiment/train_async.py`` 所使用的路径。


为什么需要 Weight Syncer
------------------------------

在具身 RL 中，actor 每完成一次训练更新，通常都需要把最新权重同步给 rollout。
对于 OpenPI、OpenVLA、OpenVLA-OFT、GR00T 等大模型，如果每次都执行全量权重传输，
代价会很高：

- 模型参数量大，全量同步很容易成为 step 时间中的主要瓶颈。
- rollout 侧如果逐次加载完整 ``state_dict``，还会引入额外的显存和 CPU 开销。
- 在 async 场景下，阻塞式全量同步会直接影响 rollout 吞吐与策略新鲜度。

因此，RLinf 将这部分能力抽象为统一的 ``WeightSyncer`` 接口，让不同同步策略
复用相同的发送端 / 接收端流程。


整体接口
------------------------------

``WeightSyncer`` 的核心职责有四个：

- ``init_sender(...)``：发送端一次性初始化。
- ``init_receiver(...)``：接收端一次性初始化。
- ``sync(...)``：发送当前版本权重。
- ``apply(...)``：接收并应用权重，同时返回本次应用的 ``version``。

这意味着 rollout 侧不需要关心底层到底是 patch 同步还是 bucket 同步，
只需要在初始化后统一调用 ``apply(...)`` 即可。

当前实现代码位于 ``rlinf/hybrid_engines/weight_syncer/``，但 YAML 配置入口
仍保持为独立的 ``weight_syncer`` Hydra config group。


当前支持的同步策略
------------------------------

目前 RLinf 提供两种策略：

``patch``
  增量同步。发送端维护一份 snapshot，仅发送相对于 snapshot 发生变化的参数位置与数值。

``bucket``
  全量同步。把完整 ``state_dict`` 按 bucket 切分后顺序发送。


State Dict 设备要求
------------------------------

不同 ``weight_syncer`` 对发送端 ``state_dict`` 的设备要求不同：

``bucket``
  对 ``sync(...)`` 传入的发送端 ``state_dict`` 设备没有特殊要求。参数可以位于
  CPU 或 GPU，bucket syncer 会在发送前按 ``bucket_device`` 和 ``bucket_dtype``
  搬运 / 转换。接收端 ``apply(...)`` 通过 ``load_state_dict`` 加载权重，PyTorch
  会把输入 tensor 拷贝到目标模型参数所在设备并转换为目标参数 dtype。

``patch``
  要求 ``init_sender(...)`` 和 ``sync(...)`` 传入的发送端 ``state_dict`` 位于
  GPU。即使配置为 ``snapshot_device: cpu``，也只是让 sender 侧 snapshot 常驻
  CPU；patch 的差异比较、``nonzero`` 和新值收集仍然在 GPU 上完成。
  如果把发送端 ``state_dict`` 放在 CPU 上，patch 构建会退化为 CPU 扫描，
  无法利用当前优化路径，也不符合 patch 模式的设计目标。

接收端 ``apply(...)`` 会把收到的 patch payload 搬到目标模型参数所在设备后再写入，
但模型结构、参数顺序和初始权重仍需满足 patch 模式的初始化约束。


推荐结论
------------------------------

对目前具身训练中的主流 VLA 配置，推荐默认使用 ``patch``，原因是：

- actor 每步更新后的权重变化通常非常稀疏。
- 通常 actor 与 rollout 会从同一份 checkpoint / model path 初始化。
- patch 模式在同步数据量上通常远小于全量传输。

但需要特别注意一点：

.. warning::

   当前 ``patch`` 模式 **不是独立的完整权重快照** 。  
   它发送的是“相对于发送端 snapshot 的增量”，因此要求 actor 与 rollout 在
   patch 同步开始前已经处于**一致的初始权重状态**。  
   如果你不能保证这一点，就应该先做一次全量同步，或直接使用 ``bucket``。


如何在 YAML 中启用
------------------------------

``weight_syncer`` 被做成了独立的 Hydra config group。

在具身 YAML 中，推荐这样写：

.. code-block:: yaml

   defaults:
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer

对应的配置文件位于：

- ``examples/embodiment/config/weight_syncer/patch_syncer.yaml``
- ``examples/embodiment/config/weight_syncer/bucket_syncer.yaml``


Patch 模式
------------------------------

一个典型的 patch 配置如下：

.. code-block:: yaml

   weight_syncer:
     type: patch
     patch:
       snapshot_device: cpu
       transport_device: cpu
       delta_encoding: true
       compression: none

各字段含义如下：

``type``
  固定为 ``patch``，表示使用增量同步。

``patch.snapshot_device``
  snapshot 所在设备。可选 ``cpu`` 或 ``cuda``。当前推荐优先使用 ``cpu``：
  它可以避免 sender 侧额外占用一份模型大小的 GPU 显存，并且经过 GPU 侧比较、
  异步预取与后台写回等优化后，同步耗时已经接近 ``snapshot_device: cuda``。
  如果 GPU 显存非常充足，``cuda`` 仍然是最直接的低延迟路径。

``patch.transport_device``
  patch 发送前所搬运到的设备。默认可设为 ``cpu``。若需要 GPU 上压缩或 GPU 通信，
  通常设为 ``cuda``。

``patch.delta_encoding``
  是否对 COO 坐标做 delta encoding。默认建议开启。

``patch.compression``
  压缩算法。当前可用值包括：

  - ``none``：不压缩
  - ``nvcomp_lz4``：使用 nvCOMP 在 GPU 上做无损压缩


Patch 模式的工作流程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patch 模式大致分为两个阶段：

1. 一次性初始化阶段

   - 接收端在 ``init_receiver(...)`` 中发送本地模型 metadata。
   - 发送端在 ``init_sender(...)`` 中接收 metadata，并据此建立 snapshot。

   当前 metadata 主要包括：

   - 参数键的固定顺序 ``ordered_keys``
   - 每个张量的原始形状 ``original_shapes``
   - 每个张量在接收端的 dtype

   接收端 **不会保存发送端 snapshot** ，它只保存足够把 patch 正确落到本地模型上的结构信息。
   发送端 snapshot 的 dtype 与接收端对应权重的 dtype 保持一致，因此可以正确支持
   ``bfloat16`` 与 ``float32`` 混合精度模型。

2. 每次同步阶段

   - 发送端比较当前 ``state_dict`` 与 snapshot 的差异。
   - 将变化项组织成 patch 后发送。
   - 接收端收到 patch 后，直接把变化应用到本地模型参数。


CPU Snapshot 的优化路径
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当 ``snapshot_device: cpu`` 时，发送端 snapshot 常驻 CPU，但当前 ``state_dict``
仍然位于 GPU。为了避免把 patch 构建主路径退回到 CPU，RLinf 对这一场景做了专门优化：

- CPU snapshot 使用 pinned memory 保存，便于 CPU 与 GPU 之间进行异步拷贝。
- 每个 tensor 在比较前，会先把对应的 CPU snapshot 异步预取到 state tensor 所在 GPU。
- snapshot 预取使用独立 CUDA copy stream，与其他 tensor 的 GPU 侧比较尽量重叠。
- 差异比较、``nonzero`` 以及新值收集都在 GPU 上完成，避免 CPU 逐元素扫描。
- 生成 patch 所需的 ``rows``、``cols`` 和 ``values`` 会异步拷贝到 pinned CPU staging buffer，
  并通过 ``torch.cuda.Event`` 标记拷贝完成时刻。
- sender 会在 patch 构造完成后立即返回并继续执行后续传输；CPU snapshot 的写回由后台线程完成。
- 下一次 patch 构建开始前，会等待上一轮后台写回完成，从而保证 snapshot 一致性。

因此，``snapshot_device: cpu`` 不再意味着“在 CPU 上做比较”。它的实际路径是：

.. code-block:: text

   CPU snapshot -> GPU prefetch -> GPU compare/nonzero/gather
   -> pinned CPU staging -> background snapshot flush

这种方式用少量额外的异步拷贝和后台写回，换取了显著更低的 GPU 显存占用。
在当前具身 VLA 训练配置中，CPU snapshot 的权重同步耗时已经可以接近 GPU snapshot；
因此当 GPU 显存紧张时，优先选择 ``snapshot_device: cpu`` 通常是更稳妥的默认配置。


Patch 的数据组织方式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前 patch 使用扁平化后的张量索引信息进行表示，核心字段包括：

- ``ordinals``：哪个 tensor 发生了变化
- ``nnz_per_tensor``：该 tensor 里有多少个非零变化项
- ``rows`` / ``cols``：变化位置的二维坐标
- ``values``：这些坐标上的新值
- ``version``：本次同步对应的版本号

这里的“二维坐标”来自内部的 2D COO 视图。张量会被转换成如下形态：

- 标量：按 ``(1, 1)`` 看待
- 一维张量：按 ``(1, N)`` 看待
- 二维张量：保持不变
- 三维及以上张量：按 ``(shape[0], prod(shape[1:]))`` 看待

这样做的目的是把不同形状的参数统一到同一套 patch 表达方式中。


Delta Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当 ``delta_encoding=true`` 时，``rows`` 与 ``cols`` 不直接发送绝对坐标，而是发送差分编码后的结果：

- ``rows`` 发送相邻行坐标的增量
- 如果当前元素仍在同一行内，``cols`` 发送列增量
- 如果切换到新行，``cols`` 发送该行中的绝对列起点

这样做的好处是：

- 索引数值通常会更小
- 更容易 downscale 到 ``uint8`` / ``int32`` 等更紧凑的数据类型
- 对后续压缩也更友好


压缩
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patch 模式的压缩只作用于 patch 本身，而不是完整模型权重。

当前 RLinf 已实现的压缩器有：

- ``none``：直接发送 patch tensor
- ``nvcomp_lz4``：对 ``rows``、``cols``、``values`` 分别执行 GPU 侧无损压缩

如果你启用 ``nvcomp_lz4``，需要满足：

- ``transport_device: cuda``
- 运行环境已安装 ``nvidia-nvcomp-cu12``

如果你是通过 ``bash requirements/install.sh embodied ...`` 安装具身环境，
该依赖会自动随具身公共依赖安装。


什么时候 patch 不合适
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下场景不建议直接使用 patch：

- actor 与 rollout 的初始权重不一致
- 你需要一个显式、稳妥的 bootstrap / full sync
- 参数变化并不稀疏，增量 patch 的收益不明显
- 你希望在排查训练正确性问题时先使用最保守的同步策略


Bucket 模式
------------------------------

一个典型的 bucket 配置如下：

.. code-block:: yaml

   weight_syncer:
     type: bucket
     bucket:
       bucket_size: 536870912
       bucket_dtype: null
       bucket_device: cuda
       is_agent: false
       load_instant: true

各字段含义如下：

``type``
  固定为 ``bucket``，表示全量分桶同步。

``bucket.bucket_size``
  每个 bucket 的最大字节数。

``bucket.bucket_dtype``
  发送时使用的数据类型。若设置为 ``null``，则保留每个张量的原始 dtype；
  若设置为 ``bfloat16``、``float16`` 或 ``float32``，则仅对浮点张量进行转换，
  ``int`` / ``bool`` 等非浮点 buffer 会保留原 dtype，避免状态信息被破坏。

``bucket.bucket_device``
  bucket 所在设备，通常为 ``cuda``。

``bucket.is_agent``
  面向 agent 路径的一些命名兼容选项。具身训练通常保持 ``false``。

``bucket.load_instant``
  是否在接收每个 bucket 后立刻 ``load_state_dict``。


Bucket 模式的特点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bucket 模式会把完整 ``state_dict`` 切成多个块顺序发送。它的特点是：

- 优点：语义简单，适合做全量同步与自举。
- 优点：不依赖发送端 snapshot，也不要求稀疏更新。
- 缺点：通信量通常远大于 patch。

如果 ``load_instant=true``，接收端会在每个 bucket 到达时立即加载。  
如果 ``load_instant=false``，接收端会先暂存，最后再统一加载。


Async 场景下的行为
------------------------------

在 async 具身训练中，如果启用了 ``actor.sync_weight_no_wait=true``，
rollout 侧的权重接收与应用会放到后台 ``asyncio`` task 中执行。

这意味着：

- actor 发起同步请求后，rollout 不一定立刻阻塞等待。
- 新权重要等到后台任务完成后，才真正对 rollout 生效。
- 权重的“请求时刻”与“生效时刻”之间可能存在一个小延迟。

因此在 async 场景下，``version`` 的传递就比较重要。  
当前 ``WeightSyncer.apply(...)`` 会返回本次真正应用到 rollout 上的版本号，
rollout 再据此更新自身版本状态。


性能建议
------------------------------

如果你的目标是优先优化同步耗时，建议按下面顺序调：

1. 先使用 ``patch``，确认初始权重一致。
2. 默认优先使用 ``snapshot_device: cpu``，在不额外占用一份模型大小 GPU 显存的前提下，
   获得接近 GPU snapshot 的同步耗时。
3. 保持 ``delta_encoding: true``。
4. 先用 ``compression: none`` 跑通，再评估是否需要 ``nvcomp_lz4``。
5. 如果 GPU 显存非常充足且追求最低同步延迟，可以评估 ``snapshot_device: cuda``。
6. 若确实需要 full sync 或排查问题，再切回 ``bucket``。

需要注意的是，patch 模式会额外保存一份 sender 侧 snapshot。若
``snapshot_device: cuda``，这部分会占用 GPU 显存，大小约为模型参数量乘以
接收端对应权重 dtype 的字节数。
因此在大模型或显存较紧张的配置下，需要为 snapshot 预留显存，避免训练或同步时
OOM。

若 ``snapshot_device: cpu``，这部分 snapshot 不占用 GPU 显存，但会占用一份
CPU pinned memory。其大小同样约为模型参数量乘以接收端对应权重 dtype 的字节数。
该模式下 patch 比较仍在 GPU 上完成，并通过预取、事件同步和后台写回减少 CPU snapshot
带来的额外延迟。对于显存紧张的训练任务，这是当前更推荐的配置。此外，
``nvcomp_lz4`` 需要 ``transport_device`` 为 ``cuda``。

限制与注意事项
------------------------------

当前实现有以下限制需要注意：

- ``patch`` 模式默认假设 actor 与 rollout 以相同初始权重启动。
- ``patch`` 模式当前主要为具身 HuggingFace rollout 路径设计。
- 高维张量在内部会被转成 2D 视图；若 trailing 维无法以 view 的方式展平，patch 模式会报错。
- 当前文档中的压缩配置仅指 patch payload 压缩，不是模型权重本体压缩。
- 如果你只是希望最稳妥地“先跑通同步”，请优先选 ``bucket``；如果你要追求高效同步，再切到 ``patch``。


推荐使用方式
------------------------------

可以用下面这条经验法则快速决策：

- 默认训练：先用 ``patch``
- 首次自举 / 排障：先用 ``bucket``
- 确认稀疏度高且追求极致性能：``patch + delta_encoding + 可选 nvcomp``

如果你不确定当前链路是否满足 patch 的前提，最安全的做法是：

- 先确保 actor 与 rollout 加载同一模型路径 / checkpoint
- 先用 ``bucket`` 验证正确性
- 再切到 ``patch`` 做性能优化
