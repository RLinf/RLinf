Nsight Systems
==============================

本文介绍 RLinf 中基于 ``cluster.nsight`` 的系统级 Profiling 配置，用于通过
NVIDIA Nsight Systems 对指定 Ray worker group 执行 ``nsys profile`` 包装。

借助这套机制，你可以采集 CUDA kernel、cuDNN、cuBLAS、NVTX，以及可选的
CPU runtime 相关时间线。

如何启用
------------------------------

在具身 YAML 的 ``defaults`` 中引入 Nsight 预设：

.. code-block:: yaml

   defaults:
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - nsight/default@cluster.nsight

对应的配置文件是：

- ``examples/embodiment/config/nsight/default.yaml``


默认预设
------------------------------

内置的默认预设如下：

.. code-block:: yaml

   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup, Actor, Rollout, Env]
   options:
     t: cuda,cudnn,cublas,nvtx,osrt
     sample: process-tree
     cpuctxsw: process-tree
     cudabacktrace: all
     osrt-threshold: 1000
   flags: []

这份默认配置会优先采样具身训练里最常见的计算 worker 和通信 worker：

- ``ActorGroup``
- ``RolloutGroup``
- ``EnvGroup``
- ``Actor``
- ``Rollout``
- ``Env``

这里的名字必须和真实的 worker group 名一致，例如 ``actor.group_name``、
``rollout.group_name``，而不是组件别名 ``actor`` 或 ``rollout``。

这份 preset 默认会保留 CPU sampling，并额外开启 ``cudabacktrace``，因此第一轮
profiling 时就能同时看到 CUDA 侧时间线和 CUDA API 调用栈。


``enabled`` 开关
------------------------------

``enabled`` 是 Nsight 的总开关：

.. code-block:: yaml

   cluster:
     nsight:
       enabled: false

当 ``enabled: false`` 时：

- RLinf 不会用 ``nsys profile`` 包装 worker
- RLinf 不会预留默认的 Nsight 输出目录
- 其余 profiling 配置可以保留，方便后续再次开启

如何覆盖 worker_groups
------------------------------

你可以直接在主 YAML 里覆盖这份预设：

.. code-block:: yaml

   cluster:
     nsight:
       worker_groups: [EnvGroup, RolloutGroup, ActorGroup, Env, Rollout, Actor]

这对以下场景很有用：

- 采 actor / rollout 这类计算 worker
- 采 ``Env``、``Rollout``、``Actor`` 这类 channel worker
- 采 ``EnvGroup`` 这类环境 worker

如果省略 ``worker_groups``，RLinf 会对所有 worker group 开启 profiling。

这里有一个容易混淆的点：当前实现里的 ``ChannelWorker`` 不是
``ActorGroup`` / ``RolloutGroup`` 某个 rank 的子进程，而是通过
``Channel.create(name)`` 单独 launch 出来的独立 worker group，名字通常就是
``Env``、``Rollout``、``Actor``。因此只 profile ``ActorGroup`` 并不会自动
覆盖 ``Actor`` 这个 channel worker；如果你想看 channel 本身，需要把这些名字
显式加进 ``worker_groups``。

对于内置的具身 runner，这几类名字和实际含义可以直接对应起来：

- ``ActorGroup``: actor 计算 worker
- ``RolloutGroup``: rollout 计算 worker
- ``EnvGroup``: env 计算 worker
- ``Actor``: 由 ``Channel.create("Actor")`` 创建出来的 channel worker
- ``Rollout``: 由 ``Channel.create("Rollout")`` 创建出来的 channel worker
- ``Env``: 由 ``Channel.create("Env")`` 创建出来的 channel worker

所以 ``worker_groups: [Actor]`` 的含义是“profile Actor 这个 channel worker”，
而不是“profile 所有 actor 侧计算”。当前这套匹配机制本来就是按 worker group
name 做判断，因此这里必须填写真实的 group name。若你在自定义 runner 里创建了
别的 channel 名字，也应当把那个精确名字写进 ``worker_groups``。


如何覆盖 Nsight 参数
------------------------------

``cluster.nsight.options`` 会被直接映射到那些“带值”的 ``nsys profile`` 参数，
而 ``cluster.nsight.flags`` 则用于输出裸 flag：

.. code-block:: yaml

   cluster:
     nsight:
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         cpuctxsw: process-tree
         cudabacktrace: all

常用参数包括：

- ``t``: 需要采集的 API，例如 ``cuda``、``cudnn``、``cublas``、``nvtx``、``osrt``
- ``sample``: CPU sampling 模式
- ``backtrace``: CPU sampling 搭配使用的回溯方式，例如 ``lbr``、``fp``、``dwarf``
- ``cpuctxsw``: CPU 线程调度时间线
- ``cudabacktrace``: 采集 CUDA API 调用栈；它依赖 CPU sampling，并且可能明显增加 overhead
- ``capture-range`` 和 ``capture-range-end``: 用 NVTX 或 CUDA profiler API 控制采样窗口
- ``o`` 或 ``output``: 显式指定输出前缀

如果你开启了 ``capture-range: nvtx``，请确认代码里确实发出了 NVTX range；
否则 Nsight 很可能只会生成几乎没有内容的空 report。

你并不局限于 ``nsight/default`` 里已经出现的那些 key。RLinf 会把
``cluster.nsight.options`` 中的任意新增项继续透传给 ``nsys profile``：

.. code-block:: yaml

   cluster:
     nsight:
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         backtrace: fp
         capture-range: cudaProfilerApi
         capture-range-end: stop
         samples-per-backtrace: 4
       flags: [python-backtrace]

这是因为 RLinf 会把 ``cluster.nsight.options`` 当作一个自由字典来渲染：

- 单字符 key 会被渲染成 ``-t cuda,...`` 这种形式
- 多字符 key 会被渲染成 ``--backtrace=fp`` 这种形式
- ``flags`` 里的项会被渲染成 ``--python-backtrace`` 这种形式

如果你想输出一个“不带值”的 flag，可以把它写进 ``cluster.nsight.flags``：

.. code-block:: yaml

   cluster:
     nsight:
       flags: [python-backtrace]

同样也可以通过 Hydra CLI 覆盖：

.. code-block:: bash

   python ... 'cluster.nsight.flags=[python-backtrace]'

这对那些“值是可选的”，并且裸 flag 形式有特殊语义的 ``nsys`` 参数尤其有用。

不同 Nsight Systems 版本、不同主机平台支持的参数和推荐取值并不完全一致。尤其是
``backtrace`` 的可用模式和效果，可能需要按机器调整。如果目标节点上的某个参数
不被接受，或者 ``lbr`` 效果不好，请直接在目标节点执行 ``nsys profile --help``，
然后覆盖 ``cluster.nsight.options``，例如把 ``backtrace`` 改成 ``fp`` 或
``dwarf``。


如何打 NVTX Range
------------------------------

RLinf 已经提供了一个现成的 NVTX helper：

.. code-block:: python

   from rlinf.utils.utils import nvtx_range

   with nvtx_range("actor.forward", color="green"):
       run_actor_forward()

如果你准备开启 ``capture-range: nvtx``，或者只是希望在 Nsight 时间线里看到更高层
的命名区间，这通常是最方便的用法。

这个 helper 会优先尝试可选的 ``nvtx`` Python 包；如果没装这个包，则会在 CUDA
可用时回退到 ``torch.cuda.nvtx``；如果两者都不可用，它就会退化成 no-op。因此把
它留在同时支持“带 NVTX / 不带 NVTX”两种环境的代码路径里通常也是安全的。

如果你使用了 ``capture-range: nvtx``，请确认被 profile 的 worker 确实会执行到
这些 range；否则 Nsight 可能只会采到很少的数据，甚至得到几乎为空的 report。


输出路径
------------------------------

当 ``cluster.nsight.enabled`` 为 true，且没有显式指定 ``o`` / ``output`` 时，
RLinf 默认会把 report 写到：

.. code-block:: text

   runner.logger.log_path/runner.logger.experiment_name/nsights

例如：

.. code-block:: text

   ../results/libero_spatial_ppo_openpi/nsights/

如果你希望写入固定目录，可以显式覆盖：

.. code-block:: yaml

   cluster:
     nsight:
       options:
         o: /mnt/public/profiles/my_run/worker_trace


推荐使用方式
------------------------------

第一轮定位问题时，最简单的用法通常是：

- 先用 ``nsight/default@cluster.nsight``
- 保持 ``enabled: true``
- 如果你既想看 CUDA timeline，也想看 CPU/channel 侧 runtime 行为，默认 preset 可以直接用
- 在确认目标 worker 已经打出 NVTX 之前，不要急着加 ``capture-range: nvtx``
