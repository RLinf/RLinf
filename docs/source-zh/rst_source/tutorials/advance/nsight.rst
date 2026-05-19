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


如何只 Profile 特定训练 step
------------------------------

默认情况下 ``nsys profile`` 会跟随 worker 进程整个生命周期采样。对长时间训练任务
而言，最终产生的 ``.nsys-rep`` 很快就会膨胀到几十甚至上百 MB，关心的那一步反而
不容易找。``cluster.nsight.steps`` 可以把采样窗口收窄到指定的几个训练 step：

.. code-block:: yaml

   cluster:
     nsight:
       enabled: true
       steps: [3]               # 只 profile global step 3

也可以一次列出多个：

.. code-block:: yaml

   cluster:
     nsight:
       steps: [3, 10, 50]

或者直接在 Hydra CLI 上覆盖：

.. code-block:: bash

   python ... '+cluster.nsight.steps=[3]'

当 ``steps`` 被设置时：

- RLinf 会自动把 ``capture-range=cudaProfilerApi`` 和
  ``capture-range-end=stop`` 注入 ``cluster.nsight.options``，你不必再手动写
  对应的 CLI flag。
- 具身 runner 会在每个列出的 step 之前调用 ``torch.cuda.profiler.start()``，
  在该 step 之后调用 ``torch.cuda.profiler.stop()``，Nsight 只在这些窗口内
  写入数据。
- 最终的 trace 大小由列出的几个 step 决定，与训练总时长无关。

如果不设置 ``steps``\ （默认行为），则保持和过去一致 —— nsys 会采集 worker 进程
整个生命周期。

如果你想在采样窗口内同时看到命名区间，下一节描述的 worker NVTX 注解会
只在 step 窗口打开期间发射，所以一份 ``steps=[3]`` 产生的 trace 会精确显示
step 3 时 actor / rollout / env 的行为。


Compute 路径上的 NVTX 注解
------------------------------

RLinf 在 actor、rollout、env worker 的关键方法上通过
``@NsightProfiler.annotate("...")`` 装饰器加了一批命名 NVTX range。它们会在
Nsight 时间线上以带名字的区间形式显示，也会出现在
``nsys stats --report nvtx_sum`` 的输出里。

装饰器和 ``cluster.nsight.steps`` 共用同一个 step 窗口 —— 只有在
``is_profiling_active()`` 为 true 时才会真正发射 NVTX 事件；关闭 profiling 时
基本没有额外开销（只多一次 bool 读取）。

内置注解一览：

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Worker group
     - NVTX label
     - 覆盖范围
   * - Actor
     - ``actor/recv_traj``
     - 从 rollout / env 侧接收一批 trajectory
   * - Actor
     - ``actor/compute_adv``
     - Advantage / return 计算
   * - Actor
     - ``actor/run_training``
     - Policy / value 优化（forward + backward + optimizer）
   * - Actor
     - ``actor/sync_model_to_rollout``
     - 从 actor 向 rollout 广播权重
   * - Rollout
     - ``rollout/recv_obs``
     - 从 env channel 拉 observation
   * - Rollout
     - ``rollout/predict``
     - 单步 policy forward
   * - Rollout
     - ``rollout/generate``
     - 多步 generation / unroll
   * - Rollout
     - ``rollout/generate_epoch``
     - 一个完整 rollout epoch
   * - Rollout
     - ``rollout/send_actions``
     - 向 env 侧回传 action
   * - Rollout
     - ``rollout/send_traj``
     - 把完成的 trajectory 发回 actor 侧
   * - Rollout（async）
     - ``rollout/poll_weight_sync`` / ``rollout/request_weight_sync``
     - 和 actor 的异步权重同步握手
   * - Env
     - ``env/recv_actions``
     - 从 rollout 侧接收下一批 action
   * - Env
     - ``env/step`` / ``env/bootstrap_step``
     - 单步仿真器步进（以及 episode 起始的 warm-up step）
   * - Env
     - ``env/interact`` / ``env/interact_once``
     - 完整的环境交互循环（以及其中的一次子迭代）
   * - Env
     - ``env/send_obs`` / ``env/send_rollout_trajectories``
     - 向下游发送 observation / 完成的 rollout

在自定义 worker 上加注解的写法是一样的：

.. code-block:: python

   from rlinf.utils.nsight_profiler import NsightProfiler

   class MyWorker(Worker):
       @NsightProfiler.annotate("my_worker/my_phase", color="green")
       def my_phase(self, batch):
           ...

``message`` 参数就是 timeline 上显示的名字，``color`` / ``domain`` 会原样转给
``nvtx.start_range``。如果 worker 进程里没有安装可选的 ``nvtx`` 包，装饰器会
退化成 no-op。


如何写出 Ad-Hoc NVTX Range
------------------------------

如果你想在某个函数内部临时圈出一段区间，并且不希望它受 step 窗口控制，可以用
RLinf 提供的 context manager：

.. code-block:: python

   from rlinf.utils.utils import nvtx_range

   with nvtx_range("actor.forward", color="green"):
       run_actor_forward()

适合用来标注某个函数内部的临时区域，或在代码路径还没稳定到值得加装饰器的时候
作为过渡。

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
