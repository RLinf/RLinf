性能分析
==============================

RLinf 提供两种互补的性能理解工具，面向不同层次：

- **GPU Profiling**（``cluster.profiling``）：对*单个* worker 进程的低层系统级
  profiling。它用特定后端的 profiler（``nsys``/``rocprof-sys``）包装指定的 worker
  group，采集 CUDA kernel、显存流量、NVTX range 和 CPU 活动。用于回答“某个
  GPU/进程逐个 kernel 在做什么？”。
- **Tracing**（``cluster.tracer``）：*跨所有 worker 和 runner* 的 RLinf 自身计时
  区间的高层时间线——step、阶段，以及 ``generate``、``run_training``、``interact``
  等 worker RPC。开销极低，用于回答“各 worker 的阶段如何重叠、关键路径在哪？”。
  见下文 `Tracing`_。

两者相互独立，可同时开启。本页其余部分介绍 GPU Profiling；Tracing 见文末。

RLinf 支持使用特定 profiling 工具包装指定的 worker group。通过必填字段 ``backend``
来选择后端：

- ``nsight`` — NVIDIA Nsight Systems（``nsys profile``）
- ``rocprof_sys`` — AMD ROCm Systems Profiler（``rocprof-sys-python``）

所有后端共享相同的公共字段（``enabled``、``worker_groups``、``steps``、
``output_dir``）。后端专属选项各自独立。


如何启用
------------------------------

在 YAML 的 ``defaults`` 中引入 profiling 预设：

.. code-block:: yaml

   defaults:
     - hybrid_engines/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - profile/default@cluster.profiling

对应的配置文件是 ``examples/embodiment/config/profile/default.yaml``。
如需切换后端，在主 YAML 或 Hydra CLI 中覆盖 ``cluster.profiling.backend``
即可（详见下方各后端专属章节）。


公共字段
------------------------------

以下字段适用于所有 profiling 后端：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 字段
     - 默认值
     - 说明
   * - ``backend``
     - *(必填)*
     - 后端标识：``"nsight"`` 或 ``"rocprof_sys"``。
   * - ``enabled``
     - ``true``
     - 总开关。设置为 ``false`` 可临时关闭，无需删除配置。
   * - ``worker_groups``
     - ``null``
     - 需要 profile 的 worker group 名称列表。``null`` 表示不 profile 任何 worker。
   * - ``steps``
     - ``null``
     - 限定 profiling 的训练 step 索引。``null`` 表示覆盖整个 worker 生命周期。
   * - ``output_dir``
     - *(自动推导)*
     - 输出目录。省略时默认为 ``<log_path>/<experiment_name>/profiling/``。


``enabled`` 开关
------------------------------

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       enabled: false

当 ``enabled: false`` 时：

- 不会用 profiler 命令包装任何 worker。
- 不会创建输出目录。
- 其余 profiling 配置可以保留，方便后续再次开启。


输出目录
------------------------------

默认情况下，报告写入：

.. code-block:: text

   runner.logger.log_path/runner.logger.experiment_name/profiling/

例如：

.. code-block:: text

   ../results/libero_spatial_ppo_openpi/profiling/

如果希望写入固定目录，可以显式设置 ``output_dir``：

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       output_dir: /mnt/public/profiles/my_run


如何覆盖 worker_groups
------------------------------

在主 YAML 中覆盖 ``worker_groups``：

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       worker_groups: [ActorGroup, RolloutGroup]

如果省略 ``worker_groups`` 或设为 ``null``，则不会 profile 任何 worker。

这里有一个容易混淆的点：``ChannelWorker`` 不是 ``ActorGroup`` / ``RolloutGroup``
某个 rank 的子进程，而是通过 ``Channel.create(name)`` 单独 launch 出来的独立
worker group，名字通常就是 ``Env``、``Rollout``、``Actor``。因此只 profile
``ActorGroup`` 并不会自动覆盖 ``Actor`` 这个 channel worker；如果你想看
channel 本身，需要把这些名字显式加进 ``worker_groups``。

对于内置的具身 runner：

- ``ActorGroup``: actor 计算 worker
- ``RolloutGroup``: rollout 计算 worker
- ``EnvGroup``: env 计算 worker
- ``Actor``: 由 ``Channel.create("Actor")`` 创建出来的 channel worker
- ``Rollout``: 由 ``Channel.create("Rollout")`` 创建出来的 channel worker
- ``Env``: 由 ``Channel.create("Env")`` 创建出来的 channel worker


如何只 Profile 特定训练 step
------------------------------

默认情况下，profiling 会覆盖 worker 进程整个生命周期。``steps`` 可以把采样窗口
收窄到指定的训练 step：

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       enabled: true
       steps: [3]            # 只 profile global step 3

多个 step：

.. code-block:: yaml

   cluster:
     profiling:
       steps: [3, 10, 50]

Hydra CLI：

.. code-block:: bash

   python ... '+cluster.profiling.steps=[3]'

当 ``steps`` 被设置时：

- 对于 ``nsight`` 后端，RLinf 会自动把 ``capture-range=cudaProfilerApi`` 和
  ``capture-range-end=stop`` 注入 ``options``。
- 具身 runner 会在每个列出的 step 之前调用 ``torch.cuda.profiler.start()``，
  在该 step 之后调用 ``torch.cuda.profiler.stop()``。
- 最终的 trace 大小由列出的几个 step 决定，与训练总时长无关。


NVIDIA：Nsight Systems（``backend: nsight``）
---------------------------------------------

默认预设
~~~~~~~~~~~~~~~~~~

内置的 ``profile/default`` 预设如下：

.. code-block:: yaml

   backend: nsight
   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup, Actor, Rollout, Env]
   options:
     t: cuda,cudnn,cublas,nvtx,osrt
     sample: process-tree
     cpuctxsw: process-tree
     cudabacktrace: all
     osrt-threshold: 1000
   flags: []

覆盖 ``nsys profile`` 参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``options`` 会被直接映射到带值的 ``nsys profile`` 参数，``flags`` 则用于输出裸 flag：

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         backtrace: fp
         capture-range: cudaProfilerApi
         capture-range-end: stop
       flags: [python-backtrace]

渲染规则：

- 单字符 key → ``-t cuda,...``
- 多字符 key → ``--backtrace=fp``
- ``flags`` 里的项 → ``--python-backtrace``

常用参数：

- ``t``: 需要采集的 API（``cuda``、``cudnn``、``cublas``、``nvtx``、``osrt``）
- ``sample``: CPU sampling 模式
- ``backtrace``: CPU sampling 搭配使用的回溯方式（``lbr``、``fp``、``dwarf``）
- ``cpuctxsw``: CPU 线程调度时间线
- ``cudabacktrace``: CUDA API 调用栈（会增加 overhead）
- ``capture-range`` / ``capture-range-end``: 用 NVTX 或 CUDA profiler API 控制采样窗口

Compute 路径上的 NVTX 注解
~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 在 actor、rollout、env worker 的关键方法上使用
``@Worker.timer("...")``。这个装饰器会记录 RLinf timer 指标，并通过
``AcceleratorUtil.profiling_range`` 打开当前加速器对应的 profiling range。
对 ``nsight`` 后端来说，这些 range 会在时间线和
``nsys stats --report nvtx_sum`` 中显示为带标签的 NVTX 区间。

profiling range 只会在 profiling 窗口打开时发射。profiling 关闭时，
``Worker.timer`` 仍然会记录 timing 指标，而加速器 profiling range 会退化为
no-op。

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

在自定义 worker 上加注解：

.. code-block:: python

   from rlinf.scheduler.worker import Worker

   class MyWorker(Worker):
       @Worker.timer("my_worker/my_phase")
       def my_phase(self, batch):
           ...

Ad-Hoc Profiling Range
~~~~~~~~~~~~~~~~~~~~~~

在函数内部临时圈出一段区间：

.. code-block:: python

   from rlinf.scheduler.hardware import AcceleratorUtil

   class MyWorker(Worker):
       def my_phase(self, batch):
           with AcceleratorUtil.profiling_range(
               self._accelerator_type, "my_worker/inner_phase"
           ):
               run_inner_phase(batch)

``AcceleratorUtil.profiling_range`` 会分发到当前加速器后端。如果该加速器没有
注册 profiling range 实现，或当前 profiling 未开启，它会退化为 no-op。


AMD：ROCm Systems Profiler（``backend: rocprof_sys``）
------------------------------------------------------

最小配置
~~~~~~~~~~~~~~~~~~

在运行配置中内联配置 ``rocprof_sys``：

.. code-block:: yaml

   backend: rocprof_sys
   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup]
   args:
     T: hip           # 采集 HIP API 调用

RLinf 会把每个匹配 worker 的 Python 解释器包装成：

.. code-block:: text

   rocprof-sys-python [args] -- <python_interpreter>

覆盖 ``rocprof-sys-python`` 参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``args`` 映射到带值的 ``rocprof-sys-python`` 参数：

.. code-block:: yaml

   cluster:
     profiling:
       backend: rocprof_sys
       args:
         T: hip,hsa,rccl     # 单字符 key → -T hip,hsa,rccl
         output-format: json  # 多字符 key → --output-format=json

渲染规则：

- 单字符 key → ``-T hip,hsa,rccl``
- 多字符 key → ``--output-format=json``

注入环境变量
~~~~~~~~~~~~~~~~~~

使用 ``env`` 向被 profile 的 worker 注入额外环境变量：

.. code-block:: yaml

   cluster:
     profiling:
       backend: rocprof_sys
       env:
         ROCPROFSYS_SAMPLING_FREQ: "100"

RLinf 会自动从 ``output_dir`` 推导 ``ROCPROFSYS_OUTPUT_PATH`` 和
``ROCPROFSYS_OUTPUT_PREFIX``；``env`` 中显式设置的值优先级更高。


推荐使用方式
------------------------------

NVIDIA 第一轮定位：

- 先用 ``profile/default@cluster.profiling``。
- 保持 ``enabled: true``，用默认 preset 同时采集 CUDA timeline 和 CPU runtime。
- 在确认目标 worker 已经发出 NVTX range 之前，不要急着加 ``capture-range: nvtx``。
- 对长时间训练任务，使用 ``steps: [3]`` 限制 trace 大小。

AMD 第一轮定位：

- 内联配置 ``rocprof_sys``：

  .. code-block:: yaml

     cluster:
       profiling:
         backend: rocprof_sys
         enabled: true
         worker_groups: [ActorGroup, RolloutGroup, EnvGroup]
         args:
           T: hip

- 确认 ``rocprof-sys-python`` 已经在每个 worker 环境的 ``PATH`` 中。
- 运行结束后检查 ``output_dir`` 下的 ``.json`` 或二进制 trace 文件。


Tracing
------------------------------

与 GPU Profiling 聚焦单个进程不同，Tracing 给出全局视角：它记录 RLinf 自身计时
区间在 runner 和所有 Ray worker（环境、Rollout、Actor 等）上的执行时间线，并导出
为 Chrome Trace 格式做交互式可视化。开销极低，是查看各阶段如何重叠、关键路径在哪的
最快方式。

工作原理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 启用追踪后，cluster 会在启动时把 **tracer**（一个调度器 manager，位于节点
   rank 0 的单个 Ray actor）与其他 manager（``WorkerManager``、``NodeManager`` 等）
   一起启动。
2. 每个 worker 和 driver 都通过 ``Tracer.get_proxy()`` 自主访问它，与其他 manager
   完全一致——没有进程级客户端、缓冲区或后台线程。
3. 所有计时区间——runner 的 ``ScopedTimer`` 区间（如 ``step``、
   ``generate_rollouts``）与 worker 的 ``Worker.timer`` 区间（如
   ``run_training``、``interact``、``predict``）——都会发出追踪事件，直接发送给
   tracer manager，由其追加写入 JSONL 追踪文件。

如何启用追踪
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

由于 tracer 是一个 cluster manager，其配置位于 ``cluster.tracer``，无需单独启动
服务器进程：

.. code-block:: yaml

   cluster:
     tracer:
       enable: true
       output_file: null    # 默认为 <log_path>/<experiment_name>/trace/trace_events.jsonl

也可以直接通过 Hydra CLI 开启：

.. code-block:: bash

    python3 examples/embodiment/train_embodied_agent.py \
        --config-name libero_spatial_ppo_openpi_pi05 \
        +cluster.tracer.enable=true

若未设置 ``output_file``，追踪事件会写入节点 rank 0 上的
``runner.logger.log_path/runner.logger.experiment_name/trace/trace_events.jsonl``。

如何添加自定义 Span
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

计时区间会被自动追踪。在 worker 中，``@Worker.timer(...)`` 和
``with self.worker_timer(...)`` 默认会发出追踪事件；传入 ``trace=False``
可以让某个 timer 不参与追踪。若只需要追踪而不需要计时，``Tracer``
提供了 ``trace_begin`` / ``trace_end``、``trace_span`` 上下文管理器和
``trace_func`` 装饰器：

.. code-block:: python

   from rlinf.scheduler import Tracer

   with Tracer.trace_span("data_preprocess", cat="actor"):
       preprocess()

追踪关闭时，所有追踪 API 都是 no-op，因此这些调用可以安全地留在
不开启追踪的代码路径中。

追踪可视化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在运行完成或执行期间，你可以对追踪数据进行可视化：

1. 找到输出的 JSONL 文件（例如 ``trace_events.jsonl``）。
2. 使用 ``jq`` 将 JSON Lines 格式转换为单一 JSON 数组（Perfetto 需要该格式）：

   .. code-block:: bash

       jq -s . trace_events.jsonl > trace_events.json

3. 打开基于 Chrome 的浏览器并导航至 `Perfetto UI <https://ui.perfetto.dev/>`_
   或 ``chrome://tracing``。
4. 点击 **Open trace file** 并选择你转换后的 ``trace_events.json`` 文件。

你将看到一个交互式甘特图，展示执行层次、Actor 等待时间和系统瓶颈，每个进程按其
worker 名（如 ``ActorGroup:0``）或 ``driver`` 标注。
