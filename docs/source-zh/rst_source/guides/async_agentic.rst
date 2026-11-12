异步 agentic RL
===============

本页先说明如何在 RLinf 里打开异步 agentic RL、哪些配置可能需要改，再解释底下的机制：rollout 与训练怎样重叠，两个准入上限各管什么，训练数据会旧到什么程度。如果只需要一份能直接跑的配置，先看 :doc:`Search-R1 异步训练 </rst_source/examples/agentic/searchr1/async>`；等到要改某个默认值、想知道该改哪个以及为什么时，再回到这里。

如何开启
--------

在 agentic 配置上打开异步，只需要一个 runner 开关，外加两条部署要求：actor 和 rollout 必须放在不同的 GPU 上，因为两者同时运行；rollout 必须使用 SGLang server 后端，因为权重同步要通过 server 的 HTTP 接口暂停和恢复生成。以同步的 Search-R1 配置为基础，需要改的是下面几行：

.. code-block:: yaml

   runner:
     async_mode: True

   cluster:
     component_placement:
       actor: 0-7             # actor 与 rollout 分开部署
       rollout,reward: 8-15

   rollout:
     rollout_backend: sglang
     sglang:
       backend_type: server

   agentloop:
     task_name: searchr1      # 异步模式下必填

``component_placement`` 总要求有 ``reward`` 一项，所以这里保留了它；Search-R1 的 reward 在 agent loop 里计算，不会启动 reward worker。缺少上面任何一项要求的配置，会在配置校验阶段直接报错，不会等到 worker 启动之后。

其余配置都有默认值。下表是最可能需要调整的几项，后文会逐一说明什么时候需要改。

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - 配置项
     - 默认值
     - 作用
   * - ``rollout_buffer.max_inflight_groups``
     - ``ceil(max_running_requests x instances x 1.2 / group_size)``
     - 给 rollout 引擎多少活
   * - ``rollout_buffer.max_buffered_groups``
     - 2 × ``data.rollout_batch_size``
     - 训练数据最多多旧
   * - ``agentloop.max_actor_accept_staleness``
     - 不设置（只统计 staleness，不设上限）
     - 训练前丢弃比它更旧的组
   * - ``rollout_buffer.max_dropped_groups_per_step``
     - 10 × ``data.rollout_batch_size``
     - 丢弃速度超过训练消耗时终止运行
   * - ``rollout.pause_mode``
     - ``retract``
     - 权重同步时如何处理正在生成的请求

运行起来之后，最先要看的是 ``rollout/buffer/admission_wait_inflight_limit`` 和 ``rollout/buffer/admission_wait_buffer_full`` 两个指标，它们说明当前是哪个上限在限制投放。后面两节分别讲遇到哪种情况该怎么调。

异步为什么快
------------

要理解这两个上限，先看异步改变了什么。同步的 agent 训练里，训练和生成这两个最耗时的环节只能轮流进行，总有一边空着：actor 训练一批轨迹时，rollout 引擎没有活干；训练结束后引擎重新满载，actor 又得等这一批里最慢的那条轨迹跑完，才能开始下一步。多轮 agentic rollout 会让情况更糟，一条反复调用工具的轨迹就能把整批拖到最后。

异步训练把两边解耦，不再互相等待。prompt 持续送进 rollout 引擎，生成完的轨迹先放进缓冲，actor 每准备好一步，就取走缓冲里最早的 ``rollout_batch_size`` 个组。

实测的重叠效果接近理想情况。以参考运行为例（第 2–19 步，数据取自 TensorBoard），同步运行的 ``time/step`` 为 278.6 s，由 ``time/rollout`` 181.9、``time/training`` 93.3 和 ``time/sync_weights`` 1.4 组成；异步运行的 ``time/step`` 为 139.0 s，由 ``time/training`` 91.7、``time/wait_data`` 46.0 和 ``time/sync_weights`` 1.3 组成。两者的训练时间几乎一样，省下来的正是双方互相等待的时间。

组件如何协作
------------

异步训练涉及四个组件，我们先认识它们，再看数据怎么在其间流动：

- rollout worker：运行 SGLang server 后端，持有生成引擎；
- agent loop：驱动每条多轮轨迹，依次发出生成请求、解析工具调用、调用工具、把工具返回拼回对话，直到轨迹结束；
- buffer worker（``RolloutBufferWorker``）：持有 dataloader，决定投放哪些 prompt、把哪些已完成的组交给训练；
- actor：完成训练，再把新权重同步回 rollout 引擎。

一个 prompt 的路径很短。buffer worker 从 dataloader 读出 prompt，准入上限允许时，通过 ``dataloader_channel`` 投放给 agent loop；agent loop 用 rollout 引擎跑完这个组的所有轨迹，再通过 ``agent_output_channel`` 把整组交回；buffer worker 把完成的组放进缓冲，每一步把最早的 ``rollout_batch_size`` 个组交给 actor。

重叠之所以可控，关键在 buffer worker。dataloader 由它而不是 runner 持有，所以 actor 训练期间它仍能继续投放；它同时统计在途的组和缓冲里的组，所以两边之间积压的数据有上限；它总是先交出最早的组，所以 actor 拿到的数据变旧的速度由上限决定。

两个准入上限
------------

buffer worker 只在两个上限同时满足时才投放新 prompt。两个上限管的是不同的事：一个决定给引擎多少活，一个决定训练数据最多多旧。每当可能空出额度时，准入都会重新检查：收到一个完成的组、丢弃一个组、向 actor 交付一批。

在途上限：引擎负载
~~~~~~~~~~~~~~~~~~

第一个上限要求在途组数（已投放、尚未交回的组）低于 ``rollout_buffer.max_inflight_groups``。默认值把引擎的请求容量换算成组数，即 ``ceil(max_running_requests x instances x 1.2 / group_size)``；多出的 1.2 倍余量，留给那些轨迹正在等工具返回、或在等组内最慢那条轨迹的组。

只有引擎还有空闲时，调高这个上限才有用。引擎一旦忙满，多投放的组只会在引擎里排队，让数据变旧。在 GLM 参考运行上，把 ``max_inflight_groups`` 从 77 提到 160 后，每个 DP rank 的运行请求数从约 75 升到 128（另有 33–78 个在排队），吞吐没有变化，平均 staleness 却从约 1.3 升到约 2.0：GPU 在约 75 个请求时就已经忙满了。对应的实测图见 :doc:`Search-R1 异步页 </rst_source/examples/agentic/searchr1/async>` 的「两个上限怎么调」一节。

缓冲上限：数据新鲜度
~~~~~~~~~~~~~~~~~~~~

第二个上限要求在途组数加上缓冲里的组数低于 ``rollout_buffer.max_buffered_groups``，默认为 2 × ``rollout_batch_size``。在途组也要算进去，因为它们迟早会进缓冲：组内轨迹一跑完，就无条件放入缓冲。如果只限制缓冲本身，在途部分就没有约束，一旦训练成为瓶颈，缓冲会越堆越多。我们实测过这种错误配置，缓冲涨到 338 和 369 个组，staleness 达到 8。把在途组算进上限后，缓冲不会超过 ``max_buffered_groups``，staleness 也就有了上界。

这个上限怎么取，要看瓶颈在哪一边。生成是瓶颈时，actor 一直在等数据，缓冲随进随出，这时 ``max_buffered_groups`` 至少要取 ``max_inflight_groups + rollout_batch_size``；取得更小，每步末尾缓冲里的组会占掉在途的额度，引擎反而吃不饱。训练是瓶颈时，靠缓冲让生成一直有活干，这时至少要取 2 × ``rollout_batch_size``，下一批才能在本批训练时开始生成。如果只取 1 倍，下一批要等本批交付之后才能开始投放，每一步都要等整批（包括最慢的组）生成完。在缩小规模的 Qwen2.5-1.5B 配置上（``rollout_batch_size`` 为 64）实测，1×B 的步时中位数约 11 s，2×B 为 3.94 s；后者付出的代价是 staleness 从约 1 升到均值峰值 2.09。

判断哪个上限在起作用
~~~~~~~~~~~~~~~~~~~~

瓶颈在哪一边通常不用自己推算。``rollout/buffer/admission_wait_inflight_limit`` 和 ``rollout/buffer/admission_wait_buffer_full`` 分别给出每一步里投放被两个上限挡住的时间占比。``admission_wait_inflight_limit`` 高，说明引擎已经按允许的上限满载，调高上限之前先看 GPU 利用率；``admission_wait_buffer_full`` 高，说明训练是瓶颈，staleness 由缓冲上限控制。

Staleness
---------

默认情况下，staleness 只统计、不设上限：``agentloop.max_actor_accept_staleness`` 不设置，数据有多旧由上面的缓冲上限决定。训练是瓶颈时，缓冲里最早的组大约已经等了 ``max_buffered_groups / rollout_batch_size`` 步。

``rollout/data_staleness/*`` 这组指标记录的就是这个值。对每一步训练的每个组，staleness 等于本步开始时 actor 的权重版本减去该组最早的生成版本，指标给出均值、最大值和各取值的占比。它可以拆成两部分：``rollout/version_span/mean`` 是组在生成过程中跨过的版本数，``rollout/queue_steps/mean`` 是组生成完之后在缓冲里等待的版本数。

需要硬性上限时，设置 ``agentloop.max_actor_accept_staleness``。比上限更旧的组会在训练前被丢弃，丢弃会归还准入额度，于是会有别的 prompt 补投进来；被丢弃的 prompt 要到之后的 epoch 才会再次出现。丢弃的代价是这些组已经花掉的生成算力，所以上限要高于缓冲上限本身带来的 staleness（即上面说的 ``max_buffered_groups / rollout_batch_size`` 步）。如果设得更低，每一批里最早的组都会被丢掉，而不只是偶尔出现的过旧的组，配置校验时也会给出警告。

权重同步与 ``pause_mode``
-------------------------

每个训练步结束后，新权重要在生成仍在进行时同步进 rollout 引擎。``rollout.pause_mode`` 决定同步期间正在生成的请求怎么处理：

- ``retract`` 是默认模式：把正在生成的请求退回引擎队列，再同步权重。已经生成的 token 保留；KV cache 随旧权重一起清空，恢复后重新计算，所以每个被退回的请求要多做一次 re-prefill。
- ``in_place`` 暂停引擎的事件循环：正在运行的请求保留原有 KV cache，同步完成后用新权重继续解码。
- ``drain`` 等所有正在生成的请求结束，再同步权重，之后才接收新请求。

默认使用 ``retract``，原因与多轮 rollout 有关。``in_place`` 模式下，引擎在有请求运行时无法清空 cache，同步前开始、同步后继续的请求会拿旧版本的前缀 KV cache 配新权重继续生成，而调度器完全看不到这部分数据有多旧。``retract`` 让每个请求多做一次 re-prefill，换来的是每个请求都有明确的版本边界。

异步解决不了什么
----------------

异步消除的是生成和训练之间的空闲时间，并不会让任何一边本身变快。生成是瓶颈时，步时由 rollout 吞吐决定，准入怎么调都降不下来。以 GLM 参考配置为例，每步生成约 250k token，吞吐约 2000 tok/s，步时下限约为 125–140 s，而训练本身只需要约 85 s。要想更快，只能调整 actor 与 rollout 的 GPU 分配：给 rollout 更多算力，或者用更小的模型生成。

限制
----

异步 agentic RL 目前只支持它所基于的组合：Megatron actor、多轮 agent loop、SGLang server 后端，以及 actor 与 rollout 分离部署的 placement。训练时不做 off-policy 修正，rollout 返回的 logprob 直接作为 PPO 的 old logprob，因此要靠前面介绍的 staleness 上界，让训练数据与当前 policy 保持接近。
