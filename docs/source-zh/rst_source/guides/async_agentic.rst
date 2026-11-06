异步 agentic RL
===============

本页介绍 RLinf 的异步 agentic RL 如何工作：rollout 与训练如何重叠、两个准入上限各控制什么、怎么取值。如果你要的是一份能直接跑的配置而不是原理，先看 :doc:`Search-R1 异步训练 </rst_source/examples/agentic/searchr1/async>`，等到需要调上限时再回到这里。

异步为什么快
------------

同步的 agent 训练里，循环里最贵的两侧轮流闲置。actor 在一批轨迹上训练时，rollout 引擎没有东西可生成；训练结束后引擎重新填满，actor 又要等整批里最慢的那条轨迹走完才能进入下一步。多轮 agentic rollout 会放大这个问题：一条不停调工具的轨迹，会在每步末尾拖住整批。

异步训练把这种互相等待拆掉了。prompt 持续流入 rollout 引擎；完成的轨迹进缓冲等待；actor 一准备好下一步，就取走最旧的 ``rollout_batch_size`` 个组。生成和训练从此按各自的节奏推进，不再轮流等待。

实测效果接近理想的重叠。参考运行（第 2–19 步，TensorBoard）里，同步运行的 ``time/step`` 278.6 s 拆开是 ``time/rollout`` 181.9 + ``time/training`` 93.3 + ``time/sync_weights`` 1.4；异步运行的 ``time/step`` 139.0 s 拆开是 ``time/training`` 91.7 + ``time/wait_data`` 46.0 + ``time/sync_weights`` 1.3。训练时长两边几乎相同，消失的正是各自等对方的那部分时间。

结构：四个组件怎么衔接
----------------------

四个组件通过两个 channel 协作。

rollout worker 运行 SGLang server 后端，持有生成引擎。agent loop 驱动多轮交互：发出生成请求、解析工具调用、查询工具、把工具输出拼回对话，直到每条轨迹结束。``RolloutBufferWorker`` 持有 dataloader 和调度状态：从 ``dataloader_channel`` 读 prompt，按准入上限决定投放哪些，从 ``agent_output_channel`` 收完成的组，每步把最旧的 ``rollout_batch_size`` 个组交给 actor。actor 在这些组上训练，再把新权重同步回 rollout 引擎。

buffer worker 是让重叠安全运转的枢纽。dataloader 由它而不是 runner 持有，所以 actor 训练时它仍能继续投放 prompt。它同时数在途组和缓冲组，所以两侧之间能积压多少数据是有界的。它交付时先取最旧的组，actor 看到的数据就以受控的方式变旧——这就引出了下面两个上限。

准入：两个上限
--------------

buffer worker 只有在两个条件同时满足时才投放新的 prompt，而两个条件回答的是不同的问题。

第一个条件——在途组数低于 ``rollout_buffer.max_inflight_groups``——决定给 rollout 引擎多少活。默认按 ``ceil(max_running_requests x instances x 1.2 / group_size)`` 推导：把引擎的请求容量换算成组数，1.2 的余量留给在等工具或在等组内最慢轨迹的组。

第二个条件——在途组数加上已在缓冲里的组数低于 ``rollout_buffer.max_buffered_groups``——决定数据最多多旧。默认取 2 × ``rollout_batch_size``。

注意第二个上限把在途组也算进去了，而不只数缓冲中的组。这是刻意的。今天在途的组，明天就是缓冲里的组：轨迹一生成完，它就无条件进缓冲。如果投放只数缓冲中的组，在途部分就不受约束，一旦训练成为瓶颈，缓冲会不受控地越堆越高——曾实测过这种错配，缓冲堆到 338 和 369 个组，staleness 冲到 8。把在途组计入上限后，缓冲不会超过 ``max_buffered_groups``，staleness 也就有了界。

投放在容量可能释放的每个时点重新检查：收到一个结果、丢弃一个组、向 actor 交付一批。

上限怎么取
----------

哪个上限需要认真调，取决于瓶颈在循环的哪一侧。

生成是瓶颈时，actor 始终缺数据，缓冲随填随空。此时 ``max_buffered_groups`` 的约束是算术上的：至少取 ``max_inflight_groups + rollout_batch_size``。低于这个值，缓冲和在途额度抢同一个预算，每步末尾缓冲中的组会挤占在途名额，本该打满的引擎反而吃不饱。

训练是瓶颈时，生成快于消费，缓冲才是维持流水线运转的东西。``max_buffered_groups`` 至少取 2 × ``rollout_batch_size``，让下一批能在本批训练时开始生成。取 1 × 时，下一批只能在本批交付后开始投放，每步都要等一整批（含最慢的组）生成完——生成和训练又变回串行，只是绕了一圈。在 Qwen 上以 ``rollout_batch_size`` 64 实测，1×B 的步时中位数约 11 s，2×B 为 3.94 s；快的那档代价是 staleness 从约 1 升到峰值均值 2.09。

其实不必猜自己在哪种情形：看 ``rollout/buffer/admission_wait_inflight_limit`` 和 ``rollout/buffer/admission_wait_buffer_full``，即步窗口内投放分别被两个上限挡住的比例。哪个高，当前就是哪个上限在起作用。

在途上限也不是越大越好。把它调过引擎忙起来的那个点，只是在引擎里多排队、让数据变旧。GLM 参考运行上，把 ``max_inflight_groups`` 从 77 提到 160，每个 DP rank 的在飞请求从约 75 涨到 128（另有 33–78 个排队），吞吐不变，平均 staleness 从约 1.3 升到约 2.0——GPU 在 75 个请求时就已经忙满了。这组实测图在 :doc:`Search-R1 异步页 </rst_source/examples/agentic/searchr1/async>` 的「两个上限怎么调」一节。

Staleness
---------

actor 训练的是缓冲里的数据，而引擎正用更新的权重生成，所以每个被训练的组都落后几个权重版本。``rollout/data_staleness/*`` 序列量的就是这件事：对本步的每个组，步开始时的 actor 版本减去组内最早的生成版本——有均值、最大值和按取值的占比。它可以分解为两个分量：``rollout/version_span/mean``，即组自身生成跨越了几个版本；``rollout/queue_steps/mean``，即组完成后在缓冲里等了几个版本。

有界的 staleness 是重叠的代价，而给它定界的正是缓冲上限。如果要硬性限制，``agentloop.max_actor_accept_staleness`` 会在训练前丢掉比阈值更旧的组；丢弃返还的是准入额度，补上来的是别的 prompt，被丢的 prompt 本身要到下一个 epoch 才会再出现。代价是这些组上已经花掉的生成算力，所以阈值要设在缓冲上限本身产生的 staleness 之上（训练是瓶颈时约为 ``max_buffered_groups / rollout_batch_size`` 步的等待），否则丢掉的就不是例外的组，而是每次最旧的那批。actor 本身只在过旧的组多到缓冲凑不满一批时才需要等数据；按上面的方式取上限时，投放会先被挡住，轮不到它等。

权重同步与 ``pause_mode``
-------------------------

每个训练步结束后，新权重要在生成正酣时同步进 rollout 引擎。``rollout.pause_mode`` 决定同步期间在飞的请求怎么办：

- ``retract`` 是默认模式：把在飞请求放回引擎队列并同步权重。已生成的 token 保留；KV cache 随旧权重一起 flush、恢复后自动重算，代价是每个被收回的请求多一次 re-prefill。
- ``in_place`` 暂停引擎的事件循环：在飞请求原地保留 KV cache，同步完成后用新权重继续解码。
- ``drain`` 等所有在飞请求结束、同步完成，再放新请求进来。

默认取 ``retract`` 有一个多轮 rollout 特有的原因：``in_place`` 下，引擎在有请求运行时无法 flush cache，同步前开始、同步后继续的请求就带着旧版本的前缀 KV cache 对上新权重——由此产生的过期没有调度器能看到的界。``retract`` 为此付出每个请求一次 re-prefill，换来一个系统其余部分可以依赖的版本边界。

异步解决不了什么
----------------

异步消除的是生成与训练之间的空闲时间，它不会让任何一侧变快。生成是瓶颈时，步时由 rollout 吞吐决定，准入怎么调都压不下去：GLM 参考配置每步生成约 250k token、吞吐约 2000 tok/s，步时下限约 125–140 s，而训练只需要约 85 s。要更快，只能改 actor 与 rollout 的 GPU 配比——给 rollout 更多卡，或者换更小的模型来生成。

限制
----

异步 agentic RL 当前要求它所基于的组合：Megatron actor、多轮 agent loop、SGLang server 后端，以及 actor 与 rollout 的分离式 placement。训练侧对数据不做 off-policy 修正：rollout 的 logprob 直接作为 PPO 的 old logprob，上文那些 staleness 的界就是让这个差距保持小的护栏。
