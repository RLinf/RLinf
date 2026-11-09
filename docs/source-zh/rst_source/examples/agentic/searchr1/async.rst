Search-R1 异步训练
================================

以 rollout 与训练重叠的方式跑 Search-R1：prompt 持续流入 rollout 引擎，actor 同时在更早的权重版本上训练。本页使用 ``examples/agent/searchr1/config/train_glm4.7_flash_2node_async.yaml``，在两台 8 卡机器上把 actor 和 rollout 分别放在不同节点。请先按 :doc:`index` 搭建 RLinf 环境和 Qdrant wiki 检索服务；模型相关的配置（数据格式、tokenizer、actor 并行方式）与 :doc:`glm4.7_flash` 相同。异步模式的工作原理和两个调度上限怎么取，见 :doc:`异步 agentic RL </rst_source/guides/async_agentic>`。

概述
--------------------------------

使用本配方在两台 8 卡机器上异步训练 GLM-4.7-Flash：Megatron actor 占用 node 0，SGLang server rollout 和 reward worker 占用 node 1，生成持续进行，训练消费的是更早权重版本上的数据。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GLM-4.7-Flash（MoE，激活参数约 3B）

   .. grid-item-card:: 算法
      :text-align: center

      带搜索工具调用的多轮强化学习，异步 rollout

   .. grid-item-card:: 工具
      :text-align: center

      Qdrant 本地 wiki server

   .. grid-item-card:: 硬件
      :text-align: center

      参考运行使用 2×8 A100-80G

| **你要做的：** 搭建 Ray 集群 → 填三处路径 → 启动 → 观察 ``rollout/buffer/*``。
| **前提：** :doc:`RLinf 安装 </rst_source/start/installation>` · :doc:`多机指南 </rst_source/guides/multi_node>` · :doc:`index` 中的检索服务。

前提条件
--------------------------------

异步训练依赖以下四点协同工作：

- 分离式 placement：actor 与 rollout 跑在不同的 GPU 上（本配置在不同节点上），训练不与生成争抢同一批设备。
- Megatron actor：异步训练循环由 Megatron 的 actor worker 实现。
- ``rollout.sglang.backend_type: server``：SGLang server 后端，支持异步 runner 所用的权重同步与暂停协议。
- ``agentloop.task_name``：任务名，作为 rollout buffer 按任务记录状态的 key，并盖在每次 rollout 请求上；异步训练下未设置时启动即报错。

reward 在 agent loop 内计算（Search-R1 的规则奖励在轨迹和检索文本上解析），因此 rollout 与训练之间没有独立的 reward 模型推理步骤。

启动训练
--------------------------------

按 :doc:`多机指南 </rst_source/guides/multi_node>` 搭建 Ray 集群：每台机器在 ``ray start`` 之前设置互不相同的 ``RLINF_NODE_RANK``，head 节点为 0，并激活 RLinf 环境。检索服务在训练前于 node 0 上启动；搜索工具 worker 也运行在 node 0 上，因此 ``tools.search.server_addr`` 只需要从 node 0 可达。

填写数据路径、模型路径和检索服务地址：

.. code-block:: yaml

   data:
     train_data_paths: ["/path/to/data/searchr1_train_glm.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/GLM-4.7-Flash
   tools:
     search:
       server_addr: <server_ip>:8000

然后在 head 节点上启动训练：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_glm4.7_flash_2node_async

配置
--------------------------------

在同步配方之外，这些键控制异步模式：

.. list-table::
   :header-rows: 1
   :widths: 40 18 42

   * - 键
     - 默认值
     - 作用
   * - ``runner.async_mode``
     - ``False``
     - 切换到异步 agent runner，让 rollout 与训练重叠。
   * - ``agentloop.task_name``
     - 必填
     - 作为 rollout buffer 按任务记录状态的 key；盖在每次 rollout 请求上。
   * - ``rollout.pause_mode``
     - ``retract``
     - 权重同步时如何暂停生成：``retract`` 把在飞请求放回队列并保留已生成的 token；``in_place`` 暂停引擎事件循环、在飞请求保留 KV cache，同步后用新权重继续解码；``drain`` 等所有在飞请求结束后再同步。
   * - ``rollout_buffer.max_inflight_groups``
     - 推导
     - 允许已投放但未返回的组数。留空时按 ``ceil(max_running_requests x instances x 1.2 / group_size)`` 推导（本配置下为 77）。
   * - ``rollout_buffer.max_buffered_groups``
     - ``2 x rollout_batch_size``
     - 在途加缓冲的组数上限，同时约束缓冲显存和数据最多多旧。必须不小于 ``rollout_batch_size``，config 校验会拒绝更小的值；``agentloop.max_actor_accept_staleness`` 小于 ``ceil(max_buffered_groups / rollout_batch_size)`` 时启动告警。
   * - ``rollout_buffer.max_dropped_groups_per_step``
     - ``10 x rollout_batch_size``
     - 两次交付之间丢弃组数的安全上限；超出时运行直接失败。
   * - ``agentloop.max_actor_accept_staleness``
     - 不限
     - 训练前丢弃比它更旧的组；丢弃返还的是准入额度，别的 prompt 会补上来，被丢的 prompt 本身要到下一个 epoch 才会再出现。
   * - ``down_sampling.down_sampling_config.reject_equal_reward``
     - ``False``
     - 丢弃组内各轨迹 reward 全相等的组（组内基线没有梯度信号）。

两个调度上限怎么取，取决于瓶颈在生成还是训练；机制与实测证据见 :doc:`异步 agentic RL </rst_source/guides/async_agentic>`。

要看的指标
--------------------------------

这些 TensorBoard 序列反映异步管线的运行状态：

- ``rollout/buffer/inflight_groups``：已投放但未返回的组数。稳定在推导出的上限上，说明 rollout 引擎已经打满。
- ``rollout/buffer/size``：缓冲中等待的完成组数，与 ``max_buffered_groups`` 对照看。
- ``rollout/buffer/admission_wait_inflight_limit`` 与 ``rollout/buffer/admission_wait_buffer_full``：步窗口内投放分别被两个上限挡住的比例，据此判断当前是哪个上限在起作用。
- ``rollout/data_staleness/mean``：本步训练的组，步开始时的 actor 版本减去组内最早生成版本，取平均。配套的两个分量：``rollout/version_span/mean``，即组自身生成跨越的版本数；``rollout/queue_steps/mean``，即完成后在缓冲里等了多久。
- ``rollout/buffer/dropped_groups``：每步丢弃的组数（staleness 上限或等 reward 过滤所致）。持续偏高说明在浪费生成算力。

可视化与结果
--------------------------------

参考运行在 2×8 A100-80G 上训练 GLM-4.7-Flash，``rollout_batch_size`` 128 × ``group_size`` 4，``pause_mode: retract``，共 20 步（统计第 2–19 步，去掉首步预热）。吞吐按每步 ``total_num_sequence x response_length ÷ time/step`` 计算。

.. list-table::
   :header-rows: 1
   :widths: 30 16 16 16 11 11

   * - 运行
     - 平均步时
     - 吞吐
     - 平均 data staleness
     - reward 范围
     - ``ratio_abs`` 范围
   * - 同步 ``AgentRunner``
     - 278.6 s
     - 1159 tok/s
     - —
     - 0.25–0.47
     - 0.052–0.062
   * - 异步
     - 139.0 s
     - 2024 tok/s
     - 1.3–1.4
     - 0.26–0.42
     - 0.053–0.063

训练质量保持住了：reward 范围和训推 logprob 差（``ratio_abs``）与同步运行相当，步时砍半（278.6 s 到 139.0 s），吞吐提升 1.75 倍（1159 到 2024 tok/s）。在途组数稳定在推导出的上限 77 上，每个步窗口有 98–100% 的时间被在途上限挡住——rollout 引擎全程处于打满状态。缓冲持有 72–118 个组，远低于 256 的上限。20 步内回答长度从约 180 涨到约 380 token，因此步时缓慢变长而 token 吞吐保持稳定。

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_step_time.png" style="width: 100%;"/>
       <p><em>每步耗时：同步与异步对比</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_walltime.png" style="width: 100%;"/>
       <p><em>轨迹 reward 随墙钟时间的变化</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_step.png" style="width: 100%;"/>
       <p><em>每步的轨迹 reward</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_ratio_abs.png" style="width: 100%;"/>
       <p><em>训推 logprob 差（ratio_abs）</em></p>
     </div>
   </div>
   <p style="text-align: center;"><em>GLM-4.7-Flash 异步 Search-R1</em></p>

这些图的横轴是参数更新次数：每个 rollout 步更新 4 次（``algorithm.n_minibatches: 4``），与 :doc:`glm4.7_flash` 的口径一致。

两个上限怎么调
--------------------------------

两组实测说明瓶颈在不同侧时两个上限各自的表现。数字背后的原理见 :doc:`异步 agentic RL </rst_source/guides/async_agentic>`，本节只给证据。

生成是瓶颈时（GLM，本页这套配置），提高在途上限换不来吞吐。把 ``max_inflight_groups`` 从推导出的 77 提到 160，每个 DP rank 的在飞请求从约 75 涨到 128（另有 33–78 个排队），吞吐不变，平均 staleness 从约 1.3 升到约 2.0——GPU 在 75 个请求时就已经忙满了。

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_step_time.png" style="width: 100%;"/>
       <p><em>每步耗时：在途上限 77 对 160</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_staleness.png" style="width: 100%;"/>
       <p><em>平均 data staleness：在途上限 77 对 160</em></p>
     </div>
   </div>

训练是瓶颈时（Qwen2.5-1.5B-Instruct，8 个 SGLang 实例 TP1，``rollout_batch_size`` 64 × ``group_size`` 4），缓冲上限决定两边能否重叠。``max_buffered_groups`` 取 1 × ``rollout_batch_size`` 时，下一批只能在本批交付后开始投放，生成与训练重新串行，步时中位数约 11 s；取 2 × 时为 3.94 s。缓冲分别稳定在 54 和 108 个组，都不超上限。

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_step_time.png" style="width: 100%;"/>
       <p><em>每步耗时：缓冲上限 1×B 对 2×B</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_size.png" style="width: 100%;"/>
       <p><em>缓冲大小：上限 1×B 对 2×B</em></p>
     </div>
   </div>
   <p style="text-align: center;"><em>在途与缓冲上限实验</em></p>

恢复训练
--------------------------------

rollout buffer worker 会把 dataloader 位置存进 checkpoint（``data/data.pt``）。恢复时跳过保存前已经投放出去的 prompt，数量在每次存档时打日志；在 Qwen 默认配置下实测为 192 个（在途加缓冲 128 个，加预取的一批 64 个）。

限制
--------------------------------

- 异步训练基于 Megatron actor 与 Search-R1 使用的多轮 agent loop 实现。
- rollout 的 logprob 直接作为 PPO 的 old logprob（``algorithm.recompute_logprobs: False``）。
- 开启 ``reject_equal_reward`` 后丢弃率可能很高；长跑之前先看 ``rollout/buffer/dropped_groups``。
