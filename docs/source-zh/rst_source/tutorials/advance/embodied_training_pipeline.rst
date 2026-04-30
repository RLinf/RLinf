具身训练 Pipeline 模式
==============================

本文介绍 RLinf 在同步具身训练中的 ``use_training_pipeline`` 模式。它的目标是让 actor training
尽早开始，从而同时利用两类重叠：

- ``env -> actor`` 通信 与 actor training 的重叠
- 当 ``rollout_epoch > 1`` 时，后续 ``rollout <-> env`` 数据生成 与 actor training 的重叠

因此，它尤其适合多机、低带宽，或者 rollout step 较长、actor 容易空等的场景。

当前这套能力主要面向 ``examples/embodiment/train_embodied_agent.py`` 所使用的
同步具身训练路径。


为什么需要这个模式
------------------------------

默认的同步具身训练流程可以简化为：

1. env 与 rollout 完成一整轮 rollout。
2. env 将 rollout 数据整体发送给 actor。
3. actor 收齐完整 rollout 后，再统一计算优势并开始训练。

这种方式的优点是语义直接，但如果 ``env -> actor`` 的网络传输很慢，actor 会在较长时间内处于等待状态。

``use_training_pipeline`` 的核心思路是尽早把可训练的 micro batch 送到 actor，
让 actor 在后续数据仍在传输时就开始训练；当 ``rollout_epoch > 1`` 时，还可以让 actor training
与后续 rollout epoch 的数据生成过程重叠。


实现原理
------------------------------

这套模式不会改变训练入口，也不会改变 rollout 与 env 的基本交互方式；它主要改动的是
``env -> actor`` 这一段的数据组织方式。

整体流程如下：

1. env 和 rollout 正常执行一个 ``rollout_epoch``。
2. 当前 ``rollout_epoch`` 结束后，env 不再等待整个 step 全部结束，而是立即 flush。
3. env 按 actor 本地 ``micro_batch_size`` 对齐切分数据。
4. env 侧先计算该 split 所需的 ``advantages`` / ``returns``，并整理成 train-ready micro batch。
5. actor 收到一个 micro batch 后立即执行训练。
6. 如果 ``update_epoch > 1``，只有第 1 个 update epoch 能与传输重叠；剩余 epoch 会在数据全部到齐后本地 replay。

这意味着，这个模式主要优化的是：

- ``env -> actor`` 通信
- actor 第 1 个 update epoch
- 当 ``rollout_epoch > 1`` 时，actor training 与后续 rollout epoch 的 env-rollout 数据生成

它 **不等同于** rollout 与 env 之间的 ``pipeline_stage_num``，也不负责优化
``rollout <-> env`` 的那段推理流水。


两类重叠
------------------------------

这个模式里实际有两层不同的 overlap：

1. ``env -> actor`` 通信 与 actor training 的 overlap

   这是最直接的一层。env 在发送当前 ``rollout_epoch`` 的多个 pipeline micro batch 时，
   actor 可以在收到前几个 micro batch 后立刻开始训练，而不必等整批数据全部发送完成。

2. 后续 ``rollout_epoch`` 的 ``rollout <-> env`` 交互 与 actor training 的 overlap

   当 ``rollout_epoch > 1`` 时，env 会在每个 ``rollout_epoch`` 结束后立即 flush。
   这意味着 actor 可以在训练第 ``k`` 个 ``rollout_epoch`` 已经完成的数据时，
   env 和 rollout 继续生成第 ``k+1`` 个乃至后续 ``rollout_epoch`` 的数据。

第二层 overlap 很重要，因为它不只是“隐藏一段网络发送时间”，还会把 actor training
与后续的 env-rollout 数据生成过程重叠起来。

如何启用
------------------------------

在具身训练配置中开启：

.. code-block:: yaml

   runner:
     use_training_pipeline: true
     pipeline_schedule: global_batch

最常见的启动方式是直接在 YAML 中写入，或者在命令行追加 override：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-path examples/embodiment/config \
     --config-name <your_config> \
     +runner.use_training_pipeline=true \
     +runner.pipeline_schedule=global_batch

其中 ``runner.pipeline_schedule`` 当前支持两种取值：

- ``global_batch``：以 global batch 为调度单位，更接近原始 step 边界
- ``micro_batch``：以 micro batch 为调度单位，更激进，也更容易填补 bubble


当前支持范围
------------------------------

当前 ``use_training_pipeline`` 支持以下 ``adv_type``：

- ``gae``
- ``raw``
- ``grpo``

当前 ``use_training_pipeline`` 不支持以下 ``adv_type``：

- ``grpo_dynamic``

常见的组合关系如下：

- ``gae``：通常配合 ``actor_critic`` 或 ``decoupled_actor_critic``
- ``raw``：通常配合 ``actor``
- ``grpo``：通常配合 ``actor``

当前不建议将该模式用于以下路径：

- ``embodied_sac`` / ``embodied_dagger`` / ``embodied_nft`` 等非当前同步 on-policy 主路径
- 依赖完整全局 shuffle 或完整全局 normalize 阶段的自定义训练流程


使用限制与注意事项
------------------------------

1. 需要显式指定 ``runner.pipeline_schedule``。

   当前可选值为 ``global_batch`` 和 ``micro_batch``。

2. ``gae`` 和 ``raw`` 路径下，不再做额外的全局 ``normalize_advantages``。

   这套模式的目标是尽早开始训练，因此不会在 actor 侧等待完整 rollout 后再做一次额外的
   全局 normalize。对于 ``grpo``，其组内归一化逻辑仍然由对应的
   advantage 定义本身负责。

3. ``shuffle_rollout`` 不是严格的 full-rollout 全局 shuffle。

   当 ``pipeline_schedule=global_batch`` 时，shuffle 只会发生在 ready 的 global batch 之间；
   当 ``pipeline_schedule=micro_batch`` 时，shuffle 只会发生在 ready 的 micro batch 之间。
   因此它更接近一种 pipeline 内部调度打散，而不是默认同步路径里的整步全局 shuffle。

4. ``update_epoch > 1`` 时，收益会下降。

   只有第 1 个 update epoch 会参与这条 pipeline 的重叠过程。
   剩余的 ``update_epoch - 1`` 轮仍然要在数据全部到齐后本地 replay，因此无法继续与
   ``env -> actor`` 通信或后续 rollout epoch 的数据生成重叠。

5. ``rollout_epoch > 1`` 时，这个模式通常更容易体现收益。

   当前实现会在每个 ``rollout_epoch`` 结束后向 actor flush 一次数据，而不是等整个 step 全部结束。
   因此当 ``rollout_epoch > 1`` 时，actor 往往能更早拿到第一批可训练 micro batch，更容易形成有效重叠。

6. ``micro_batch_size`` 需要与当前 rollout 形状对齐。

   在当前实现里，env 是按“每个 ``rollout_epoch`` flush 一次”来切分的，因此一个 actor 本地
   pipeline micro batch 大致对应：

   .. code-block:: text

      env_batch_per_pipeline_micro_batch
      = actor.micro_batch_size / n_train_chunk_steps

   如果使用 group 类 advantage（如 ``grpo``），这个 ``env_batch_per_pipeline_micro_batch``
   必须能整除 ``group_size``，这样每个 group 才不会在切分时被打散。


与 ``pipeline_stage_num`` 的区别
--------------------------------

两者解决的是不同问题：

``runner.use_training_pipeline``
  优化 ``env -> actor`` 通信与 actor 训练之间的重叠。

``rollout.pipeline_stage_num``
  优化 rollout 与 env 之间的交互与推理流水。

它们可以同时存在，但不是同一件事，也不互相替代。


什么时候值得启用
------------------------------

推荐在以下情况优先尝试：

- actor 与 env 分布在不同机器上
- 网络带宽较低，``env -> actor`` 传输时间明显偏长
- rollout_epoch > 1

如果你的训练主要瓶颈既不在 ``env -> actor`` 通信，也不在“后续 rollout epoch 生成与 actor 空等”
这一类同步等待上，或者你非常依赖完整全局 shuffle / 额外全局 normalize 语义，那么默认同步路径通常更简单，
也更容易和已有实验严格对齐。
