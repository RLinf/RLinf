降低智能体批次的存储与通信开销
==============================

使用变长批次，让轨迹按真实长度保存，直到计算损失时再补齐当前小批次。本指南介绍如何在 Megatron 动态智能体训练路径中启用这一功能，按十个步骤对照新旧流程，并说明如何使用 Search-R1 复现验证结果。

启用真实长度批次
----------------

固定长度批次在存储和 DP 重分配期间也会把短回答补齐到最大序列长度。变长批次（ragged batch）则为每条序列分别保存一个张量，例如长度为 101、201、301 的三条序列，在模型并行对齐之前共占用 603 个 token 位置。DP 是数据并行：多个模型副本分别训练不同部分的数据。

以 :doc:`GLM-4.7-Flash 示例 </rst_source/examples/agentic/searchr1/glm4.7_flash>` 为起点，设置以下字段：

.. code-block:: yaml

   actor:
     enable_ragged_batch: True
     enable_dp_load_balance: True
   agentloop:
     is_dynamic_rollout_batch: True
   runner:
     enable_dynamic_batch_size: True
   algorithm:
     adv_type: grpo_dynamic
     advantage_mode: trajectory
     normalize_advantages: False

``actor.enable_ragged_batch`` 在 ``MAMegatronActor`` 中选择新的数据路径；不设置或设为 false 时使用固定长度路径。Search-R1 的 GLM 示例显式启用了该开关。``actor.enable_dp_load_balance`` 独立控制是否在 DP 副本间移动题组。关闭负载均衡后，每个副本保留收到的题组，仍然可以使用变长批次。

当前路径适用于 Megatron 动态智能体 Actor，要求启用动态 microbatch、使用 ``grpo_dynamic``、关闭优势归一化，并将流水线并行度设为 1。GPU 验证使用 actor 与 rollout 共用显卡的部署方式，张量并行度为 4、DP 为 2、上下文并行度为 1。CPU 打包测试还覆盖了上下文并行度为 2 的情况；多机及独立推理 Worker 部署仍需单独验证。这里的支持范围是这条 Actor 数据路径，不能由此推导 RLinf 中所有任务和训练后端均已验证。

对照十步训练流程
----------------

下表使用示例中的最大序列长度 8192。turn 表示一次模型响应；trajectory（轨迹）包含一次回答尝试的所有轮次和工具返回；group（题组）包含同题的四次回答尝试，由 ``algorithm.group_size: 4`` 指定。在合并同一轨迹中兼容的轮次之前，每一行对应一个 turn。两条路径都以配置的最大长度作为上限。

设置 ``actor.enable_ragged_batch: False`` 时：

.. list-table:: 固定长度路径
   :header-rows: 1
   :widths: 6 26 68

   * - 步骤
     - 执行位置
     - 数据与操作
   * - 1
     - ``Searchr1AgentLoopWorker``
     - 完成生成、搜索和评分，在 ``DynamicRolloutResult`` 中保存真实长度列表，通过 Channel（Worker 之间的队列）传输。
   * - 2
     - ``MAMegatronActor.get_batch``
     - 接收结果，广播给模型并行组内的进程。
   * - 3
     - ``DynamicRolloutResult.to_actor_batch``
     - 将文本编号、标记及已有的生成概率等逐 token 字段补齐到 [N, 8192]，合并收到的批次。
   * - 4
     - ``MAMegatronActor.run_inference``
     - 开启重算时：去除补齐、对齐并打包 token、执行模型前向，将旧概率恢复成 [B, 8192]，再传给训练阶段。
   * - 5
     - ``MAMegatronActor.run_training``
     - 转换并合并批次，计算组内优势和损失缩放，合并兼容的轮次；逐 token 字段的宽度仍为 8192。
   * - 6
     - ``RolloutDataBalance.from_rollout_batches_dynamic``
     - 开启负载均衡时，在 DP 进程组上用 all_gather_object 收集 CPU 字段，合并全局数据，再按长度选择本地行。使用 NCCL 时，序列化数据会暂存到 GPU 上进行通信。
   * - 7
     - ``MAMegatronActor`` 与批次迭代器
     - 切分用于参数更新的 minibatch，再按 token 预算切 microbatch；存储中的逐 token 字段宽度仍为 8192。
   * - 8
     - ``MAMegatronActor.get_forward_step_func``
     - 去除补齐，对齐并打包 token，执行训练前向，再将当前概率恢复成矩形张量。
   * - 9
     - ``policy_loss``
     - 让当前概率、旧概率、优势、损失缩放和响应标记逐位置对应，计算损失。
   * - 10
     - Megatron 前后向调度与优化器
     - 沿前向计算图反向传播，同步梯度，更新权重。

设置 ``actor.enable_ragged_batch: True`` 时：

.. list-table:: 变长路径
   :header-rows: 1
   :widths: 6 26 68

   * - 步骤
     - 执行位置
     - 数据与操作
   * - 1
     - ``Searchr1AgentLoopWorker``
     - 产生相同格式的真实长度题组结果和奖励。
   * - 2
     - ``MAMegatronActor.get_ragged_batches``
     - 数据接收进程先接收原始题组。开启负载均衡时先重分配，再广播给模型并行进程；关闭时保留收到的分配。
   * - 3
     - ``balance_rollout_groups`` 与 ``DynamicRolloutResult.to_ragged_actor_batch``
     - 汇总题组长度，分配完整题组，通过 all_to_all_single 先交换字节数，再交换变长数据，随后广播；构造按真实长度保存的 CPU 张量列表。本地保留的题组不发送。
   * - 4
     - ``MAMegatronActor.run_inference``
     - 开启重算时，打包真实长度输入、计算旧概率，通过本地 Channel 返回变长概率行；训练阶段沿用这次 DP 分配。
   * - 5
     - ``MAMegatronActor.run_training``
     - 使用完整题组计算优势和损失缩放，以变长行合并兼容轮次，并保留来源 DP 的归一化语义。
   * - 6
     - ``MAMegatronActor._pad_ragged_rows``
     - 仅在批次数量需要整除时增加损失为零的占位行。DP 重分配已经完成，逐 token 字段仍是变长的。
   * - 7
     - 批次迭代器与 ``split_dynamic_batch_size``
     - 根据真实长度及模型并行对齐要求，切分 minibatch 和符合 token 预算的 microbatch。
   * - 8
     - ``preprocess_ragged_seqs`` 与 ``MAMegatronActor.get_forward_step_func``
     - 对齐并打包当前 microbatch，执行前向，恢复输出行，再通过 ``materialize_ragged_batch`` 将当前损失输入补齐到 [B, 8192]。
   * - 9
     - ``policy_loss``
     - 沿用矩形损失接口和标记语义。变长路径关闭了可选的“移除整行无训练位置数据”操作，以保持归一化语义。
   * - 10
     - Megatron 前后向调度与优化器
     - 使用原生梯度同步与参数更新路径。小批次组成及浮点归约顺序可能变化。

开启重算时，第 2 步在推理阶段首次接收 rollout 结果时执行；训练读取本地结果并沿用分配。关闭重算时，训练入口接收原始结果，在那里执行第 2 步。使用变长批次且关闭 DP 负载均衡时，这两个入口都不重新分配题组。

保持完整题组
------------

前置重分配发生在优势计算之前，因此按完整题组移动数据。例如同题奖励为 [1, 0, 0, 0] 时，平均值是 0.25。如果把这四条轨迹拆到两个 DP 副本，再分别使用 0.5 和 0 作为平均值，就改变了 GRPO 对同题回答的比较。题组还需要包含这些轨迹的所有轮次。计算完优势和依赖题组的损失缩放后，minibatch 可以再拆分这些行。

损失通过矩形标记选择训练位置。mask（标记）控制某个位置是否贡献损失，并不自动跳过补齐位置上的所有算术操作。packing（打包）在模型前向之前去除大部分序列补齐；张量并行和上下文并行仍可能要求少量尾部对齐，损失计算时则只为当前 microbatch 恢复配置宽度。

验证改动
--------

使用仓库中的 `tests/e2e_tests/agent/ragged_batch/README.md <https://github.com/RLinf/RLinf/blob/main/tests/e2e_tests/agent/ragged_batch/README.md>`_ 准备成对配置、执行 CPU 接口检查，并回放已有通信基准。同目录的 ``REPORT.md`` 分别记录通信测量、固定轨迹正确性及长跑结果。

比较端到端运行时，固定输入题目、种子、模型、检索服务和算法设置，只切换变长开关。独立采样仍可能产生不同回答；检查字段保留和数值一致性时，应单独使用固定的生成结果。DP 交换阶段显存、整卡显存和模型前向峰值也应分别记录。
