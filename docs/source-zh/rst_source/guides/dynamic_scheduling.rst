动态弹性调度
============

动态调度（Dynamic Scheduling）在训练运行期根据系统各组件（actor / rollout / inference）的实时状态，秒级动态调整与迁移资源，以提升整体吞吐与资源利用率。它依托于 Megatron-LM 的在线扩缩容能力（秒级扩缩）与 SGLang/vLLM 的请求迁移功能，在不终止训练的前提下弹性重分配集群中的 GPU 资源。

.. warning::

   **旧版环境前置要求**：动态调度依赖独立的 **Megatron-LM 0.11** 分支与外部预编译重分片包（``params_resharding_release``），不兼容默认 ``bash requirements/install.sh agentic`` 环境（安装 Megatron-LM 0.18）。可运行的参考配置见 ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron-dynamic.yaml``\ （配好下方前置依赖后，运行 ``bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron-dynamic`` 启动）。

在线扩缩机制
------------

在线扩缩（也称弹性训练）可以在 1 秒内完成 GPU 切换，动态调整训练资源。你可以根据集群可用性、任务需求或资源优化目标，实时调整训练使用的 GPU 和节点数量。

什么是在线扩缩？
~~~~~~~~~~~~~~~~

自动扩缩指的是在训练过程中能够 **向上扩缩** （增加更多资源）或 **向下缩减** （释放部分资源），同时保持训练的连续性和模型状态的一致性。

在使用 Megatron-LM 进行 RL 训练时，这包括：

- **向上扩缩**：增加节点/GPU 来提升训练吞吐量
- **向下缩减**：释放节点/GPU，将资源腾出来给其他任务
- **并行策略调整**：动态改变 Megatron 的并行策略（TP/PP/DP/CP）

系统会自动处理以下内容：

- 模型参数在新的并行配置中的重新分布
- 优化器状态的迁移
- 通信组的重建
- 训练状态的同步

为什么在线扩缩很重要？
~~~~~~~~~~~~~~~~~~~~~~

在 RLinf 的分离式模式下结合细粒度流水线时，rollout 和 inference 阶段通常会在 actor 阶段结束前就完成。此时，可以在 **几秒内** 将 rollout 和 inference 所使用的资源重新分配给 actor 阶段，从而加速 actor 的训练，并提升整个系统的性能。

优势与收益
~~~~~~~~~~

**性能收益：**

- **吞吐提升**：为瓶颈组件临时增加 GPU，加快整体训练速度
- **资源利用率更高**：及时将空闲资源分配给有效计算
- **总时长缩短**：动态调度通常可减少 20–50% 的总训练时长（视任务与集群而定）

**运行特性：**

- **训练不断点**：扩缩与迁移过程无需停止训练
- **一致性保障**：在扩缩过程中保持模型与训练状态一致

动态调度实践
------------

动态调度指在训练过程中，根据不同阶段的瓶颈与负载变化，按需将 GPU 资源在各组件之间迁移与扩缩：

- 扩容：在某组件成为性能瓶颈时，为其临时增加 GPU
- 缩容：当某组件阶段性闲置时，回收 GPU 以服务其他组件

如何使用动态调度
~~~~~~~~~~~~~~~~

前置依赖
^^^^^^^^

.. warning::

   在线扩缩目前需要独立的 **Megatron-LM 0.11** 环境，不兼容默认 ``install.sh agentic`` 虚拟环境（默认安装 Megatron-LM 0.18）。

1) 准备 Megatron-LM 在线扩缩依赖（编译产物）：

.. code-block:: bash

    WORKSPACE=YourWorkspace
    cd $WORKSPACE
    git clone https://github.com/i-Taozi/params_resharding_release.git
    export PYTHONPATH=$PYTHONPATH:$WORKSPACE/params_resharding_release

该仓库提供 Megatron-LM 在线扩缩相关的编译产物。在线扩缩容相关源码即将开源。

2) Megatron 版本要求为 0.11，请单独获取 0.11：

.. code-block:: bash

    WORKSPACE=YourWorkspace
    cd $WORKSPACE
    git clone -b core_r0.11.0 https://github.com/NVIDIA/Megatron-LM.git
    export PYTHONPATH=$PYTHONPATH:$WORKSPACE/Megatron-LM

.. important::
    如果你使用 ``torch >= 2.6.0``，Megatron-LM 0.11 可能会因默认的 ``torch.load`` 行为报错。请改用适配后的 Megatron-LM 0.11 分支：

    .. code-block:: bash

        git clone -b core_v0.11.0_rlinf https://github.com/RLinf/Megatron-LM.git

配置示例
^^^^^^^^

原有分离流水模式配置如下：

.. code-block:: yaml

    cluster:
      num_nodes: 1
      component_placement:
        rollout: 0-3
        inference: 4-5
        actor: 6-7

基于原有分离流水模式配置，打开自动调度相关配置，并保证组件顺序满足 ``actor -> rollout -> inference``。如果组件顺序不满足，actor 无法扩容。

.. code-block:: yaml

    cluster:
      num_nodes: 1
      auto_scheduler: True
      use_pre_process_policy: True
      use_wait_before_last_iter_policy: False
      component_placement:
        actor: 0-1
        rollout: 2-5
        inference: 6-7

调度策略
~~~~~~~~

开启动态调度后，运行时调度器会依据各组件的进展与队列长度决定是否调整资源。典型动作包括：

- rollout 待执行任务量较少时，触发 rollout 迁移，释放部分 rollout 资源以扩容 actor
- rollout 或 inference 执行结束时，释放资源给 actor 扩容

可选策略
~~~~~~~~

- ``use_pre_process_policy``

  1. 每轮迭代的前期，优先将 actor 资源临时转移给 rollout；
  2. 调度器检测到合适时机后，再从 rollout 归还部分资源给 actor；
  3. 适用于序列较长（rollout 开销大）的场景，最大化流水效率。

- ``use_wait_before_last_iter_policy``

  1. 每轮迭代中，actor 最后一个 iter 开始前，先等待 rollout 与 inference 完成；
  2. 随后 actor 获得全部资源扩容训练；
  3. 得益于流水特性，rollout/inference 通常早于 actor 完成，调度恰当时可利用全集群资源完成最后一次 actor 计算。
