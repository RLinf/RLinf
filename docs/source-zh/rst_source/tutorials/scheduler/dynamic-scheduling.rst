动态调度
========================


动态调度(Dynamic Scheduling)
是在训练运行期根据系统各组件(actor / rollout / inference)的实时状态,
对资源进行秒级动态调整与迁移,以提升整体吞吐与资源利用率的机制。
它依托于 Megatron-LM 的在线扩缩容能力(秒级扩缩)与 SGLang 的 migrate 功能,
在不终止训练的前提下,对集群中的 GPU 资源进行弹性重分配。

什么是动态调度？
-----------------------

动态调度指在训练过程中,根据不同阶段的瓶颈与负载变化,
按需将 GPU 资源在各组件之间迁移与扩缩:

- 扩容:在某组件成为性能瓶颈时,为其临时增加 GPU
- 缩容:当某组件阶段性闲置时,回收 GPU 以服务其他组件


收益与影响
--------------------

**性能收益:**

- 吞吐提升:为瓶颈组件临时加速,整体训练更快
- 资源利用率更高:空闲资源被及时挪用于有效计算
- 总时长缩短:动态调度常带来 20~50% 的总时长优化(视任务与集群而定)

**运行特性:**

- 训练不断点:扩缩/迁移过程无须停止训练
- 一致性保障:保持模型与训练状态在扩缩过程中的一致性

如何使用动态调度
--------------------

前置依赖
^^^^^^^^^^^^

1) 准备 Megatron-LM 在线扩缩依赖(编译产物):

.. code-block:: bash

    cd YourPath
    git clone git@github.com:i-Taozi/params_resharding_release.git
    export PYTHONPATH=$PYTHONPATH:YourPath/params_resharding_release

该仓库提供 Megatron-LM 在线扩缩相关的编译产物(源码暂未开源)。

2) Megatron 版本要求为 0.11。如果你的环境不是 0.11,请单独准备 0.11:

.. code-block:: bash

    cd YourPath
    git clone git@github.com:NVIDIA/Megatron.git
    cd Megatron-LM
    git checkout core_r0.11.0
    export PYTHONPATH=$PYTHONPATH:YourPath/Megatron-LM

配置示例(基于分离流水模式)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

将原有分离流水模式配置:

.. code-block:: yaml

    cluster:
      num_nodes: 1
      component_placement:
        rollout: 0-3
        inference: 4-5
        actor: 6-7

修改为开启自动调度,并保证组件顺序满足 actor -> rollout -> inference:

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
--------------------

当开启动态调度后,运行时调度器会依据各组件的进展与队列长度,
判断是否需要资源调整。典型动作包括:

- 当rollout待执行任务量较少,触发rollout迁移,释放部分rollout资源以扩容actor
- 当rollout或者inference执行结束,释放资源给actor扩容

可选策略
^^^^^^^^^^^^

- use_pre_process_policy

  1. 每轮迭代的前期,优先将actor资源临时转移给rollout;
  2. 当调度器检测到合适时机,再从rollout归还部分资源给actor;
  3. 适用于序列较长(rollout开销大)场景,最大化流水效率。

- use_wait_before_last_iter_policy

  1. 在每轮迭代中,actor最后一个iter开始前,先等待rollout与inference完成;
  2. 随后actor获得全部资源进行扩容训练;
  3. 得益于流水特性,rollout/inference会早于actor完成,如调度恰当可充分利用全集群资源完成最后一次actor计算。
