共享式模式
===============

.. image:: ../../../_static/svg/collocate.svg
   :width: 600px
   :align: center
   :class: col-img

所有 Worker 都被调度到 **同一组** GPU 上。在任意阶段，
只有一种类型的 Worker 运行，并占用整个设备的计算能力，直到该阶段结束。
这里有两种执行模式：同时常驻于 GPU 内存，或者通过卸载/重新加载在 GPU 内存中切换驻留。

**优点**

* 设计简单；无需复杂的数据依赖管理。

**缺点**

* 通常需要为每个组件实现卸载/重新加载。
* rollout 阶段的长尾延迟会延长端到端 RL 训练时间。

**示例配置**

下面是一个 Worker 放置的示例配置。本次训练任务有两个节点，每个节点有 8 个 GPU。`actor` 使用所有 16 个 GPU，`rollout` 也使用所有 16 个 GPU：

.. code:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor,rollout: all # or 0-15

另外，一些 Worker 支持卸载，以在某些时间段释放 GPU 给其他 Worker 使用。
例如，math RL actor 支持一些卸载选项：

.. code:: yaml

   actor:
     offload_optimizer: True
     offload_weight: True
     offload_grad: True

如果为 actor 启用了卸载，actor 会在运行前被加载到 GPU 内存中，运行结束后再被卸载到 CPU 内存。
如果没有启用卸载，共享式的 Worker（假设运行在 GPU 上）会争夺 GPU 内存，可能导致 OOM 错误。
完整配置请参考 :doc:`../user/yaml`。

**ComponentPlacement 编程**

基于以上的放置配置，用户可以使用合适的 `ComponentPlacement` 类来解析配置，
并将放置策略应用到 Worker，如下所示：

.. code:: python

   from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode

   component_placement = ModelParallelComponentPlacement(cfg, cluster)
   rollout_placement_strategy = component_placement.get_strategy("rollout")
   rollout_group = SGLangWorker.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

`ModelParallelComponentPlacement` 支持两种放置方式：共享式和分离式。
更重要的是，它会处理 rank 的排列，从而实现从训练到 rollout 的高效模型权重更新。
它会解析配置并为不同组件生成放置策略。生成的放置策略会在 Worker 启动时生效。
完整代码请参考 `Math RL 训练代码 <https://github.com/RLinf/RLinf/blob/main/examples/reasoning/main_grpo.py>`_。
