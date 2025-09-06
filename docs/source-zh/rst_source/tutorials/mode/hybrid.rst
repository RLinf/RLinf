混合式模式
===========

.. image:: ../../../_static/svg/hybrid.svg
   :width: 600px
   :align: center
   :class: hyb-img

RLinf 在共享式模式和分离式模式的基础上，进一步引入了混合式模式：  
有些任务共享同一组 GPU，而有些任务使用独立的 GPU。

上图展示了一个具身 RL 训练的具体放置与执行示例。  
仿真 Worker 被放置在 GPU 0-1 上，生成 Worker 被放置在 GPU 2-3 上。两个 *数据队列* 将生产者和消费者的速率解耦，帮助平滑流水线、平衡负载，并几乎消除性能瓶颈。在 rollout 阶段（即仿真+生成）结束后，推理 Worker 被放置并运行在 GPU 0-3 上，随后训练 Worker 也运行在 GPU 0-3 上。可以看到，混合式模式结合了共享式和分离式模式。RLinf 中的通信工具 (:doc:`../communication/index`) 支持这种灵活的放置和执行方式。

**示例配置**

混合式模式的配置风格与共享式/分离式模式一致，如下所示：  
`env` （即仿真 Worker）放置在 GPU 0-3 上，`rollout` （即生成 Worker）放置在 GPU 4-7 上，它们通过流水线运行。`actor` （即训练 Worker）放置在 GPU 0-7 上。当 rollout 阶段结束后，`env` 和 `rollout` 会卸载到 CPU 内存，`actor` 会加载到 GPU 内存。

.. code:: yaml

  cluster:
    num_nodes: 1
    num_gpus_per_node: 8
    component_placement:
      actor: 0-7
      env: 0-3
      rollout: 4-7

在大多数情况下，`env`、`rollout` 和 `actor` 应该启用如下的卸载功能，以避免 OOM 错误。

.. code:: yaml

   env:
     enable_offload: True
   rollout:
     enable_offload: True
   actor:
     enable_offload: True

完整配置请参考 :doc:`../user/yaml`。

**ComponentPlacement 编程**

与共享式和分离式模式不同，混合式模式使用 `HybridComponentPlacement`，  
它对 Worker 放置的限制更少。

.. code:: python 

   from rlinf.utils.placement import HybridComponentPlacement

   component_placement = HybridComponentPlacement(cfg)
   # 创建 actor Worker 组
   actor_placement = component_placement.get_strategy("actor")
   actor_group = FSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

完整代码请参考  
`具身模型训练 <https://github.com/RLinf/RLinf/blob/main/examples/embodiment/train_embodied_agent.py>`_。
