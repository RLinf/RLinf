Placement 接口
================================

本节介绍 RLinf 中的 GPU 和节点放置（placement）策略。
无论是在 **共置模式（collocated mode）**、**分离模式（disaggregated mode）**
还是 **混合模式（hybrid mode）** 下，``ComponentPlacement``
都是面向用户的接口，用于为不同组件的 worker（例如 actor、env、rollout、inference）
生成放置信息；而各类 placement 策略则是实现“每个节点、每个 GPU 资源精确分配”的
底层机制。
生成的 **placement 元数据（placement metadata）** 随后会用于配合 Ray 启动远程任务。


组件 Placement
-----------------------------------

``ComponentPlacement`` 接口负责解析配置文件中的 ``cluster.component_placement``
字段，并为不同组件的 worker 生成精确的放置信息。

需要注意的是，``ComponentPlacement`` 还通过 ``cluster.node_groups``
配置中的 ``node_group`` 字段，支持对异构集群的放置描述。

关于语法的详细说明，可参考异构集群教程和下方自动文档。

.. autoclass:: rlinf.utils.placement.ComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

在 **具身智能（embodied intelligence）** 和 **数学推理（MATH reasoning）**
场景中，分别使用 ``HybridComponentPlacement`` 和
``ModelParallelComponentPlacement`` 来生成 worker 放置方案。
``HybridComponentPlacement`` 直接继承自 ``ComponentPlacement``，
而 ``ModelParallelComponentPlacement`` 在其基础上扩展了放置逻辑，
以支持在多张 GPU 上进行推理引擎的模型并行。


HybridComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.utils.placement.HybridComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

ModelParallelComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.utils.placement.ModelParallelComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

Placement 策略
-----------------------------------

Placement 策略是 ``ComponentPlacement`` 用来获得“每个节点、每个 GPU 资源
精确分配方案”的底层机制。
如果你希望自定义更细粒度的放置方式，可以参考以下内置策略：
``FlexiblePlacementStrategy``、``PackedPlacementStrategy`` 和
``NodePlacementStrategy``。
其中，``FlexiblePlacementStrategy`` 与 ``PackedPlacementStrategy``
用于在加速器/GPU 上放置 worker 进程，而 ``NodePlacementStrategy``
则在仅关注“节点位置”而不关心底层加速器资源时使用，
因此非常适合只依赖 CPU 的 worker。


FlexiblePlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.flexible.FlexiblePlacementStrategy
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

PackedPlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.packed.PackedPlacementStrategy
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

NodePlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.node.NodePlacementStrategy
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

Placement 元数据
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.placement.Placement
   :members:
   :member-order: bysource
   :exclude-members: __init__