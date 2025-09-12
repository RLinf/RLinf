Placement 接口
================================

本节介绍 RLinf 中的 GPU 和节点分配策略。
无论是在**共置模式 (collocated mode)**、**分离模式 (disaggregated mode)** 还是**混合模式 (hybrid mode)** 下，``ComponentPlacement`` 都是面向用户的接口，用于生成不同组件工作器（例如 actor、env、rollout、inference）的分配方案，而分配策略 (placement strategies) 则是获取每个节点和每个 GPU 资源精确分配的底层机制。
生成的**分配元数据 (placement metadata)** 随后会用于通过 Ray 进行远程启动。


组件 Placement
-----------------------------------

在**具身智能 (embodied intelligence)** 和**数学推理 (MATH reasoning)** 场景中，
分别使用 ``HybridComponentPlacement`` 和 ``ModelParallelComponentPlacement`` 来生成工作器分配方案。
两种分配接口都接受由 OmegaConf 解析的 ``DictConfig``，并将 ``cluster.component_placement`` 字段转换为精确的 GPU 和节点分配。

组件Placement接受 ``cluster.component_placement`` 的字典语法：

- 键 (key) 是组件的名称，例如 ``rollout``，或 ``rollout,inference,actor``
- 值 (value) 是分配给这些组件的全局 GPU ID，可以是：
   - "all"：使用集群中的所有 GPU
   - 单个整数，例如 "3"：使用 GPU 3
   - 逗号分隔的整数列表，例如 "0,2,3"：使用 GPU 0、2 和 3
   - 连字符分隔的整数范围，例如 "0-3"：使用 GPU 0、1、2 和 3
   - 上述两种方式的组合，例如 "0-3,5,7"：使用 GPU 0、1、2、3、5 和 7

HybridComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.utils.placement.HybridComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

ModelParallelComponentPlacement
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.utils.placement.ModelParallelComponentPlacement
   :show-inheritance:
   :members:
   :member-order: bysource
   :class-doc-from: class

Placement 策略
-----------------------------------

Placement策略是为获取组件分配所使用的每个节点和每个 GPU 资源的精确分配的底层机制。
通常，用户无需直接使用分配策略。
但如果您希望实现更自定义的分配方案，可以参考以下两种内置策略：``PackedPlacementStrategy`` 和 ``FlexiblePlacementStrategy``。


FlexiblePlacementStrategy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.packed.FlexiblePlacementStrategy
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

使用示例
---------

.. autoclass:: rlinf.scheduler.placement.PlacementStrategy
   :no-members:
   :no-inherited-members:
   :exclude-members: __init__, __new__

Placement 元数据
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlinf.scheduler.placement.placement.Placement
   :members:
   :member-order: bysource
   :exclude-members: __init__