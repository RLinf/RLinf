Worker 放置策略
========================================

Placement 模块用于定义 Worker 在集群的硬件资源（节点、GPU、NPU、机器人等）上的分布方式。
RLinf 的核心接口是 ``ComponentPlacement``，它通过 YAML 配置中的 ``cluster.component_placement`` 字段进行配置。
接下来会详细介绍配置的语法。


基本形式
~~~~~~~~

放置策略有两种等价的写法。

1. **简写形式** —— 直接把组件名映射到资源编号：

.. code-block:: yaml

	cluster:
	  num_nodes: 1
	  component_placement:
	    actor,inference: 0-7

在这里，``actor`` 与 ``inference`` *共用同一条放置规则*。
字符串 ``0-7`` 被解释为一段 **资源编号** 范围。
RLinf 会为每个组件分别创建 8 个进程（进程编号为 ``0-7``），
并将它们平均映射到资源 ``0-7`` 上。

2. **节点组形式** —— 显式选择某个节点组：

关于节点组的内容，见 :doc:`../advance/hetero`。

.. code-block:: yaml

	cluster:
	  num_nodes: 2
	  component_placement:
	    actor:
	      node_group: a800
	      placement: 0-8
	    rollout:
	      node_group: 4090
	      placement: 0-8
	    env:
	      node_group: robot
	      placement: 0-3:0-7
	    agent:
	      node_group: node
	      placement: 0-1:0-200,2-3:201-511

其含义是：

* ``actor`` 使用节点组 ``a800`` 中编号为 ``0-8`` 的加速卡；
* ``rollout`` 使用节点组 ``4090`` 中编号为 ``0-8`` 的加速卡；
* ``env`` 使用节点组 ``robot`` 中编号为 ``0-3`` 的机器人硬件，
  进程编号 ``0-7`` 被平均分配到这些机器人上（每台机器人对应 2 个进程）；
* ``agent`` 使用特殊节点组 ``node``。进程 ``0-200`` 放在节点编号
  ``0-1`` 上，进程 ``201-511`` 放在节点编号 ``2-3`` 上。


Resource Rank (资源编号) 和 Process Rank (进程编号)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个放置条目的通用形式为::

	resource_ranks[:process_ranks]

其中，``resource_ranks`` 和 ``process_ranks`` 均支持以下写法：

- ``a-b`` —— 闭区间整数范围，例如 ``0-3`` 表示 0,1,2,3；
- ``a-b,c-d`` —— 多段闭区间范围，可通过逗号连接，例如 ``0-3,5-7`` 表示 0,1,2,3,5,6,7；

同时，``resource_ranks`` 还支持：

- ``all`` —— 表示所选节点组中的全部有效资源。

* ``resource_ranks``：要使用的物理资源（GPU、机器人或节点）的编号，“资源”的具体含义由节点组决定：

  - 若节点组声明了某类 ``hardware``，则编号指这一类硬件
    （例如机器人索引）；
  - 若未声明 ``hardware`` 但存在加速卡，则编号指加速卡索引；
  - 若既没有声明 ``hardware`` 又没有加速卡，则编号指节点索引。

* ``process_ranks``：要放置到这些资源上的 **组件进程编号**，
  语法与 ``resource_ranks`` 相同，但不能写成 ``all``。

  如果省略 ``process_ranks``，RLinf 会自动分配一段连续的进程编号，
  其长度与资源数量相同。例如有如下两段::

	 0-3,4-7

  则第一段隐式对应进程 ``0-3``，第二段对应进程 ``4-7``。

可以通过逗号连接多段配置，并混合使用显式/隐式 ``process_ranks``::

	 0-1:0-3,3-5,7-10:7-14

其含义为：

* 进程 ``0-3`` 均匀分配到资源 ``0-1`` 上；
* 进程 ``4-6`` 被隐式分配到资源 ``3-5`` （每个资源对应一个进程，
  由调度器自动推断）；
* 进程 ``7-14`` 均匀分配到资源 ``7-10`` 上。

对于同一组件，所有进程编号必须从 ``0`` 到 ``N-1`` **连续且互不重复**
（其中 ``N`` 为该组件的总进程数），否则放置解析会报错。

此外，对于每一对 ``resource_ranks:process_ranks``，资源数量与进程数量
必须满足“互为整数倍”的关系：

* 要么一个进程占用多个资源；
* 要么多个进程共享同一个资源。


在代码中引用放置配置
~~~~~~~~~~~~~~~~~~~~~~~~

在配置文件中定义好 ``cluster`` 与 ``component_placement`` 后，
可以在 RLinf 代码中通过 ``rlinf.scheduler.ComponentPlacement``
获取放置信息，如下所示：

.. code-block:: python

	import hydra
	from rlinf.scheduler import Cluster, ComponentPlacement, Worker

	class TestWorker(Worker):
		def __init__(self):
			super().__init__()

		def run(self):
			self.log_info(f"Hello from TestWorker rank {self._rank}!")

	# 示例：YAML 配置文件名为 "conf.yaml"
	@hydra.main(version_base=None, config_path=".", config_name="conf")
	# cfg 即解析后的 YAML 配置
	def main(cfg):
		# 使用配置初始化集群对象
		cluster = Cluster(cluster_cfg=cfg.cluster)

		# 创建 ComponentPlacement，解析放置规则
		placement = ComponentPlacement(cfg, cluster)

		# 获取组件 "test_worker" 对应的放置策略
		strategy = placement.get_strategy("test_worker")

		# 按照放置策略启动 TestWorker
		worker = TestWorker.create_group().launch(
			cluster, placement_strategy=strategy
		)
		worker.run().wait()

	main()

对应的 ``conf.yaml`` 如下所示：

.. code-block:: yaml

	cluster:
	  num_nodes: 1
	  component_placement:
	    test_worker:  # 组件名称
	      node_group: a800
	      placement: 0-3  # 在节点组 a800（仅包含节点 0）上的 GPU 0-3 各启动一个进程

	  node_groups:
	    - label: a800
	      node_ranks: 0


放置策略如何执行
------------------

在内部，:class:`rlinf.scheduler.placement.ComponentPlacement`
会解析放置字符串，并根据所选节点组是否有硬件资源来决定使用哪种具体策略：

* 如果节点组 **没有专门的硬件资源** （既无机器人也无加速卡），
  RLinf 使用 :class:`rlinf.scheduler.placement.NodePlacementStrategy`
  仅通过节点编号进行放置。每个节点被视为一个资源，单个进程不能跨节点运行。

* 如果节点组 **有硬件资源** （如加速卡或自定义硬件），
  RLinf 使用 :class:`rlinf.scheduler.placement.FlexiblePlacementStrategy`
  将每个进程映射到某个节点上的一个或多个本地硬件编号上。

无论使用哪种策略，最终都会生成一组底层的
:class:`rlinf.scheduler.placement.Placement` 对象，其中包含：

* 该进程在整个集群中的全局进程编号；
* 该进程所在节点的全局节点编号及其在节点上的本地索引；
* 分配给该进程的硬件编号及其本地索引；
* 是否通过 ``CUDA_VISIBLE_DEVICES`` 等环境变量隐藏未分配给
  该 worker 的加速卡。

在典型使用场景下，你只需要正确编写 ``cluster`` 段落和
``component_placement`` 配置。RLinf 会基于它们自动完成异构集群中
各类组件的放置工作。

Placement 策略
---------------------------

见 :doc:`../../apis/placement` 中对Placement策略的详细描述。
