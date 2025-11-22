异构软硬件集群配置
=====================

RLinf 支持在具有异构硬件和软件环境的集群上运行。例如，你可以：

* 在支持光线追踪的 GPU（如 RTX 4090）上运行高保真模拟器；
* 在大显存计算 GPU（如 A100）上进行训练；
* 在无 GPU 的节点上运行搜索 Agent。
* 在存在特殊硬件（如机械臂）的节点上运行机器人控制器

要搭建这样的异构环境，只需要在 YAML 配置文件中正确配置 ``cluster`` 段落即可。


集群配置总览
------------

``cluster`` 段落描述了：**你拥有哪些机器**，以及 **RLinf 应该如何在这些机器上放置各个组件（actor、rollout、env、agent 等）**。

从高层来看，你需要指定：

* 集群中节点（node）的总数量；
* 一组 *节点组（node group）*，每个节点组内部拥有相同的硬件 / 运行环境；
* 一条 *组件放置（component placement）* 规则，用来把逻辑组件映射到具体的硬件资源（GPU、机器人或仅仅是节点）。


示例配置
~~~~~~~~

下面的示例展示了一个具有异构硬件、并通过
``env_configs`` 配置每个节点软件环境的集群：

.. code-block:: yaml

	cluster:
	  num_nodes: 18

	  component_placement:
	    actor:
	      node_group: a800
	      placement: 0-63           # 在 ``a800`` 节点组内部的硬件编号
	    rollout:
	      node_group: 4090
	      placement: 0-63           # 在 ``4090`` 节点组内部的硬件编号
	    env:
	      node_group: franka
	      placement: 0-1            # 在 ``franka`` 节点组内部的机器人硬件编号
	    agent:
	      node_group: node
	      placement: 0-1:0-199,2-3:200-399  # 节点编号 : 进程编号

	  node_groups:
	    - label: a800
	      node_ranks: 0-7
	      env_configs:
	        - node_ranks: 0-7
	          python_interpreter_path: /opt/venv/openpi/bin/python3
	          env_vars:
	            - GLOO_SOCKET_IFNAME: "eth0"

	    - label: 4090
	      node_ranks: 8-15
	      env_configs:
	        - node_ranks: 8-15
	          env_vars:
	            - GLOO_SOCKET_IFNAME: "eth1"

	    - label: franka
	      node_ranks: 16-17
	      hardware:
	        type: Franka
	        configs:
	          - robot_ip: "10.10.10.1"
	            node_rank: 16
	            camera_serials:
	              - "322142001230"
	              - "322142001231"
	          - robot_ip: "10.10.10.2"
	            node_rank: 17
	            camera_serials:
	              - "322142001232"
	              - "322142001233"


配置解释
--------

上面的配置表达了如下含义：

* ``num_nodes: 18`` —— 集群中共有 18 个节点。节点编号从 0 开始，
  并且需要在每个节点启动 Ray 之前，通过环境变量 ``RLINF_NODE_RANK``
  指定对应的节点编号。

* ``node_groups`` —— 每一项定义了一个 **节点组（node group）**，
  表示一组具备相同硬件与环境的节点。节点组包含：

  - ``label``：在 ``component_placement`` 中引用该节点组时使用的唯一字符串标识，
    例如 ``a800``、``4090``、``franka``。标签区分大小写。

    标签 ``cluster`` 与 ``node`` 由调度器保留，用户不能自定义使用。
    其中 ``node`` 是一个特殊的节点组，表示“所有节点但不关心硬件”，
    适合放置只依赖 CPU 的进程（如 agent 等）。

  - ``node_ranks``：属于该节点组的全局节点编号列表或范围。
    在示例中，``a800`` 覆盖 ``0-7``，``4090`` 覆盖 ``8-15``，``franka`` 覆盖 ``16-17``。

  - ``env_configs`` （可选）：用于描述该节点组内部不同节点子集的软件环境。
    每一项是一个 ``NodeGroupEnvConfig`` ，包含其自身的 ``node_ranks``、``env_vars`` 与 ``python_interpreter_path``：

	* 每个 ``env_configs`` 项的 ``node_ranks`` 必须是父节点组 ``node_ranks`` 的子集；同一节点组中不同 ``env_configs`` 的 ``node_ranks`` 之间不能重叠；

	* ``env_vars`` 是一组只包含单个键值对的字典列表；在同一节点上，环境变量的键对于同一个组内的同一个节点必须是唯一的；
	
	* ``python_interpreter_path`` 用于指定这些节点上使用的 Python 解释器。

  - ``hardware`` （可选）：描述该节点组上 *非加速卡* 硬件（如机器人）的结构化配置。
    其具体字段由 ``type`` 决定（例如 ``Franka``）。当指定了 ``hardware`` 时，
    该节点组被视为仅包含这一类硬件资源，并在组内为其定义 **硬件编号** （0, 1, ...）。

* 如果节点组 **未** 指定 ``hardware`` 字段，RLinf 的行为如下：

  - 如果节点上检测到加速卡硬件（GPU、NPU 等），则这些加速卡成为默认资源，
    它们的本地索引用作硬件编号；
  - 如果没有任何加速卡存在，则每个 **节点本身** 被视为一个硬件资源，
    在本节点内的硬件编号为 0。

当你在 ``component_placement`` 中引用某个 ``node_group`` 时，
其 ``placement`` 字段中写下的始终是 **该节点组内部的硬件编号** ：

* 若节点组配置了 ``hardware``，则编号指的是该类型硬件的编号
  （例如机器人 0–3）；
* 若未配置 ``hardware`` 但存在加速卡，则编号指的是自动检测到的
  加速卡索引；
* 若也没有加速卡，则编号对应的是节点本身。

当在 ``component_placement`` 中使用保留标签 ``node`` 时，
调度器不再进行任何硬件层面的放置，而是直接把编号解释为“节点编号”。
这非常适合将与硬件无关的进程（例如 agent 或只用 CPU 的 worker）
精确放到某些节点上，而不考虑 GPU 分布。