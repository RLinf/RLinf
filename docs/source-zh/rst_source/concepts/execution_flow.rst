RLinf 执行流程
==============

RLinf 的每次训练任务都按四个运行时阶段推进：入口脚本加载配置并启动 Ray 集群、为各角色拉起 worker 组、建立分布式通道并交由 runner 编排，随后进入训练迭代循环。本页梳理端到端的运行生命周期；Python 编程模型见 :doc:`flow`，GPU 部署模式的权衡见 :doc:`execution_modes`。

端到端架构
----------

下图将左侧的启动脚本、中间的主控进程调度逻辑与右侧的分布式运行时对应起来。``Worker``、``WorkerGroup`` 与 ``Channel`` 三个核心抽象把逻辑 RL 工作流与物理硬件布局解耦。

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf_exec_flow.jpg
   :alt: RLinf execution flow
   :width: 95%
   :align: center

阶段 1：配置加载与集群启动
--------------------------

启动脚本（例如具身训练的 ``examples/embodiment/run_embodiment.sh`` 或数学推理的 ``examples/reasoning/run_main_grpo_math.sh``）调用对应的 Python 入口脚本（``examples/embodiment/train_embodied_agent.py`` 或 ``examples/reasoning/main_grpo.py``）。入口脚本校验 Hydra YAML 配置后，初始化全局单例 ``Cluster``。

``Cluster`` 连接 Ray、探测各节点的 CPU 与加速器拓扑，并拉起负责节点管理、worker 生命周期管理与集合通信的全局服务。多节点运行时，需先在每个节点导出唯一的 ``RLINF_NODE_RANK`` 并启动 Ray，再仅在头节点运行入口脚本。

阶段 2：Placement 解析与启动 Worker 组
--------------------------------------

随后，入口脚本将 ``cluster.component_placement`` 解析为 actor、rollout、env、reward、critic 或 agent 等角色的 placement 策略。推理任务按模型并行规则排列共享式或分离式 rank，具身任务则按混合布局分配 GPU、CPU 与真机节点。

每个组件在目标设备上启动远程进程，并向主控进程返回一个 ``WorkerGroup`` 句柄。主控进程通过该句柄即可并行触发整组 worker 的方法，无需手写 Ray 调度代码：

.. code-block:: text

   入口脚本（Hydra YAML）
       │
       ├──► Cluster(num_nodes=...)              # 发现节点并启动全局管理器
       ├──► ComponentPlacement(cfg, cluster)    # 将各角色映射到节点与 GPU
       ├──► Worker.create_group().launch(...)   # 拉起 Actor、Rollout、Env、Reward
       └──► Runner(cfg, actor, rollout, ...)    # 接管训练与评估主循环

阶段 3：Channel 构建与 Runner 编排
----------------------------------

入口脚本将所有 worker 组句柄传入任务对应的 runner（面向具身、推理、智能体或 SFT 任务），完成 worker 初始化后启动训练主循环。

观测、动作和轨迹等大体积数据不经过主控进程中转。Runner 创建分布式 ``Channel`` 队列，并将队列句柄传给各 worker：

- 具身训练通过独立队列分别向环境发送动作、向 rollout worker 传递观测，并将完整轨迹汇总给 actor。
- Worker 直接在队列上读写数据，权重同步和大张量传输则交由底层集合通信组完成。

阶段 4：单步训练迭代
--------------------

进入 runner 主循环后，每个同步训练步按五个阶段顺序推进：

1. **权重同步**：actor 将最新 policy 权重广播给 rollout worker。
2. **Rollout 与环境交互**：rollout worker 与环境 worker（或工具与智能体循环）通过队列交换观测与动作，输出完整轨迹。
3. **奖励与优势计算**：若配置了独立 reward worker 则先打分，再由 actor 计算 GAE、GRPO 等优势与回报。
4. **Policy 更新**：actor（以及存在时的 critic）执行前向、反向与优化器更新，返回汇总指标与阶段耗时。
5. **评估与保存**：到达设定间隔时，runner 触发验证并保存 checkpoint。

在流水线或异步 runner 中，环境步进、rollout 推理和 actor 训练通过同一套队列接口持续并发重叠，runner 负责控制权重同步节奏与样本陈旧度边界。

延伸阅读
--------

- :doc:`flow`：M2Flow 编程模型如何用 worker 组与队列编排训练逻辑。
- :doc:`cluster`、:doc:`worker` 与 :doc:`channel`：调度与数据交换原语的详细机制。
- :doc:`placement` 与 :doc:`execution_modes`：placement 语法与共享式、分离式、混合式部署的权衡。
- :doc:`../reference/api/runners`：内置 runner 的 Python API 参考。
