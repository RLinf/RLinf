API 手册
========

查阅 RLinf 的公开 Python 接口，按自底向上的层次涵盖调度原语、worker、模型、真机机器人、算法注册表与数据容器。

基础调度 API
------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - API
     - 内容
   * - :doc:`Worker <worker>`
     - 分布式执行与通信的 ``Worker`` 与 ``WorkerGroup`` 抽象。
   * - :doc:`Placement <placement>`
     - 将 worker 组映射到节点与加速器的 placement 策略。
   * - :doc:`Cluster <cluster>`
     - Ray 集群发现、节点组与 worker 分配。
   * - :doc:`Channel <channel>`
     - 支持 collector 与 dispatcher 的异步生产者-消费者队列。

RL 组件、模型、机器人与数据
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - API
     - 内容
   * - :doc:`Actor <actor>`
     - 基于 Megatron 与 FSDP 的 actor、SFT worker 与模型管理类。
   * - :doc:`Rollout <rollout>`
     - 基于 SGLang、vLLM 与 Hugging Face 的 rollout worker。
   * - :doc:`Env <env>`
     - 环境 worker、环境工厂（``get_env_cls``、``SupportedEnvType``）与 ``RealWorldEnv``。
   * - :doc:`Runners <runners>`
     - 面向具身、推理、智能体与 SFT 任务的顶层训练循环 runner。
   * - :doc:`模型与 Policy <models>`
     - 模型注册表（``register_model``、``get_model``）与具身 ``BasePolicy`` 接口。
   * - :doc:`Robotics <robotics>`
     - 真机 ``Connection``、``RobotPart``、``Robot`` 与硬件发现 API。
   * - :doc:`算法与注册表 <algorithms>`
     - 优势函数、policy 损失、损失缩放与工具调用解析器注册表。
   * - :doc:`Agentic 数据 <data>`
     - Agentic rollout 请求与结果数据结构。
   * - :doc:`Embodied Data <embodied_data>`
     - 具身环境与 rollout 数据结构。
   * - :doc:`Replay Buffer <replay_buffer>`
     - ``TrajectoryReplayBuffer``、``TrajectoryCache`` 与预加载数据集。

.. toctree::
   :hidden:
   :maxdepth: 1

   Worker <worker>
   Placement <placement>
   Cluster <cluster>
   Channel <channel>
   Actor <actor>
   Rollout <rollout>
   Environment <env>
   Runners <runners>
   模型与 Policy <models>
   Robotics <robotics>
   算法与注册表 <algorithms>
   Data <data>
   Embodied Data <embodied_data>
   Replay Buffer <replay_buffer>
