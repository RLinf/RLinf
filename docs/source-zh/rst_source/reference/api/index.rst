Python API
==========

按字母顺序查阅 RLinf 的公开 Python 接口，涵盖调度原语、worker、模型、真机机器人、算法注册表与数据容器。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - API
     - 内容
   * - :doc:`Actor <actor>`
     - 基于 Megatron 与 FSDP 的 actor、SFT worker 与模型管理类。
   * - :doc:`Agentic Data <data>`
     - Agentic rollout 请求与结果数据结构。
   * - :doc:`Algorithm <algorithms>`
     - 优势函数、policy 损失、损失缩放与工具调用解析器注册表。
   * - :doc:`Channel <channel>`
     - 支持 collector 与 dispatcher 的异步生产者-消费者队列。
   * - :doc:`Cluster <cluster>`
     - Ray 集群发现、节点组与 worker 分配。
   * - :doc:`Embodied Data <embodied_data>`
     - 具身环境与 rollout 数据结构。
   * - :doc:`Env <env>`
     - 环境 worker、环境工厂（``get_env_cls``、``SupportedEnvType``）与 ``RealWorldEnv``。
   * - :doc:`Model <models>`
     - 模型注册表（``register_model``、``get_model``）与具身 ``BasePolicy`` 接口。
   * - :doc:`Placement <placement>`
     - 将 worker 组映射到节点与加速器的 placement 策略。
   * - :doc:`Replay Buffer <replay_buffer>`
     - ``TrajectoryReplayBuffer``、``TrajectoryCache`` 与预加载数据集。
   * - :doc:`Robotics <robotics>`
     - 真机 ``Connection``、``RobotPart``、``Robot`` 与硬件发现 API。
   * - :doc:`Rollout <rollout>`
     - 基于 SGLang、vLLM 与 Hugging Face 的 rollout worker。
   * - :doc:`Runner <runners>`
     - 面向具身、推理、智能体与 SFT 任务的顶层训练循环 runner。
   * - :doc:`Worker <worker>`
     - 分布式执行与通信的 ``Worker`` 与 ``WorkerGroup`` 抽象。

.. toctree::
   :hidden:
   :maxdepth: 1

   Actor <actor>
   Agentic Data <data>
   Algorithm <algorithms>
   Channel <channel>
   Cluster <cluster>
   Embodied Data <embodied_data>
   Env <env>
   Model <models>
   Placement <placement>
   Replay Buffer <replay_buffer>
   Robotics <robotics>
   Rollout <rollout>
   Runner <runners>
   Worker <worker>
