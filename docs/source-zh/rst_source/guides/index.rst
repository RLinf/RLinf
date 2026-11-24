使用指南
========

按实际跑任务的先后顺序查阅指南：先把训练配好并能看日志、续训和导出，再扩到多机集群，遇到吞吐瓶颈时先做性能诊断再对症提速，最后按需切换 Rollout 后端。

按当前阶段快速定位
------------------

.. list-table::
   :header-rows: 1
   :widths: 32 44 24

   * - 当前阶段
     - 常见需求
     - 入口
   * - 把任务配好并稳定运行
     - 日志曲线、断点续训、Checkpoint 转换、LoRA 省显存、多 Env 混训
     - :doc:`配置与运行 <configure/index>`
   * - 单卡 / 单机不够，要上集群
     - 多节点训练、异构集群、云边协同、自动 Placement、5D 并行、动态调度
     - :doc:`扩展到多机集群 <launch-scale/index>`
   * - 训练太慢或 GPU 空等
     - 先做 Profiling 查瓶颈，再按需开 Pipeline、Env 解耦、观测压缩、异步 Agent RL、RTC、FUSCO
     - :doc:`性能诊断与提速 <performance/index>`
   * - 切换或部署独立推理引擎
     - SGLang / vLLM 版本、SGLang 服务与路由、HTTP 推理客户端、Session Server、PhyAI、ApxInf
     - :doc:`Rollout 后端与服务 <rollout-engines/index>`

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 1. 配置与运行
      :link: configure/index
      :link-type: doc

      配置日志与可视化、断点续训、Checkpoint 转换、LoRA 微调与多 Env 混合训练。

   .. grid-item-card:: 2. 扩展到多机集群
      :link: launch-scale/index
      :link-type: doc

      扩展到多节点、异构集群与云边协同，配置自动 Placement、5D 并行与动态调度。

   .. grid-item-card:: 3. 性能诊断与提速
      :link: performance/index
      :link-type: doc

      先用 Profiling 定位瓶颈，再开启 Pipeline、Env 解耦、观测压缩、异步 Agent RL、RTC 或 FUSCO。

   .. grid-item-card:: 4. Rollout 后端与服务
      :link: rollout-engines/index
      :link-type: doc

      管理 SGLang 与 vLLM 版本、部署独立推理服务，或启用 PhyAI / ApxInf 具身推理后端。

如果是真机遥操作、数据采集或真机多机启动，请前往 :doc:`真机通用流程 <../examples/real_world_workflows_index>`。
