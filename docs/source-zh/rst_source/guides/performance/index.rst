性能诊断与提速
==============

先用 Profiling 查清耗时落在仿真、Rollout、Actor 更新还是通信，再按瓶颈开启对应的提速手段。

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Profiling 性能剖析
      :link: ../profile
      :link-type: doc

      采集各阶段耗时与 PyTorch Trace，先定位真正的瓶颈环节。

   .. grid-item-card:: 训练 Pipeline 模式
      :link: ../training_pipeline
      :link-type: doc

      让具身 Rollout 与 FSDP Actor 更新流水线重叠，减少 GPU 空等。

   .. grid-item-card:: Env 解耦执行
      :link: ../env_decoupled_mode
      :link-type: doc

      仿真步进较慢时，将 Env Worker 与策略 Rollout / 训练解耦运行。

   .. grid-item-card:: 观测数据压缩
      :link: ../obs_compression
      :link-type: doc

      在 Env 与 Rollout 之间用 LZ4 / Zstd 压缩图像观测，降低传输开销。

   .. grid-item-card:: 异步 Agent RL
      :link: ../async_agentic
      :link-type: doc

      用全异步训练消除多轮 Agent Rollout 中的长尾拖慢问题。

   .. grid-item-card:: 实时控制（RTC）
      :link: ../rtc
      :link-type: doc

      让当前动作块执行与下一动作块推理重叠，隐藏部署与评测延迟。

   .. grid-item-card:: FUSCO 集合通信加速
      :link: ../../examples/system/fusco
      :link-type: doc

      优化大规模多机训练中的集合通信吞吐。

.. toctree::
   :hidden:

   Profiling 性能剖析 <../profile>
   训练 Pipeline 模式 <../training_pipeline>
   Env 解耦执行 <../env_decoupled_mode>
   观测数据压缩 <../obs_compression>
   异步 Agent RL <../async_agentic>
   实时控制（RTC） <../rtc>
   FUSCO 集合通信加速 <../../examples/system/fusco>
