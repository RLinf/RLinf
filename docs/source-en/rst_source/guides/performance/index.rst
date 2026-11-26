Profile & Speed Up
==================

Start with profiling to see whether time goes to simulator stepping, policy rollout, actor updates, or communication—then turn on the targeted optimization below.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Performance Profiling
      :link: ../profile
      :link-type: doc

      Capture step timings and PyTorch traces to pinpoint the real bottleneck first.

   .. grid-item-card:: Training Pipeline Mode
      :link: ../training_pipeline
      :link-type: doc

      Overlap embodied rollout stages and packed FSDP actor micro-batches so GPUs stay busy.

   .. grid-item-card:: Decoupled Env Execution
      :link: ../env_decoupled_mode
      :link-type: doc

      Decouple slow simulators from policy rollout and training workers.

   .. grid-item-card:: Observation Compression
      :link: ../obs_compression
      :link-type: doc

      Compress high-resolution image observations with LZ4 or Zstd between env and rollout workers.

   .. grid-item-card:: Async Agentic RL
      :link: ../async_agentic
      :link-type: doc

      Eliminate long-tail stragglers in multi-turn agent rollouts with asynchronous training.

   .. grid-item-card:: Real-Time Control (RTC)
      :link: ../rtc
      :link-type: doc

      Overlap action chunk execution with next-chunk inference to hide deployment latency.

   .. grid-item-card:: FUSCO Communication
      :link: ../../examples/system/fusco
      :link-type: doc

      Accelerate collective communication in large-scale multi-node runs.

.. toctree::
   :hidden:

   Performance Profiling <../profile>
   Training Pipeline Mode <../training_pipeline>
   Decoupled Env Execution <../env_decoupled_mode>
   Observation Compression <../obs_compression>
   Async Agentic RL <../async_agentic>
   Real-Time Control (RTC) <../rtc>
   FUSCO Communication <../../examples/system/fusco>
