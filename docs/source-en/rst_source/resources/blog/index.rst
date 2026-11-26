Blog
====

Explore RLinf engineering reports, system optimizations, and model integrations.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: MUSA × WoVR
      :link: musa_wovr
      :link-type: doc

      MUSA adaptation, accuracy alignment, and 8- and 16-GPU scaling on Moore Threads S5000.

   .. grid-item-card:: WorldLoop VLM Reward
      :link: worldloop_topreward
      :link-type: doc

      Use a training-free VLM reward for world-model RL with WoVR.

   .. grid-item-card:: SGLang × Cosmos3
      :link: cosmos3_sft_and_sglang_evaluate
      :link-type: doc

      Fine-tune Cosmos3 and evaluate it with parallel SGLang inference.

   .. grid-item-card:: BEHAVIOR 25× Speedup
      :link: behavior_system_optimization
      :link-type: doc

      Accelerate environment interaction for household manipulation tasks.

   .. grid-item-card:: Ascend Embodied
      :link: ascend_embodied_optimization
      :link-type: doc

      Boost embodied RL throughput on Ascend NPUs by reusing attention metadata, overlapping FSDP communication, and trimming host-device syncs.

   .. grid-item-card:: Cursor Coding RL
      :link: build_a_coding_online_rl_case
      :link-type: doc

      Connect code completion feedback to online policy training.

   .. grid-item-card:: RLinf vs. veRL
      :link: compare_with_verl
      :link-type: doc

      Benchmark setup, configuration, and throughput comparisons.

.. toctree::
   :hidden:
   :maxdepth: 1

   MUSA × WoVR <musa_wovr>
   WorldLoop VLM Reward <worldloop_topreward>
   SGLang × Cosmos3 <cosmos3_sft_and_sglang_evaluate>
   BEHAVIOR 25× Speedup <behavior_system_optimization>
   Ascend Embodied <ascend_embodied_optimization>
   Cursor Coding RL <build_a_coding_online_rl_case>
   RLinf vs. veRL <compare_with_verl>
