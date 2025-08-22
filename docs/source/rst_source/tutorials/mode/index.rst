Flexible Execution Modes
========================

Conventional RL post-training systems can typically be categorized—based on their GPU placement strategy—into two primary modes: :doc:`collocated` and :doc:`disaggregated`.

In co-located mode, all major components (i.e., workers), such as generator, actor inference, and actor training, share the same set of GPUs or nodes. In contrast, disaggregated mode places these components on separate GPUs or nodes. However, these modes cannot well deal with the RL workload such as embodied intelligence, which has more components (e.g., simulators) and more complex communication among components, e.g., fine-grained interactions between the **simulator** and **generator**.

To efficiently adapt to variable RL workloads, RLinf supports flexible component placement and execution mode, enabling diverse component orchestration. The components can be placed on any GPUs. When two components are placed on the same set of GPUs, users can configure either the two components are resident in GPU memory or the two components swith the usage of the GPUs based on offloading/reloading mechanism. When the two components use separate GPUs, they can run sequentially one after the other, which induces GPU idle time, or they can run in a pipelined manner, so all GPUs stay busy. :doc:`hybrid` enables flexible configuration of placement and execution mode.


.. toctree::
   :hidden:
   :maxdepth: 1

   collocated
   disaggregated
   hybrid