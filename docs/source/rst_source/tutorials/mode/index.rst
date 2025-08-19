Flexible Execution Modes
==========================

Conventional RL post-training systems can typically be categorized—based on their GPU placement strategy—into two primary modes: :doc:`colocated` and :doc:`disaggregated`.

In colocated mode, all major components such as the simulator, generator, and actor modules share the same GPU or machine. 
In contrast, disaggregated mode places these components on separate GPUs or nodes, enabling more flexibility.

However, in complex, multi-step RL workloads—such as those involving embodied intelligence—the *inference* phase becomes a bottleneck. This stage requires frequent, fine-grained interactions between the **simulator** and **generator**, significantly slowing down rollout throughput.

To address this challenge, RLinf introduces a third execution strategy: :doc:`hybrid` mode.  
This mode features fine-grained pipelining and flexible component placement, balancing communication cost with hardware efficiency. It allows for overlapping computation and communication across pipeline stages, maximizing utilization and minimizing rollout latency in demanding workloads.

.. toctree::
   :hidden:
   :maxdepth: 1

   colocated
   disaggregated
   hybrid