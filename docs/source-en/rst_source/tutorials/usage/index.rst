Usage and Programming Tutorial
===============================

This section introduces the core programming model and deployment patterns of RLinf.
You will learn the fundamental concepts—Workers, WorkerGroups, placement, and
communication—and how to scale from a single node to multi-node clusters with
flexible execution modes.

- :doc:`worker`
   Introduces the *Worker*, the modular execution unit in RLinf. Multiple similar
   Workers form a *WorkerGroup*, simplifying distributed execution.

- :doc:`placement`
   Explains how RLinf strategically assigns hardware resources across tasks and workers
   to ensure efficient utilization across GPUs, NPUs, robotic hardware, and CPU-only nodes.

- :doc:`flow`
   Integrates WorkerGroup, Placement, and Cluster concepts to present the complete
   programming flow of RLinf.

- :doc:`channel`
   Introduces the *Channel* abstraction for asynchronous producer-consumer communication
   between workers, essential for fine-grained pipelining across RL stages.

- :doc:`collective`
   Covers low-level, high-performance Python object exchange between workers,
   using optimized point-to-point backends such as CUDA IPC and NCCL.

- :doc:`multi_node`
   Start a multi-machine Ray cluster, configure environment variables and code sync,
   and launch RLinf training tasks across nodes.

- :doc:`collocated`
   All major components share the same set of GPUs or nodes.

- :doc:`disaggregated`
   Components are assigned to separate GPUs or nodes for dedicated resource usage.

- :doc:`hybrid`
   A flexible combination of collocated and disaggregated placement and execution strategies,
   enabling fine-grained pipelining and optimal GPU utilization.


.. toctree::
   :hidden:
   :maxdepth: 1

   worker
   placement
   flow
   channel
   collective
   multi_node
   collocated
   disaggregated
   hybrid
