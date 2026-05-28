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

- :doc:`convertor`
   Describes how to convert a saved checkpoint file into HuggingFace safetensors format,
   which can be used for checkpoint evaluation or uploading to the HuggingFace Hub.

- :doc:`multi_node`
   Start a multi-machine Ray cluster, configure environment variables and code sync,
   and launch RLinf training tasks across nodes.

- :doc:`execution_modes`
   Covers all three execution modes in RLinf: collocated, disaggregated, and hybrid,
   with example configurations and programming patterns for each.


.. toctree::
   :hidden:
   :maxdepth: 1

   worker
   placement
   flow
   channel
   convertor
   multi_node
   execution_modes
