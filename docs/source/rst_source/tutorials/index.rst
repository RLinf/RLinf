Tutorials
=========

This section offers an in-depth exploration of **RLinf**.  
It provides a collection of hands-on tutorials covering all the core components and features of the library.

- :doc:`user/index`  
   From a user’s perspective, this tutorial introduces the fundamental components of RLinf, 
   including how to configure tasks using YAML, assign workers for each RL task, 
   and manage GPU resources from a global, cluster-level viewpoint.

- :doc:`mode/index`  
   Learn about the different execution modes supported by RLinf, 
   including task-colocated mode, task-disaggregated mode, 
   and a hybrid mode with fine-grained pipelining.

- :doc:`scheduler/index`  
   Understand RLinf’s automatic scheduling mechanisms, 
   featuring the online scaling mechanism and auto scheduling policy 
   to dynamically adapt to workload changes.

- :doc:`communication/index`  
   Explore the underlying logic of RLinf’s elastic communication system, 
   covering peer-to-peer communication and the implementation of 
   producer/consumer-based channels built on top of it.

- :doc:`advance/index`  
   Dive into RLinf’s advanced features, 
   such as 5D parallelism configuration and LoRA integration, 
   designed to help you achieve optimal training efficiency and performance.

- :doc:`rlalg/index`  
   Follow comprehensive tutorials for each supported RL algorithm, 
   including PPO, GRPO, and more—complete with ready-to-use configurations and practical performance tuning tips.

- :doc:`extend/index`  
   Learn how to extend RLinf by integrating your own training algorithms, 
   simulation environments, and model architectures to suit your specific research needs.

.. toctree::
   :hidden:
   :maxdepth: 4

   user/index
   mode/index
   scheduler/index
   communication/index
   advance/index
   rlalg/index
   extend/index
