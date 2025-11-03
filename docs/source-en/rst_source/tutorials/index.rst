Tutorials
=========

This section offers an in-depth exploration of **RLinf**.  
It provides a collection of hands-on tutorials covering all the core components and features of the library.
Below, we first give an overview of RLinf execution flow to help users understand how RLinf executes an RL training.

RLinf Execution Overview
------------------------

The following figure demonstrates the overview of RLinf execution flow, including the main code flow (left), the main process corresponding to the code flow (middle), and the concept of Worker, WorkerGroup, and Channel (right).

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf_exec_flow.jpg
   :alt: RLinf execution flow
   :width: 80%
   :align: center

- **Main Code Flow.** Let's first look at the main code flow shown in the left of the figure. The `run.sh` script calls `main_grpo.py` which is the entry point main function. In `main_grpo.py`, the main function first generates placement for Workers (e.g, actor, rollout) following the specified placement in yaml configuration file (i.e., `cluster/component_placement`). Specifically, a Worker can be flexibly placed on any number of GPUs (or other types of accelerators) through the yaml configuration file. Following the generated placement, the main function launches WorkGroups, where a WorkGroup is a group of spawned Worker processes of a Worker. Then it passes the launched WorkGroups into Runner and executes Runner's key function `run()`. The main RL training workflow is wrapped in the `run()` function.

- **Main Process.** Then, let's look at the middle part. The middle part is a more detailed illustration of the main function. It shows that the placement is created through `Worker Placement Strategy`, WorkerGroup (e.g., ActorGroup, RolloutGroup) is launched through `create_group()`. On the other hand, Runner is defined under `rlinf/runners/`. It has WorkerGroups passed in. Communication Channels are created (`Channel.Create()`) in Runner for the WorkerGroups to communicate with each other. The WorkerGroups run their core functions (e.g., `train`, `rollout`) and communicate through the corresponding channels. After each iteration, ActorGroup synchronizes model weights to RolloutGroup (`sync_model`).

- **Key Concepts and Features.** At last, the right part demonstrates three key functionalities provided by RLinf. The first is Worker placement, a WorkerGroup can be elastically placed on any node and any GPU. The second is easy-to-use send/recv communication, users can simply specify WorkerGroup's name to send or receive data. The third is distributed data channel, users can simply use `channel.put` and `channel.get` in Workers to communicate with other Workers.

RLinf adopts a modular design that abstracts distributed complexity through the Worker, WorkerGroup, and Channel features.
This design enables users to build large-scale RL training pipelines with minimal distributed programming effort, especially for embodied intelligence and agent-based systems.

- :doc:`user/index`  
   From a user’s perspective, this tutorial introduces the fundamental components of RLinf, 
   including how to configure tasks using YAML, assign workers for each RL task, 
   and manage GPU resources from a global, cluster-level viewpoint.
   Finally, we provide a complete programming flow to illustrate the high-level workflow of RLinf.

- :doc:`mode/index`  
   Learn about the different execution modes supported by RLinf, 
   including collocated mode, disaggregated mode, 
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
