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
   :width: 95%
   :align: center

- **Code Flow Overview.** Let's first look at the main code flow shown in the left of the figure. The `run.sh` script runs `main_grpo.py` which serves as the entry point. In `main_grpo.py`, the main function first determines the placement for Workers (e.g, actor, rollout) based on the YAML configuration (i.e., `cluster/component_placement`). Specifically, each Worker can be flexibly assigned to any number of GPUs (or other types of accelerators) through the YAML configuration. After determining the placement, the script launches WorkGroups, each consisting of one or more Worker processes of the same type. These WorkerGroups are then passed to the Runner, where the main RL training workflow is encapsulated in the `run()` function.

- **Main Process.** Here, Worker placement defined in the YAML `placement` configuration is translated into our `Worker Placement Strategy`, which dictates on which node and/or which GPU a worker process should run. Based on this, worker processes are launched in the cluster via `Worker`'s `launch()` API. The `launch` API returns a handle to collectively manage all remote processes of the same Worker class, e.g. RolloutWorker, termed a `WorkerGroup`. You can command a group's processes to simultaneously execute any public functions of the Worker class via this `WorkerGroup` handle. The `Runner` then obtains these handles and orchestrates the execution of Worker processes remotely. However, communication between Workers is not conducted via the main `Runner` process. Instead, `Runner` establishes communication Channels (`Channel.Create()`) for inter-worker data exchange. For example, in a typical `Runner` iteration, it first calls RolloutGroup's `rollout` to execute the `rollout` function on all RolloutWorker processes, and then similarly executes ActorGroup's `train` function. For each function, it passes in the created Channels for them to communicate.

- **Key Concepts and Features.** The right side of the figure highlights three core features of RLinf. (i) Flexible Worker placement, a WorkerGroup can be elastically placed on any node or GPU. (ii) Easy-to-use communication interface, users can send or receive data by referencing only the WorkerGroup's name. (iii) Distributed data Channels, Workers can easily exchange data using `channel.put` and `channel.get`.

RLinf adopts a modular design that abstracts distributed complexity through the Worker, WorkerGroup, and Channel features.
This design enables users to build large-scale RL training pipelines with minimal distributed programming effort, especially for embodied intelligence and agent-based systems.


.. toctree::
   :hidden:
   :maxdepth: 4

   user/index
   mode/index
   scheduler/index
   communication/index
   components/index
   advance/index
   rlalg/index
   extend/index
