RLinf Execution Flow
====================

Every RLinf job runs through four runtime stages: an entry script loads configuration and boots a Ray-backed cluster, launches each role as a worker group, wires distributed channels into a task runner, and executes the training loop. Read this page for the end-to-end lifecycle of a run; see :doc:`flow` for the Python programming model and :doc:`execution_modes` for GPU placement trade-offs.

End-to-End Architecture
-----------------------

The diagram below maps the launch script on the left to the driver process in the middle and the distributed runtime on the right. Three core abstractions—``Worker``, ``WorkerGroup``, and ``Channel``—keep the logical RL workflow separate from physical hardware placement.

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf_exec_flow.jpg
   :alt: RLinf execution flow
   :width: 95%
   :align: center

Stage 1: Configuration and Cluster Boot
---------------------------------------

A launcher script (such as ``examples/embodiment/run_embodiment.sh`` or ``examples/reasoning/run_main_grpo_math.sh``) invokes a Python entry script (such as ``examples/embodiment/train_embodied_agent.py`` or ``examples/reasoning/main_grpo.py``). The entry script resolves and validates the Hydra YAML configuration, then initializes a singleton ``Cluster``.

``Cluster`` connects to Ray, inspects the available nodes and accelerators, and starts the cluster-wide management actors that track node topology, worker lifecycles, and collective communication groups. On multi-node jobs, start Ray on every node first with a unique ``RLINF_NODE_RANK``, then run the entry script on the head node only.

Stage 2: Placement and Launching Worker Groups
----------------------------------------------

Next, the entry script translates ``cluster.component_placement`` into per-component placement strategies for actor, rollout, environment, reward, critic, or agent roles. LLM reasoning jobs map model-parallel ranks for collocated or disaggregated layouts, while embodied jobs use flexible hybrid layouts across GPU, CPU, and robot nodes.

Each component spawns its remote processes onto the assigned devices and returns a ``WorkerGroup`` handle to the driver. Through that handle, the driver triggers worker methods across all ranks in parallel without writing Ray scheduling logic directly:

.. code-block:: text

   Entry Script (Hydra YAML)
       │
       ├──► Cluster(num_nodes=...)              # Discover nodes & start managers
       ├──► ComponentPlacement(cfg, cluster)    # Map roles to GPU / node ranks
       ├──► Worker.create_group().launch(...)   # Spawn Actor, Rollout, Env, Reward
       └──► Runner(cfg, actor, rollout, ...)    # Own the training / eval loop

Stage 3: Channel Setup and Runner Orchestration
-----------------------------------------------

The entry script hands the launched worker-group handles to a task runner (for embodied, reasoning, agentic, or SFT workloads), initializes the workers, and starts the training loop.

Bulk observations, actions, and trajectories never pass through the driver process. Instead, the runner creates distributed ``Channel`` queues and passes those handles into worker methods:

- In embodied training, separate channels carry actions to environments, observations to rollout workers, and completed trajectories to actor workers.
- Workers exchange payloads directly through put and get calls on those channels, while collective communication groups handle high-throughput weight synchronization and tensor transfers between worker groups.

Stage 4: One Training Iteration
-------------------------------

Inside the runner loop, each synchronous training step advances through five phases:

1. **Weight sync**: the actor broadcasts updated policy weights to the rollout workers.
2. **Rollout and environment interaction**: rollout workers and environment workers (or tool and agent loops) exchange actions and observations over channels and emit completed trajectories.
3. **Reward and advantage**: optional reward workers score trajectories, and actor workers compute advantages and returns (such as GAE or GRPO).
4. **Policy update**: actor workers (and critic workers, if configured) run forward, backward, and optimizer steps, then return aggregated metrics and stage timings to the runner.
5. **Evaluation and checkpointing**: at configured intervals, the runner runs validation and writes actor checkpoints.

In pipelined or asynchronous runners, environment stepping, rollout inference, and actor training overlap continuously over the same channel interfaces while the runner governs weight-sync cadence and sample staleness bounds.

Next Steps
----------

- :doc:`flow`: how the M2Flow programming model composes worker groups and channels in Python.
- :doc:`cluster`, :doc:`worker`, and :doc:`channel`: deep dives into the scheduling and data-exchange primitives.
- :doc:`placement` and :doc:`execution_modes`: placement syntax and collocated, disaggregated, and hybrid trade-offs.
- :doc:`../reference/api/runners`: Python API reference for RLinf's built-in runners.
