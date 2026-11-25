M2Flow Programming Flow
=======================

RLinf uses a Macro-to-Micro Flow (M2Flow) programming model so you can express an RL algorithm as high-level interactions among worker groups while leaving device mapping, pipelining, and collective communication to the runtime. Read this page to understand how to compose worker groups, channels, and runners in Python; see :doc:`execution_flow` for the four-stage job lifecycle and :doc:`../reference/configuration` for Hydra YAML options.

Macro Logic vs. Micro Execution
-------------------------------

Writing a distributed RL loop usually tangles algorithm steps with hardware layout: moving rollout workers to separate GPUs or overlapping environment stepping with inference forces rewrites across the training loop.

M2Flow splits those two concerns cleanly:

- **Macro flow**: the driver script and runner declare *what* runs and in *what order*—create worker groups, connect them with asynchronous channels, and invoke group methods across stages.
- **Micro flow**: placement strategies, channel queues, and collective groups decide *where* each rank executes, how batches split across stages, and when tensors move between devices.

Because the macro code talks only to ``WorkerGroup`` and ``Channel`` handles, switching between collocated, disaggregated, and hybrid execution changes ``cluster.component_placement`` in YAML instead of changing runner code.

Composing Worker Groups in an Entry Script
------------------------------------------

An entry script translates configuration into a cluster view, launches each component with its placement strategy, and passes the resulting handles to a runner. The example below from ``examples/embodiment/train_embodied_agent.py`` shows the standard pattern for an embodied training job:

.. code-block:: python

   cluster = Cluster(cluster_cfg=cfg.cluster)
   component_placement = HybridComponentPlacement(cfg, cluster)

   actor_placement = component_placement.get_strategy("actor")
   actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
       cluster, placement_strategy=actor_placement
   )

   rollout_placement = component_placement.get_strategy("rollout")
   rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
       cluster, placement_strategy=rollout_placement
   )

   env_placement = component_placement.get_strategy("env")
   env_group = EnvWorker.create_group(cfg).launch(
       cluster, placement_strategy=env_placement
   )

   runner = EmbodiedRunner(
       cfg=cfg,
       actor=actor_group,
       rollout=rollout_group,
       env=env_group,
   )
   runner.init_workers()
   runner.run()

Notice two properties of this setup. First, every worker group follows the same ``create_group(...).launch(...)`` call regardless of whether it runs on one GPU or across multiple nodes. Second, the runner receives abstract group handles rather than device IDs, so the same runner class works under any valid placement strategy (see :doc:`placement` for placement syntax).

Driving Asynchronous Stages with Handles and Channels
-----------------------------------------------------

Calling a method on a ``WorkerGroup`` returns an asynchronous future handle immediately instead of blocking the driver. A runner uses that behavior to start producer and consumer stages concurrently, connect them through ``Channel`` queues, and synchronize only at stage boundaries.

Inside the embodied training loop, each iteration runs four macro operations:

.. code-block:: python

   for step in range(max_steps):
       update_rollout_weights()
       generate_rollouts()
       actor_group.compute_advantages_and_returns().wait()
       actor_group.run_training().wait()

Two helper patterns make the micro execution overlap cleanly without extra boilerplate in the loop:

1. **Paired weight synchronization**: both the rollout receiver and actor sender launch together and wait at the end of the transfer.

   .. code-block:: python

      def update_rollout_weights():
          rollout_futures = rollout_group.sync_model_from_actor()
          actor_futures = actor_group.sync_model_to_rollout()
          actor_futures.wait()
          rollout_futures.wait()

2. **Channel-backed rollout pipeline**: environment workers, rollout workers, and actor receivers all start in parallel, passing observations, actions, and trajectories through shared channels until the rollout epoch completes.

   .. code-block:: python

      def generate_rollouts():
          env_futures = env_group.interact(
              input_channel=env_channel, output_channel=rollout_channel
          )
          rollout_futures = rollout_group.generate(
              input_channel=rollout_channel,
              output_channel=env_channel,
              actor_channel=actor_channel,
          )
          actor_futures = actor_group.recv_rollout_batch(
              input_channel=actor_channel
          )
          env_futures.wait()
          rollout_futures.wait()
          actor_futures.wait()

When ``rollout.pipeline_stage_num`` is greater than 1 and environments and rollout run on separate GPUs, those three concurrent calls automatically overlap environment stepping with policy inference across pipeline chunks. Changing the pipeline depth or device split requires no changes to the Python loop above.

Next Steps
----------

- :doc:`execution_flow`: full four-stage runtime lifecycle from launcher script to checkpointing.
- :doc:`execution_modes`: how collocated, disaggregated, and hybrid modes trade off memory and throughput.
- :doc:`worker` and :doc:`channel`: full contracts for worker groups, futures, and distributed queues.
- :doc:`../reference/api/runners`: API reference for RLinf's built-in training runners.
