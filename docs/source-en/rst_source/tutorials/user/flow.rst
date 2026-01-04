High-Level Programming Flow
===========================

This section walks you through RLinf’s top-level programming logic.
We avoid low-level details and focus on the highest-level APIs so you can
understand the overall control flow and customize your own algorithms or projects.

The running example highlights RLinf’s core capability: **hybrid mode with fine-grained pipelining**
for training VLA models in an embodied-intelligence environment.


YAML Configuration 
-----------------------

Before launching any training script, the most important step is to prepare
the configuration file.
For example:

- Configs for training a **VLA** agent in embodied tasks live under
  ``examples/embodiment/config``.
- Configs for training an **LLM** on math reasoning live under
  ``examples/reasoning/config/math``.

As a starting point, we recommend getting familiar with the YAML structure of
these examples, then iterating toward your custom task. Key options include
(but are not limited to):

**1. Execution mode and the number of nodes/GPUs to use**

.. code:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-7
       env: 0-3
       rollout: 4-7

**2. Models, tokenizer and output paths**

- ``rollout.model.model_path``
- ``actor.tokenizer.tokenizer_model``
- ``actor.model.model_path``
- ``runner.logger.log_path``

**3. Training hyperparameters such as max steps and batch sizes**

- ``runner.max_epochs``
- ``runner.max_steps``
- ``actor.global_batch_size``
- ``actor.micro_batch_size``

As a first run, keep the defaults and iterate. For a full parameter reference,
see :doc:`yaml`.

Worker Launch Orchestration
---------------------------

The following Python snippet is distilled from
``examples/embodiment/train_embodied_agent.py`` and mirrors the pattern used by
all RLinf ``main`` entry points:

.. code:: python

   cluster = Cluster(num_nodes)
   component_placement = HybridComponentPlacement(cfg, cluster)

   # Create actor worker group
   actor_placement = component_placement.get_strategy("actor")
   actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
       cluster, placement_strategy=actor_placement
   )

   # Create rollout worker group
   rollout_placement = component_placement.get_strategy("rollout")
   rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
       cluster, placement_strategy=rollout_placement
   )

   # Create env worker group
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

The entry point performs three major tasks:

1. Initializes the ``Cluster`` (global resource view) and
   ``HybridComponentPlacement`` (GPU placement for all RL workers) from config.
2. Creates the **actor**, **rollout**, and **env** worker groups and manages
   them via ``WorkerGroup``.
3. Builds an ``EmbodiedRunner`` and starts the main training loop via
   ``runner.run()``.


Training Loop Overview
----------------------

The high-level logic inside ``runner.run()`` (from
``rlinf/runners/embodied_runner.py``) looks like:

.. code:: python

   for step in range(training_step):
       update_rollout_weights()
       generate_rollouts()

       actor_group.compute_advantages_and_returns()

       actor_group.run_training()

It consists of four steps:

1. **Model sync between actor and rollout** via ``update_rollout_weights()``:

   .. code:: python

      def update_rollout_weights():
          rollout_futures = rollout_group.sync_model_from_actor()
          actor_futures = actor_group.sync_model_to_rollout()
          actor_futures.wait()
          rollout_futures.wait()

2. **Fine-grained rollout pipeline** in hybrid mode via ``generate_rollouts()``:

   .. code:: python

      def generate_rollouts(self):
          env_futures = env_group.interact()
          rollout_futures = rollout_group.generate()
          actor_futures = actor_group.recv_rollout_batch()
          env_futures.wait()
          actor_futures.wait()
          rollout_futures.wait()

   Here, the crucial pieces are ``env_group.interact()`` and
   ``rollout_group.generate()``, which connect through two producer–consumer
   queues to implement **fine-grained pipelining** for fast rollout.
   See :doc:`../mode/hybrid` for details.

3. **Advantage/return computation** with
   ``actor_group.compute_advantages_and_returns()`` based on the collected
   rollouts.

4. **Policy update** with
   ``actor_group.run_training()`` using rollouts plus the computed advantages
   and returns.
