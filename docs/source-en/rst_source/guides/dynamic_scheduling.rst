Dynamic Scheduling
==================

Dynamic scheduling adjusts and migrates resources among components (actor / rollout / inference)
in real time during training to improve overall throughput and resource utilization.
It relies on Megatron-LM's online scaling (second-level elasticity) and SGLang/vLLM's migrate capability
to reallocate GPU resources without stopping training.

.. warning::

   **Legacy environment prerequisite:** Dynamic scheduling requires a dedicated **Megatron-LM 0.11** checkout plus the external prebuilt resharding package (``params_resharding_release``), which is incompatible with the default ``bash requirements/install.sh agentic`` environment (Megatron-LM 0.18). For a runnable reference config, see ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron-dynamic.yaml`` (launch with ``bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron-dynamic`` after preparing the prerequisites below).

Online Scaling Mechanism
------------------------

Online scaling (also known as elastic training)
adjusts training resources dynamically, completing GPU switching within 1 second.
You can scale the number of GPUs and nodes used for training in real time
based on cluster availability, workload demands, or resource optimization goals.

What is Online Scaling?
~~~~~~~~~~~~~~~~~~~~~~~

Online scaling refers to the ability to **scale up** (add more resources) or **scale down** (remove resources)
during training while maintaining training continuity and model state consistency.

In the context of RL training with Megatron-LM, this involves:

- **Scaling up**: Adding nodes/GPUs to increase training throughput
- **Scaling down**: Releasing nodes/GPUs to free up resources for other tasks
- **Parallel strategy adjustment**: Dynamically changing Megatron's parallel strategies (TP/PP/DP/CP)

The system automatically handles:

- Model parameter redistribution across the new parallel configuration
- Optimizer state migration
- Communication group reconstruction
- Training state synchronization

Why is Online Scaling Important?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using RLinf's disaggregated mode with fine-grained pipelining,
the rollout and inference stages are completed before the actor stage finishes.
At this point, the resources used for rollout and inference can be reallocated to the actor stage **within seconds**,
accelerating actor training and improving overall system performance.

Benefits
~~~~~~~~

**Performance benefits:**

- **Higher throughput**: temporarily accelerate bottleneck components to speed up training
- **Better utilization**: promptly reassign idle resources to effective computation
- **Shorter end-to-end time**: often yields 20–50% total time reduction (task/cluster dependent)

**Operational benefits:**

- **Zero training interruption**: scaling and migration occur without stopping training
- **Consistent training progress**: maintains convergence and model state consistency throughout scaling

Dynamic Scheduling in Practice
------------------------------

Dynamic scheduling adjusts GPU resources across components during training based on stage-specific
bottlenecks and workload changes:

- Scaling up: temporarily add GPUs when a component becomes the bottleneck
- Scaling down: reclaim GPUs when a component is temporarily idle

How to Use Dynamic Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prerequisites
^^^^^^^^^^^^^

.. warning::

   Online scaling requires a dedicated **Megatron-LM 0.11** environment rather than the default ``install.sh agentic`` virtual environment (which installs Megatron-LM 0.18).

1) Prepare Megatron-LM online scaling dependency (prebuilt):

.. code-block:: bash

    WORKSPACE=YourWorkspace
    cd $WORKSPACE
    git clone https://github.com/i-Taozi/params_resharding_release.git
    export PYTHONPATH=$PYTHONPATH:$WORKSPACE/params_resharding_release

This repository provides the compiled artifacts for Megatron-LM online scaling. The source code will be released in future.

2) Megatron must be version 0.11. Fetch 0.11 separately:

.. code-block:: bash

    WORKSPACE=YourWorkspace
    cd $WORKSPACE
    git clone -b core_r0.11.0 https://github.com/NVIDIA/Megatron-LM.git
    export PYTHONPATH=$PYTHONPATH:$WORKSPACE/Megatron-LM

.. important::
    If you use ``torch >= 2.6.0``, Megatron-LM 0.11 may raise errors due to the default ``torch.load`` behavior.
    Clone the modified Megatron-LM 0.11 branch instead:

    .. code-block:: bash

        git clone -b core_v0.11.0_rlinf https://github.com/RLinf/Megatron-LM.git

Configuration Example
^^^^^^^^^^^^^^^^^^^^^

Disaggregated Pipeline configuration:

.. code-block:: yaml

    cluster:
      num_nodes: 1
      component_placement:
        rollout: 0-3
        inference: 4-5
        actor: 6-7

Based on the disaggregated pipeline configuration, change the component order and enable the auto scheduler.
Make sure the component order is ``actor -> rollout -> inference``, otherwise the actor can't scale up.

.. code-block:: yaml

    cluster:
      num_nodes: 1
      auto_scheduler: True
      use_pre_process_policy: True
      use_wait_before_last_iter_policy: False
      component_placement:
        actor: 0-1
        rollout: 2-5
        inference: 6-7

Scheduling Logic
~~~~~~~~~~~~~~~~

When dynamic scheduling is enabled, the runtime scheduler monitors component progress and queues
and decides whether to adjust resources. Typical actions include:

- When the rollout backlog is small: trigger rollout migration, release part of rollout resources, and expand the actor
- When rollout or inference finishes: release resources to expand the actor

Optional Policies
~~~~~~~~~~~~~~~~~

- ``use_pre_process_policy``

  1. In the early phase of each iteration, temporarily transfer actor resources to rollout
  2. When the scheduler detects an appropriate time, reassign part of rollout resources back to the actor
  3. Effective for long sequence length (expensive rollout) scenarios to maximize pipeline efficiency

- ``use_wait_before_last_iter_policy``

  1. Before the last actor iter in an iteration, the actor waits for rollout and inference to finish
  2. Then the actor takes all resources for an expanded final step
  3. Thanks to pipelining, rollout/inference typically finish earlier; with proper scheduling, the actor can fully utilize the entire cluster for the last iter
