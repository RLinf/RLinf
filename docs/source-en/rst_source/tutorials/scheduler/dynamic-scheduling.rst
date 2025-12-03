Dynamic Scheduling
==================


Dynamic scheduling adjusts and migrates resources among components (actor / rollout / inference)
in real time during training to improve overall throughput and resource utilization.
It relies on Megatron-LM's online scaling (second-level elasticity) and SGLang/vLLM's migrate capability
to reallocate GPU resources without stopping training.

What is Dynamic Scheduling?
---------------------------

Dynamic scheduling adjusts GPU resources across components during training based on stage-specific
bottlenecks and workload changes:

- Scaling up: temporarily add GPUs when a component becomes the bottleneck
- Scaling down: reclaim GPUs when a component is temporarily idle

Benefits and Effects
--------------------

**Performance benefits:**

- Higher throughput: temporarily accelerate bottleneck components to speed up training
- Better utilization: promptly reassign idle resources to effective computation
- Shorter end-to-end time: often yields 20–50% total time reduction (task/cluster dependent)

**Operational characteristics:**

- No training interruption: scaling/migration occurs without stopping training
- Consistency preserved: training/model state remains consistent during scaling

How to Use Dynamic Scheduling
-----------------------------

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Prepare Megatron-LM online scaling dependency (prebuilt):

.. code-block:: bash

    WORKSPACE=YourWorkspace
    cd $WORKSPACE
    git clone git@github.com:i-Taozi/params_resharding_release.git
    export PYTHONPATH=$PYTHONPATH:$WORKSPACE/params_resharding_release

This repository provides the compiled artifacts for Megatron-LM online scaling. The source code will be released in future.

2) Megatron must be version 0.11. If your environment is not 0.11, fetch 0.11 separately:

.. code-block:: bash

    WORKSPACE=YourWorkspace
    cd $WORKSPACE
    git clone -b core_r0.11.0 git@github.com:NVIDIA/Megatron-LM.git
    export PYTHONPATH=$PYTHONPATH:$WORKSPACE/Megatron-LM

.. important::
    If you use `torch >= 2.6.0`, Megatron-LM 0.11 may raise errors due to the default `torch.load` behavior.
    You can clone a modified Megatron-LM 0.11 version from 

    .. code-block:: bash

        git clone -b core_v0.11.0_rlinf git@github.com:RLinf/Megatron-LM.git

Configuration Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Disaggregated Pipeline configuration:

.. code-block:: yaml

    cluster:
      num_nodes: 1
      component_placement:
        rollout: 0-3
        inference: 4-5
        actor: 6-7

Based on the disaggregated pipeline configuration, change the component order and enable the auto scheduler.
Make sure the component order is `actor -> rollout -> inference`, otherwise the actor can't scale up.

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
----------------

When dynamic scheduling is enabled, the runtime scheduler monitors component progress and queues
and decides whether to adjust resources. Typical actions include:

- When the rollout backlog is small: trigger rollout migration, release part of rollout resources, and expand the actor
- When rollout or inference finishes: release resources to expand the actor

Optional Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `use_pre_process_policy`

  1. In the early phase of each iteration, temporarily transfer actor resources to rollout
  2. When the scheduler detects an appropriate time, reassign part of rollout resources back to the actor
  3. Effective for long sequence length (expensive rollout) scenarios to maximize pipeline efficiency

- `use_wait_before_last_iter_policy`

  1. Before the last actor iter in an iteration, the actor waits for rollout and inference to finish
  2. Then the actor takes all resources for an expanded final step
  3. Thanks to pipelining, rollout/inference typically finish earlier; with proper scheduling, the actor can fully utilize the entire cluster for the last iter 