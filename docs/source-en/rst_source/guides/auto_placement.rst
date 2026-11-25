Auto Placement
==============

Generate an optimized ``cluster.component_placement`` layout before launching a large RL run. The autotuner searches collocated, disaggregated, and pipelined GPU splits using measured per-iteration timings and reports the layout with the lowest step time.

Scope and Prerequisites
-----------------------

- ``examples/reasoning/run_placement_autotune.sh`` targets Megatron math-reasoning YAML configs under ``examples/reasoning/config/math/`` (default: ``qwen2.5-1.5b-grpo-megatron``).
- Run the autotuner on the target Ray cluster (or a cluster with the same ``cluster.num_nodes`` and per-node GPU count), because the tool queries the active Ray cluster to validate GPU ranks.
- Collect baseline collocated timings for ``actor``, ``inference``, and ``rollout`` first (see Step 1).

How It Works
------------

The autotuner models the task workflow graph (``rollout -> inference -> actor`` when ``algorithm.recompute_logprobs: True``, or ``rollout -> actor`` otherwise), estimates how each component scales with GPU count subject to its model-parallel and batch-size constraints, and searches collocated and disaggregated GPU partitions to minimize iteration time.

Step 1: Collect Profile Data
----------------------------

Measure the average wall-clock time (in seconds per iteration) of each component in collocated mode (``actor,rollout,reward: all``):

1. Launch your baseline math-reasoning config in collocated mode for 3–5 iterations (after the first warm-up step).
2. Read the per-iteration timings from TensorBoard or training logs (or capture them with :doc:`Profiling <profile>`):

   - ``actor_cost``: time spent in actor training per iteration (seconds).
   - ``inference_cost``: time spent in reference / old-logprob forward passes per iteration (seconds). If ``algorithm.recompute_logprobs`` is ``False``, keep a numeric placeholder such as ``0.0`` because the config loader still reads this field.
   - ``rollout_cost``: time spent in rollout generation per iteration (seconds).

3. Add a top-level ``profile_data`` block to your config YAML:

.. code-block:: yaml

   profile_data:
     actor_cost: 95.7      # Actor training time (seconds per iteration)
     inference_cost: 30.8  # Logprob inference time (seconds per iteration)
     rollout_cost: 59.9    # Rollout generation time (seconds per iteration)

Step 2: Run the Autotuner
-------------------------

Run the autotuner script from the repository root with the config name (without ``.yaml``) from ``examples/reasoning/config/math/``:

.. code-block:: bash

   bash examples/reasoning/run_placement_autotune.sh qwen2.5-1.5b-grpo-megatron

For a config in another directory, call ``toolkits/auto_placement/auto_placement_worker.py`` directly with ``--config-path`` and ``--config-name``.

Expected output:

.. code-block:: text

   ==================================================
   Best placement for this task is:
   rollout, actor : all

When a disaggregated or hybrid layout is faster, the output prints explicit GPU rank ranges per component (for example, ``rollout: 0-7`` and ``actor: 8-15``).

Step 3: Apply the Placement
---------------------------

Copy the recommended ``cluster.component_placement`` block into your training YAML and launch training as usual. For placement syntax and hybrid layouts, see :doc:`Placement Strategies <../concepts/placement>`.

Troubleshooting
---------------

- **Missing ``profile_data`` key**: ensure ``profile_data.actor_cost``, ``profile_data.inference_cost``, and ``profile_data.rollout_cost`` are all defined as positive numbers (or ``0.0`` for unused ``inference_cost``).
- **No valid placement found**: check that the active Ray cluster matches ``cluster.num_nodes`` and that ``data.rollout_batch_size * algorithm.group_size / algorithm.n_minibatches`` is divisible by at least one valid data-parallel size for your model-parallel configuration.
