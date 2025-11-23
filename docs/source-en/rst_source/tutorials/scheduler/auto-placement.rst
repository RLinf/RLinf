Auto Placement
==============

Auto Placement before RL training
---------------------------------

This tool automatically generates optimal component placement configurations for RL training workflows. It analyzes the computational costs of different components (rollout, inference, training, etc.) and determines the best placement strategy to minimize overall training time.

Overview
~~~~~~~~

The auto placement tool consists of three main components in `toolkits/auto_placement`:

- **scheduler_task.py**: Main scheduler that performs time and space division multiplexing to find optimal placements
- **resource_allocator.py**: Handles resource allocation for different components
- **workflow.py**: Manages workflow graphs and cost calculations

Usage
~~~~~

Step 1: Collect Profile Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before running the auto placement tool, you need to collect profile data for your components. This includes measuring the computation time for each component (rollout, inference, training, etc.) in collocated mode for one iteration.

Add the profile data to your YAML configuration file under the ``profile_data`` section:

.. code-block:: yaml

   profile_data:
     actor_cost: 95.7    # Training component cost (seconds per iteration)
     inference_cost: 30.8  # Inference component cost (seconds per iteration)
     rollout_cost: 59.9    # Rollout component cost (seconds per iteration)

**How to collect profile data:**

1. Run your training with origin cluster in collocated mode for several iterations
2. Use profiling tools to measure the time each component takes per iteration
3. Record the average time per iteration for each component

Step 2: Run Auto Placement
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the provided shell script to run the auto placement tool:

.. code-block:: bash

   cd examples/reasoning
   ./run_placement_autotune.sh [config_name]

Where ``config_name`` is the name of your configuration file.

The output of this script is like:

.. code-block:: text

   Best placement for this task is:

   cluster:
     num_nodes: 1
     component_placement:
       rollout,actor: all

Step 3: Apply the Results
^^^^^^^^^^^^^^^^^^^^^^^^^

The tool will output a new configuration with optimized component placement. Copy the ``cluster.component_placement`` section from the output and replace the corresponding section in your original YAML file.

Replace the ``cluster.component_placement`` section in your original configuration file with this optimized placement.

Troubleshooting
~~~~~~~~~~~~~~~

1. **Profile data not provided error**: Ensure your YAML file includes the ``profile_data`` section with all three cost values.

2. **Invalid placement**: Check that the total GPU allocation doesn't exceed your cluster capacity.

