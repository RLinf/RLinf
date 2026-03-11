Auto Placement
==============

Auto Placement before RL training
---------------------------------

This tool automatically generates optimal component placement configurations for RL training workflows. It analyzes the computational costs of different components (rollout, inference, training, etc.) and determines the best placement strategy to minimize overall training time.

Overview
~~~~~~~~

The auto placement tool consists of three main components in `toolkits/auto_placement`:

- **auto_placement_worker.py**: Main scheduler that performs time and space division multiplexing to find optimal placements
- **placement.py**: Handles resource allocation for different components
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

For embodiment tasks, you need to provide the target number of parallel environments and collect profile data for different parallel environment configurations.

.. code-block:: yaml

   
   data:
      env_num: 16             # Target number of parallel environments
   profile_data:
      actor_cost: 70.3        # Training component cost under target parallel environment configuration (seconds per iteration)
      env_profile_data:       # Key-value pairs of env cost under different parallel environment configurations: number of parallel environments per instance (GPU) and corresponding cost (seconds per iteration)
         4: 25.8              
         8: 30.3
         16: 36.5
      rollout_profile_data:   # Key-value pairs of rollout cost under different parallel environment configurations: number of parallel environments per instance (GPU) and corresponding cost (seconds per iteration)
         4: 26.0
         8: 30.7
         16: 37.2

**How to collect profile data:**

1. Run your training with origin cluster in collocated mode for several iterations
2. Use profiling tools to measure the time each component takes per iteration
3. Record the average time per iteration for each component

For embodiment tasks, use  the provided shell script to collect profile data with auto profiling tool:

.. code-block:: bash

   cd toolkits/auto_placement
   ./auto_profile.sh [your_config_name]

Where ``your_config_name`` is the name of your configuration file.

The script will output the cost of each component in the above format and append them to a new YAML file ``your_config_name_profiled.yaml`` that copies your original configuration.

Step 2: Run Auto Placement
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the provided shell script to run the auto placement tool:

.. code-block:: bash

   cd examples/reasoning
   ./run_placement_autotune.sh [your_config_name]

Where ``your_config_name`` is the name of your configuration file.

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

