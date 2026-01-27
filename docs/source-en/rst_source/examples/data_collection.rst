Data Collection (Sim)
======================

This document provides a guide to using the Data Collection feature in RLinf.
The ``DataCollectorWrapper`` enables transparent collection of environment interaction data
during RL training, which is useful for offline learning, reward model training, or behavior analysis.

Overview
--------

The Data Collection system automatically records all environment interactions (observations,
actions, rewards, terminations) and saves them as pickle files. Each parallel environment's
episodes are saved as separate files, with configurable sampling rates for successful and
failed episodes.

Configuration
-------------

Enable data collection by adding the ``data_collection`` section under ``env`` in your YAML config:

.. code-block:: yaml

   env:
     group_name: "EnvGroup"
     enable_offload: False

     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       mode: "all"
       sample_rate_success: 1.0
       sample_rate_fail: 0.1

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Data Collection Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``False``
     - Whether data collection is enabled.
   * - ``save_dir``
     - N/A
     - Directory where episode data will be saved.
   * - ``mode``
     - ``"all"``
     - Collection mode: ``"train"``, ``"eval"``, or ``"all"``.
   * - ``sample_rate_success``
     - ``1.0``
     - Probability of saving successful episodes (0.0-1.0).
   * - ``sample_rate_fail``
     - ``0.1``
     - Probability of saving failed episodes (0.0-1.0).

FullStateWrapper
~~~~~~~~~~~~~~~~

In ``obs_mode: "rgb"`` mode, ManiSkill returns a 29-dimensional partial state (without object pose).
To use MLP policy while collecting image data, enable ``use_full_state`` to replace with full 42-dim state.

.. list-table:: FullStateWrapper Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``use_full_state``
     - ``False``
     - When enabled, replaces partial states with full 42-dim states in rgb mode

Data Format
-----------

Episode File Naming
~~~~~~~~~~~~~~~~~~~

Each episode is saved as a pickle (``.pkl``) file:

.. code-block:: text

   rank_{worker_rank}_env_{env_idx}_episode_{episode_id}_{success|fail}.pkl

Example: ``rank_0_env_5_episode_42_success.pkl``

Episode Data Structure
~~~~~~~~~~~~~~~~~~~~~~

The pickle file contains a dictionary:

.. code-block:: python

   {
       "mode": str,           # "train", "eval", or "all"
       "rank": int,           # Worker rank
       "env_idx": int,        # Environment index within the worker
       "episode_id": int,     # Episode number for this env
       "success": bool,       # Whether the episode was successful
       "observations": list,  # List of observations (len = num_steps + 1)
       "actions": list,       # List of actions (len = num_steps)
       "rewards": list,       # List of rewards (len = num_steps)
       "terminated": list,    # List of termination flags (len = num_steps)
       "truncated": list,     # List of truncation flags (len = num_steps)
       "infos": list,         # List of info dicts (len = num_steps)
   }

Usage
-----

1. Add data collection configuration to your YAML:

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       mode: "all"
       sample_rate_success: 1.0
       sample_rate_fail: 0.1

2. Run training as normal:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

3. Collected data will be saved in the specified ``save_dir``.

Loading Collected Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pickle

   with open("collected_data/rank_0_env_5_episode_42_success.pkl", "rb") as f:
       episode = pickle.load(f)

   print(f"Episode ID: {episode['episode_id']}")
   print(f"Success: {episode['success']}")
   print(f"Number of steps: {len(episode['actions'])}")

Example Configuration
---------------------

.. code-block:: yaml

   defaults:
     - env/maniskill_pick_cube@env.train
     - env/maniskill_pick_cube@env.eval
     - model/mlp_policy@actor.model
     - training_backend/fsdp@actor.fsdp_config
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-0
       env: 0-0
       rollout: 0-0

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "maniskill_ppo_mlp_collect"
       logger_backends: ["tensorboard"]
     max_epochs: 1000

   env:
     group_name: "EnvGroup"
     enable_offload: False
     use_full_state: True  # Use full 42-dim state in rgb mode

     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       mode: "all"
       sample_rate_success: 1.0
       sample_rate_fail: 0.1

     train:
       total_num_envs: 256
       init_params:
         obs_mode: "rgb"
     eval:
       total_num_envs: 32
       init_params:
         obs_mode: "rgb"

   actor:
     model:
       obs_dim: 42  # Full state dimension when use_full_state is enabled

Best Practices
--------------

1. **Disk Space**: RGB images consume significant storage. Adjust ``sample_rate_fail`` to reduce data volume.

2. **Balanced Datasets**: Use high ``sample_rate_success`` (1.0) and low ``sample_rate_fail`` (0.01-0.1) to focus on successful trajectories.

3. **Environment Count**: Reduce ``total_num_envs`` when collecting high-resolution images to manage storage.

4. **MLP + Image Collection**: Use ``obs_mode: "rgb"`` + ``use_full_state: True`` + ``obs_dim: 42`` to train MLP policy while collecting images for Reward Model training.
