Automated Testing
====================

The RLinf automated testing toolkit is located in the ``tests/parity_tests/`` directory. It batch-runs embodied training experiments, detects run status, and performs automatic baseline comparison analysis.
The toolkit supports automatic experiment running, automatic Python environment switching, and log-based experiment completion or crash detection.

Environment and Configuration
------------------------------

Task List Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Define the experiment list in the ``TASKS`` array within ``run_all.sh``. Each line follows the format ``ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE``:

- ``ENV_NAME``: Environment name (e.g., ``maniskill_libero``, ``behavior``, ``isaaclab``, ``metaworld``, ``calvin``)
- ``MODEL_NAME``: Model name (e.g., ``openvla``, ``openvla-oft``, ``openpi``, ``gr00t``, ``mlp``)
- ``YAML_ARG``: Corresponding YAML configuration file name
- ``T_NODES``: Number of nodes required
- ``T_STEPS``: Total training steps
- ``T_SAVE``: Save interval (``-1`` means no checkpoint saving)

Example configuration:

.. code-block:: bash

   TASKS=(
       "maniskill_libero openvla-oft maniskill_ppo_openvlaoft 1 100 -1"
       "maniskill_libero openpi libero_goal_ppo_openpi 1 100 -1"
       "maniskill_libero openpi maniskill_ppo_mlp 1 100 -1"
   )

Python Environment and Model Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the corresponding Python virtual environments and update ``VENV_BASE_DIR`` in ``run_all.sh``:

.. code-block:: bash

   # Install a single model+environment combination
   bash requirements/install.sh embodied --model openvla --env maniskill_libero

   # Update VENV_BASE_DIR to point to your virtual environments directory
   export VENV_BASE_DIR="/path/to/venvs"

.. # Or batch install all environments
.. bash requirements/install_all_envs.sh

Download model weights and asset files, and update paths in the YAML configuration files:

.. code-block:: bash

   # Download model weights
   hf download gen-robot/openvla-7b-rlvla-warmup --local-dir openvla-7b-rlvla-warmup

   # Download ManiSkill assets
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

Running the Tests
-----------------

Start automated testing with the following command:

.. code-block:: bash

   bash ./tests/parity_tests/run_all.sh

``run_all.sh`` accepts a ``--similarity-method`` parameter to specify the similarity metric for baseline comparison (default: ``pearson``). Available options: ``spearman``, ``mse``, ``mae``, ``cosine``, ``dtw``, ``all``:

.. code-block:: bash

   bash ./tests/parity_tests/run_all.sh --similarity-method pearson

.. For multi-node execution, set the ``RANK`` environment variable on each node: ``RANK=0`` (default) for the head node, ``RANK!=0`` for worker nodes. The head node schedules tasks, starts the Ray cluster, and runs training; worker nodes poll the synchronization signal file ``ray_utils/task_sync.txt`` to automatically join the cluster and wait for task completion.

Execution flow: the node runs each task sequentially, checking logs to determine whether the threshold has been reached or the run has crashed. If the threshold is reached, the task is skipped; if all runs crashed, it is marked as failed; otherwise, the matching virtual environment is activated and training begins. After all tasks complete, a final summary is output and automatic log analysis, curve plotting, and baseline comparison are performed.

Output Results
--------------

After all tasks complete, a final summary is output:

.. code-block:: text

   =========================================================
                       FINAL SUMMARY
   =========================================================
   Total tasks:        6
   Success:            4
   Skipped (crashed):  2
   =========================================================

A detailed running summary is also generated, showing the status of each task and their final status.
And comparison analysis is automatically performed, generating a ``success_once`` curve plot in the ``logs/`` directory and calculating similarity metrics against the baseline.