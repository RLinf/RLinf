Evaluation on RoboUniverse-10K
==============================

.. figure:: https://raw.githubusercontent.com/Einsia/RoboUniverse-10K/main/assets/task-taxonomy.png
   :align: center
   :width: 90%
   :alt: RoboUniverse-10K task taxonomy

   RoboUniverse-10K task families across Franka Panda and AgileX dual-Piper environments (`Einsia/RoboUniverse-10K <https://github.com/Einsia/RoboUniverse-10K>`__).

Evaluate π₀.₅ / OpenPI policies across RoboUniverse-10K's 10,807 Isaac Sim manipulation tasks.

Overview
--------

Evaluate embodied policies from two RGB views, joint states, and a natural-language task instruction.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: Algorithms
      :text-align: center

      Evaluation only

   .. grid-item-card:: Tasks
      :text-align: center

      10,307 training tasks · 500 ``heldout500`` tasks

   .. grid-item-card:: Hardware
      :text-align: center

      Linux x86_64 · NVIDIA RTX GPU (Isaac Sim 5.1.0)

| **You'll do:** install → download the dataset and checkpoint → configure EULA and paths → launch → check ``eval/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Review the `NVIDIA Isaac Sim License <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html>`_.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Robot Family
     - Tasks
     - Config Base
     - Action Space
   * - Franka Panda
     - 4,132 total (200 in ``heldout500``)
     - ``env/robouniverse_franka``
     - 9-DoF joint position targets (7 arm + 2 fingers)
   * - AgileX dual-Piper
     - 6,675 total (300 in ``heldout500``)
     - ``env/robouniverse_agilex``
     - 16-DoF joint position targets (2 × 6 arm + 2 × 2 gripper)

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Specification
   * - Observation
     - External and wrist 640×480 RGB cameras, measured joint positions (9D Franka or 16D AgileX), and optional joint velocities.
   * - Action
     - Joint-position targets at 20 Hz on Franka (0.05 s step) and 10 Hz on AgileX (0.1 s step, per each task's ``env_spec.json``), clamped to each task's joint limits.
   * - Reward
     - Sparse terminal reward of 1 on verified task success and 0 on failure; simulator or verifier infrastructure errors are logged separately in ``eval/infra_error``.
   * - Prompt
     - Natural-language task instruction packaged with each task.

Installation
------------

Run RLinf on a Linux x86_64 host or inside NVIDIA's ``nvcr.io/nvidia/isaac-sim:5.1.0`` container so the simulator subprocess can invoke ``/isaac-sim/python.sh`` directly.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install RLinf's OpenPI environment for RoboUniverse-10K:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robouniverse
   source .venv/bin/activate

Download the Dataset
--------------------

Download the catalog and ``heldout500`` split index from Hugging Face into ``/path/to/RoboUniverse-10K``:

.. code-block:: bash

   hf download Einsia/RoboUniverse-10K catalog.csv splits/heldout500.csv \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

Download either a single smoke-test task archive or the full ``tasks/`` tree into the same directory:

.. code-block:: bash

   # Single Franka smoke-test task
   hf download Einsia/RoboUniverse-10K tasks/franka/000247_rotate_twist.zip \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

   # Or download all task archives listed in catalog.csv
   hf download Einsia/RoboUniverse-10K --include "tasks/**" \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

Download the Model
------------------

Prepare a PyTorch OpenPI π₀.₅ checkpoint fine-tuned for the target robot's joint space along with its ``norm_stats.json`` assets:

- **Franka Panda (9-DoF):** place the checkpoint at ``/path/to/openpi_robouniverse_franka_checkpoint`` (loaded with ``openpi.config_name: pi05_robouniverse_franka``).
- **AgileX dual-Piper (16-DoF):** place the checkpoint at ``/path/to/openpi_robouniverse_agilex_checkpoint`` (loaded with ``openpi.config_name: pi05_robouniverse_agilex``).

See :doc:`OpenPI SFT </rst_source/examples/embodied/sft_openpi>` for training or converting OpenPI checkpoints.

Run It
------

Edit ``evaluations/robouniverse/robouniverse_franka_eval_openpi.yaml`` to set your dataset path, checkpoint directory, Isaac Sim Python path, and NVIDIA EULA acceptance:

.. code-block:: yaml

   env:
     eval:
       init_params:
         data_root: /path/to/RoboUniverse-10K
         sim_python: /isaac-sim/python.sh
         accept_isaac_eula: true

   rollout:
     model:
       model_path: /path/to/openpi_robouniverse_franka_checkpoint

Launch Franka evaluation:

.. code-block:: bash

   bash evaluations/run_eval.sh robouniverse robouniverse_franka_eval_openpi

To run a single smoke-test task, pass ``env.eval.init_params.cases='["000247_rotate_twist"]'`` as an override. To evaluate AgileX dual-Piper tasks, update ``evaluations/robouniverse/robouniverse_agilex_eval_openpi.yaml`` with your dataset and checkpoint paths and run:

.. code-block:: bash

   bash evaluations/run_eval.sh robouniverse robouniverse_agilex_eval_openpi

.. warning::

   Allocate at least 8 CPU cores and 24 GiB RAM per simulator worker. Fewer cores can cause cold Isaac Sim startup to time out before the first step.

Visualization and Results
-------------------------

Inspect evaluation metrics in the terminal and TensorBoard:

- ``eval/success_once``: fraction of evaluated episodes whose task verifier returned ``success``.
- ``eval/return``: mean episodic return (1.0 on verified success, 0.0 otherwise).
- ``eval/infra_error``: fraction of episodes that terminated early because the simulator or task verifier raised an error rather than failing the task.

See :doc:`Logger </rst_source/guides/logger>` for logging backends and output configuration.
