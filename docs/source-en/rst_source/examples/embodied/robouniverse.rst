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

      10,807 tasks (500 in ``heldout500``)

   .. grid-item-card:: Hardware
      :text-align: center

      Linux x86_64 · NVIDIA RTX GPU (Isaac Sim 5.1.0)

| **You'll do:** install → download the dataset → prepare a checkpoint → configure EULA and paths → launch → check ``eval/success_once``.
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
     - Sparse terminal reward of 1 on verified task success and 0 on failure.
   * - Prompt
     - Natural-language task instruction packaged with each task.

Installation
------------

Run RLinf on a Linux x86_64 host with Isaac Sim 5.1.0 installed, or inside NVIDIA's ``nvcr.io/nvidia/isaac-sim:5.1.0`` container.

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

Download either the first Franka ``heldout500`` task archive for a smoke test or the full ``tasks/`` tree into the same directory:

.. code-block:: bash

   # First Franka heldout500 task (smoke test)
   hf download Einsia/RoboUniverse-10K tasks/franka/000/000095_rotate_twist.zip \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

   # Or download all task archives listed in catalog.csv
   hf download Einsia/RoboUniverse-10K --include "tasks/**" \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

Prepare the Model
-----------------

Supply a PyTorch OpenPI π₀.₅ checkpoint fine-tuned for the target robot's joint space along with its normalization statistics:

- **Franka Panda (9-DoF):** place the checkpoint at ``/path/to/openpi_robouniverse_franka_checkpoint`` with ``norm_stats.json`` at ``<model_path>/robouniverse_franka/norm_stats.json`` (loaded by ``openpi.config_name: pi05_robouniverse_franka``).
- **AgileX dual-Piper (16-DoF):** place the checkpoint at ``/path/to/openpi_robouniverse_agilex_checkpoint`` with ``norm_stats.json`` at ``<model_path>/robouniverse_agilex/norm_stats.json`` (loaded by ``openpi.config_name: pi05_robouniverse_agilex``).

If ``norm_stats.json`` lives elsewhere, pass ``rollout.model.openpi_data.norm_stats_path=/path/to/norm_stats.json``. For the general OpenPI SFT and checkpoint-conversion workflow, see :doc:`OpenPI SFT </rst_source/examples/embodied/sft_openpi>`.

Run It
------

Edit ``evaluations/robouniverse/robouniverse_franka_eval_openpi.yaml`` to set your dataset path, checkpoint directory, Isaac Sim Python path (``/isaac-sim/python.sh`` inside the container, or your host Isaac Sim ``python.sh``), and NVIDIA EULA acceptance:

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

By default, each config evaluates 1 episode from ``heldout500`` (``rollout_epoch: 1``, ``total_num_envs: 1``). Override ``env.eval.rollout_epoch`` or ``env.eval.total_num_envs`` to run more episodes, or pass an explicit case list:

.. code-block:: bash

   bash evaluations/run_eval.sh robouniverse robouniverse_franka_eval_openpi \
     'env.eval.init_params.cases=["000095_rotate_twist"]'

To evaluate AgileX dual-Piper tasks, update ``data_root``, ``sim_python``, ``accept_isaac_eula: true``, and ``model_path`` in ``evaluations/robouniverse/robouniverse_agilex_eval_openpi.yaml`` and run:

.. code-block:: bash

   bash evaluations/run_eval.sh robouniverse robouniverse_agilex_eval_openpi

.. warning::

   Allocate at least 8 CPU cores and 24 GiB RAM per simulator worker. Fewer cores can cause cold Isaac Sim startup to time out before the first step.

Visualization and Results
-------------------------

Inspect evaluation logs in ``logs/<timestamp>-<config_name>/`` and TensorBoard:

- ``eval/success_once`` and ``eval/return`` correspond to ``env/success_once`` and ``env/return`` in :doc:`Training Metrics </rst_source/reference/metrics>`.
- ``eval/infra_error`` reports the fraction of episodes aborted by a simulator or task-verifier error rather than policy failure.

See :doc:`Training Visualisation </rst_source/guides/logger>` for logging backends and output configuration.
