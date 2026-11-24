RL with RoboUniverse-10K
========================

.. figure:: https://raw.githubusercontent.com/Einsia/RoboUniverse-10K/main/assets/task-taxonomy.png
   :align: center
   :width: 90%
   :alt: RoboUniverse-10K task taxonomy

   RoboUniverse-10K task families across Franka Panda and AgileX dual-Piper environments (`Einsia/RoboUniverse-10K <https://github.com/Einsia/RoboUniverse-10K>`__).

Run embodied evaluation and RL training with π₀.₅ / OpenPI across RoboUniverse-10K's 10,807 Isaac Sim manipulation tasks.

Overview
--------

Evaluate or train embodied policies from two RGB views, joint states, and a natural-language task instruction.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: Algorithms
      :text-align: center

      Evaluation, PPO, GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      10,307 training tasks · 500 ``heldout500`` tasks

   .. grid-item-card:: Hardware
      :text-align: center

      Linux x86_64 · NVIDIA RTX GPU (Isaac Sim 5.1.0)

| **You'll do:** install → download the dataset → configure EULA and paths → launch → check ``eval/success_once``.
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
     - Joint-position targets at 20 Hz, clamped to each task's joint limits.
   * - Reward
     - Sparse terminal reward of 1 on verified task success and 0 on failure; simulator infrastructure errors are logged separately in ``infra_error``.
   * - Prompt
     - Natural-language task instruction packaged with each task.

Installation
------------

.. include:: _setup_common.rst

Install the RLinf environment with OpenPI and RoboUniverse support:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robouniverse
   source .venv/bin/activate

Prepare an Isaac Sim 5.1.0 Python interpreter (for example ``/isaac-sim/python.sh`` inside ``nvcr.io/nvidia/isaac-sim:5.1.0``).

Download the Dataset
--------------------

Download the catalog and ``heldout500`` split file from Hugging Face:

.. code-block:: bash

   hf download Einsia/RoboUniverse-10K catalog.csv splits/heldout500.csv \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/robouniverse-10k

Download the task ZIPs listed in ``catalog.csv`` into the same directory.

Run It
------

Edit ``examples/embodiment/config/robouniverse_franka_eval_openpi.yaml`` to set your dataset path, model checkpoint, simulator Python path, and NVIDIA EULA acceptance:

.. code-block:: yaml

   env:
     eval:
       init_params:
         data_root: /path/to/robouniverse-10k
         sim_python: /isaac-sim/python.sh
         accept_isaac_eula: true

   rollout:
     model:
       model_path: /path/to/model

Start evaluation:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robouniverse_franka_eval_openpi

To run a single smoke-test task, set ``env.eval.init_params.cases: ["000247_rotate_twist"]``. To run AgileX dual-Piper tasks, set the env config to ``env/robouniverse_agilex`` and ``rollout.model.action_dim: 16``.

.. warning::

   Allocate at least 8 CPU cores and 24 GiB RAM per simulator worker. Fewer cores can cause cold Isaac Sim startup to time out before the first step.

Visualization and Results
-------------------------

Check ``eval/success_once``, ``eval/return``, and ``eval/infra_error`` in TensorBoard. See :doc:`Training Metrics </rst_source/reference/metrics>` for logging configuration and metric definitions.
