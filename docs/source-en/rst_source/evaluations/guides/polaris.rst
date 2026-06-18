PolaRiS Evaluation
==================

PolaRiS is a tabletop manipulation simulation platform with DROID-style tasks such as TapeIntoContainer and MoveLatteCup. RLinf supports evaluating OpenPI policies on PolaRiS.

Related training doc: :doc:`../../examples/embodied/polaris`

Environment Setup
-----------------

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate
   export POLARIS_DATA_PATH=/path/to/dataset/PolaRiS-Hub

Example Configs
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config file
     - Task
     - Model
   * - ``polaris_tapeintocontainer_openpi_pi05_eval.yaml``
     - TapeIntoContainer
     - π₀.₅
   * - ``polaris_movelattecup_openpi_eval.yaml``
     - MoveLatteCup
     - π₀

End-to-End Workflow
-------------------

**Step 1: Download dataset and model**

Follow :doc:`../../examples/embodied/polaris` to download the PolaRiS dataset and OpenPI checkpoints.

**Step 2: Set environment variables**

.. code-block:: bash

   source .venv/bin/activate
   export POLARIS_DATA_PATH=/path/to/dataset/PolaRiS-Hub

**Step 3: Edit the config**

Set ``rollout.model.model_path`` to your local checkpoint.

**Step 4: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh polaris polaris_tapeintocontainer_openpi_pi05_eval

Or:

.. code-block:: bash

   bash evaluations/run_eval.sh polaris polaris_movelattecup_openpi_eval

**Step 5: Check results**

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs.

FAQ
---

- **Dataset path:** ``POLARIS_DATA_PATH`` must point to the PolaRiS-Hub root; ``run_eval.sh`` reads it automatically.
- **Model conversion:** JAX checkpoints must be converted to PyTorch format per the training doc before evaluation.
