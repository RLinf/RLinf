Installation
============

Evaluation shares the same embodied environment installation flow as training. From the repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model <model> --env <env>
   source .venv/bin/activate

Choose ``<model>`` and ``<env>`` to match your target benchmark:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Benchmark
     - Recommended ``--model``
     - Recommended ``--env``
   * - LIBERO
     - ``openpi`` / ``openvla-oft`` / ``starvla`` / ``dreamzero``
     - ``maniskill_libero`` or ``libero``
   * - RoboTwin
     - ``openvla-oft`` / ``openpi`` / ``lingbotvla``
     - ``robotwin``
   * - BEHAVIOR-1K
     - ``openpi``
     - ``behavior``
   * - ManiSkill OOD
     - ``openvla-oft``
     - ``maniskill_libero``
   * - RealWorld
     - ``openpi`` / ``dreamzero``
     - ``franka``
   * - PolaRiS
     - ``openpi``
     - ``polaris``

For more installation options, see :doc:`../../start/installation`.

Benchmark-Specific Environment Variables
----------------------------------------

Configure the following when your benchmark requires them (LIBERO is handled automatically by ``install.sh``; see below):

**LIBERO**

With ``--env libero``, ``install.sh`` clones LIBERO into ``.venv/libero`` by default and appends it to ``PYTHONPATH`` in the activate script. To use an existing checkout instead, set ``LIBERO_PATH`` before installation:

.. code-block:: bash

   export LIBERO_PATH=/path/to/LIBERO

**RoboTwin**

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export ROBOT_PLATFORM=ALOHA

**BEHAVIOR-1K**

Set ``OMNIGIBSON_DATA_PATH`` and related OmniGibson paths. See :doc:`../../examples/embodied/behavior`.

**DreamZero**

.. code-block:: bash

   export DREAMZERO_PATH=/path/to/DreamZero

**PolaRiS**

.. code-block:: bash

   export POLARIS_DATA_PATH=/path/to/dataset/PolaRiS-Hub

``run_eval.sh`` reads these variables automatically and configures ``PYTHONPATH``.

Next Steps
----------

After installation, follow :doc:`quick_tour` for your first evaluation, or see the benchmark guide: :doc:`../guides/index`.
