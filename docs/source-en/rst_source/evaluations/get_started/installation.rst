Installation
============

Evaluation shares the same embodied environment installer as training. Pick a benchmark tab below, or pull a prebuilt image from `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__. In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

Setup by Benchmark
------------------

.. tab-set::

   .. tab-item:: LIBERO (OpenPI)
      :sync: eval-libero

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

   .. tab-item:: ManiSkill OOD (OpenVLA-OFT)
      :sync: eval-maniskill

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

   .. tab-item:: RoboTwin (OpenPI)
      :sync: eval-robotwin

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate

   .. tab-item:: BEHAVIOR-1K (OpenPI)
      :sync: eval-behavior

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

   .. tab-item:: RealWorld (Franka)
      :sync: eval-realworld

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

Supported Benchmark Combinations
--------------------------------

For other model-and-benchmark pairs, pass ``--model`` and ``--env`` from the table below (see :doc:`../../start/installation` for all options):

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Benchmark
     - Recommended ``--model``
     - Recommended ``--env``
   * - LIBERO
     - ``openpi`` / ``openvla-oft`` / ``starvla`` / ``dreamzero`` / ``molmoact2``
     - ``maniskill_libero`` or ``libero``
   * - LIBERO / LIBERO-Plus
     - ``fastwam``
     - ``libero`` or ``liberoplus``
   * - RoboTwin
     - ``openvla-oft`` / ``openpi`` / ``lingbotvla`` / ``lingbotvlav2``
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
   * - RoboLab
     - ``openpi``
     - ``robolab``
   * - SIMPLE
     - ``psi0``
     - ``simple``

.. dropdown:: Benchmark-specific environment variables (optional)

   ``run_eval.sh`` reads these variables automatically and configures ``PYTHONPATH`` when needed:

   .. list-table::
      :header-rows: 1
      :widths: 22 48 30

      * - Benchmark
        - Environment Variable(s)
        - Notes
      * - RoboTwin
        - ``ROBOTWIN_PATH=/path/to/RoboTwin``, ``ROBOT_PLATFORM=ALOHA``
        - Path to cloned RoboTwin checkout
      * - BEHAVIOR-1K
        - ``OMNIGIBSON_DATA_PATH``
        - See :doc:`../../examples/embodied/behavior`
      * - DreamZero
        - ``DREAMZERO_PATH=/path/to/DreamZero``
        - Path to DreamZero checkout
      * - PolaRiS
        - ``POLARIS_DATA_PATH=/path/to/PolaRiS-Hub``
        - Path to downloaded PolaRiS-Hub assets

Next Steps
----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 5-Minute Quick Tour
      :link: quick_tour
      :link-type: doc

      Evaluate OpenPI π₀.₅ on LIBERO Spatial to verify your setup.

   .. grid-item-card:: Benchmark Guides
      :link: ../guides/index
      :link-type: doc

      Open step-by-step evaluation guides for every supported benchmark.
