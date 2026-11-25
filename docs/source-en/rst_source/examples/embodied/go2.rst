Go2 High-Level Control
======================

Read Go2 EDU state and check six directions through the Unitree controller. The Gymnasium task ``Go2DriveEnv-v1`` reads state by default. A direction check enables ``SportClient`` velocity commands for one run, and the operator confirms the motion and stop.

Overview
--------

.. grid:: 2 4 4 4

   .. grid-item-card:: Models

      Manual velocity commands.

   .. grid-item-card:: Algorithms

      No training recipe.

   .. grid-item-card:: Tasks

      State reception and six directions.

   .. grid-item-card:: Hardware

      Go2 EDU and a Linux controller.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Check
     - Result
   * - Read-only
     - State samples without motion commands.
   * - One direction
     - Forward, backward, left, right, turn-left, or turn-right, followed by a stop request.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Contract
   * - Observation
     - Orientation, angular velocity, twelve joint positions and velocities, battery voltage, and four raw foot-force readings.
   * - Action
     - Body velocity in m/s, m/s, rad/s, bounded by absolute values of 0.3, 0.3, and 0.5.
   * - Reward
     - Zero. This check does not train a policy.

Setup and Run
-------------

**Prerequisites:** Follow the :doc:`installation guide </rst_source/start/installation>` to prepare the Linux controller. Use a dedicated network interface connected to the Go2 EDU.

.. tab-set::

   .. tab-item:: Go2 EDU

      **1. Install the SDK**

      On the Linux controller, follow the `Unitree SDK instructions <https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206>`_ to install native CycloneDDS 0.10.x and export ``CYCLONEDDS_HOME`` in your shell. The Python 3.11 bindings build against this library. From the RLinf repository root, install the SDK in the same shell:

      .. code-block:: bash

         bash requirements/install.sh embodied --env go2_sport
         source .venv/bin/activate

      Configure the network interface as described in the SDK instructions. The controller computer needs no GPU or policy weights.

      **2. Check state and motion**

      Replace ``<interface>`` with the network interface attached to the Go2. Preview a command without opening SDK channels:

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_sport --interface <interface> --direction forward --dry-run

      The plan reports ``hardware_access: false``. Check state reception separately:

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_sport --interface <interface> --readonly

      This writes ten state-validation results to ``outputs/go2_sport/``. To inspect a single checksum-verified joint and IMU sample from ``rt/lowstate``, run the read-only subscriber in a fresh process.

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_lowstate --interface <interface>

      For motion, keep the original controller active, verify standing posture and clear space, and have the remote ready on site. Close other motion programs before running one direction:

      .. code-block:: bash

         python -m toolkits.realworld_check.check_go2_sport --interface <interface> --direction forward

      Enter ``GO`` only after the on-site checks.

      For translation, choose ``forward``, ``backward``, ``left``, or ``right``.

      For rotation, choose ``turn_left`` or ``turn_right``.

      Each invocation checks one direction for one second at 10 Hz. ``--seconds 2`` selects two seconds. ``--output <path>`` selects a fresh report path and refuses to overwrite a file.

      .. warning::

         A stop request or a low SDK velocity estimate does not establish physical rest. Observe the robot directly and use the remote if motion continues. Network loss or process failure can defeat software stop requests.

Visualization and Results
-------------------------

The report records interface errors and the SDK velocity estimate after the stop request. Record your observation of the motion and stop separately. Measure travel distance, turn angle, and stopping distance separately if your application requires them.
