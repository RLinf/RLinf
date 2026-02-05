Data Collection (RealWorld)
=================================

This document provides a guide for performing real-world data collection in RLinf. Real-world data collection is typically used for RLPD (Reinforcement Learning from Prior Data) experiments, where human intervention (e.g., via SpaceMouse) is used to acquire initial successful trajectories, providing prior data for subsequent training.

Overview
--------

The real-world data collection system is launched via the ``examples/embodiment/collect_data.sh`` script. Unlike the large-scale parallel collection used in simulation, real-world collection typically runs on a single control node and supports real-time human intervention through a SpaceMouse. A core feature of this system is its support for setting a **target number of successes**; the script will continue running until the specified number of successful episodes has been collected.

Configuration
-------------

Real-world collection is configured via ``examples/embodiment/config/realworld_collect_data.yaml``:

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Core Configuration for Real-World Data Collection
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``runner.num_data_episodes``
     - ``20``
     - The number of **successful** trajectories to collect. The script stops once this number is reached.
   * - ``cluster.node_groups.hardware.configs.robot_ip``
     - N/A
     - The IP address of the Franka robot.
   * - ``env.eval.use_spacemouse``
     - ``True``
     - Whether to enable SpaceMouse for manual intervention.
   * - ``env.eval.override_cfg.target_ee_pose``
     - N/A
     - The target end-effector pose for the task [x, y, z, rx, ry, rz].
   * - ``env.eval.override_cfg.success_hold_steps``
     - ``1``
     - The number of consecutive steps the robot must maintain the target pose to be judged as a successful trajectory.

Data Format
-----------

File Naming
~~~~~~~~~~~

A single data collection session generates one pickle (``.pkl``) file. All episodes collected during the session are stored sequentially within this file. The file is located in the ``logs`` directory, typically named: ``data.pkl``.

Data Structure
~~~~~~~~~~~~~~

The real-world collection pickle file contains a list. Each element in the list represents data for a single time step (Step). The structure is as follows:

.. code-block:: python

   {
       "transitions": {
           "obs": {
               "states":        # Robot state (shape=[19], includes pose, force/torque, etc.)
               "main_images":   # Wrist camera image (shape=[128, 128, 3])
           },
           "next_obs": {
               "states":        # Robot state at the next step
               "main_images":   # Image observation at the next step
           }
       },
       "action":                # Action taken at the current step (shape=[6], matches control mode)
       "rewards":               # Reward value (shape=[1], success is usually 1.0)
       "dones":                 # Done flag (shape=[1], bool)
       "terminations":          # Termination flag (shape=[1], bool)
       "truncations":           # Truncation flag/max steps reached (shape=[1], bool)
   }

Usage
-----

1. Environment Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~

On the controller node, you need to source the virtual python environment and the relevant ROS/Robot control setup scripts:

.. code-block:: bash

   source <path_to_your_venv>/bin/activate
   source <your_catkin_ws>/devel/setup.bash

2. Configure Robot and Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify ``examples/embodiment/config/realworld_collect_data.yaml``. Ensure the ``robot_ip`` is correct and set the ``target_ee_pose``. You can acquire the ``target_ee_pose`` using the toolkit: ``toolkits/realworld_check/test_controller.py``.

3. Launch Collection Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the collection script:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh

4. Manual Intervention
~~~~~~~~~~~~~~~~~~~~~~

After the script starts, the robot will reset. Once the camera feed appears on the screen, you can use the SpaceMouse to control the robot and complete the task. When the robot detects a success (entering the target zone and maintaining it for the required steps), the success counter will increment.

**Note:** It is normal for the camera feed to stop updating during the robot reset phase after a success is detected.

5. Access Data
~~~~~~~~~~~~~~

Once the number of successes reaches ``num_data_episodes``, the data will be automatically saved to:
``logs/[running-timestamp]/data.pkl``.

6. Upload Data
~~~~~~~~~~~~~~

After collection is complete, upload ``data.pkl`` to your GPU training server and point to this path in your training configuration to start RLPD training.