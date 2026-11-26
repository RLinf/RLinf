AgileX Piper Setup
==================

.. figure:: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
   :align: center
   :width: 70%

   Piper arm and gripper model. Image: AgileX Robotics, piper_ros.

Set up an AgileX Piper 6-DoF arm over Linux SocketCAN and verify joint and gripper feedback and motion with the RLinf hardware test script.

Overview
--------

This guide covers hardware setup and motion checks on the Linux machine connected to the arm. No GPU or model checkpoint is required; RLinf does not ship a training task for AgileX Piper yet.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      AgileX Piper 6-DoF · AgxGripper

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Joint & TCP feedback · optional cameras

   .. grid-item-card:: Workflow
      :text-align: center

      CAN bring-up · Read-only · Joint & gripper check

   .. grid-item-card:: Hardware
      :text-align: center

      Linux PC · USB-to-CAN (SocketCAN) · no GPU

Hardware and Safety
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Requirement
   * - Arm and gripper
     - AgileX Piper with its power supply; AgxGripper if fitted.
   * - Controller machine
     - Linux computer with a USB-to-CAN adapter in SocketCAN mode and a CAN cable to the arm.

Mount the arm on a stable surface with space for small joint moves, and connect power and CAN wiring according to the manufacturer's instructions before powering it on.

.. warning::

   Connecting enables the motors even with ``--read-only``. Clear the workspace before connecting, and keep hands away during joint and gripper commands. Closing the test script is not an emergency stop and does not cut motor power.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: AgileX Piper

      **1. Install RLinf**

      Install the Piper runtime (includes the pinned ``pyAgxArm`` driver) on the controller machine connected to the arm:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env piper && source .venv/bin/activate

      **2. Bring up the SocketCAN interface**

      Find the adapter with ``ip -brief link`` (normally ``can0``; replace below if yours differs), bring it up at 1 Mbit/s, and confirm five feedback frames arrive with the arm powered on:

      .. code-block:: bash

         sudo ip link set can0 down
         sudo ip link set can0 type can bitrate 1000000
         sudo ip link set can0 up
         ip -details link show can0
         candump -n 5 can0

      The ``ip`` output should show an enabled interface at ``bitrate 1000000``. The SDK does not configure CAN; see the `pyAgxArm documentation <https://github.com/agilexrobotics/pyAgxArm>`__ for adapter setup.

      **3. Check feedback, joints, and gripper**

      Read feedback first without sending position targets or resetting the arm (prints six joint angles in degrees, tool pose in metres plus xyzw quaternion, gripper opening in ``[0, 1]``, and detected firmware profile):

      .. code-block:: bash

         python -m toolkits.realworld_check.test_piper_controller \
           --channel can0 --read-only

      Add ``--gripper-max-width 0.1`` for an AgxGripper with a 0.1 m stroke, or ``--no-gripper`` if no gripper is fitted.

      Next, omit ``--read-only`` to test small movements at 10% speed:

      .. code-block:: bash

         python -m toolkits.realworld_check.test_piper_controller \
           --channel can0 --speed-percent 10

      Enter these commands one at a time, waiting for each move to finish:

      .. code-block:: text

         where
         joint 1 2
         where
         joint 1 -2
         grip 0.5
         where
         quit

      ``joint 1 2`` steps joint 1 by 2 degrees (accepts joints 1–6 within 5 degrees, clipped to travel limits); ``grip 0.5`` requests half stroke; ``where`` reads updated feedback; ``quit``, EOF, or Ctrl+C closes the CAN session while motors hold position.

Results and Troubleshooting
---------------------------

Setup is verified once ``candump -n 5 can0`` prints five frames, ``--read-only`` reports live joint and gripper readings, and ``where`` reflects the commanded 2-degree step and gripper move.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Fix
   * - ``candump -n 5 can0`` blocks with no frames
     - Press Ctrl+C and check arm power, CAN wiring, interface name from ``ip -brief link``, and ``bitrate 1000000``.
   * - Gripper scaling mismatch or no gripper installed
     - Pass ``--gripper-max-width 0.1`` for a 0.1 m AgxGripper stroke, or ``--no-gripper`` if no gripper is fitted.
   * - Motors remain holding torque after ``quit``
     - Expected behavior; use the arm's hardware power or emergency stop switch to cut motor power.

To control Piper programmatically or attach cameras, see :doc:`Robotics Interface <../../concepts/robotics>`; to define a real-world task before training, see :doc:`New Real-World Tasks <../../extending/new_task>`.
