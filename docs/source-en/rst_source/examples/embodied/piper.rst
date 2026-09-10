AgileX Piper Setup and Control
==============================

This doc walks you through setting up an AgileX Piper arm and running the RLinf
hardware test script to read feedback and check joint and gripper movement.

.. figure:: https://raw.githubusercontent.com/agilexrobotics/piper_ros/noetic/asserts/pictures/piper_urdf_zero.png
   :align: center
   :width: 70%

   Piper arm and gripper model. Image: AgileX Robotics, piper_ros.

Overview
--------

The test runs on the Linux machine connected to the arm. It requires no GPU or
model checkpoint. RLinf does not yet provide a supported real-world task or
training workflow for AgileX Piper; this guide covers hardware checks only.

Hardware Setup
--------------

Prepare the arm and controller machine before installing the software.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Requirement
   * - Arm and gripper
     - AgileX Piper with its power supply; AgxGripper if fitted.
   * - Controller machine
     - Linux computer with a USB-to-CAN adapter in SocketCAN mode and a CAN cable to the arm.

Mount the arm on a stable surface and leave space for small joint movements.
Connect the power supply and CAN cable according to the manufacturer's wiring
instructions before powering it on.

Installation
------------

Install the robot dependencies on the controller machine using the custom-environment
workflow below. Run the checks from the same checkout and activated environment.

Robot Controller Node
~~~~~~~~~~~~~~~~~~~~~

Clone RLinf on the machine connected to the robot, then install its environment.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the robot dependencies from the repository root:

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env piper
   source .venv/bin/activate
   export REPO_PATH="$PWD"

Run subsequent commands from this checkout with the environment activated.
The installer includes the pinned ``pyAgxArm`` driver used by ``PiperArm``; see
:doc:`Installation </rst_source/start/installation>` for platform prerequisites.

Prepare the CAN Connection
^^^^^^^^^^^^^^^^^^^^^^^^^^

Find the SocketCAN interface with ``ip -brief link``. A USB-to-CAN adapter in
SocketCAN mode normally appears as ``can0``; replace that name below if yours
differs. Bring this interface up at 1 Mbit/s before opening the robot:

.. code-block:: bash

   sudo ip link set can0 down
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set can0 up
   ip -details link show can0

The output should show an enabled interface and ``bitrate 1000000``. With the
arm powered on, run ``candump -n 5 can0`` to inspect five feedback frames. If it
waits without receiving frames, press Ctrl+C and check the arm's power, CAN
wiring, selected adapter, and bitrate before continuing. The SDK does not
configure the CAN interface. Consult the
`pyAgxArm documentation <https://github.com/agilexrobotics/pyAgxArm>`_ for adapter
setup and supported hardware.

Run It
------

With CAN configured and the arm powered on, read its feedback first:

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --read-only

The command prints six joint angles in degrees, a tool pose (position in metres
and an xyzw quaternion), and the gripper opening in ``[0, 1]``, then exits.
It enables the motors and reads feedback without sending a position
target or resetting the arm. The detected firmware profile appears in the log.
For an AgxGripper with a 0.1 m stroke, add ``--gripper-max-width 0.1``; for an arm
without a gripper, add ``--no-gripper``.

To test a small movement, omit ``--read-only`` and enter the commands below one
at a time, waiting for the arm after each movement:

.. code-block:: bash

   python -m toolkits.realworld_check.test_piper_controller \
     --channel can0 --speed-percent 10

.. code-block:: text

   where
   joint 1 2
   where
   joint 1 -2
   grip 0.5
   where
   quit

``joint 1 2`` moves joint 1 by two degrees from its measured position. The
script accepts joints 1 through 6 and increments within five degrees; the arm
driver also clips targets to its travel. ``grip 0.5`` requests half the fitted
gripper's stroke. ``where`` reads feedback, so repeat it after the gripper has
settled. ``quit``, EOF, or Ctrl+C closes the CAN session and leaves the motors
holding position.

.. warning::

   Connection enables the motors even with ``--read-only``. Clear the workspace
   before connecting, and keep hands away during joint and gripper commands.
   Closing the test is not an emergency stop and does not cut motor power.

To use the arm in your own program or attach cameras, see
:doc:`Robotics Interface </rst_source/concepts/robotics>`.

New Real-World Tasks
--------------------

After the hardware checks pass, follow
:doc:`New Real-World Tasks </rst_source/extending/new_task>` to define a task's
reset procedure, observations, actions, reward, and success condition. A working
real-world task is required before starting training.
