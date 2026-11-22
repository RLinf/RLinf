Franka GELLO Teleop
===================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gello.jpeg
   :align: center
   :width: 80%

   GELLO joint-level teleoperation device used to collect Franka demonstrations.

Overview
--------

Use a GELLO leader arm for joint-level Franka demonstration collection instead of a SpaceMouse. Install ``gello`` and ``gello-teleop`` on the controller node, verify the USB-FTDI serial stream, update the collection config, and record demonstrations with live progress monitoring.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · GELLO leader arm · gripper

   .. grid-item-card:: Connection
      :text-align: center

      USB-FTDI serial (``/dev/serial/by-id/``)

   .. grid-item-card:: Workflow
      :text-align: center

      Joint-level teleop data collection

   .. grid-item-card:: Requires
      :text-align: center

      :doc:`franka` · external servo power · ``dialout`` access

Hardware and Safety
-------------------

Power the GELLO Dynamixel servos with their external power supply and plug the USB-FTDI adapter into the Franka controller node (typically the NUC). Identify the stable device path under ``/dev/serial/by-id/`` instead of ``/dev/ttyUSB*`` so the port stays fixed across reboots and USB reconnects:

.. code-block:: bash

   ls /dev/serial/by-id/
   # Example output:
   # usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

Grant serial access for the current session or persistently via the ``dialout`` group (log out and back in after adding the group):

.. code-block:: bash

   sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
   # Or persistent access:
   sudo usermod -aG dialout $USER

For motor ID assignment, servo config, and port mapping, refer to the `gello_software README <https://github.com/wuphilipp/gello_software#readme>`_.

.. warning::

   Keep the Franka e-stop button within reach and clear obstacles from the workspace before starting teleoperation. Match the GELLO leader arm close to a safe Franka start pose so the follower arm does not jump on activation, and confirm Franka Desk is in programmable mode with no other process holding arm control.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Franka + GELLO

      **1. Install RLinf Franka and GELLO packages (in order)**

      Install the ``franka`` environment on the controller node, then clone and install ``gello_software`` (with its Dynamixel SDK submodule) and ``gello-teleop`` from the RLinf repository root:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

         git clone https://github.com/wuphilipp/gello_software.git ../gello_software
         git -C ../gello_software submodule update --init third_party/DynamixelSDK
         uv pip install -e ../gello_software -e ../gello_software/third_party/DynamixelSDK/python

         git clone https://github.com/RLinf/gello-teleop.git ../gello-teleop
         uv pip install -e ../gello-teleop

      .. note::

         ``git -C ../gello_software submodule update --init third_party/DynamixelSDK`` fetches only the Dynamixel SDK and skips ``third_party/mujoco_menagerie``; ``gello-teleop`` ships its own Franka MJCF.

      **2. Verify the serial device and joint stream**

      From the RLinf repository root, run the built-in GELLO check script with your ``/dev/serial/by-id/`` path:

      .. code-block:: bash

         python -m rlinf.robotics.parts.teleop.gello \
             --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

      Move the GELLO arm and confirm continuously updating TCP and gripper output:

      .. code-block:: text

         pos=[0.500 0.000 0.300]  quat=[1.000 0.000 0.000 0.000]  gripper=[0.040]

      **3. Configure and launch collection**

      Set ``robot_ip`` and your GELLO serial path in ``examples/embodiment/config/realworld_collect_data_gello.yaml``:

      .. code-block:: yaml

         env:
           eval:
             teleop: gello
             gello_port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"

      Start data collection in terminal 1 (logging to a file because Ray's log monitor batches stdout and breaks ``tqdm`` in-place refresh), and watch live episode counts in terminal 2:

      .. code-block:: bash

         # Terminal 1 — launch single-arm collection (or realworld_collect_data_gello_joint_dual_franka for dual-arm)
         bash examples/embodiment/collect_data.sh \
             realworld_collect_data_gello 2>&1 \
             | tee logs/collect.log

         # Terminal 2 — live progress bar (pass --no-replay to tail from EOF)
         python -m toolkits.realworld_check.collect_monitor logs/collect.log

.. dropdown:: GELLO YAML fields and dual-arm reference

   .. list-table::
      :header-rows: 1
      :widths: 25 15 60

      * - Field
        - Default
        - Description
      * - ``env.eval.teleop``
        - ``spacemouse``
        - Set to ``gello`` to drive the arm from a GELLO leader arm.
      * - ``env.eval.gello_port``
        - ``null``
        - Serial port path of the GELLO device. Required when ``teleop`` is ``gello``.

   For dual-arm joint-space collection, pass ``realworld_collect_data_gello_joint_dual_franka`` to ``examples/embodiment/collect_data.sh``. Full demonstration collection options and replay-buffer exports are covered in :doc:`franka`.

.. dropdown:: Cluster setup notes

   For multi-node setups, follow the Ray cluster setup in :doc:`franka`, with one order requirement: install both ``gello`` and ``gello-teleop`` in the virtual environment on the controller node (NUC) **before** running ``ray start``.

   .. warning::

      Ray captures the Python interpreter and environment variables at ``ray start`` time. Packages installed after ``ray start`` are invisible to Ray workers. Install ``gello`` and ``gello-teleop`` first, then start Ray.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Fix
   * - GELLO device not detected
     - Check ``ls /dev/serial/by-id/`` and ``lsusb`` for an ``FTDI`` device, and verify the Dynamixel servos have external power switched on.
   * - Permission denied on serial port
     - Run ``sudo chmod 666 /dev/serial/by-id/<your-device>``, or run ``sudo usermod -aG dialout $USER`` and log out and back in.
   * - ``No module named 'gello'``
     - Verify ``gello`` from ``gello_software`` is installed in the active virtual environment with ``pip show gello`` (restart Ray in multi-node setups).
   * - ``No module named 'gello_teleop'``
     - Verify ``gello-teleop`` is installed with ``pip show gello-teleop`` and check the editable checkout path.
   * - GELLO readings do not update
     - Confirm Dynamixel servo IDs match ``gello_software``, lower the baud rate if serial traffic drops packets, and close any other process holding the port.
