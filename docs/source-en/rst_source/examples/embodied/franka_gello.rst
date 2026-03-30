Real-World Franka with GELLO Teleoperation
============================================

This guide explains how to set up and use the **GELLO** teleoperation device
with the Franka real-world environment in RLinf. It extends the base
:doc:`franka` documentation with hardware-specific installation,
configuration, and verification steps.

.. note::

   If you have not read the base Franka guide yet, please start with
   :doc:`franka` first. This page only covers the **additional** steps
   required for the GELLO hardware.


Hardware Overview
-----------------

`GELLO <https://github.com/wuphilipp/gello_software>`_ is a joint-level
teleoperation device that mirrors the kinematic structure of the Franka arm.
It provides more intuitive and precise control than a SpaceMouse, with
native gripper support.

A typical GELLO deployment connects the device to the **controller node**
(usually the NUC or the machine physically connected to the robot) via a
USB serial adapter (FTDI).

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Node
     - Role
     - Hardware
   * - **GPU server** (node 0)
     - Actor, rollout, env worker; camera capture
     - NVIDIA GPU (e.g. RTX 4090), RealSense cameras
   * - **NUC** (node 1)
     - FrankaController, GELLO teleoperation
     - Franka arm, GELLO device (USB-FTDI)


GELLO Software Installation
------------------------------

GELLO depends on two packages that must be installed **in order**:

1. ``gello`` — the low-level driver from `gello_software <https://github.com/wuphilipp/gello_software>`_.
2. ``gello-teleop`` — the forward-kinematics and teleoperation agent used by RLinf.

Both packages should be installed on the node that runs the GELLO device
(typically the NUC / controller node).

1. Install ``gello`` (gello_software)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose a directory to install the GELLO software, then clone the repository
and initialize its submodules (which include the **Dynamixel SDK**):

.. code-block:: bash

   cd /path/to/install/gello
   git clone https://github.com/wuphilipp/gello_software.git
   cd gello_software
   git submodule init && git submodule update

Install the ``gello`` package and the **Dynamixel SDK** (bundled as a
third-party submodule):

.. code-block:: bash

   pip install -e .
   pip install -e third_party/DynamixelSDK/python

The Dynamixel SDK is required for communicating with the Dynamixel servos
inside the GELLO device. Without it, the ``GelloAgent`` will not be able
to read joint positions.

.. note::

   If you encounter permission issues accessing the serial device, add
   your user to the ``dialout`` group:

   .. code-block:: bash

      sudo usermod -aG dialout $USER

   Then log out and back in for the change to take effect.

For additional hardware configuration (e.g. setting unique motor IDs,
DynamixelRobotConfig, and port mapping), refer to the
`gello_software README <https://github.com/wuphilipp/gello_software#readme>`_.

2. Install ``gello-teleop``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``gello-teleop`` wraps the ``gello`` driver with Franka forward kinematics
(using dm_control/MuJoCo) and a teleoperation agent interface. Install it
directly from the GitHub repository:

.. code-block:: bash

   pip install git+https://github.com/RLinf/gello-teleop.git

Or, if you prefer an editable installation:

.. code-block:: bash

   git clone https://github.com/RLinf/gello-teleop.git
   cd gello-teleop
   pip install -e .

To also install the ``gello`` dependency automatically (if not already
installed separately):

.. code-block:: bash

   pip install "gello-teleop[gello] @ git+https://github.com/RLinf/gello-teleop.git"


3. Set up the serial device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plug the GELLO device into the controller node via the USB-FTDI adapter.
Identify the serial port:

.. code-block:: bash

   ls /dev/serial/by-id/
   # Look for something like:
   # usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

Grant permission:

.. code-block:: bash

   sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0
   # Or add your user to the dialout group for persistent access:
   sudo usermod -aG dialout $USER

.. tip::

   Using the ``/dev/serial/by-id/`` path is recommended over ``/dev/ttyUSB*``
   because it is stable across reboots and does not change when other USB
   devices are plugged in.

4. Verify the GELLO device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the built-in test script to confirm that the GELLO device is
communicating correctly and producing valid joint readings:

.. code-block:: bash

   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0

You should see continuously updating output like:

.. code-block:: text

   pos=[0.500 0.000 0.300]  quat=[1.000 0.000 0.000 0.000]  gripper=[0.040]

If the output is updating as you move the GELLO device, the installation
is successful.


YAML Configuration
-------------------

To use GELLO for data collection, use the config file
``examples/embodiment/config/realworld_collect_data_gello.yaml``.
The key differences from the standard SpaceMouse config are:

.. code-block:: yaml

   env:
     eval:
       use_spacemouse: False
       use_gello: True
       gello_port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"

.. list-table:: GELLO-specific configuration fields
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``use_gello``
     - ``False``
     - Enable GELLO teleoperation. Set to ``True`` to use GELLO instead of
       SpaceMouse.
   * - ``gello_port``
     - ``null``
     - Serial port path of the GELLO device. Required when ``use_gello``
       is ``True``.
   * - ``use_spacemouse``
     - ``True``
     - Must be set to ``False`` when using GELLO.

For full data collection instructions, refer to the
**Data Collection with GELLO** section in :doc:`franka`.


Cluster Setup Notes
---------------------

The cluster setup procedure is the same as described in :doc:`franka`. The
key additional requirement is:

- On the **controller node** (NUC): make sure both ``gello`` and
  ``gello-teleop`` are installed in the virtual environment **before**
  running ``ray start``.

.. warning::

   Remember that Ray captures the Python interpreter and environment
   variables at ``ray start`` time. Any package installed **after**
   ``ray start`` will not be visible to Ray workers. Always install
   ``gello`` and ``gello-teleop`` first, then start Ray.


Troubleshooting
----------------

**GELLO device not detected**

- Verify the USB-FTDI adapter is connected: ``ls /dev/serial/by-id/``.
- Check ``lsusb`` for ``FTDI`` devices.
- Ensure the Dynamixel servos are powered on (the GELLO device needs
  external power for the servos).

**Permission denied on serial port**

- Run ``sudo chmod 666 /dev/serial/by-id/<your-device>``.
- Or add your user to the ``dialout`` group:
  ``sudo usermod -aG dialout $USER`` (requires re-login).

**Import error: ``No module named 'gello'``**

- Ensure the ``gello`` package (from ``gello_software``) is installed in
  the same virtual environment. Run:
  ``pip show gello`` to verify.

**Import error: ``No module named 'gello_teleop'``**

- Ensure ``gello-teleop`` is installed:
  ``pip show gello-teleop`` to verify.
- If using an editable install, make sure the repository path is correct.

**GELLO readings are not updating**

- Check that the Dynamixel servo IDs match the configuration in
  ``gello_software``.
- Try lowering the baud rate in the GELLO configuration if communication
  is unreliable.
- Ensure no other process is using the same serial port.
