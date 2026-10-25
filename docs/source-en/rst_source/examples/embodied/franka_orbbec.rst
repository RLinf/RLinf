.. Copyright 2026 The RLinf Authors.
   Licensed under the Apache License, Version 2.0.

Using Orbbec Cameras with Franka
================================

Read color images and aligned depth from an Orbbec USB camera through RLinf's
camera interface. Start with a standalone camera, then use the same backend in
a robot configuration. The backend uses Orbbec SDK v2; the hardware validation
target is Femto Bolt. Other models need compatible SDK v2 firmware and stream
profiles.

Install and Connect
-------------------

Install the Python SDK in the environment that will own the camera. For a camera
hosted on another node, use that node's worker environment:

.. code-block:: bash

   python -m pip install --no-deps pyorbbecsdk2==2.1.2

``requirements/install.sh`` does not install this SDK. ``--no-deps`` keeps out
the packages its examples use (Open3D, pygame, PyAV, pynput) and its NumPy pin,
which can conflict with the versions RLinf installs. The camera backend needs
only NumPy and OpenCV, which the RLinf robot environments already include. The
distribution is named ``pyorbbecsdk2`` and its import is ``pyorbbecsdk``.
Follow the `official Linux installation guide
<https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html>`_
for device permissions and platform requirements. The SDK is imported when
discovering or connecting a camera.

Connect Femto Bolt with a USB 3 data cable and its 12 V adapter for depth testing.
On Linux, ``lsusb -t`` reports the negotiated link speed: ``5000M`` indicates a
5 Gbit/s connection, while ``480M`` indicates USB 2.0. Select the camera by the
device serial number reported by the Orbbec SDK; USB enumeration order and
``/dev/videoN`` names can change after reconnecting.

.. warning::

   Give each physical camera one owner. Opening an Orbbec device can initialize
   its sensors, so close other applications using that camera before connecting.
   RLinf opens only the configured serial and never substitutes another camera.

Find the Camera Serial Number
-----------------------------

Run this command on the computer physically connected to the camera, using the
Python environment where you installed the SDK:

.. code-block:: bash

   python - <<'PY'
   import pyorbbecsdk as ob

   context = ob.Context()
   context.enable_net_device_enumeration(False)
   devices = context.query_devices()
   for index in range(devices.get_count()):
       print(devices.get_device_serial_number_by_index(index))
   PY

The command prints the SDK-reported device serial number for each connected USB
camera without starting image capture. Replace ``YOUR_CAMERA_SERIAL`` in the
Python and YAML examples below with that exact value. To identify a camera among
several devices, connect only that camera, run the command and record its serial,
then reconnect the others. If no serial is printed, check the USB connection and
device permissions on this computer.

Read Color, Then Enable Depth
-----------------------------

First verify color capture with ``enable_depth=False``. Declare the camera with
``Camera.of(CameraInfo(...))``, then call ``connect()`` to open it. Construction
itself does not access the device:

.. code-block:: python

   from rlinf.robotics import Camera, CameraInfo

   camera = Camera.of(
       CameraInfo(
           name="scene",
           serial_number="YOUR_CAMERA_SERIAL",
           camera_type="orbbec",
           resolution=(1280, 720),
           fps=15,
           enable_depth=False,
       )
   )
   try:
       camera.connect()
       observation = camera.get_observation()
       image = observation["frame"]  # (720, 1280, 3), BGR uint8
   finally:
       camera.disconnect()

``get_observation()`` returns a fresh camera observation, and ``disconnect()``
stops capture and releases the device. The camera can then be connected again.
With color working, create the descriptor with ``enable_depth=True`` to also
receive ``observation["depth"]``: a ``float32`` array of shape ``(720, 1280)`` in
metres. Depth pixels are aligned to the color image through SDK calibration;
zero denotes unavailable depth.

``resolution`` describes the output size in ``(width, height)`` order. The driver
selects the native color resolution nearest to it at the requested ``fps``. Depth
uses the device's default depth resolution at that ``fps`` (640 × 576 on Femto
Bolt), so changing the output size never changes the depth mode. Depth is
aligned to color, and both are resized to the output size. Color uses area
resampling; depth uses nearest-neighbor resampling to preserve measured values.
The selected native profiles are logged when the camera connects. An output
aspect ratio different from the native color stream stretches the image and
logs a warning. Unsupported frame rates fail with the available profiles rather
than silently selecting a different rate.

Power Over USB Only
-------------------

For color-only operation, leave ``enable_depth=False``. This starts only the
color stream. It does not start a depth stream or change laser and power settings.

The `Femto Bolt datasheet
<https://www.orbbec.com/wp-content/uploads/2023/08/ORBBEC_Datasheet_Femto-Bolt-0816-v01-1.pdf>`_
specifies a 12 V / 2 A DC adapter, or a Type-C source supplying 5 V / 3 A.
USB-only operation supports depth/IR up to 640 × 576 in Y16 and color up to
1920 × 1080 in YUY2/MJPG. A USB 3 data connection alone does not guarantee that
power budget. Other Orbbec models have their own power requirements.

When depth is requested, a missing depth stream is an error; RLinf does not
silently switch to color-only observations. If startup fails or frames stop,
check the adapter, cable, USB bandwidth, permissions, and competing camera
processes. After correcting the connection, reconnect the camera. Reads use the
shared camera timeout and reconnect behavior described in
:doc:`../../concepts/robotics_architecture`.

Use the Camera in a Robot
-------------------------

Once the standalone read succeeds, compose the camera with other robot parts
using the interface in :doc:`../../concepts/robotics`. To place this same camera on
a worker node, pass ``node_rank`` to ``Camera.of``; the worker owns the SDK and
USB connection, while observation keys and units stay the same.

For an existing Franka run, update the camera fields in the selected
``cluster.node_groups[].hardware.configs[]`` entry:

.. code-block:: yaml

   camera_type: orbbec
   camera_serials: ["YOUR_CAMERA_SERIAL"]

These fields select the registered backend and the devices the robot will open.
In the Franka environment's ``init_params``, set ``enable_camera_depth: true``
when depth is needed; its default is ``false``. The Franka environment exposes
depth in its ``depths`` observation mapping. A policy must consume that mapping
to use depth for learning. Keep the arm and task settings from your existing
configuration; :doc:`../../guides/realworld_robot` describes the complete launch workflow.
