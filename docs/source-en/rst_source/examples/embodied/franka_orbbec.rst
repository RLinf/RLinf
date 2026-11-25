.. Copyright 2026 The RLinf Authors.
   Licensed under the Apache License, Version 2.0.

Franka Orbbec Cameras
=====================

Overview
--------

Read color images and aligned depth from an Orbbec USB camera through RLinf's camera interface, then wire the same backend into a Franka configuration. The backend uses Orbbec SDK v2 and is validated on Femto Bolt; other Orbbec models need compatible SDK v2 firmware and stream profiles.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Hardware
      :text-align: center

      Orbbec Femto Bolt (SDK v2 USB)

   .. grid-item-card:: Connection
      :text-align: center

      USB 3.0 (5 Gbit/s) + 12 V / 2 A DC

   .. grid-item-card:: Workflow
      :text-align: center

      RGB & aligned depth for Franka

   .. grid-item-card:: Requires
      :text-align: center

      :doc:`franka` · ``pyorbbecsdk2==2.1.2``

Hardware and Safety
-------------------

Connect Femto Bolt with a USB 3 data cable and its 12 V / 2 A DC adapter (or a 5 V / 3 A Type-C source) for depth testing. Verify link speed on Linux with ``lsusb -t``: ``5000M`` confirms a 5 Gbit/s USB 3 link, while ``480M`` indicates USB 2.0. Select cameras by the SDK serial number rather than USB enumeration order or ``/dev/videoN`` names, which change across reconnects.

.. note::

   **Power over USB only.** Color-only operation (``enable_depth=False``) starts only the color stream and leaves laser and power settings unchanged. Per the `Femto Bolt datasheet <https://www.orbbec.com/wp-content/uploads/2023/08/ORBBEC_Datasheet_Femto-Bolt-0816-v01-1.pdf>`_, USB-only power supports depth/IR up to 640 × 576 (Y16) and color up to 1920 × 1080 (YUY2/MJPG), and a USB 3 data port alone does not guarantee 5 V / 3 A. When depth is enabled, a missing depth stream raises an error rather than switching to color only. Other Orbbec models have their own power limits.

.. warning::

   Give each physical camera one owner and close competing camera applications before connecting, because opening an Orbbec device initializes its sensors. RLinf opens only the configured serial and never substitutes another camera. When mounting cameras near a Franka arm, route cables clear of the workspace and keep the e-stop within reach.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Orbbec SDK Setup & Check

      **1. Install RLinf Franka and Orbbec SDK v2**

      Install the ``franka`` environment on the host that owns the USB camera (use that worker node's environment for remote cameras), then install ``pyorbbecsdk2==2.1.2`` with ``--no-deps`` so its example dependencies (Open3D, pygame, PyAV, pynput) and NumPy pin do not conflict with RLinf (NumPy and OpenCV are already in RLinf robot environments). The package name is ``pyorbbecsdk2`` and the Python import is ``pyorbbecsdk``; follow the `official Linux installation guide <https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html>`_ for udev permissions and platform prerequisites:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate
         uv pip install --no-deps pyorbbecsdk2==2.1.2

      **2. Find the camera serial number**

      On the host physically connected to the camera, print the SDK-reported serial number for each connected USB device without starting capture (plug in one camera at a time to distinguish multiple units):

      .. code-block:: bash

         python - <<'PY'
         import pyorbbecsdk as ob

         context = ob.Context()
         context.enable_net_device_enumeration(False)
         devices = context.query_devices()
         for index in range(devices.get_count()):
             print(devices.get_device_serial_number_by_index(index))
         PY

      Replace ``YOUR_CAMERA_SERIAL`` below with the printed value.

      **3. Verify RGB, then aligned depth**

      First verify color capture with ``enable_depth=False`` (declaring the camera descriptor does not touch hardware until connected):

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

      Once color capture succeeds, switch ``enable_depth=True`` to also receive ``observation["depth"]``: a ``float32`` array of shape ``(720, 1280)`` in metres, aligned to the color image via SDK calibration (zero marks unavailable depth). Disconnecting releases the camera cleanly so it can be reopened.

      **4. Use the camera in a Franka config**

      Update the camera fields in your ``cluster.node_groups[].hardware.configs[]`` entry:

      .. code-block:: yaml

         camera_type: orbbec
         camera_serials: ["YOUR_CAMERA_SERIAL"]

      Set ``enable_camera_depth: true`` in the Franka environment's ``init_params`` when depth is needed (default ``false``); the environment exposes aligned depth in its ``depths`` observation mapping for policies that consume depth. To compose the camera with other robot parts or place it on a remote worker via ``node_rank``, see :doc:`../../concepts/robotics`; for the full launch workflow, see :doc:`../../guides/realworld_robot`.

.. dropdown:: Stream resolution, alignment, and resampling details

   - ``resolution`` specifies the output size in ``(width, height)`` order. The driver picks the native color resolution nearest to it at the requested ``fps`` and uses the device's default depth resolution at that ``fps`` (640 × 576 on Femto Bolt).
   - Depth is aligned to color first, then both streams are resized to ``resolution`` using area resampling for color and nearest-neighbor resampling for depth so measured values stay intact.
   - Selected native profiles are logged on connect. A mismatched output aspect ratio stretches the image and logs a warning; unsupported frame rates fail with the list of available profiles.
   - Read timeouts and reconnect behavior follow the shared camera interface described in :doc:`../../concepts/robotics_architecture`.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Fix
   * - Serial query prints nothing
     - Check the USB 3 cable, run ``lsusb -t`` to verify the link, and apply udev device permissions from the `official Linux installation guide <https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html>`_.
   * - Depth startup fails or frames stop
     - Connect the 12 V / 2 A DC adapter (or 5 V / 3 A Type-C power), check USB 3 link speed (``5000M`` in ``lsusb -t``) and bandwidth, close competing processes, then reconnect.
   * - Device busy or sensor fails to initialize
     - Close any other application accessing that Orbbec camera; each physical camera must have a single owner.
   * - Unsupported frame rate or stretched image
     - Match ``fps`` to a native profile printed in the error message, and set ``resolution`` to an aspect ratio matching the native color stream.
