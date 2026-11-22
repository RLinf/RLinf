LeKiwi Mobile Base
==================

Take a LeKiwi 3-omniwheel mobile base and its front UVC camera from unboxing to a running ``LeKiwiDriveEnv-v1`` environment by checking the wheels, camera, and keyboard teleoperation in order.

Overview
--------

Work through the checks in order—wheels, camera, then keyboard—so any later failure narrows directly to the new layer. The base driver converts planar velocity commands into three Feetech STS3215 wheel speeds and scales all three down together when any wheel exceeds ``max_raw`` so the heading and turn rate stay intact.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      LeKiwi · 3× Feetech STS3215 omniwheels

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Front UVC webcam · local / TCP keyboard

   .. grid-item-card:: Workflow
      :text-align: center

      Motor IDs · Part checks · Teleop · DriveEnv

   .. grid-item-card:: Hardware
      :text-align: center

      Raspberry Pi / Linux · USB servos · no GPU

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Measured planar body velocity plus RGB camera frames (captured at ``640x480`` by default, center-cropped to aspect ratio, and resized to ``frame_size``, default ``128x128``).
   * - Action
     - Planar base velocity ``[x, y, theta]`` (forward ``m/s``, lateral ``m/s``, yaw ``deg/s``), held for one control period at ``step_frequency`` and stopped afterward.
   * - Reward
     - Always ``0`` with no early termination; task success is defined by the caller or a custom task.

Hardware and Safety
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Requirement
   * - Base & camera
     - LeKiwi mobile base (three Feetech STS3215 wheel servos, servo controller, kit power supply) and a front-mounted UVC webcam on the same host.
   * - Controller computer
     - Linux host with USB (typically a Raspberry Pi 5 on the base); no GPU required.
   * - Keyboard computer (optional)
     - Any LAN host (including macOS) with Python 3.11+ and ``pynput``; no RLinf install required.

For mechanical assembly, follow the `LeKiwi assembly guide <https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md>`__ (and LeRobot's `LeKiwi guide <https://huggingface.co/docs/lerobot/en/lekiwi>`__ for upstream usage).

.. warning::

   Put the base on blocks so all three wheels spin free for initial wheel and keyboard checks, and keep the servo power switch within reach. Run only one motor-control program on the serial port at a time. Normal exit, lost input, and exceptions stop the wheels and drop torque, but a cut USB cable or a killed process can leave wheels spinning—use the servo power switch immediately if that happens.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Install & Hardware Check

      **1. Install RLinf on the base controller computer**

      Run on the computer wired to the servo controller (for example a Raspberry Pi 5). ``--no-root`` skips sudo system GL libraries because the headless OpenCV build only captures frames (drop ``--no-root`` if you also train or run simulators on this host):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env lekiwi --install-rlinf --no-root && source .venv/bin/activate

      **2. Find the servo port and assign wheel motor IDs**

      Identify the servo controller serial port (examples assume ``/dev/ttyACM0``; prefer ``/dev/serial/by-id/...`` for fixed setups), then number factory-fresh STS3215 motors one at a time (writes left 7, back 8, right 9; skip if already numbered):

      .. code-block:: bash

         lerobot-find-port
         lerobot-setup-motors --robot.type=lekiwi --robot.port=/dev/ttyACM0

      **3. Check the wheels**

      With the base on blocks and all three motors connected, spin each wheel for 0.6 s at raw command 100 (adjust with ``--raw`` and ``--seconds``):

      .. code-block:: bash

         python -m toolkits.realworld_check.test_lekiwi_env wheels --port /dev/ttyACM0

      Expected output:

      .. code-block:: text

         Spinning the left wheel at raw 100
         left  commanded    100  measured     98  ok
         back  commanded    100  measured     97  ok
         right commanded    100  measured     99  ok

      **4. Check the front camera**

      Verify the camera actually delivers frames and inspect ``check.png`` (use ``--device`` when multiple cameras are attached):

      .. code-block:: bash

         python -m toolkits.realworld_check.test_lekiwi_env camera --save check.png

      Expected output (copy the printed identifier into ``camera_names`` below):

      .. code-block:: text

         Webcams found: /dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0
         Using /dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0 at 640x480
         Captured 10 frames in 0.34 s (29.4 fps)
         Frame shape (480, 640, 3), dtype uint8, mean brightness 96.2

   .. tab-item:: 2. Keyboard Drive & Env

      **1. Test keyboard input and drive the base**

      Keep the wheels off the ground for the first test. Verify key capture with ``--dry-run`` before engaging the motors (override defaults with ``--wheel-radius`` 0.05, ``--base-radius`` 0.125, ``--max-raw``, and ``--fps``):

      .. code-block:: bash

         python -m toolkits.realworld_check.test_lekiwi_env teleop --dry-run
         python -m toolkits.realworld_check.test_lekiwi_env teleop --port /dev/ttyACM0

      .. list-table::
         :header-rows: 1
         :widths: 35 65

         * - Key
           - Action
         * - ``W / S`` · ``A / D`` · ``Z / X``
           - Forward / backward · strafe left / right · turn left / right
         * - ``R / F``
           - Step speed up / down across Slow (0.1 m/s, 30 deg/s), Medium (0.2 m/s, 60 deg/s), and Fast (0.3 m/s, 90 deg/s)
         * - ``Q / Esc / Ctrl+C`` · release keys
           - Quit, stop wheels, and drop torque · command zero velocity

      .. dropdown:: Remote keyboard teleop (headless SSH / Wayland / macOS)

         Local capture requires a desktop X11 session. For headless SSH, Wayland, or driving from another machine (start the receiver first; TCP port 5555 via ``--remote-port``, ``--fps`` 30 Hz, ``--connect-timeout``, ``--input-timeout`` 2 s):

         .. code-block:: bash

            # On the robot (receiver):
            python -m toolkits.realworld_check.test_lekiwi_env teleop --listen --port /dev/ttyACM0

            # On the keyboard host (replace with robot IP):
            python -m toolkits.realworld_check.test_lekiwi_env teleop --remote 192.168.1.100

         On a separate laptop or macOS (where ``install.sh`` is unsupported), install ``pynput`` in a Python 3.11+ venv and run from a checkout (on macOS, grant Terminal Accessibility and Input Monitoring permissions and test with receiver ``--dry-run`` first):

         .. code-block:: bash

            git clone https://github.com/RLinf/RLinf.git && cd RLinf
            pip install pynput
            python -m toolkits.realworld_check.test_lekiwi_env teleop --remote 192.168.1.100

      **2. Configure LeKiwiDriveEnv-v1**

      Select ``LeKiwiDriveEnv-v1`` with ``env_type: real`` (same layout as ``examples/embodiment/config/env/so101_reach.yaml``), declare ``hardware.type: LeKiwi``, and map the camera identifier to ``front`` so frames appear under ``main_image_key``:

      .. code-block:: yaml

         cluster:
           node_groups:
             - label: lekiwi
               node_ranks: 1
               hardware:
                 type: LeKiwi
                 configs:
                   - serial_port: /dev/ttyACM0
                     camera_serials: ["/dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0"]
                     node_rank: 1

         env:
           eval:
             env_type: real
             main_image_key: front
             init_params:
               id: "LeKiwiDriveEnv-v1"
             override_cfg:
               camera_names:
                 "/dev/v4l/by-id/usb-USB_Camera_USB_Camera_2505280051-video-index0": front
               frame_size: [128, 128]
               max_linear_velocity: 0.5
               max_angular_velocity: 90.0
               step_frequency: 10.0

      .. dropdown:: RPent deployment (optional)

         LeKiwi ships no training config under ``examples/embodiment`` because RPent runs ``LeKiwiDriveEnv-v1`` directly. For environment server, robot configuration, and planner setup, follow the `RPent LeKiwi guide <https://github.com/RLinf/RPent/blob/main/docs/source-en/rst_source/real_world_robots/lekiwi.rst>`__.

Results and Troubleshooting
---------------------------

Hardware setup is verified once all three wheels report ``ok``, ``camera --save check.png`` delivers live frames with non-zero brightness, and ``teleop --dry-run`` registers key presses before enabling the motors.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Fix
   * - Local ``teleop`` does not receive keys over SSH or Wayland
     - Run ``teleop --listen`` on the robot and ``teleop --remote <robot-ip>`` on a desktop host.
   * - macOS remote sender fails to capture keys
     - Enable Accessibility and Input Monitoring for your terminal in System Settings > Privacy & Security, restart the terminal, and verify with ``--dry-run``.
   * - Camera opens but frames appear under ``camera_1`` instead of ``front``
     - Map the exact identifier printed by ``test_lekiwi_env camera`` to ``front`` in ``override_cfg.camera_names``.
   * - Receiver exits mid-session
     - Default ``--input-timeout`` is 2 s after lost packets or disconnect; restart both receiver and sender to begin a new session.

For part observation and action contracts, see :doc:`Robotics Interface <../../concepts/robotics>`; to add rewards or termination logic, see :doc:`New Real-World Tasks <../../extending/new_task>`.
