LeKiwi Mobile Base Setup and Control
====================================

This page takes a LeKiwi omnidirectional base and its front camera from the
box to a running ``LeKiwiDriveEnv-v1`` environment. It installs the environment,
numbers the wheel motors, checks each part with one command, and then shows the
config that hands the base to a policy or a controller.

Work through the three checks in order. Each one covers a single part, so when
a later step fails you already know the wheels, the camera, and the keyboard
work. For assembly, follow the `LeKiwi assembly guide
<https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md>`_; LeRobot's
`LeKiwi guide <https://huggingface.co/docs/lerobot/en/lekiwi>`_ covers the same
robot for its own stack.

What you need
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Requirement
   * - Base
     - LeKiwi mobile base with three Feetech STS3215 wheel servos, its servo
       controller, and the kit's power supply.
   * - Camera
     - UVC webcam mounted on the front, connected to the same computer as the
       servo controller.
   * - Controller computer
     - Linux computer with a USB port for the servo controller. A Raspberry Pi
       on the robot is the usual choice; it needs no GPU.

How the pieces fit
------------------

The base is three wheel servos at 120 degree spacing on one serial bus, so
only per-wheel velocities reach the hardware. RLinf converts a body velocity
``[x, y, theta]``, in m/s, m/s, and deg/s, into wheel speeds, and scales every
wheel by the same factor when one would exceed its limit, so a direction the
base cannot reach at full speed runs slower but still in a straight line. The
robotics layer composes ``LeKiwiRobot`` from the base and the cameras; the env
layer wraps that robot as ``LeKiwiDriveEnv-v1``.

Install the environment
-----------------------

Run these steps on the computer (for example, a Raspberry Pi 5) wired to the
servo controller, not on your laptop.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the robot dependencies from the repository root:

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env lekiwi --install-rlinf --no-root
   source .venv/bin/activate

The installer includes LeRobot with Feetech support and OpenCV's headless
build, which captures frames without the system GL libraries a camera window
needs. ``--install-rlinf`` installs RLinf itself, which is what provides the
three ``rlinf-lekiwi-*`` commands used below; without it the installer installs
only dependencies. ``--no-root`` skips the system libraries the installer would
otherwise add with sudo: the checks below use only the serial port and that
headless build, so a base computer needs none of them. Drop the flag if you also
train or run simulators on this computer. Platform prerequisites are in
:doc:`Installation </rst_source/start/installation>`.

Find the servo port
-------------------

Every command takes the serial port of the servo controller. With the
controller plugged in and nothing else on its USB path, run:

.. code-block:: bash

   lerobot-find-port

Unplug the controller when prompted, then reconnect it. The device it reports
is the port to use below; these examples assume ``/dev/ttyACM0``, and on macOS
the path looks like ``/dev/tty.usbmodem...``. A name under
``/dev/serial/by-id/`` survives replugging into a different socket, so prefer
it when the base stays in one place. The port stays the same for all the checks
below.

Assign the wheel motor IDs
--------------------------

A factory-fresh STS3215 answers on the same default ID, so the wheels are
numbered one at a time. LeRobot's setup command prompts for each wheel; connect
only the wheel it names:

.. code-block:: bash

   lerobot-setup-motors --robot.type=lekiwi --robot.port=/dev/ttyACM0

It writes left wheel 7, back wheel 8, and right wheel 9, the IDs this driver
expects. Skip this step on a base whose wheels are already numbered.

Check the wheels
----------------

Put the base on blocks so the wheels spin free, connect all three motors, and
run:

.. code-block:: bash

   rlinf-lekiwi-test-wheels --port /dev/ttyACM0

Each wheel turns for half a second and the command reports the velocity its
encoder sent back:

.. code-block:: text

   INFO: Spinning the left wheel at raw 100
   INFO: left  commanded    100  measured     98  ok
   INFO: back  commanded    100  measured     97  ok
   INFO: right commanded    100  measured     99  ok

``--raw`` makes the spin faster and ``--seconds`` makes it longer.

Check the camera
----------------

.. code-block:: bash

   rlinf-lekiwi-check-camera --save check.png

The command lists the webcams it finds, opens one, and reports the frames it
actually delivers, because a camera can open and still send nothing:

.. code-block:: text

   INFO: Webcams found: usb-USB_Camera_USB_Camera_2505280051-video-index0
   INFO: Using usb-USB_Camera_USB_Camera_2505280051-video-index0 at 640x480
   INFO: Captured 10 frames in 0.34 s (29.4 fps)
   INFO: Frame shape (480, 640, 3), dtype uint8, mean brightness 96.2

Write the printed identifier into ``camera_names`` later in this page. Open
``check.png`` to see what the robot sees, and adjust the mount before
collecting data. With several cameras attached, choose one with
``--device``.

Drive the base with the keyboard
--------------------------------

Keep the wheels off the ground for the first run and keep servo power within
reach. Check the key handling without touching the motors first:

.. code-block:: bash

   rlinf-lekiwi-teleoperate --dry-run

Then drive:

.. code-block:: bash

   rlinf-lekiwi-teleoperate --port /dev/ttyACM0

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Action
   * - W / S
     - Forward / backward
   * - A / D
     - Left / right
   * - Z / X
     - Turn left / right
   * - R / F
     - Increase / decrease speed, once per press
   * - Q / Esc / Ctrl+C
     - Quit, stop the wheels, and drop torque
   * - Release movement keys
     - Command zero velocity

The three speed levels start at the slowest:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Level
     - Translation
     - Rotation
   * - Slow (initial)
     - 0.1 m/s
     - 30 deg/s
   * - Medium
     - 0.2 m/s
     - 60 deg/s
   * - Fast
     - 0.3 m/s
     - 90 deg/s

Use ``--wheel-radius`` and ``--base-radius`` in metres if your geometry
differs from the defaults of 0.05 and 0.125, ``--max-raw`` to lower the servo
limit, and ``--fps`` to change the control rate.

The command stops and releases every wheel on quit, on lost keyboard input,
and on an exception. A cut USB cable or a killed process can still leave the
wheels turning: use the servo power switch, and run one motor-control program
on the port at a time.

The keyboard listener reads the local desktop, so it needs a local X11
session; it does not read a headless SSH session or Wayland. Drive the same
base from another machine instead: ``--listen`` on the robot drives the wheels
from keys sent over TCP, and ``--remote HOST`` where the keyboard is sends
them.

.. code-block:: bash

   # On the robot, wired to the motors:
   rlinf-lekiwi-teleoperate --listen --port /dev/ttyACM0

   # On the machine with the keyboard:
   rlinf-lekiwi-teleoperate --remote <robot-ip>

The keys and speed levels are the same. Both sides use TCP port 5555 unless you
change it with ``--remote-port``. ``--listen`` also takes ``--fps``,
``--wheel-radius``, ``--base-radius``, ``--max-raw``, and ``--dry-run``, and
``--connect-timeout`` sets how long ``--remote`` keeps retrying. The listener
watches the sender's heartbeat: if the sender is killed, the network drops, or
the laptop sleeps, it zeroes the wheels and waits for a new sender. Stop the
sender with Ctrl+C; it also exits when the listener closes.

The drive environment
---------------------

``LeKiwiDriveEnv-v1`` is the registered Gymnasium task: it composes the robot from
the hardware config, reports the measured body velocity and camera frames, and
commands bounded velocities. A step holds the commanded velocity for one
control period and then stops the wheels, so the base moves only while the
caller keeps stepping.

Select the task with ``init_params.id`` in an env config file, as every
real-world task does; ``examples/embodiment/config/env/so101_reach.yaml`` shows
an arm configured this way:

.. code-block:: yaml

   # examples/embodiment/config/env/lekiwi_drive.yaml
   env_type: real
   total_num_envs: null
   main_image_key: front

   init_params:
     id: "LeKiwiDriveEnv-v1"
     num_envs: null

The port, the cameras, and the wheel geometry come from the hardware block of
the node group that runs the environment:

.. code-block:: yaml

   cluster:
     node_groups:
       - label: lekiwi
         node_ranks: 1
         hardware:
           type: LeKiwi
           configs:
             - port: /dev/ttyACM0
               camera_serials: ["<camera-serial>"]
               node_rank: 1

Per-run settings are listed under ``override_cfg`` in the training config:

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         camera_names:
           "<camera-serial>": front
         frame_size: [128, 128]
         max_linear_velocity: 0.5
         max_angular_velocity: 90.0
         step_frequency: 10.0

The keys of ``camera_names`` are the identifiers printed by
``rlinf-lekiwi-check-camera``. Name the camera ``front`` so its frames appear
under the key ``main_image_key`` points at; a camera left unnamed reports under
``camera_1`` and so on.

``frame_size`` sets the width and height of every frame in the observation, 128
by 128 pixels by default. A policy usually wants a small frame; a live viewer
wants a larger one. The camera driver asks the device for that size and resizes
in software when the device cannot deliver it, so pick a resolution the camera
supports natively.

The reward is always zero and the episode never terminates early: a task
decides success itself, as
:doc:`New Real-World Tasks </rst_source/extending/new_task>` describes.

Drive the base from your own program
------------------------------------

To use the base outside an experiment, build ``LeKiwiRobot`` from the robotics
layer and call the parts directly; the same velocity conversion and limits
apply:

.. code-block:: python

   from rlinf.robotics.parts.cameras import CameraInfo
   from rlinf.robotics.robots import LeKiwiRobot

   robot = LeKiwiRobot.build(
       port="/dev/ttyACM0",
       node_rank=None,  # Open the hardware in this process.
       cameras={
           "front": CameraInfo(
               name="front",
               serial_number="<camera-serial>",
               camera_type="webcam",
               resolution=(320, 240),
           )
       },
   )
   robot.connect()
   try:
       robot.send_action({"base": {"velocity": [0.1, 0.0, 0.0]}})
   finally:
       robot.disconnect()

See :doc:`Robotics Interface </rst_source/concepts/robotics>` for the
observation and action contracts of each part.
