Real-World Dual-Franka: GELLO Collection, π₀.₅ SFT, Deployment
================================================================

This guide is the end-to-end recipe for the **dual-arm Franka** rig in
RLinf — bringing up two physical compute nodes, collecting bimanual
GELLO joint-space teleoperation data at 1 kHz, fine-tuning π₀.₅ on the
data in a 20-D rot6d action space, and deploying the trained policy
back to the real robot driven by a foot pedal.

This page assumes you have already read:

* :doc:`franka` — single-arm Franka basics, Ray cluster setup, the
  RealSense + SpaceMouse data-collection path. Read this first if
  none of the words "FrankaController", "FCI", or "RLINF_NODE_RANK"
  ring a bell.
* :doc:`franka_gello` — GELLO hardware install, Dynamixel SDK,
  ``gello-teleop`` package, USB-FTDI permissions.

This page focuses on what changes for the dual-arm rig:

* the **franky** low-level backend (libfranka via ``franky-control``)
  shared by both arms — replacing the legacy ROS / serl path used by
  :doc:`franka`,
* three new dual-arm environments — ``DualFrankaEnv`` (legacy 14-D
  Cartesian delta), ``DualFrankaJointEnv`` (16-D joint, used at
  collection), ``DualFrankaTcpEnv`` (20-D TCP-rot6d, used at SFT
  and deployment),
* the **rot6d / SE(3) body-frame delta** action representation that
  replaces openpi's component-wise ``DeltaActions``,
* resume-aware data collection driven by a 3-key foot pedal,
* a 2-physical-node Ray cluster where each node owns one Franka
  controller and the env worker / GPU live on node 0.


Why this rig (and what it is *not*)
-----------------------------------

The dual-Franka rig in RLinf is designed for **bimanual manipulation
SFT** — collecting high-quality teleoperation demonstrations and
fine-tuning a foundation VLA (π₀ / π₀.₅) on them. Compared with the
single-arm SAC / PPO loops in :doc:`franka`, this rig:

* **Targets imitation learning, not online RL.** The collection path
  is GELLO joint teleop; the deployment path autonomously runs the
  SFT policy with a foot pedal that owns episode boundaries. There is
  no reward labelling and no RL update on the collected data.
* **Uses one libfranka backend (``franky-control``) for both arms.**
  All low-level control runs in C++ at 1 kHz inside ``franky``;
  Python only updates references. This avoids the GIL-induced jitter
  that pure-Python control loops on top of ROS suffer from.
* **Encodes orientation as 6D rotation, not Euler.**
  Euler-based state/action pollutes π₀ / π₀.₅ with ±π wrap
  discontinuities (one frame's roll = +3.14 rad → next frame's roll
  = −3.14 rad ⇒ a "−2π" pseudo-delta the policy memorises). Switching
  to rot6d + SE(3) body-frame deltas removes that class of bug
  entirely.
* **Splits two arms across two compute nodes.** Each node has a
  direct Ethernet link to one Franka's FCI port (one cable per arm,
  one NIC per node). The two nodes share a LAN used only by Ray and
  tensor sync.

If you want online RL on a single Franka with SAC/PPO, you want
:doc:`franka`, not this page.


Hardware topology
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Node
     - Role
     - Hardware on this node
   * - **node 0** (head)
     - Ray head; env worker; left ``FrankyController``;
       actor / rollout (during eval); all camera + GELLO capture
     - 1× GPU (e.g. RTX 4090) — only used at SFT and deployment;
       left Franka FR3 on a directly-cabled NIC reaching its FCI port;
       left Robotiq 2F-85 (USB-RS485 Modbus);
       **both left and right GELLO** Dynamixel chains (USB-FTDI);
       **all three cameras** — base RealSense D435i (third-person)
       and left + right wrist Lumos USB-3;
       PCsensor 3-key FootSwitch (on node 0)
   * - **node 1** (worker)
     - Ray worker; right ``FrankyController`` only
     - Optional GPU (not used for inference);
       right Franka FR3 on its own directly-cabled NIC reaching its
       FCI port;
       right Robotiq 2F-85

.. note::

   FCI IPs and NIC names depend on your actual network setup — fill in
   whatever matches your rig in the Hardware YAML below.

Camera roles (the wrapper stack uses
``main_image_key: left_wrist_0_rgb`` so π₀.₅'s ``observation/image``
slot is the *left* wrist; ``base_0_rgb`` and ``right_wrist_0_rgb`` go
into ``observation/extra_view_image-{0,1}``). All three cameras'
USB cables terminate on **node 0** — the env worker there is the
single process that opens ``/dev/v4l/by-id/...`` and
``rs.pipeline()``, so the right-wrist Lumos cable still runs back
to node 0 even though the right arm itself lives on node 1:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Camera slot
     - Backend
     - Purpose
   * - ``base_0_rgb``
     - RealSense D435i
     - Third-person view shared by both arms
   * - ``left_wrist_0_rgb``
     - Lumos USB 3 (XVisio vSLAM)
     - Left arm wrist camera, used as π₀.₅'s primary ``image``
   * - ``right_wrist_0_rgb``
     - Lumos USB 3 (XVisio vSLAM)
     - Right arm wrist camera

Foot pedal: a 3-key PCsensor FootSwitch flashed to send key codes
``a`` / ``b`` / ``c`` (one-time flash via the vendor Windows tool;
codes are stored in firmware and persist across reboots).
**The pedal must be plugged into node 0** (the env worker is pinned
there by the shipped placement). Node 0 must export
``RLINF_KEYBOARD_DEVICE=/dev/input/eventXX`` *before* ``ray start``
so Ray captures the path. ``KeyboardListener`` reads ``evdev``
directly — no ``DISPLAY``, no ``xev``, no terminal focus.


Software stack
--------------

The data path during **collection** is::

  GELLO arm (Dynamixel)                     env worker (node 0)
        │                                          │
        ▼                                          ▼
  GelloJointExpert (1 kHz read)            DualFrankaJointEnv.step
        │ ±2π unwrap                              │ 10 Hz
        ▼                                          │
  DualGelloJointIntervention                       │
   (direct_stream daemon, 1 kHz)                   │  (env.step reads
        │                                          │   state + grippers
        └─move_joints─► FrankyController(left)  ◄──┘   only — does NOT
        └─move_joints─► FrankyController(right)        forward motion)
                              │ C++ 1 kHz JointImpedanceTracker
                              ▼
                        Franka FR3

The data path during **deployment** is::

  observation (state[20] + 3 cams)
        │
        ▼
  DualFrankaRot6dInputs ─► RigidBodyDeltaActions ─► π₀ / π₀.₅
                                                       │
                                                       ▼
                            RigidBodyAbsoluteActions ◄┘  (T_abs = T_state @ T_delta)
                                       │
                                       ▼
                            DualFrankaRot6dOutputs (slice 20-D)
                                       │
                                       ▼
                  DualFrankaTcpEnv.step (per-arm move_tcp_pose)
                                       │ C++ 1 kHz CartesianImpedanceTracker
                                       ▼
                                 Franka FR3

The two trackers (``JointImpedanceTracker`` and
``CartesianImpedanceTracker``) are **mutually exclusive** inside
``FrankyController`` — switching from collection (joint impedance)
to deployment (Cartesian impedance) automatically stops the previous
tracker, so you do not need to restart franky between sessions.


Installation (per node)
-----------------------

Repeat this section on **both** ``node 0`` and ``node 1``. The nodes
have separate checkouts and separate venvs; they only share the LAN.

1. PREEMPT_RT kernel and rtprio limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The franky backend assumes the host is running an unrestricted
PREEMPT_RT kernel. Build and boot one per Franka's official guide
`Setting up the real-time kernel
<https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html>`_;
this project has been validated on ``5.15.133-rt69``. Verify:

.. code-block:: bash

   uname -a | grep -o PREEMPT_RT   # must print PREEMPT_RT

A direct gigabit NIC must reach the Franka FCI port (typically
``172.16.0.2``) with no switch in between, and ``/proc/cmdline``
should not pass odd ``iommu`` / ``apic`` flags that would interfere
with the RT thread.

Drop ``/etc/security/limits.d/99-realtime.conf`` so PAM grants
``rtprio 99`` and ``memlock unlimited`` to your user:

.. code-block:: text

   *  -  rtprio    99
   *  -  memlock   unlimited

Log out and back in to let PAM re-read the limits; ``ulimit -r``
must then return ``99`` (or ``unlimited``) and ``ulimit -l`` must
return ``unlimited``. Without these,
``FrankyController.__init__`` logs ``SCHED_FIFO denied`` /
``mlockall failed`` and falls back to default scheduling — the
controller still runs, but RT jitter returns.

.. note::

   These limits are checked at startup by
   ``_apply_rt_hardening()`` in
   ``rlinf/envs/realworld/franka/franky_controller.py``. If
   ``SCHED_FIFO`` is denied or ``mlockall`` fails, the controller
   continues in best-effort mode and emits a warning rather than
   aborting; see the warning text for the exact remediation.

2. Per-boot RT tuning
~~~~~~~~~~~~~~~~~~~~~

These knobs revert on every reboot. Run them once per session, or
wire them into a systemd one-shot / ``rc.local``:

.. code-block:: bash

   # 1. CPU governor → performance (suppress P-state µs-jitter)
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
       echo performance > "$g"
   done'
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # expect: performance

   # 2. Lift the SCHED_FIFO 95% throttle (default 950000/1000000)
   sudo sysctl -w kernel.sched_rt_runtime_us=-1
   cat /proc/sys/kernel/sched_rt_runtime_us                    # expect: -1

   # 3. Disable NIC interrupt coalescing on the Franka link
   sudo ethtool -C eno1 rx-usecs 0 tx-usecs 0                  # change eno1

Use ``ip -br a`` to confirm the actual NIC name. To persist the
``rt_runtime`` change across reboots:

.. code-block:: bash

   echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

.. note::

   ``requirements/install.sh embodied --env franka-franky`` (which
   internally invokes ``requirements/embodied/franky_install.sh``)
   prints these three commands at the end of the install. This
   section is the authoritative copy.

3. RLinf + franky
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # One command does it all: system deps (rt-tests, ethtool, eigen,
   # pinocchio, ... — install.sh invokes franky_install.sh internally,
   # which needs sudo) + RLinf Python deps + franky-control wheel.
   # Non-root users will get a sudo password prompt mid-install.
   bash requirements/install.sh embodied --env franka-franky --use-mirror

   source .venv/bin/activate

The ``--env franka-franky`` target pins the franky path
(``franky-control >= 0.15.0`` from PyPI) and **skips** the legacy
``serl_franka_controllers`` ROS / catkin build used by
:doc:`franka`. The ``--use-mirror`` flag is for mainland China users
(switches PyPI / GitHub / HuggingFace mirrors).

.. note::

   ``requirements/install.sh embodied --env franka-franky`` is a
   **one-command install**: uv venv → invokes ``franky_install.sh``
   for system deps (``rt-tests``, ``ethtool``, ``cmake``,
   ``libeigen3-dev``, ``libpoco-dev``, ``libfmt-dev``, pinocchio,
   ...) → installs the ``franky-control`` wheel. **No need to run**
   ``franky_install.sh`` standalone.

**libfranka is not installed by either script.** ``franky-control`` is
distributed on PyPI as manylinux wheels with **libfranka bundled
inside the wheel** — a standard Ubuntu + mainstream Python setup
(default match: Python 3.11 + libfranka 0.15.x) works out of the
box, no extra steps required.

Only when your Python / system ABI does not match the wheel does
``pip`` fall back to a source build of **franky-control** (which
also handles its libfranka dependency in the process).
``franky_install.sh`` has already preinstalled cmake / eigen / poco /
fmt / pinocchio for this fallback; install the source build of franky
directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/TimSchneider42/franky.git

For full source-build options, libfranka submodule versions, and
CMake configuration, see the franky upstream README:
https://github.com/TimSchneider42/franky .

.. warning::

   **Avoid libfranka 0.18.0 specifically.** Franka's official 0.18.0
   release notes flag a regression in the impedance / Cartesian
   control path; under the joint / Cartesian impedance trackers we
   use, the arm presents as severely under-torqued — limp, sagging
   under its own gravity, unable to track even gentle GELLO motion.
   Pick whichever ``libfranka`` version matches your Franka firmware
   per the official `compatibility matrix
   <https://frankarobotics.github.io/docs/compatibility.html>`_,
   just **not 0.18.0** (verified on 0.19.0). Check
   ``franky.__libfranka_version__`` after install if you are unsure.

4. GELLO (env-worker node)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both GELLO USB-FTDI cables plug into the env-worker node (**node 0**
in the shipped placement) and stay there during data collection.
``DualGelloJointIntervention`` opens both serial ports from inside
the env-worker process and reads them at ~1 kHz — routing through
the LAN to a GELLO physically wired to node 1 would blow the
real-time budget, drop samples, and cause tracker reference jumps.

For the actual install commands (``gello`` + ``gello-teleop`` +
USB-FTDI permission, with the rationale for why only the
``DynamixelSDK`` submodule is initialised), see :doc:`franka_gello`.
Run those commands on **node 0 only**, in the same venv as RLinf —
``DualGelloJointIntervention`` imports both packages in-process when
the env wrapper stack is built.

5. Foot pedal
~~~~~~~~~~~~~

The PCsensor FootSwitch is wired so its three pedals send Linux key
codes ``a`` / ``b`` / ``c`` (re-flashable via the vendor-supplied
Windows tool — flash once, the codes are stored in firmware and
persist across reboots). Verify and grant access:

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   #  expect: usb-PCsensor_FootSwitch-event-kbd → ../eventXX

   sudo chmod 666 /dev/input/eventXX

   # Export BEFORE `ray start` so Ray captures the path:
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

The ``KeyboardListener`` reads ``evdev`` directly, supports auto-reopen
on ``ENODEV`` (handles a USB hiccup mid-session), and uses an
edge-triggered press queue so a tap shorter than the polling period
is never missed.


Hardware verification
---------------------

Before bringing up Ray, smoke-test each hardware piece per node.

Cameras
~~~~~~~

.. code-block:: bash

   # RealSense: enumerate the bus and confirm USB-3 negotiation.
   rs-enumerate-devices | grep -E "Name|Serial|USB Type"

   # Lumos (XVisio vSLAM): confirm both /dev/v4l/by-id nodes exist.
   ls /dev/v4l/by-id/

   # USB topology: check that Lumos and RealSense negotiate at 5000M.
   # Anything dropping to "480M" is USB-2 fallback (bad cable / hub).
   lsusb -t

GELLO
~~~~~

**1. Find the FTDI serial paths and tell left from right**

GELLO talks through an FTDI USB→Dynamixel adapter; each GELLO chain
shows up under ``/dev/serial/by-id/`` as
``usb-FTDI_..._<unique_id>-if00-port0`` (``<unique_id>`` is the
FTDI chip serial — unique per cable, stable across reboots):

.. code-block:: bash

   # List every FTDI converter on the bus.
   ls -l /dev/serial/by-id/ | grep -i ftdi

With both GELLO chains plugged in you will see two candidates.
**Disambiguate left vs right** with the unplug-and-recheck trick:

.. code-block:: bash

   # Plug only the LEFT GELLO, record what shows up.
   ls /dev/serial/by-id/ | grep -i ftdi    # → LEFT_PATH
   # Now plug the RIGHT GELLO too.
   ls /dev/serial/by-id/ | grep -i ftdi    # → the new entry is RIGHT_PATH

Write down the two ``<unique_id>`` values. **Put both by-id paths in
the yaml** (``env.eval.left_gello_port`` / ``right_gello_port`` in
``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``):

.. code-block:: yaml

   env:
     eval:
       left_gello_port:  /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0
       right_gello_port: /dev/serial/by-id/usb-FTDI_..._<RIGHT_ID>-if00-port0

**2. Smoke-test each GELLO for real-time joint reads**

.. code-block:: bash

   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0

You should see continuously updating joint readings as you move the
GELLO arm. If readings freeze or jump by ±2π, run the calibration
toolkit (covered in the next section). Repeat with ``<RIGHT_ID>``
for the right arm.

Per-arm Franka link
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   FRANKA_ROBOT_IP=172.16.0.2 \
   FRANKA_GRIPPER_TYPE=robotiq \
   FRANKA_GRIPPER_PORT=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id>-if00-port0 \
       python toolkits/realworld_check/test_franky_controller.py

Inside the REPL:

* ``getjoint`` — print current joint angles
* ``home`` — synchronous reset to ``HOME_JOINTS``
* ``hold 30`` — hold for 30 s, listen for buzzing
* ``stream 4 0.001 500`` — push 500 J4 += 0.001 rad commands at 1 kHz
  (streaming preemption stress test)
* ``impedance 300 300 300 300 150 80 30`` — drop joint impedance, repeat
* ``open`` / ``close`` — gripper sanity

Run this on each node against its own arm: a passing run is silent
on hold, sustains ≥ 800 Hz on ``stream 4 0.001 1000``, and lets
``home`` recover cleanly from any legal start pose. **Both arms
must pass before you bring up Ray.**


GELLO calibration
-----------------

1. **Calibrate** (once per GELLO unit, or after replacing a motor):

   .. code-block:: bash

      python toolkits/realworld_check/test_gello.py calibrate

   The script moves the robot to two known poses (``POSE_A`` =
   Franka home, ``POSE_B`` = π/4 multiples), prompts you to physically
   match the GELLO leader to each pose, then solves
   ``joint_signs`` and ``joint_offsets`` from the difference. Output
   is a paste-ready ``DynamixelRobotConfig`` block to drop into
   ``gello_software/gello/agents/gello_agent.py``::

       "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0":
           DynamixelRobotConfig(
               joint_ids=(1, 2, 3, 4, 5, 6, 7),
               joint_offsets=(...),
               joint_signs=(...),
               gripper_config=(8, ..., ...),
               baudrate=1_000_000,
           ),

   The ``gello`` package is editable-installed, so no reinstall is
   needed after pasting — just restart the next process that imports
   ``gello``.

2. **Align** (run when the GELLO leader and the arm visibly disagree
   — e.g. someone hand-moved the arm, the rig has been idle for a
   while, or you just want to confirm before a fresh collection
   session):

   .. code-block:: bash

      python toolkits/realworld_check/test_gello.py align-sequential

   Drives the robot to a fixed alignment HOME pose (J4 = −π/2,
   J6 = +π/2 etc.), then walks you through aligning J1 → J7 one at
   a time with a per-joint progress bar and live deltas. As soon as
   each joint stays within ±0.10 rad for 8 consecutive frames, it
   advances to the next.

Both scripts auto-discover the local Robotiq port by globbing
``/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*-if00-port0``, so you do
not need to know whether you are on the left or right node.


Hardware YAML
-------------

The dual-Franka hardware configuration lives in
``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
(joint collection) and
``examples/embodiment/config/env/realworld_dual_franka_rot6d.yaml``
(rot6d eval). Use these as templates — placeholders to replace for
your rig:

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP`` — the per-arm FCI IPs
  (e.g. ``172.16.0.2``).
* ``BASE_CAMERA_SERIAL`` — base-camera serial (RealSense uses what
  ``rs.context().devices`` reports; replace with the SDK serial that
  matches ``base_camera_type``).
* ``LEFT_CAMERA_SERIAL`` / ``RIGHT_CAMERA_SERIAL`` — wrist-camera
  serials (Lumos uses the
  ``/dev/v4l/by-id/usb-XVisio_..._video-index0`` path; replace with
  the SDK serial that matches ``*_camera_type``).
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION`` —
  the Robotiq 2F-85 RS-485 USB ports. Always use
  ``/dev/serial/by-id/usb-FTDI_..._<id>-if00-port0``; **do not**
  use ``/dev/ttyUSB*`` (the index reshuffles on reboot / hot-plug).
* ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT`` — GELLO leader
  ``/dev/serial/by-id`` paths (both plugged into the env-worker
  node, i.e. ``node_rank: 0``).
* ``ee_pose_limit_min`` / ``ee_pose_limit_max`` in the override
  block — set to your rig's workspace safety boxes; row 0 is the
  left arm, row 1 is the right arm, each row is
  ``[x, y, z, roll, pitch, yaw]``.

``left_controller_node_rank`` / ``right_controller_node_rank``
(default ``0`` / ``1``, one arm per node) and ``node_rank`` (where
the env worker + cameras run) usually do not need to change.


Ray cluster bring-up
--------------------

Ray captures the active Python interpreter and the *exported
environment variables* when ``ray start`` runs, and worker actors
inherit that snapshot. Packages added to the venv after ``ray start``
are picked up at next import (Ray does not freeze ``site-packages``),
but env vars are not — anything you forget to export before
``ray start`` will be missing from the worker's environment forever.
Order:

1. **On every node**: activate the venv, export
   ``RLINF_NODE_RANK``, export ``RLINF_COMM_NET_DEVICES`` (optional,
   only needed when the host has multiple NICs — pin the one the
   two nodes use to talk to each other), and export
   ``RLINF_KEYBOARD_DEVICE`` if this node owns the foot pedal.
   Verify ``franky``, ``gello``, ``gello_teleop`` import.
2. **Then** ``ray start`` — head on node 0, worker on node 1.

On each node, activate the venv, export the rank-specific env vars,
then run ``ray start``. ``HEAD_IP`` / ``WORKER_IP`` are the LAN IPs the
two machines use to reach each other (not ``127.0.0.1`` and not the
public IP).

.. code-block:: bash

   # node 0 (Ray head)
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX  # if pedal lives here

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1 (Ray worker)
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1

   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

Verify on node 0:

.. code-block:: bash

   ray status
   # expected: 2 nodes, both ALIVE, with the cluster GPU/CPU resources
   # you expect.

.. warning::

   The two physical nodes have **independent checkouts**. After every
   code change on node 0, sync to node 1 (``rsync -av --delete RLinf/
   <node1>:/path/to/RLinf/``) **and** restart Ray on the affected
   node so the new code is captured by Ray. Forgetting this leads to
   cryptic ``ImportError`` or "feature works on node 0 but not on
   node 1" symptoms.


Data Collection (GELLO joint-space)
------------------------------------

The collection path uses ``DualFrankaJointEnv-v1`` with
``teleop_direct_stream: true``. A daemon thread inside
``DualGelloJointIntervention`` reads the GELLO Dynamixel servos at
~1 kHz and pushes joint targets straight into both
``FrankyController`` actors (which forward them to franky's
``JointImpedanceTracker``). ``env.step`` runs at 10 Hz and only reads
state, fires gripper open/close on edge transitions, and dispatches
camera captures — it does **not** call ``move_joints``.

Why direct-stream and not env-step-gated? At 10 Hz, sampling a
freehand teleop trajectory under-samples high-frequency wrist motion.
The 1 kHz daemon path samples the operator's actual hand motion at
GELLO's native rate, then env.step reads the *resulting* joint state
at 10 Hz — so the dataset captures what the operator did, not a
1-tap-per-100ms aliased view of it. The 10 Hz state read is what
π₀.₅ then sees as input.

Configuration
~~~~~~~~~~~~~

Use the shipped config:
``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``.

Key fields you will edit before each session:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - YAML field
     - Meaning
   * - ``runner.num_data_episodes``
     - Target episode count. Combined with
       ``data_collection.resume``, this is the *total target across
       all sessions*, not just this session.
   * - ``env.eval.left_gello_port`` / ``right_gello_port``
     - Override here if you swap GELLO units this session. Otherwise
       inherits from the env yaml.
   * - ``env.eval.override_cfg.task_description``
     - Prompt recorded into every frame's ``task`` field.
   * - ``env.eval.override_cfg.joint_action_mode``
     - ``absolute`` (1:1 GELLO mapping for collection); ``delta`` for
       offline RL on the same env class.
   * - ``env.eval.override_cfg.teleop_direct_stream``
     - ``true`` for the 1 kHz GELLO daemon path. Setting this to
       ``false`` falls back to env.step gating and is **not the
       recommended collection path**.
   * - ``data_collection.save_dir``
     - Base directory for the dataset. Each session by default writes
       to ``${runner.logger.log_path}/collected_data``; override on
       the command line to accumulate sessions under one root.
   * - ``data_collection.resume``
     - ``true`` to seed the episode counter from existing ``id_*``
       shards under ``save_dir/rank_0``.

Running
~~~~~~~

Two terminals once Ray is up.

**Terminal 1** — launcher (on node 0):

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka 2>&1 \
        | tee logs/collect.log

**Terminal 2** — live progress monitor (on node 0):

.. code-block:: bash

   python toolkits/realworld_check/collect_monitor.py logs/collect.log

The monitor exists because the collector runs as a Ray worker, whose
stdout is batched (~500 ms) by Ray's log monitor — that batching
breaks ``tqdm``'s ``\r`` in-place refresh. The monitor tails the log
file in its own TTY and renders a clean tqdm bar showing success
count, latest keyboard event, and last reward. By default it replays
the existing log at startup so episodes saved before the monitor
came up are reflected in the bar's initial position; pass
``--no-replay`` to start from EOF instead. ``--source=worker``
(default ``auto``) tails the Ray per-worker stdout file under
``/tmp/ray/session_latest/logs/worker-*-<pid>.out`` to bypass log
monitor batching entirely (~1–2 min faster), falling back to the
tee log when the worker file is on a different node.

Per-episode workflow
~~~~~~~~~~~~~~~~~~~~

Once both arms are tracking GELLO (``test_gello.py
align-sequential`` reports "ALL JOINTS ALIGNED"):

1. **(pre)** The arms align to the GELLO operator pose at every
   ``reset`` (``KeyboardStartEndWrapper`` +
   ``DualGelloJointIntervention`` skip the home slew via
   ``options["skip_reset_to_home"]=True``). The arms stay where the
   operator's hands hold them.
2. **press ``a``** to begin recording the current pose as frame 0.
3. **demonstrate the task.** Per-frame data is buffered. The robot
   tracks the GELLO leader at 1 kHz; cameras capture at 10 Hz.
4. **press ``b``** at sub-task boundaries — increments the per-frame
   ``segment_id`` (debounced at 1 s; back-to-back presses are
   ignored). Use this to mark "approach" / "grasp" / "transfer" /
   "place" so a downstream policy can be conditioned on segment id.
5. **press ``c``** to mark success → reward = 1.0,
   ``terminated=True``, ``CollectEpisode`` flushes the buffer to a
   LeRobot shard.
6. **press ``a`` again** during recording to **abort** — drops the
   buffer, returns to ``pre`` phase. The arms stay where they are
   (no home reset) so the operator can immediately re-attempt
   without GELLO discontinuity.

Output format
~~~~~~~~~~~~~

LeRobot v2.1, one shard per session under
``<save_dir>/rank_0/id_{N}/``:

* ``meta/info.json`` — feature schema. ``state`` is fixed-size
  ``[68]``; ``actions`` is ``[16]`` for joint or ``[20]`` for rot6d.
* ``meta/episodes_stats.jsonl`` — per-episode min / max / mean / std
  for ``state`` and ``actions``.
* ``data/episode_NNNNNN.parquet`` — per-step rows.

Per-frame fields:

* ``state`` — ``DualFrankaJointEnv.STATE_LAYOUT`` flat concat
  ``[gripper_position(2), joint_position(14), joint_velocity(14),
  tcp_force(6), tcp_pose(14), tcp_torque(6), tcp_vel(12)]`` = 68.
  The first 2 slots are deliberately laid out as
  ``[L_grip, R_grip]`` to match the rot6d-policy's
  ``_rearrange_state`` slicing assumption.
* ``actions`` — what the GELLO daemon dispatched at each step
  (16-D for joint mode: ``[L_jpos(7), L_grip, R_jpos(7), R_grip]``).
* ``image`` — ``left_wrist_0_rgb`` (the ``main_image_key``).
* ``wrist_image-0`` / ``wrist_image-1`` — fanned-out per-arm wrist
  views via ``CollectEpisode._expand_multi_view_images``.
* ``extra_view_image-0`` / ``extra_view_image-1`` — base + right
  wrist views, ordered as ``("base_0_rgb", "right_wrist_0_rgb")``.
  The order is asserted in
  ``DualFrankaRot6dInputs._extract_extra_views`` so a rig rename
  fails loudly instead of silently swapping camera meanings.
* ``task`` — the prompt for this episode's task.
* ``is_success`` — sticky flag; ``True`` for every frame of an
  episode that ended via pedal ``c``.
* ``done`` — only the *last* frame of an episode has ``True``.
* ``intervene_flag`` — always ``True`` for collection (the GELLO
  daemon's command is the action).
* ``segment_id`` — uint8; advances on pedal ``b``.

Resume
~~~~~~

Set ``data_collection.resume: true`` and re-launch with the same
``save_dir`` — ``CollectEpisode._count_existing_lerobot_episodes``
sums ``total_episodes`` across existing ``id_*`` shards (skipping
malformed shards, so an aborted session that left a corrupt shard
does not break resume), and the new session writes to a fresh
``id_{N}`` shard so previously-finalised data is never touched.

The progress bar's initial position is seeded from the existing
count, so ``num_data_episodes: 200`` plus 50 already-saved successes
means the new session targets 150 more.


Backfill rot6d and norm_stats
-----------------------------

Collection produces 16-D joint actions and 68-D state; π₀.₅ SFT
expects 20-D rot6d for both state and action
(``[xyz(3) + rot6d(6) + grip(1)] × 2``). Convert the dataset offline
before SFT with:

.. code-block:: bash

   export PYTHONPATH=$(pwd)
   python toolkits/dual_franka/backfill_rot6d.py \
       --src $HF_LEROBOT_HOME/<repo_id>/joint_v1 \
       --dst $HF_LEROBOT_HOME/<repo_id>/rot6d_v1

The script does three things: rewrites the first 20 state slots into
the rot6d layout (sliced directly from the tcp_pose columns already
in state, no FK); widens actions 16 → 20 with xyz / rot6d taken from
the **next frame's** tcp_pose as an approximation of the operator's
target at the current frame, while the gripper slots reuse the
original triggers; updates the parquet schema's ``actions.length``
and recomputes per-episode stats. Pointing the script at an
already-backfilled dataset errors out instead of double-writing.

Once backfilled, compute norm stats:

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_rot6d \
       --repo-id <repo_id>/rot6d_v1

This iterates the dataset through the SFT data pipeline
(``RepackTransform`` → ``DualFrankaRot6dInputs`` →
``RigidBodyDeltaActions``) and saves ``norm_stats.json`` under
``<openpi_assets_dirs>/<data_config.repo_id>/``. The same
``<repo_id>`` becomes the lookup key the rollout worker uses to
load these stats at deployment — see "Checkpoint / norm_stats
lock-step" below for the full path-resolution rule.

The norm stats must be recomputed **after** backfill, not before —
they need to see the body-frame deltas the policy will actually
predict, not the absolute targets on disk.


SFT (π₀.₅, rot6d_v1)
--------------------

Configuration
~~~~~~~~~~~~~

``examples/sft/config/realworld_sft_openpi_dual_franka_rot6d.yaml``. Edit
before launch:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Set to
   * - ``data.train_data_paths``
     - LeRobot root containing your backfilled rot6d_v1 dataset.
       This value is exported as ``HF_LEROBOT_HOME`` by
       ``train_vla_sft.py`` before validation, so openpi's
       data loader picks it up automatically.
   * - ``actor.model.model_path``
     - π₀ / π₀.₅ base ckpt (the torch-converted weights, e.g.
       ``checkpoints/torch/pi05_base/``).
   * - ``actor.model.action_dim``
     - ``20`` (must match the rot6d data layout).
   * - ``actor.model.num_action_chunks``
     - ``20`` (matches the model's ``action_horizon`` from the
       ``pi05_dualfranka_rot6d`` TrainConfig).
   * - ``actor.model.openpi.config_name``
     - ``pi05_dualfranka_rot6d``.
   * - ``actor.optim.lr``
     - ``7.91e-6`` is a reasonable default for π₀.₅ on this dataset.
   * - ``actor.fsdp_config.sharding_strategy``
     - ``full_shard`` (``hybrid_shard`` if you have >8 GPUs and
       want the inter-replica all-reduce instead of all-gather).
   * - ``runner.save_interval``
     - ``500`` (steps); checkpoints land in
       ``${runner.logger.log_path}/checkpoints/global_step_<N>/``.

Launch
~~~~~~

.. code-block:: bash

   # Single node, 4 GPU slots — cluster.num_nodes: 1,
   # component_placement.actor,env,rollout uses GPUs 0..3.
   bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_rot6d

The runner writes checkpoints every ``runner.save_interval`` steps
(default 500) to ``<log_path>/checkpoints/global_step_<N>/`` with
this layout:

.. code-block:: text

   <log_path>/checkpoints/global_step_<N>/
   ├── actor/
   │   └── model_state_dict/
   │       └── full_weights.pt
   └── <asset_id>/                        # e.g. "<your-hf-user>/<your-dataset>"
       └── norm_stats.json                # pinned norm stats for inference

Real-world deployment reads the policy weights from
``<model_path>/actor/model_state_dict/full_weights.pt`` and norm
stats from ``<model_path>/<asset_id>/norm_stats.json``.

Real-world deployment
---------------------

Same Ray cluster as collection. Different entry script + config.

Configuration
~~~~~~~~~~~~~

``examples/embodiment/config/realworld_eval_dual_franka.yaml``.
Placeholders are flagged with ``# Replace:`` comments. Most-edited
fields:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Set to
   * - ``rollout.model.model_path``
     - ``<sft_log>/checkpoints/global_step_<N>/`` — must contain
       ``actor/model_state_dict/full_weights.pt`` and
       ``<data_config.repo_id>/norm_stats.json`` (see
       "Checkpoint / norm_stats lock-step" below for how
       ``data_config.repo_id`` is resolved).
   * - ``actor.model.openpi_data.repo_id``
     - Forwarded as ``data_kwargs`` to ``get_openpi_config``; this
       overrides ``data_config.repo_id``, which is the lookup key
       for ``norm_stats.json`` at deployment. Keep it consistent
       with what ``calculate_norm_stats.py --repo-id`` was given.
   * - ``env.eval.override_cfg.task_description``
     - Prompt the policy was trained against.
   * - ``env.eval.override_cfg.joint_reset_qpos``
     - Recompute from your SFT dataset's first-frame joint means;
       stale values push the initial obs out of training distribution.
   * - ``env.eval.override_cfg.target_ee_pose`` / ``reset_ee_pose``
     - Match the workspace used at collection.
   * - ``cluster.node_groups[*].env_configs[0].python_interpreter_path``
     - Path to the openpi venv's Python on node 0 (the env worker /
       rollout actor read this to ensure imports resolve).

Hardware ``configs`` should be identical to the collection yaml —
same IPs, same camera serials, same gripper connections. The
wrappers attach based on ``env.eval.use_*`` flags, so the only
non-hardware difference between collection and eval yaml is:

* ``use_gello_joint: false`` (collection: ``true``)
* ``keyboard_reward_wrapper: eval_control`` (collection:
  ``start_end``)
* ``use_relative_frame: false`` — required for rot6d eval, since
  ``DualRelativeFrame`` would corrupt the rot6d state.

Launch
~~~~~~

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

   # Optional Hydra overrides:
   #   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
   #        rollout.model.model_path=/sft/global_step_5000 \
   #        env.eval.override_cfg.task_description="pour water"

Eval workflow (per episode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``KeyboardEvalControlWrapper`` swaps the keyboard wrapper for one
tuned to autonomous rollout:

1. After ``env.reset()``, both arms hold the reset pose.
   ``env.step()`` is intercepted in **idle** mode — it does *not*
   forward to the inner env (so the impedance controller keeps the
   target from the last reset; the arms stay physically still while
   you stage the workpiece). The wrapper still returns the most
   recent observation so the policy's chunked-rollout loop keeps
   cycling without committing fresh joint commands.
2. Press ``a`` → wrapper switches to **running**. The next
   ``env.step`` forwards to the policy's chunked rollout.
3. Press ``c`` → success: ``terminated=True``, ``reward=1.0``,
   ``info["eval_result"]="success"``. The wrapper internally calls
   ``env.reset()`` so the arms drive back to home immediately, then
   sits idle again — this is what makes the pedal feel "live"
   even when the eval ``env_worker`` runs with ``auto_reset=False``.
4. Press ``b`` → failure: same as success but ``reward=0.0``,
   ``info["eval_result"]="failure"``.
5. While running, the wrapper forces ``terminated`` / ``truncated``
   to ``False`` unless the pedal fires — the env's own
   ``max_episode_steps`` does not cut off the policy. Set
   ``max_episode_steps`` large enough that the pedal is always the
   boundary owner (the shipped config uses ``10000``).

Checkpoint / norm_stats lock-step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The single most common deployment failure is a ``norm_stats``
mismatch. The rollout worker resolves the norm_stats path inside
``rlinf/models/embodiment/openpi/__init__.py``::

   pinned_path = <model_path>/<data_config.asset_id>/norm_stats.json
   if pinned_path exists:
       use it
   else:
       fall back to data_config.norm_stats with a loud warning

``data_config.asset_id`` is what
``DualFrankaRot6dDataConfig.create()`` resolves at SFT time (it
inherits the ``AssetsConfig.asset_id`` field, which falls back to
``data_config.repo_id`` when not set explicitly). The same key is
used by ``calculate_norm_stats.py``, which writes its output under
``<openpi_assets_dirs>/<data_config.repo_id>/``. So the path under
``<model_path>/...`` must match what the SFT pipeline actually used.

In practice:

* If you keep the shipped ``actor.model.openpi_data.repo_id``
  (``<your-hf-user>/<your-dataset>``), norm_stats live under
  ``<model_path>/<your-hf-user>/<your-dataset>/norm_stats.json``.
* If you override ``actor.model.openpi_data.repo_id`` (passed
  through as ``data_kwargs``) to point at a local backfilled
  dataset, ``data_config.repo_id`` is replaced and the lookup key
  becomes the new value. **Run** ``calculate_norm_stats.py``
  **with the same** ``--repo-id`` **and copy the result into**
  ``<model_path>/<that_repo_id>/norm_stats.json``.

Verify before launch:

.. code-block:: bash

   # Whatever value resolves on this rollout run — grep the SFT log for
   # the exact path the rollout worker is going to look up:
   grep "norm_stats" <sft_log>/run_embodiment.log | tail
   # Or just confirm the file exists at the path the model_path implies:
   find <model_path> -maxdepth 3 -name norm_stats.json
   ls <model_path>/actor/model_state_dict/full_weights.pt

Mismatched stats silently produce out-of-distribution states; the
policy will collapse to a single canned trick (drift to a corner,
stuck-open gripper, etc.) without any error message. The fallback
path *does* log a ``"norm_stats fallback: ... verify they match
training or inference will be wrong"`` warning — grep for it
before you assume the rollout is healthy.


Troubleshooting
---------------

**GELLO daemon never starts**
   Power-cycle the GELLO arm, re-plug the FTDI, then verify both
   sides produce continuous Dynamixel readings with
   ``python -m gello_teleop.gello_expert --port /dev/...``.

**Ray worker silently dies on import**
   From the same shell that ran ``ray start``, run
   ``which python && python -c "import franky, gello, gello_teleop"``
   to confirm the venv and installed packages match; the actual
   error is in ``/tmp/ray/session_latest/logs/worker-*.err``.

**One arm hangs at reset**
   On the controller node, ``ping -c 100 <robot_ip>``; if packets
   drop, power-cycle that arm and re-launch.

**``move_joints`` rejects every command immediately after boot**
   Release the white User Stop button → open the Desk web UI
   (``http://<robot_ip>/desk/``) → click *Activate FCI* → wait for
   the joint LEDs to go from white to blue → re-launch.

**GELLO daemon and env reset race each other**
   Hold the GELLO leader steady on its rest plate during reset, and
   wait for ``KeyboardStartEndWrapper`` to report the reset finished
   before continuing.

**"Permission denied" on the foot pedal**
   ``sudo chmod 666 /dev/input/eventXX``; to make it persistent add
   a udev rule (``KERNEL=="event*", SUBSYSTEM=="input",
   ATTRS{name}=="PCsensor FootSwitch", MODE="0666"``).

**RealSense falls back to USB 2.x**
   Replace the USB cable and plug into a blue (USB-3) port;
   confirm ``lsusb -t`` shows ``5000M`` rather than ``480M``.

**Lumos cold-start fails on first invocation**
   Re-plug the USB cable.

**Eval idle forever**
   Confirm ``RLINF_KEYBOARD_DEVICE`` points at the correct
   ``/dev/input/eventXX`` and ``chmod 666`` is still in effect,
   then press pedal ``a``.

**Tracking jitter at deployment**
   Lower ``RLINF_CART_K_R``, raise ``RLINF_CART_GAINS_TC``, tighten
   ``RLINF_CART_MAX_STEP_RAD``; as a last resort, shorten the
   policy's action-chunk horizon.

**``norm_stats.json`` not found at deployment**
   Copy ``norm_stats.json`` from where ``calculate_norm_stats.py``
   wrote it (``<openpi_assets_dirs>/<repo_id>/``) to
   ``<model_path>/<repo_id>/``; grep for the
   ``"norm_stats fallback"`` warning first to confirm whether the
   fallback path actually fired.

**collect_monitor frozen**
   Add ``2>&1 | tee logs/collect.log`` to the launcher; pass
   ``--source=worker`` to the monitor when the env worker is on a
   different node.

**``sched_setaffinity failed`` warning at controller start**
   Either run on a 6+ core machine, or grant the capability with
   ``sudo setcap cap_sys_nice=eip $(which python)`` on the venv
   interpreter.

**Both arms move at reset but only one tracks GELLO afterwards**
   Run ``python toolkits/realworld_check/test_gello.py align-check``
   on each GELLO to confirm both produce continuous readings, then
   re-launch.
