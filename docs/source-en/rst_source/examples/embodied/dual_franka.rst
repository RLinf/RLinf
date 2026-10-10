Using Dual Franka
=================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy.jpg
   :align: center
   :width: 80%
   :alt: Dual-Franka deployment

   Dual-Franka data collection, fine-tuning, and deployment workflow.

Run the supported dual-Franka workflow: collect joint-space demonstrations with GELLO, convert them to tcp_rot6d data, fine-tune OpenPI π₀.₅, and deploy the checkpoint back to the host that drives both arms.

Overview
--------

Build a dual-arm dataset, train π₀.₅, and deploy on one host that drives both arms.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · eval-only deployment

   .. grid-item-card:: Tasks
      :text-align: center

      Dual-arm manipulation

   .. grid-item-card:: Hardware
      :text-align: center

      2× Franka · 1 host · GELLO

| **You'll do:** install franky deps → collect GELLO demos → convert rot6d data → run SFT → deploy eval config.
| **Prerequisites:** :doc:`franka` · :doc:`franka_gello` · two Franka arms · OpenPI assets.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - Task
     - Config / entry point
     - Description
   * - Collection
     - ``realworld_collect_data_gello_joint_dual_franka``
     - Collect dual-arm joint trajectories.
   * - SFT
     - ``realworld_sft_openpi_dual_franka_tcp_rot6d``
     - Fine-tune π₀.₅ on tcp_rot6d actions.
   * - Deployment
     - ``realworld_eval_dual_franka``
     - Run eval-only deployment on the host.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - Field
     - Description
   * - Observation
     - LeRobot ``image`` is the base camera ``base_0_rgb``; the wrists are extra views. State is the dual-arm robot state.
   * - Action
     - Dual-arm tcp_rot6d: ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``.
   * - Reward
     - Evaluation success signal or operator-gated deployment outcome.
   * - Prompt
     - Task text in the OpenPI data/config metadata.

Installation
------------

Robot Nodes
~~~~~~~~~~~

Run the robot-node installation once, on the host that drives both arms. Dual-arm
Franka always drives the arms through Franky, the backend that the default
``franka`` environment installs. The installer downloads a prebuilt Franky wheel
with libfranka bundled; these wheels exist only for libfranka ``0.15.0`` and
``0.19.0`` (the default) on x86_64. Set ``LIBFRANKA_VERSION`` to the one that the
official `Franka compatibility
matrix <https://frankarobotics.github.io/docs/compatibility.html>`_ lists for your
firmware. For other firmware, build a Franky wheel against the matching libfranka
and pass its path or URL in ``FRANKY_WHEEL``; the legacy ROS backend covers other
libfranka versions only for single-arm Franka.

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   export LIBFRANKA_VERSION=0.19.0       # or 0.15.0, matching the firmware
   bash requirements/install.sh embodied --env franka --use-mirror
   source .venv/bin/activate

Install GELLO dependencies on this host by following :doc:`franka_gello`.
The two GELLO leaders must stay on this host; do not route their
1 kHz stream over the LAN.

Real-time prerequisites
~~~~~~~~~~~~~~~~~~~~~~~

The ``franka`` environment uses franky/libfranka to communicate with each Franka
at 1 kHz. A PREEMPT_RT kernel is recommended. The RLinf installer installs
runtime dependencies only; configure the PREEMPT_RT kernel and real-time
permissions according to the official `Franka real-time kernel guide
<https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_.

The ``DualFranka`` hardware config defaults to ``realtime_config: ignore``, so
both arms also start on a kernel without PREEMPT_RT and RLinf logs a warning.
On such a kernel the 1 kHz control loop can miss deadlines under load and
trigger a robot reflex. Set ``realtime_config: enforce`` to refuse a kernel
without PREEMPT_RT.

The two arms use different FCI IPs and connect to this one host, either
directly (one cable from each arm into a NIC on the host) or through a switch
(both arms and the host on the same switch). Run the example below on that
host before launching RLinf. Do not start Ray. Replace ``<FRANKA_NIC>`` with
each NIC that faces an arm; when both arms share one NIC through a switch, use
that NIC. Ping ``LEFT_ROBOT_IP`` and ``RIGHT_ROBOT_IP``.

.. code-block:: bash

   # Per-boot tuning.
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
       echo performance > "$g"
   done'
   sudo sysctl -w kernel.sched_rt_runtime_us=-1
   sudo ethtool -C <FRANKA_NIC> rx-usecs 0 tx-usecs 0 2>/dev/null || true

   # Optional: keep the RT scheduling budget setting after reboot.
   echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

   # Check realtime permissions and the robot link before running RLinf.
   uname -a | grep -o PREEMPT_RT
   ulimit -r
   ulimit -l
   sudo cyclictest -p 80 -t 4 -i 1000 -l 300000 -m
   ping -c 1000 -i 0.001 <LEFT_ROBOT_IP> | tail -3
   ping -c 1000 -i 0.001 <RIGHT_ROBOT_IP> | tail -3

``ulimit -r`` should report ``99`` or ``unlimited``; ``ulimit -l`` should report
``unlimited``. Re-apply the per-boot tuning after every workstation reboot.

Training Node
~~~~~~~~~~~~~

Install OpenPI dependencies on the remote GPU training cluster that will
perform SFT:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror
   source .venv/bin/activate


Configuration
-------------

Use the repository-provided configs and replace the required parameters:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Config
     - Purpose
   * - ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
     - GELLO joint-space collection
   * - ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
     - π₀.₅ SFT on converted tcp_rot6d data
   * - ``examples/embodiment/config/realworld_eval_dual_franka.yaml``
     - Real-world policy deployment
   * - ``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
     - Shared joint-space hardware defaults
   * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
     - Shared tcp_rot6d hardware defaults

Replace the placeholders marked with ``# Replace:``:

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``: the two different FCI IPs, both
  reachable from this host.
* ``BASE_CAMERA_SERIAL``, ``LEFT_CAMERA_SERIAL``, ``RIGHT_CAMERA_SERIAL``:
  camera serials or stable ``/dev/v4l/by-id`` paths.
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION``: stable
  ``/dev/serial/by-id`` paths for the Robotiq adapters.
* ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT``: stable ``/dev/serial/by-id``
  paths for the two GELLO leaders.
* ``TASK_DESCRIPTION``: the natural-language task prompt used for
  collection, SFT, and deployment.
* ``SFT_DATASET_REPO_ID``: the converted dataset ID, usually
  ``<repo_id>/tcp_rot6d_v1``.
* ``MODEL_PATH``: deployment checkpoint directory on the host.
* The shared ``realworld_dual_franka_tcp_rot6d`` TCP xyz limits are
  x in [0.3, 0.9], y in [-0.4, 0.4], and z in [0.02, 0.7].
  ``target_ee_pose`` must sit inside that box.


Hardware Checks
---------------

Run these checks on the host before launching. Do not start Ray.

Foot pedal
~~~~~~~~~~

Use the vendor tool once to configure the PCsensor FootSwitch keys as
``a`` / ``b`` / ``c``. Then on the host:

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   sudo chmod 666 /dev/input/eventXX
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

.. note::

   Replace every ``eventXX`` with the actual ``eventNN`` resolved by the
   first command, for example ``event7``. Export
   ``RLINF_KEYBOARD_DEVICE`` in the shell that launches collection or
   deployment. Do not run ``ray start``.

Cameras
~~~~~~~

.. code-block:: bash

   rs-enumerate-devices | grep -E "Name|Serial|USB Type"
   ls /dev/v4l/by-id/
   lsusb -t

Expected output should identify the RealSense serial, two Lumos devices,
and USB-3 speed such as ``5000M``. ``480M`` means the device fell back to
USB 2.

GELLO leaders
~~~~~~~~~~~~~

Identify the two FTDI paths by plugging one leader at a time:

.. code-block:: bash

   ls /dev/serial/by-id/ | grep -i ftdi

Verify each leader streams smooth joint values:

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python -m rlinf.robotics.parts.teleop.gello_joint \
       --port /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0

The command continuously refreshes output, for example:

.. code-block:: text

   joints=[+0.012 -0.604 +0.031 -2.184 +0.019 +1.571 +0.781]  gripper=[0.035]

If values stop updating or jump by about ``2π``, run the calibration below.

GELLO calibration
~~~~~~~~~~~~~~~~~

.. _dual-franka-gello-calibration:

Calibrate each GELLO once, then verify it with ``align-sequential``.
Both leaders can be calibrated against the left arm on the host.

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export GELLO_PORT=/dev/serial/by-id/usb-FTDI_..._<ID>-if00-port0

   python toolkits/realworld_check/test_gello.py calibrate
   python toolkits/realworld_check/test_gello.py align-sequential

On success, ``align-sequential`` prints:

.. code-block:: text

   ALL JOINTS ALIGNED
     per-joint Δ (rad): ['+0.012', '-0.008', '+0.005', '+0.021', '-0.041', '+0.009', '-0.003']
     max |Δ| = 0.041 rad on J5 (stream gate threshold = 0.5 rad — well under)
   You can now Ctrl-C and start collect_data.sh.

Run the same two commands for the second leader by changing
``GELLO_PORT``.


Run It
------

Do not start Ray
~~~~~~~~~~~~~~~~

``collect_data.sh`` and the evaluation launcher start a local Ray instance
and inherit the shell you launch from. Do not run ``ray start`` first. If
``ray status`` still shows a cluster from an earlier run, stop it so the
launcher does not attach to that cluster:

.. code-block:: bash

   ray stop --force

Collect demonstrations
~~~~~~~~~~~~~~~~~~~~~~

Start collection on the host after
:ref:`align-sequential <dual-franka-gello-calibration>` reports
``ALL JOINTS ALIGNED``:

.. code-block:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX
   bash examples/embodiment/collect_data.sh \
       realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

In another terminal on the host, monitor progress:

.. code-block:: bash

   cd /path/to/RLinf
   python toolkits/realworld_check/collect_monitor.py logs/collect.log

Foot-pedal controls:

* ``a``: start recording; press again while recording to abort and drop the
  current buffer.
* ``b``: increment ``segment_id`` for sub-task boundaries.
* ``c``: mark success, write the LeRobot shard, and finish the episode.

Set ``data_collection.resume: true`` and keep the same
``data_collection.save_dir`` to append new ``id_*`` shards to an existing
dataset.

Backfill tcp_rot6d
~~~~~~~~~~~~~~~~~~

Collection writes joint-space data. Convert it before SFT:

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   export DATA_REPO_ID=<repo_id>
   export SFT_REPO_ID=$DATA_REPO_ID/tcp_rot6d_v1

   python toolkits/dual_franka/backfill_tcp_rot6d.py \
       --src $HF_LEROBOT_HOME/$DATA_REPO_ID/joint_v1 \
       --dst $HF_LEROBOT_HOME/$SFT_REPO_ID

Run SFT
~~~~~~~

Synchronize the converted dataset to the training node, then run SFT there:

.. code-block:: bash

   export TRAINER_IP=<trainer_ip>
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   export SFT_REPO_ID=<repo_id>/tcp_rot6d_v1

   ssh $TRAINER_IP "mkdir -p $HF_LEROBOT_HOME/$SFT_REPO_ID"
   rsync -av $HF_LEROBOT_HOME/$SFT_REPO_ID/ \
       $TRAINER_IP:$HF_LEROBOT_HOME/$SFT_REPO_ID/

On the training node:

.. code-block:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export HF_LEROBOT_HOME=/path/to/lerobot_root
   export DUAL_FRANKA_DATA_ROOT=/path/to/lerobot_root
   export PI05_BASE_CKPT=/path/to/pi05/torch
   export SFT_REPO_ID=<repo_id>/tcp_rot6d_v1

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_tcp_rot6d \
       --repo-id $SFT_REPO_ID

   mkdir -p $PI05_BASE_CKPT/$SFT_REPO_ID
   cp <openpi_assets_dirs>/pi05_dualfranka_tcp_rot6d/$SFT_REPO_ID/norm_stats.json \
      $PI05_BASE_CKPT/$SFT_REPO_ID/norm_stats.json

   bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_tcp_rot6d

Update ``SFT_DATASET_REPO_ID``, ``PI05_BASE_CKPT``, logger settings, and
cluster placement in
``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``.
Checkpoints are saved under
``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``.


Evaluation and Deployment
-------------------------

Prepare checkpoint files
~~~~~~~~~~~~~~~~~~~~~~~~

The deployment checkpoint directory on the host must contain:

.. code-block:: text

   <model_path>/
   ├── actor/model_state_dict/full_weights.pt
   └── <repo_id>/tcp_rot6d_v1/norm_stats.json

Synchronize the SFT checkpoint and matching normalization stats back to the host:

.. code-block:: bash

   export TRAINER_IP=<trainer_ip>
   export DEPLOY_CKPT=/path/to/deploy/global_step_<N>
   export SFT_REPO_ID=<repo_id>/tcp_rot6d_v1

   mkdir -p $DEPLOY_CKPT/actor/model_state_dict
   mkdir -p $DEPLOY_CKPT/$SFT_REPO_ID

   rsync -av \
       $TRAINER_IP:<train_log>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt \
       $DEPLOY_CKPT/actor/model_state_dict/full_weights.pt
   rsync -av $TRAINER_IP:<train_log>/checkpoints/global_step_<N>/$SFT_REPO_ID/norm_stats.json \
       $DEPLOY_CKPT/$SFT_REPO_ID/norm_stats.json

Set ``rollout.model.model_path`` to ``$DEPLOY_CKPT`` and
``actor.model.openpi_data.repo_id`` to ``<repo_id>/tcp_rot6d_v1`` in
``examples/embodiment/config/realworld_eval_dual_franka.yaml``.

Launch deployment
~~~~~~~~~~~~~~~~~

Do not start Ray before deployment. The evaluation launcher starts a local Ray
instance. Skip the "Starting the Ray Cluster" step in the
:doc:`real-world evaluation guide <../../evaluations/guides/realworld>`; that
step is for a split GPU node and robot node. On this host both arms share one
machine. ``realworld_eval_dual_franka`` is the deployment config. When
execution should overlap inference, use ``realworld_dual_franka_pi05_RTC``.
After homing, press ``a`` on the keyboard or pedal to start the policy.

Deployment pedal controls:

* ``a``: start policy execution from idle.
* ``b``: mark failure and reset.
* ``c``: mark success and reset.

After each reset, the wrapper waits for ``a`` again to allow scene reset before
the next episode.


Troubleshooting
---------------

**Ray worker import failure**
   In the shell that launched the script, check
   ``which python`` and
   ``python -c "import franky, gello, gello_teleop"``. Worker logs are under
   ``/tmp/ray/session_latest/logs/worker-*.err``. Do not start Ray yourself.

**Foot pedal permission denied**
   Re-run ``sudo chmod 666 /dev/input/eventXX`` and confirm
   ``RLINF_KEYBOARD_DEVICE`` points to the same device.

**RealSense appears as USB 2**
   Replace the cable or port. ``lsusb -t`` should show ``5000M`` instead of
   ``480M``.

**GELLO stops streaming**
   Power-cycle the leader, replug the FTDI adapter, and verify it with
   ``python -m rlinf.robotics.parts.teleop.gello_joint --port ...``.

**One arm does not respond during reset**
   On the host, run ``ping -c 100 <robot_ip>`` for that arm. If packets drop,
   fix the direct cable or the switch path, or power-cycle the robot.

**Deployment cannot locate ``norm_stats.json``**
   Check that the file is exactly at
   ``<model_path>/<actor.model.openpi_data.repo_id>/norm_stats.json``.

**Deployment remains idle**
   Confirm the pedal path and permission, then press ``a``. The eval wrapper
   waits in idle between episodes by design.
