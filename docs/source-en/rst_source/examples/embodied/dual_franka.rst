Real-World Dual-Franka: GELLO Collection, π₀.₅ SFT, Deployment
================================================================

This example describes the minimum supported workflow for the RLinf
dual-arm Franka setup: start a two-node real-world cluster, collect
dual-arm demonstrations with two GELLO leaders, fine-tune π₀.₅ on the
converted tcp_rot6d dataset, and deploy the trained policy back to the
robots with foot-pedal control.

This page covers the hardware layout, dependency installation, required
configuration placeholders, hardware checks, data collection, tcp_rot6d
backfill and SFT, deployment, and common troubleshooting notes.

Read these pages first if you are setting up the hardware for the first
time:

* :doc:`franka` for single-arm Franka basics, Ray cluster setup, FCI, and
  ``RLINF_NODE_RANK``.
* :doc:`franka_gello` for GELLO installation, Dynamixel permissions, and
  ``gello-teleop``.


Overview
--------

The workflow has five stages:

1. Install ``franka-franky`` dependencies on the two robot nodes.
2. Start Ray with ``RLINF_NODE_RANK=0`` on the head and
   ``RLINF_NODE_RANK=1`` on the worker.
3. Collect joint-space demonstrations with
   ``realworld_collect_data_gello_joint_dual_franka``.
4. Convert the dataset to tcp_rot6d and run π₀.₅ SFT with
   ``realworld_sft_openpi_dual_franka_tcp_rot6d``.
5. Deploy with ``realworld_eval_dual_franka``.

The repository-provided configs already select the corresponding environments.
Typically, replace only the hardware paths, task text, dataset IDs, and model
checkpoints.


Environment
-----------

Hardware layout
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Node
     - Role
     - Hardware
   * - ``node 0`` (head)
     - Ray head, env worker, left Franka controller, cameras, GELLO input,
       collection/deployment entrypoint
     - GPU machine; left Franka FR3; left Robotiq gripper; base camera;
       left and right wrist cameras; both GELLO leaders; PCsensor foot pedal
   * - ``node 1`` (worker)
     - Ray worker and right Franka controller
     - Right Franka FR3; right Robotiq gripper; GPU optional

Both Franka arms are usually wired to their local control node through a
dedicated NIC. All cameras, both GELLO leaders, and the foot pedal are
connected to ``node 0``.

Data and action spaces
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 22 20 34

   * - Stage
     - Environment
     - Shape
     - Use
   * - Collection
     - ``DualFrankaJointEnv-v1``
     - ``state=[68]``, ``actions=[16]``
     - GELLO joint-space demonstrations
   * - SFT / deployment
     - ``DualFrankaTCPEnv-v1``
     - ``state=[20]``, ``actions=[20]``
     - π₀.₅ tcp_rot6d policy

The tcp_rot6d action is ``[xyz(3), rot6d(6), gripper(1)]`` for each arm.
The main image key is ``left_wrist_0_rgb``; extra views are ordered as
``base_0_rgb`` and ``right_wrist_0_rgb``.


Dependency Installation
-----------------------

Robot nodes
~~~~~~~~~~~

Run the robot-node installation on both ``node 0`` and ``node 1``.
Choose ``LIBFRANKA_VERSION`` from the official `Franka compatibility
matrix <https://frankarobotics.github.io/docs/compatibility.html>`_; avoid
libfranka ``0.18.0``.

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   export LIBFRANKA_VERSION=0.15.0       # replace with your compatible version
   bash requirements/install.sh embodied --env franka-franky --use-mirror
   source .venv/bin/activate

Install GELLO dependencies on ``node 0`` by following :doc:`franka_gello`.
The two GELLO leaders must stay local to ``node 0``; do not route their
1 kHz stream over the LAN.

Training node
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

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``: FCI IP visible from each
  controller node.
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
* ``MODEL_PATH``: deployment checkpoint directory on ``node 0``.


Hardware Checks
---------------

Run these checks before starting Ray.

Foot pedal
~~~~~~~~~~

Use the vendor tool once to configure the PCsensor FootSwitch keys as
``a`` / ``b`` / ``c``. Then on ``node 0``:

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   sudo chmod 666 /dev/input/eventXX
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

.. note::

   Replace every ``eventXX`` with the actual ``eventNN`` resolved by the
   first command, for example ``event7``. Export
   ``RLINF_KEYBOARD_DEVICE`` before ``ray start``.

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
   python -m rlinf.envs.realworld.common.gello.gello_joint_expert \
       --port /dev/serial/by-id/usb-FTDI_..._<LEFT_ID>-if00-port0

The command continuously refreshes output, for example:

.. code-block:: text

   joints=[+0.012 -0.604 +0.031 -2.184 +0.019 +1.571 +0.781]  gripper=[0.035]

If values stop updating or jump by about ``2π``, run the calibration below.

GELLO calibration
~~~~~~~~~~~~~~~~~

.. _dual-franka-gello-calibration:

Calibrate each GELLO once, then verify it with ``align-sequential``.
Both leaders can be calibrated against the left arm on ``node 0``.

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


Quick Start
-----------

Start Ray
~~~~~~~~~

Ray captures environment variables at ``ray start`` time. Export the rank
and keyboard device before starting the cluster.

.. code-block:: bash

   # node 0
   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1
   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1

   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

On ``node 0``, run ``ray status`` and confirm that both nodes are ALIVE.

Collect demonstrations
~~~~~~~~~~~~~~~~~~~~~~

Start collection on ``node 0`` after
:ref:`align-sequential <dual-franka-gello-calibration>` reports
``ALL JOINTS ALIGNED``:

.. code-block:: bash

   cd /path/to/RLinf
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   bash examples/embodiment/collect_data.sh \
       realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

In another ``node 0`` terminal, monitor progress:

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

The deployment checkpoint directory on ``node 0`` must contain:

.. code-block:: text

   <model_path>/
   ├── actor/model_state_dict/full_weights.pt
   └── <repo_id>/tcp_rot6d_v1/norm_stats.json

Synchronize the SFT checkpoint and matching normalization stats back to ``node 0``:

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

Reuse the Ray cluster from collection, or restart it with the same
environment variables. Then execute on ``node 0``:

.. code-block:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

Hydra override example:

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
       rollout.model.model_path=/path/to/deploy/global_step_<N> \
       actor.model.openpi_data.repo_id=<repo_id>/tcp_rot6d_v1 \
       env.eval.override_cfg.task_description="handover the object"

Deployment pedal controls:

* ``a``: start policy execution from idle.
* ``b``: mark failure and reset.
* ``c``: mark success and reset.

After each reset, the wrapper waits for ``a`` again to allow scene reset before
the next episode.


Troubleshooting
---------------

**Ray worker import failure**
   In the same shell that ran ``ray start``, check
   ``which python`` and
   ``python -c "import franky, gello, gello_teleop"``. Worker logs are under
   ``/tmp/ray/session_latest/logs/worker-*.err``.

**Foot pedal permission denied**
   Re-run ``sudo chmod 666 /dev/input/eventXX`` and confirm
   ``RLINF_KEYBOARD_DEVICE`` points to the same device.

**RealSense appears as USB 2**
   Replace the cable or port. ``lsusb -t`` should show ``5000M`` instead of
   ``480M``.

**GELLO stops streaming**
   Power-cycle the leader, replug the FTDI adapter, and verify it with
   ``python -m rlinf.envs.realworld.common.gello.gello_joint_expert --port ...``.

**One arm does not respond during reset**
   On that controller node, run ``ping -c 100 <robot_ip>``. If packets drop,
   fix the NIC/FCI connection or power-cycle the robot.

**Deployment cannot locate ``norm_stats.json``**
   Check that the file is exactly at
   ``<model_path>/<actor.model.openpi_data.repo_id>/norm_stats.json``.

**Deployment remains idle**
   Confirm the pedal path and permission, then press ``a``. The eval wrapper
   waits in idle between episodes by design.
