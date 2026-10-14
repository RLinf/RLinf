.. _dual-franka-pico-dagger-en:

Dual Franka PICO Collection and DAgger
======================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-vr.jpg
   :align: center
   :width: 80%
   :alt: Dual-Franka VR teleoperation

   Collect dual-Franka teleoperation data with VR / PICO.

This guide explains how to use PICO to collect demonstrations in the dual-Franka
TCP-rot6d environment, then run online Human-Gated DAgger with PICO human
interventions. For dual-arm hardware, real-time kernel, and camera checks, start
with :doc:`dual_franka`; for the PICO / XRoboToolkit data publishing pipeline,
see :doc:`franka_vr`; for the single-arm HG-DAgger workflow, see
:doc:`hg-dagger`.

Overview
--------

Use the left and right PICO controllers to control the two Franka arms. First
collect tcp_rot6d LeRobot data, then prepare the OpenPI π₀.₅ student checkpoint
by following :doc:`dual_franka`, and finally launch online HG-DAgger on the real
robot.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · HG-DAgger

   .. grid-item-card:: Tasks
      :text-align: center

      Dual-arm manipulation

   .. grid-item-card:: Hardware
      :text-align: center

      2× Franka · PICO · 3 cameras

| **You'll do:** start the PICO publisher → collect dual-arm tcp_rot6d demos → reuse the dual-arm SFT checkpoint flow → run online HG-DAgger.
| **Prerequisites:** :doc:`dual_franka` · :doc:`franka_vr` · OpenPI π₀.₅ checkpoint · Ray cluster.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Task
     - Config / entry point
     - Description
   * - PICO stream
     - ``vr_data_publisher``
     - Publish headset, controller, and button data from PICO / XRoboToolkit.
   * - Collection
     - ``realworld_dual_franka_collect_data_pico``
     - Collect tcp_rot6d LeRobot data with two-hand PICO teleoperation.
   * - HG-DAgger
     - ``realworld_dual_franka_dagger_openpi``
     - Let the policy act autonomously with per-arm PICO intervention, archive complete successful trajectories, and train on fully intervened action chunks.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - Field
     - Description
   * - Observation
     - LeRobot ``image`` is the base camera ``base_0_rgb``. The wrists land in ``extra_view_image-0`` / ``extra_view_image-1`` after the leftover names are sorted. State is dual-arm TCP / gripper.
   * - Action
     - Dual-arm tcp_rot6d: ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``.
   * - Reward
     - a/b/c from ``keyboard_device``. PICO A/B/X/Y only drive the grippers.
   * - Prompt
     - ``task_description`` stored in the data and used as the OpenPI language condition.


Installation and Node Layout
----------------------------

Software Environment
~~~~~~~~~~~~~~~~~~~~

Robot Nodes
^^^^^^^^^^^

Run the robot-node installation on every node that directly communicates with
a Franka. Dual-arm Franka always drives the arms through Franky, the backend that
the default ``franka`` environment installs. The installer downloads a prebuilt
Franky wheel with libfranka bundled; these wheels exist only for libfranka
``0.15.0`` and ``0.19.0`` (the default) on x86_64. Set ``LIBFRANKA_VERSION`` to
the one that the official `Franka compatibility
matrix <https://frankarobotics.github.io/docs/compatibility.html>`_ lists for
your firmware. For other firmware, build a Franky wheel against the matching
libfranka and pass its path or URL in ``FRANKY_WHEEL``; the legacy ROS backend
covers other libfranka versions only for single-arm Franka.

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   export LIBFRANKA_VERSION=0.19.0       # or 0.15.0, matching the firmware
   bash requirements/install.sh embodied --env franka --use-mirror
   source .venv/bin/activate

The ``franka`` environment installs Franky with the camera and input
dependencies, including ``pyzmq`` for the PICO consumer side. See
:doc:`franka_vr` for the PICO headset, XRoboToolkit PC Service, and
``vr_data_publisher`` setup and validation.

Inference Node
^^^^^^^^^^^^^^

Run the OpenPI environment on the GPU inference node used by the online DAgger
actor / rollout components:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Ray Node Layout
~~~~~~~~~~~~~~~

The collection config uses one host. The two arms have different FCI IPs and
connect to that host either directly (one cable from each arm into a NIC on
the host) or through a switch (both arms and the host on the same switch).
Both grippers and the three cameras are on rank ``0``.
``left_controller_node_rank``, ``right_controller_node_rank``, and
``node_rank`` are all ``0``. Do not run ``ray start`` before collection; the
launch script starts a local Ray instance. Raise ``num_nodes`` and the
controller ranks only when you split the arms across machines. That split
still uses the Ray cluster below.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Rank
     - Role
     - Notes
   * - ``0``
     - Both arm controllers, env worker, three cameras, PICO consumer
     - Needs the keyboard or pedal, and a PICO ZeroMQ address reachable on this machine.

The online DAgger config uses three nodes:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Rank
     - Role
     - Notes
   * - ``0``
     - inference / rollout / actor
     - Usually the GPU node running OpenPI.
   * - ``1``
     - Left-arm control, env worker, three cameras, PICO consumer
     - Requires the foot pedal and access to the PICO ZeroMQ address.
   * - ``2``
     - Right-arm control
     - Only needs the right-arm Franka / Robotiq control path.

.. warning::

   Ray captures the Python interpreter and environment variables at
   ``ray start`` time. Before starting Ray, finish setting
   ``source .venv/bin/activate``, ``PYTHONPATH``, ``RLINF_NODE_RANK``,
   and any Franka-specific environment variables. The collection YAML writes
   ``keyboard_device`` into ``RLINF_KEYBOARD_DEVICE`` inside the env worker.
   The DAgger config has no such field, so export the keyboard or pedal on the
   env node before ``ray start``.

Cluster Setup
~~~~~~~~~~~~~

Skip this section for single-host collection. The launch script starts a local
Ray instance, and you do not run ``ray start``. Use the steps below only when
DAgger is split across machines.

Before that split run, set up the Ray cluster correctly.

.. warning::
   This step is critical. Small configuration mistakes can lead to missing
   dependencies or failure to control the robot.

RLinf uses Ray for distributed execution. When you run `ray start` on a node,
Ray records the current Python interpreter path and environment variables; all
processes that Ray launches on that node inherit the same environment.

RLinf provides ``ray_utils/realworld/setup_before_ray.sh`` to help set a
consistent environment before starting Ray on each node. Modify it for your
setup and source it on every node.

The script usually handles:

1. Sourcing the correct virtual environment when using a custom installation.

2. Loading the runtime environment required by Franka, Robotiq, and cameras on
   Franka controller nodes.

3. Setting RLinf environment variables on all nodes:

.. code-block:: bash

   export PYTHONPATH=<path_to_your_RLinf_repo>:$PYTHONPATH
   export RLINF_NODE_RANK=<node_rank_of_this_node>
   export RLINF_COMM_NET_DEVICES=<network_device_for_communication> # optional if there is only one NIC

``RLINF_NODE_RANK`` should be set to ``0 ~ N-1`` across the ``N`` nodes in the
cluster. It uniquely identifies each node in the config. The DAgger env node
has no ``keyboard_device``, so export the keyboard or pedal before
``ray start``:

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/usb-KEYBOARD-event-kbd

Collection uses ``N=1`` on this one host: rank ``0`` runs both arm controllers,
the env, the three cameras, and the PICO consumer. Do not run ``ray start``.
If ``ray status`` still shows a cluster from an earlier run, stop it with
``ray stop --force`` so the launcher does not attach to it.
DAgger uses ``N=3``: rank ``0`` is OpenPI inference / actor, rank ``1`` is the
left arm / env / PICO consumer, and rank ``2`` is the right arm. Set the DAgger
hardware ``node_rank`` and both ``*_controller_node_rank`` fields to the cluster
ranks that actually own the arms.

When DAgger is split across machines, start Ray on each node. Skip this step
for the single-host collection:

``<head_node_ip_address>`` must be reachable by all other cluster nodes.

.. code-block:: bash

   # On the head node (node rank 0)
   ray start --head --port=6379 --node-ip-address=<head_node_ip_address>

   # On worker nodes (node rank 1 ~ N-1)
   ray start --address='<head_node_ip_address>:6379'

Use `ray status` to check that the cluster started correctly.


Configuration
-------------

Main Config Files
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Config
     - Purpose
   * - ``examples/embodiment/config/realworld_dual_franka_collect_data_pico.yaml``
     - PICO dual-arm tcp_rot6d data collection.
   * - ``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml``
     - Online HG-DAgger with PICO human intervention.
   * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
     - Default real-world dual-arm TCP-rot6d environment config.

Hardware Placeholders
~~~~~~~~~~~~~~~~~~~~~

Replace the following fields in the collection and DAgger configs:

* ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``: FCI IPs for the left and right arms.
* ``BASE_CAMERA_SERIAL``, ``LEFT_CAMERA_SERIAL``, ``RIGHT_CAMERA_SERIAL``:
  RealSense values are librealsense ASIC serials, not V4L USB serials. A Lumos
  wrist uses that camera's ``/dev/v4l/by-id`` path.
* ``base_camera_type``, ``left_camera_type``, ``right_camera_type``: the
  collection example uses ``realsense`` for all three. The DAgger example uses
  ``realsense`` for the base and ``lumos`` for both wrists. Swap serials if an
  image lands on the wrong arm.
* ``left_gripper_type`` / ``right_gripper_type``: left and right gripper types.
* ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION``: stable
  ``/dev/serial/by-id`` paths for the left and right gripper adapters.
* ``keyboard_device``: the evdev node in the collection config. Episode keys
  come from this USB keyboard or pedal, not from the collection terminal.
* ``left_controller_node_rank`` / ``right_controller_node_rank`` / ``node_rank``:
  all three are ``0`` in the collection example. DAgger must use the cluster
  ranks that own the arms and the env.
* ``TASK_DESCRIPTION``: task text used by collection and DAgger. It should match
  the task text used to train the checkpoint.
* ``joint_reset_qpos``: the collection config omits this, so reset uses the
  environment default joint pose. Replace the all-zero placeholder in the
  DAgger config with first-frame joint means or a safe home. It is seven joint
  angles per arm, not ``target_ee_pose``.
* ``target_ee_pose``: per arm, ``[x, y, z, roll, pitch, yaw]``. xyz must sit
  inside the shared ``realworld_dual_franka_tcp_rot6d`` limits: x in
  [0.3, 0.9], y in [-0.4, 0.4], z in [0.02, 0.7]. The example
  ``[0.5, ±0.2, 0.5, -3.14, 0, 0]`` is inside that box.

PICO Config
~~~~~~~~~~~

The collection config uses ``env.eval.pico.zmq_addr``; the DAgger config uses
``env.train.pico.zmq_addr``. This address must match the publisher bind address.
Both examples default to ``ipc:///tmp/vr_data.ipc``, which is the same machine
as the env worker.

``teleop`` is a per-arm list, not a single ``pico`` value. ``pico.hand: dual``
binds the left and right controllers to the left and right arms.

.. code-block:: yaml

   env:
     eval:
       teleop:
         - {pico: {drives: left}}
         - {pico: {drives: right}}
       keyboard_device: /dev/input/by-id/usb-KEYBOARD-event-kbd
       keyboard_reward_wrapper: start_end
       pico:
         zmq_addr: "ipc:///tmp/vr_data.ipc"
         hand: "dual"
         hold_current_when_inactive: True
         control_trigger: "grip"
         calibration:
           button: "trigger"

DAgger puts the same ``teleop`` list and ``pico`` block under ``env.train``,
with ``hold_current_when_inactive: False`` and
``keyboard_reward_wrapper: eval_control``. Across machines, bind the publisher
to ``tcp://0.0.0.0:<port>`` and set the consumer to
``tcp://<vr_publisher_ip>:<port>``. Do not use ``0.0.0.0`` as the consumer
address.

Default controller semantics:

.. code-block:: text

   left grip  -> intervene on the left arm
   right grip -> intervene on the right arm
   left X/Y   -> close / open left gripper
   right A/B  -> close / open right gripper
   trigger    -> recalibrate the operator base from the current headset heading

``hold_current_when_inactive`` differs between collection and DAgger:

* Collection uses ``True``: an inactive arm holds the current TCP, which is
  suitable for pure teleoperation collection.
* DAgger uses ``False``: an inactive arm retains its rollout action. When either
  ``grip`` is held, only that arm's action is replaced by the corresponding
  PICO action, while the other arm retains its rollout action.

Dual-arm DAgger composes intervention records per arm. For example, when only
the left arm is being intervened on, the 20D action that is executed and written
to ``intervene_action`` is:

.. code-block:: text

   [left-arm PICO 10D action, right-arm rollout 10D action]

The reverse applies when only the right arm is being intervened on. Replacing
either arm sets ``intervene_flag=True``; no intervention record is produced only
when neither arm is being intervened on. The online LeRobot collector still
stores the complete successful episode. With ``only_save_expert: True``, the
sampler uses ``intervene_flag`` to expose only action chunks whose non-padded
frames are all human corrections.


Arm Compliance
~~~~~~~~~~~~~~

The collection config sets one shared ``compliance`` mapping on the
``DualFranka`` hardware entry: translational stiffness 1000 N/m, rotational
stiffness 60 Nm/rad, translational clip 0.02 m, rotational clip 0.06 rad,
per-step translation limit 0.02 m, per-step rotation limit 0.08 rad, and
``max_delta_tau`` 0.2. Franky's defaults only act on about 8 mm / 0.04 rad of
error, so teleoperation lags and the arm keeps creeping after grip release.
Rotational stiffness 80 with a 0.12 rad clip shakes as soon as takeover starts.
``max_step`` 0.03 m at 10 Hz can trip ``joint_velocity_violation`` with a
Robotiq payload. The DAgger config omits this mapping and therefore uses
Franky's defaults; copy the collection mapping onto the DAgger hardware entry
when the two should feel the same. Defaults and reset requests are described in
:ref:`Configure Arm Motion <franka-motion-settings>`.

A side-specific mapping replaces the shared mapping for that arm; keys omitted
from it use Franky's defaults. For example:

.. code-block:: yaml

   compliance:
     translational_stiffness: 900.0
   left_compliance:
     max_step: 0.02

Here the right arm uses 900 N/m and the default 3 cm target-change limit. The
left arm uses the default 1000 N/m and a 2 cm limit. Omitting
``left_compliance`` makes the left arm use the shared mapping as well;
``left_compliance: {}`` selects all backend defaults for that arm.
``right_compliance`` follows the same rules. These settings also govern policy
targets during DAgger. GELLO joint teleoperation uses joint control and is
unaffected by these Cartesian settings.


Start the PICO Data Stream
--------------------------

Start the XRoboToolkit PC Service on the PICO publisher machine, then start
the VR data publisher. The concrete installation paths are described in
:doc:`franka_vr`.

.. code-block:: bash

   cd /opt/apps/roboticsservice
   bash runService.sh

.. code-block:: bash

   cd /path/to/pico_software/XRoboToolkit-Teleop-Sample-Python
   source .venv/bin/activate
   cd /path/to/pico_software
   python -m vr_data_publisher --config configs/vr_bridge.yaml

On the node running the env worker, verify that PICO data is reachable:

.. code-block:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   python toolkits/realworld_check/test_pico_data.py \
       --zmq-addr tcp://<vr_publisher_ip>:<port>

Only continue to real-robot collection or DAgger after the output refreshes
continuously and ``grip``, ``trigger``, ``A/B``, and ``X/Y`` change as expected.


Collect PICO Demonstrations
---------------------------

Run Collection
~~~~~~~~~~~~~~

After the Franka arms, Robotiq grippers, cameras, ``keyboard_device``, and PICO
data stream are ready, run this on that machine. ``collect_data.sh`` calls
whatever ``python`` is on ``PATH``, so activate the Python 3.11 virtualenv
first. ``/usr/bin/python`` cannot run the script.

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_dual_franka_collect_data_pico

Keys come from ``keyboard_device``, not from the terminal and not from PICO
A/B/X/Y:

* ``a``: start recording. Another ``a`` within 1.5 s is ignored; after that it
  aborts and drops the buffer.
* ``b``: while recording, increment ``segment_id``. Presses outside recording
  are ignored.
* ``c``: mark success and write the LeRobot shard. Presses outside recording
  are ignored.

PICO operation:

1. Wear the headset and face the front of the workspace.
2. Pull ``trigger`` to calibrate the PICO base.
3. Hold left / right ``grip`` to intervene on the left / right arm.
4. Use ``X/Y`` for the left gripper and ``A/B`` for the right gripper.
5. When one hand releases ``grip``, the corresponding arm holds the current TCP.

The collection script writes under ``logs/<timestamp>/``:

* replay-buffer trajectories: ``demos/``
* LeRobot data: ``collected_data/rank_0/id_0/``; later shards are ``id_1``,
  ``id_2``

Headless OpenCV cannot open a preview, so the collection example sets
``enable_camera_player: False``.

PICO dual-arm collection already uses the ``realworld_dual_franka_tcp_rot6d``
environment, so the actions are already tcp_rot6d. You do not need to run the
``backfill_tcp_rot6d.py`` step used by the GELLO joint-data workflow.

.. note::

   ``data_collection.resume: True`` only resumes under the same ``save_dir``.
   ``collect_data.sh`` creates a new ``logs/<timestamp>`` directory by default.
   To append across runs, set ``data_collection.save_dir`` to a fixed path.


Prepare the Checkpoint
----------------------

Online DAgger requires a deployable OpenPI checkpoint. For data organization,
normalization stats, SFT, and checkpoint directory preparation, follow the SFT
and deployment-checkpoint sections in :doc:`dual_franka`.

When using data collected from this page, the data already comes from the
``realworld_dual_franka_tcp_rot6d`` environment, so do not run the GELLO
joint-data ``backfill_tcp_rot6d.py`` step again.

After the checkpoint is ready, set the following in
``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml``:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/deploy/global_step_<N>

   actor:
     model:
       openpi_data:
         repo_id: <repo_id>/tcp_rot6d_v1


Run Online HG-DAgger
--------------------

Check Key DAgger Settings
~~~~~~~~~~~~~~~~~~~~~~~~~

Before launch, confirm these fields:

.. code-block:: yaml

   algorithm:
     dagger:
       only_save_expert: True
       online_lerobot:
         enabled: True
         only_success: True
         robot_type: "dual_FR3"
         fps: 10
         finalize_interval: 1
         data_path: ${runner.logger.log_path}/online_lerobot
         rolling_lerobot_window_size: 50000
         min_frames: 1
         lerobot_num_workers: 0

   env:
     train:
       smooth_intervene: True
       teleop:
         - {pico: {drives: left}}
         - {pico: {drives: right}}
       keyboard_reward_wrapper: eval_control
       pico:
         zmq_addr: "ipc:///tmp/vr_data.ipc"
         hand: "dual"
         hold_current_when_inactive: False
     eval:
       teleop: none

``online_lerobot.enabled: True`` enables the online LeRobot data path. The env worker collects rollouts by episode and sends episodes that satisfy the configured filters to the actor; the actor adds them to ``RollingLeRobotDataset`` for training, so online training no longer uses the trajectory replay buffer.

``smooth_intervene: True`` removes action-chunk boundary stalls while PICO is active. If the final frame of a chunk is human-controlled, the env worker skips the next policy inference and executes a shape-compatible dummy chunk instead. PICO actions still override active arms, while inactive frames hold the measured TCP pose. Normal model inference resumes after the final chunk frame is no longer intervened or the episode ends. This mode is PICO-only: every ``env.train.teleop`` entry must be pico, and it currently requires one environment per env-worker pipeline stage.

``only_success: True`` discards failed rollouts and keeps only successful episodes. ``only_save_expert: True`` still archives each complete successful episode, but training only samples chunk starts where every non-padded frame in the action chunk has ``intervene_flag=True``. Because a dual-arm frame is marked as an intervention when either arm is replaced, such a chunk may combine one arm's PICO action with the other arm's rollout action. Every successful episode is archived immediately under ``${runner.logger.log_path}/online_lerobot/rank_0/id_<N>/``. ``env.eval.teleop: none`` means evaluation uses the policy alone, without human intervention.

The real-world DAgger config intentionally omits beta-related fields because it does not configure ``rollout.expert_model``. Beta only controls action mixing between a model expert and the student; human intervention here is determined by the PICO intervention wrapper.

Run DAgger
~~~~~~~~~~

After the DAgger Ray cluster is running, start online training on the head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dual_franka_dagger_openpi

During the run:

* ``a``: start one policy rollout from idle.
* left / right ``grip``: intervene on the corresponding arm; the other arm
  continues executing the policy action if it is not being intervened on.
* ``b``: mark failure and end the current rollout.
* ``c``: mark success and end the current rollout.

After each episode, the env resets and waits for ``a`` again. During policy
execution, hold ``grip`` only when you need to correct the policy, then release
it to let the policy continue. Holding either ``grip`` combines that arm's PICO
action with the other arm's rollout action into a complete 20D
``info["intervene_action"]``. After a successful termination, the complete
episode is sent in memory to the actor and written as an online LeRobot shard;
failed episodes are discarded. The actor retains the complete physical archive
but exposes only fully intervention-labeled chunks to training.

Monitoring
----------

Start TensorBoard:

.. code-block:: bash

   tensorboard --logdir ./logs

Recommended metrics:

* ``train/dagger/actor_loss``: supervised loss on expert-only action chunks.
* ``train/lerobot_dataset/total_episodes``: number of successful episodes received by the actor.
* ``train/lerobot_dataset/physical_frames``: number of received LeRobot physical frames.
* ``train/lerobot_dataset/logical_samples``: number of expert-valid trainable chunk starts in the rolling window.
* ``train/lerobot_dataset/num_sub_datasets``: number of currently loaded LeRobot shards.
* ``train/actor/lr`` and ``train/actor/grad_norm``: training stability.

During collection and online DAgger, inspect ``logs/<timestamp>/run_embodiment.log``
to confirm the successful episode count and the LeRobot write path. Online
DAgger shards are stored under
``logs/<timestamp>-realworld_dual_franka_dagger_openpi/online_lerobot/rank_0/``.


Troubleshooting
---------------

**DAgger waits too long and does not start**
   This is expected behavior for ``keyboard_reward_wrapper: eval_control``. If
   DAgger waits for a long time after launch, press the foot pedal mapped to
   keyboard ``a`` to start the rollout.
