.. _dual-franka-pico-dagger-en:

Dual-Arm Franka PICO VR HG-DAgger
=================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-vr.jpg
   :align: center
   :width: 80%
   :alt: Dual-Franka VR teleoperation

   Collect dual-Franka teleoperation data with VR / PICO.

Collect dual-Franka ``tcp_rot6d`` demonstrations with two PICO controllers, prepare an OpenPI π₀.₅ student checkpoint, and run online Human-Gated DAgger (HG-DAgger) with per-arm PICO intervention.

Overview
--------

Use left and right PICO controllers to teleoperate both Franka arms in the ``tcp_rot6d`` environment, train an initial OpenPI π₀.₅ checkpoint following :doc:`dual_franka`, and launch online HG-DAgger that archives complete successful episodes and trains on fully intervened action chunks. For hardware checks and real-time kernel tuning, see :doc:`dual_franka`; for XRoboToolkit / PICO publishing, see :doc:`franka_vr`; for single-arm HG-DAgger, see :doc:`hg-dagger`.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      2× Franka (Franky) · 2× Robotiq

   .. grid-item-card:: Teleop & Cameras
      :text-align: center

      PICO dual controllers · USB pedal · 3× cameras

   .. grid-item-card:: Workflow
      :text-align: center

      PICO Collect → π₀.₅ SFT → Online HG-DAgger

   .. grid-item-card:: Hardware
      :text-align: center

      1 host (collect) · 3 nodes (GPU + 2 robot hosts for DAgger)

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - Left wrist, right wrist, and base camera views plus dual-arm TCP and gripper state.
   * - Action
     - 20-D dual-arm ``tcp_rot6d`` target: ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``.
   * - Reward
     - Operator label from the USB keyboard or foot pedal (``keyboard_device`` / ``RLINF_KEYBOARD_DEVICE``).
   * - Prompt
     - ``task_description`` stored in the dataset and used as the OpenPI language condition.

Episode labels come from the USB keyboard or foot pedal; PICO ``X/Y`` and ``A/B`` only close and open the left and right grippers:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Mode
     - USB keyboard / pedal keys
   * - Collection (``start_end``)
     - ``a`` starts or aborts recording; ``b`` increments ``segment_id``; ``c`` marks success.
   * - HG-DAgger (``eval_control``)
     - ``a`` starts the policy; ``b`` marks failure; ``c`` marks success.

Hardware and Safety
-------------------

Collection runs on a single host by default: rank 0 owns both arms (via dual NICs or a switch), both grippers, all three cameras, the USB keyboard or pedal, and the PICO consumer. The collection script starts a local Ray instance automatically, so skip ``ray start``.

Online HG-DAgger uses three Ray nodes:

.. list-table::
   :header-rows: 1
   :widths: 14 34 52

   * - Rank
     - Role (HG-DAgger)
     - Wiring and environment requirements
   * - ``0``
     - OpenPI actor / rollout / inference
     - GPU node running the student model and online trainer.
   * - ``1``
     - Left arm, env worker, 3 cameras, PICO consumer
     - Attach USB keyboard/pedal and export ``RLINF_KEYBOARD_DEVICE`` before ``ray start``; ensure the PICO ZeroMQ address is reachable.
   * - ``2``
     - Right arm controller
     - Direct FCI and Robotiq link for the right arm.

.. warning::

   Keep an emergency stop within reach and clear the dual-arm workspace before enabling motion.

   In the DAgger config, replace the all-zero ``joint_reset_qpos`` placeholder with first-frame joint means or a verified safe home pose (7 joint angles per arm). Keep every ``target_ee_pose`` inside the ``realworld_dual_franka_tcp_rot6d`` workspace box: x in [0.3, 0.9], y in [-0.4, 0.4], and z in [0.02, 0.7] m.

   Ray captures your virtualenv and environment variables at ``ray start`` time, so export ``RLINF_NODE_RANK`` and ``RLINF_KEYBOARD_DEVICE`` before starting Ray.

.. dropdown:: Multi-node Ray cluster setup (required for HG-DAgger and 2-host collection)

   Skip this block for default single-host collection, which starts Ray automatically. For 3-node HG-DAgger—or 2-host collection with ``num_nodes: 2``, ``node_rank: 0``, ``left_controller_node_rank: 0``, and ``right_controller_node_rank: 1``—activate ``.venv`` and export the node variables on every node before starting Ray:

   .. code-block:: bash

      # On every node (set RLINF_NODE_RANK to 0, 1, or 2; omit NIC if only one exists)
      source .venv/bin/activate
      export RLINF_NODE_RANK=0
      export RLINF_COMM_NET_DEVICES=eth0

      # On the DAgger env node (rank 1), export the USB keyboard/pedal device
      export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/usb-KEYBOARD-event-kbd

      # Head node (rank 0)
      export HEAD_NODE_IP=192.168.1.10
      ray start --head --port=6379 --node-ip-address="$HEAD_NODE_IP"

      # Worker nodes (ranks 1 and 2)
      export HEAD_NODE_IP=192.168.1.10
      ray start --address="${HEAD_NODE_IP}:6379"

   Run ``ray status`` on the head node to confirm all nodes are connected.

   Ranks are global cluster ranks. The DAgger config defaults to ``node_rank: 1`` (inside the ``franka`` node group), ``left_controller_node_rank: 1``, and ``right_controller_node_rank: 2``.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Install & PICO Stream

      **Robot / env nodes** (every host wired to a Franka arm): prebuilt x86_64 Franky wheels bundle libfranka 0.15.0 or 0.19.0 (default). Match ``LIBFRANKA_VERSION`` to the `Franka compatibility matrix <https://frankarobotics.github.io/docs/compatibility.html>`_ (or pass ``FRANKY_WHEEL``); the ``franka`` environment installs camera, input, and ZeroMQ dependencies (see :doc:`franka_vr` for XRoboToolkit setup).

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **Training / rollout node** (GPU host for online DAgger):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **Start the PICO publisher** (on the publisher PC) and verify reception on the env node before touching the arms:

      .. code-block:: bash

         # Publisher PC terminal 1: XRoboToolkit PC Service
         (cd /opt/apps/roboticsservice && bash runService.sh)

         # Publisher PC terminal 2 (from the pico_software directory): VR data publisher
         source XRoboToolkit-Teleop-Sample-Python/.venv/bin/activate
         python -m vr_data_publisher --config configs/vr_bridge.yaml

         # Env node (from the RLinf root with .venv active; replace IP:port with your publisher)
         python -m toolkits.realworld_check.test_pico_data \
             --zmq-addr tcp://192.168.1.20:5555

      Continue only after the output refreshes continuously and ``grip``, ``trigger``, ``A/B``, and ``X/Y`` update as you press them.

   .. tab-item:: 2. Collect & Prep Checkpoint

      **Collect PICO demonstrations** on the single-host robot machine with ``.venv`` active (single-node runs auto-start Ray):

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_dual_franka_collect_data_pico

      **Controls during collection:**

      1. Wear the headset facing the front of the workspace and pull ``trigger`` to calibrate the operator base.

      2. Press ``a`` on the USB keyboard or pedal to start recording (a second ``a`` after 1.5 s aborts and drops the buffer), ``b`` to increment ``segment_id`` (1 s debounce), and ``c`` to mark success and write the LeRobot shard.

      3. Hold left or right ``grip`` to drive the matching arm (releasing holds the current TCP), ``X/Y`` to close or open the left gripper, and ``A/B`` to close or open the right gripper.

      Recordings save under ``logs/<timestamp>/`` in ``demo_buffer/`` and ``lerobot_dataset/rank_0/id_0/``.

      On headless machines, keep ``enable_camera_player: False``. To resume across sessions, set ``data_collection.resume: True`` with a fixed ``data_collection.save_dir``.

      **Prepare the OpenPI π₀.₅ checkpoint** using the SFT and deployment-checkpoint steps in :doc:`dual_franka`. Skip ``backfill_tcp_rot6d.py`` because PICO collection already records ``tcp_rot6d`` actions. Then point ``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml`` at your staged checkpoint (rollout inherits ``actor.model.model_path`` automatically):

      .. code-block:: yaml

         actor:
           model:
             model_path: /path/to/deploy/global_step_N
             openpi_data:
               repo_id: your_repo/tcp_rot6d_v1

   .. tab-item:: 3. Run Online HG-DAgger

      With the 3-node Ray cluster running (see the cluster dropdown under Hardware and Safety), launch online HG-DAgger from the head node:

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_dual_franka_dagger_openpi

      **Runtime controls:**

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - Control
           - Behavior
         * - ``a``
           - Start one autonomous rollout from idle.
         * - Left / right ``grip``
           - Intervene on that arm while the other arm keeps running the policy; holding either ``grip`` composes a 20-D ``[PICO 10D, rollout 10D]`` record and marks the frame as intervened.
         * - ``b``
           - Mark failure, discard the rollout, and reset.
         * - ``c``
           - Mark success, stream the complete episode to the actor as an online LeRobot shard, and reset.

      After every reset, the environment waits in idle for ``a``. The actor archives complete successful episodes and trains only on action chunks whose non-padded frames are all intervention-labeled.

.. dropdown:: Hardware placeholders and PICO configuration

   Key configs for this workflow:

   .. list-table::
      :header-rows: 1
      :widths: 52 48

      * - Config file
        - Role
      * - ``examples/embodiment/config/realworld_dual_franka_collect_data_pico.yaml``
        - PICO collection (three ``realsense`` cameras by default).
      * - ``examples/embodiment/config/realworld_dual_franka_dagger_openpi.yaml``
        - 3-node online HG-DAgger (``realsense`` base and two ``lumos`` wrists by default).
      * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
        - Shared ``tcp_rot6d`` hardware defaults.

   Replace the hardware placeholders for your workstation:

   .. list-table::
      :header-rows: 1
      :widths: 44 56

      * - Placeholder or key
        - What to set
      * - ``LEFT_ROBOT_IP``, ``RIGHT_ROBOT_IP``
        - IP addresses of the left and right Franka arms.
      * - ``BASE_CAMERA_SERIAL``, ``LEFT_CAMERA_SERIAL``, ``RIGHT_CAMERA_SERIAL``
        - librealsense ASIC serials for RealSense, or ``/dev/v4l/by-id`` paths for Lumos.
      * - ``left_gripper_type``, ``right_gripper_type``, ``LEFT_GRIPPER_CONNECTION``, ``RIGHT_GRIPPER_CONNECTION``
        - Gripper type and connection string for each arm.
      * - ``keyboard_device``
        - USB keyboard or pedal event path (collection only).
      * - ``TASK_DESCRIPTION``
        - Language prompt stored in the dataset and used by OpenPI.
      * - ``joint_reset_qpos``
        - First-frame joint means or safe 7-DOF home pose per arm (DAgger).
      * - ``target_ee_pose``
        - Target pose inside the TCP xyz box (for example ``[0.5, ±0.2, 0.5, -3.14, 0, 0]``).

   ``teleop`` binds each arm separately, and ``pico.hand: dual`` maps left and right controllers to the left and right arms.

   Collection sets ``hold_current_when_inactive: True`` under ``env.eval`` so released arms hold TCP. DAgger sets it to ``False`` under ``env.train`` so released arms follow policy rollouts.

   Across machines, bind the publisher to ``tcp://0.0.0.0:<port>`` and set ``zmq_addr`` on the consumer to ``tcp://<vr_publisher_ip>:<port>`` (never ``0.0.0.0``):

   .. code-block:: yaml

      env:
        eval:
          teleop:
            - {pico: {drives: left}}
            - {pico: {drives: right}}
          keyboard_device: null  # or /dev/input/by-id/usb-...-event-kbd
          keyboard_reward_wrapper: start_end
          pico:
            zmq_addr: "ipc:///tmp/vr_data.ipc"
            hand: "dual"
            hold_current_when_inactive: True
            control_trigger: "grip"
            calibration:
              button: "trigger"

.. dropdown:: Arm compliance tuning (optional)

   The collection config sets a shared ``compliance`` block for both arms (see :ref:`Configure Arm Motion <franka-motion-settings>` for defaults and reset behavior):

   .. list-table::
      :header-rows: 1
      :widths: 45 55

      * - Setting
        - Collection default
      * - Translational / rotational stiffness
        - 1000 N/m / 56 Nm/rad
      * - Translational / rotational clip
        - 0.024 m / 0.15 rad
      * - Per-step translation / rotation limit
        - 0.06 m / 0.15 rad
      * - Default ``max_delta_tau``
        - 0.3 Nm per cycle

   With a Robotiq payload at 10 Hz, ``max_step: 0.03`` can trigger ``joint_velocity_violation``; lower ``max_step`` if needed. The DAgger config omits ``compliance`` and uses Franky defaults—copy the collection block to match feel.

   A per-arm ``left_compliance`` or ``right_compliance`` mapping replaces the shared block for that side (omitted keys fall back to Franky defaults; ``{}`` selects all backend defaults):

   .. code-block:: yaml

      compliance:
        translational_stiffness: 900.0
      left_compliance:
        max_step: 0.02

   These Cartesian settings also apply to policy targets during DAgger; GELLO joint control is unaffected.

.. dropdown:: Online HG-DAgger advanced configuration (optional)

   ``realworld_dual_franka_dagger_openpi.yaml`` enables online LeRobot archiving, ``smooth_intervene``, per-arm PICO intervention, and policy-only evaluation:

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

   .. list-table::
      :header-rows: 1
      :widths: 32 68

      * - Setting
        - Behavior
      * - ``online_lerobot.enabled: True``
        - Streams filtered episodes directly to the actor's rolling LeRobot dataset under ``${runner.logger.log_path}/online_lerobot/rank_0/id_<N>/``.
      * - ``smooth_intervene: True``
        - Eliminates action-chunk boundary stalls during intervention by skipping policy inference whenever the last frame of a chunk is intervened. Requires all ``env.train.teleop`` entries to be ``pico`` and one environment per pipeline stage.
      * - ``only_success: True``
        - Discards failed rollouts.
      * - ``only_save_expert: True``
        - Samples only action chunks whose non-padded frames all carry ``intervene_flag=True``.
      * - ``env.eval.teleop: none``
        - Keeps evaluation policy-only. Beta fields are omitted because intervention comes from PICO, with no ``rollout.expert_model``.

Results and Troubleshooting
---------------------------

Monitor online HG-DAgger in TensorBoard with ``tensorboard --logdir ./logs`` and inspect logs and archived shards under ``./logs``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Signal or artifact
     - Metric name or path
   * - Actor loss & optimizer
     - ``train/dagger/actor_loss``, ``train/actor/lr``, ``train/actor/grad_norm``
   * - Online dataset counters
     - ``train/lerobot_dataset/total_episodes``, ``train/lerobot_dataset/physical_frames``, ``train/lerobot_dataset/logical_samples``, ``train/lerobot_dataset/num_sub_datasets``
   * - Episode log
     - ``logs/<timestamp>/run_embodiment.log``
   * - Archived shards
     - ``logs/<timestamp>-realworld_dual_franka_dagger_openpi/online_lerobot/rank_0/``

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - DAgger waits idle after launch
     - Expected under ``keyboard_reward_wrapper: eval_control``; press ``a`` on the USB keyboard or pedal to start a rollout.
   * - Episode keys have no effect
     - Confirm keys come from ``keyboard_device`` / ``RLINF_KEYBOARD_DEVICE`` (USB keyboard or pedal), not the SSH terminal or PICO face buttons, and re-export before ``ray start`` for DAgger.
   * - ``test_pico_data`` receives no packets
     - Verify XRoboToolkit PC Service and ``vr_data_publisher`` are running, and set ``zmq_addr`` to the publisher's reachable IP (never ``0.0.0.0``).
   * - ``joint_velocity_violation`` during fast motion
     - Lower ``compliance.max_step`` (or ``left_compliance`` / ``right_compliance``) for your Robotiq payload and 10 Hz control rate.
   * - Single-host collection attaches to an old cluster
     - Run ``ray stop --force`` before ``collect_data.sh`` and ensure the Python 3.11 virtualenv is active.
