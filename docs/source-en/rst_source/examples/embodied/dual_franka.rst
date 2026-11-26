Dual-Arm Franka OpenPI JAX
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy.jpg
   :align: center
   :width: 80%
   :alt: Dual-Franka deployment

   Dual-Franka data collection, fine-tuning, and deployment workflow.

Collect dual-arm joint demonstrations with GELLO, convert them to ``tcp_rot6d`` data, fine-tune OpenPI π₀.₅ on a GPU node, and deploy the checkpoint back to your two-node Franka rig.

Overview
--------

This guide walks through the end-to-end dual-Franka workflow: teleoperating two arms locally with GELLO leaders, converting recorded joint trajectories to Cartesian ``tcp_rot6d`` actions, running OpenPI π₀.₅ SFT, and deploying the trained checkpoint for evaluation. For dataset cleaning/merging, ``sft2deploy`` checkpoint conversion, and single-workstation RTC deployment, see :doc:`dual_franka_openpi_pytorch`; for single-arm setup and GELLO assembly, see :doc:`franka` and :doc:`franka_gello`.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      2× Franka (Franky) · 2× Robotiq

   .. grid-item-card:: Teleop & Cameras
      :text-align: center

      2× GELLO · FootSwitch · RealSense + 2× Lumos

   .. grid-item-card:: Workflow
      :text-align: center

      GELLO Collect → Backfill → π₀.₅ SFT → Eval Deploy

   .. grid-item-card:: Hardware
      :text-align: center

      2 robot nodes (x86_64) · 1 GPU training node

.. include:: _dual_franka_setup.rst

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Install & Check

      **Robot nodes** (``node 0`` and ``node 1``): dual-arm Franka uses Franky, whose prebuilt x86_64 wheels bundle libfranka ``0.15.0`` or ``0.19.0`` (default). Match ``LIBFRANKA_VERSION`` to the `Franka compatibility matrix <https://frankarobotics.github.io/docs/compatibility.html>`_ (or build a wheel and pass ``FRANKY_WHEEL``). Install GELLO dependencies on ``node 0`` by following :doc:`franka_gello`.

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **Training node** (remote GPU server): install OpenPI dependencies for SFT.

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **Hardware checks on** ``node 0`` (before starting Ray): grant foot-pedal permissions (replace ``event12`` with your resolved event device), verify USB-3 camera links (``5000M`` in ``lsusb -t``; ``480M`` indicates USB 2 fallback), and confirm smooth GELLO readings.

      .. code-block:: bash

         # Foot pedal (replace event12 with your resolved device)
         ls -l /dev/input/by-id/*-event-kbd
         sudo chmod 666 /dev/input/event12
         export RLINF_KEYBOARD_DEVICE=/dev/input/event12

         # Cameras (expect RealSense serial, two Lumos paths, and 5000M speed)
         rs-enumerate-devices | grep -E "Name|Serial|USB Type"
         ls /dev/v4l/by-id/
         lsusb -t

         # GELLO leaders (plug one at a time to identify FTDI paths)
         ls /dev/serial/by-id/ | grep -i ftdi
         python -m rlinf.robotics.parts.teleop.gello_joint \
             --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0  # replace with your left GELLO device

      Expected GELLO output refreshes continuously as ``joints=[+0.012 -0.604 ...]  gripper=[0.035]``. If values stall or jump by ``2π``, run ``align-sequential`` (see *GELLO calibration* above).

   .. tab-item:: 2. Collect & SFT

      **Start Ray** on both robot nodes from the RLinf repository root with ``.venv`` active (Ray captures ``RLINF_NODE_RANK`` and ``RLINF_KEYBOARD_DEVICE`` at ``ray start`` time):

      .. code-block:: bash

         # node 0 (replace IP and event device for your station)
         export RLINF_NODE_RANK=0
         export RLINF_KEYBOARD_DEVICE=/dev/input/event12
         ray start --head --port=6379 --node-ip-address=192.168.1.10

         # node 1 (replace IPs for your station)
         export RLINF_NODE_RANK=1
         ray start --address=192.168.1.10:6379 --node-ip-address=192.168.1.20

      Confirm ``ray status`` on ``node 0`` shows both nodes ``ALIVE``.

      **Collect GELLO demonstrations** on ``node 0`` after ``align-sequential`` (see *GELLO calibration* above) passes:

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh \
             realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

      In a second ``node 0`` terminal, run ``python -m toolkits.realworld_check.collect_monitor logs/collect.log``. Foot-pedal controls: ``a`` starts recording (press again to abort and drop the buffer), ``b`` increments ``segment_id`` for sub-task boundaries, and ``c`` marks success and writes the LeRobot shard. Set ``data_collection.resume: true`` with the same ``data_collection.save_dir`` to append new ``id_*`` shards.

      **Convert to tcp_rot6d and sync to the training node:**

      .. code-block:: bash

         export HF_LEROBOT_HOME="$HOME/lerobot_data"     # replace with your LeRobot data root
         export DATA_REPO_ID="my_org/dual_franka_demo"   # replace with your dataset repo ID
         export SFT_REPO_ID="$DATA_REPO_ID/tcp_rot6d_v1"
         export TRAINER_IP="192.168.1.50"                # replace with your training node IP

         python -m toolkits.dual_franka.backfill_tcp_rot6d \
             --src "$HF_LEROBOT_HOME/$DATA_REPO_ID/joint_v1" \
             --dst "$HF_LEROBOT_HOME/$SFT_REPO_ID"

         ssh "$TRAINER_IP" "mkdir -p $HF_LEROBOT_HOME/$SFT_REPO_ID"
         rsync -av "$HF_LEROBOT_HOME/$SFT_REPO_ID/" \
             "$TRAINER_IP:$HF_LEROBOT_HOME/$SFT_REPO_ID/"

      **Run SFT on the training node** (from the RLinf repository root with ``.venv`` active) after updating ``SFT_DATASET_REPO_ID``, ``PI05_BASE_CKPT``, logger settings, and placement in ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``:

      .. code-block:: bash

         export HF_LEROBOT_HOME="$HOME/lerobot_data"            # replace with your LeRobot data root
         export DUAL_FRANKA_DATA_ROOT="$HF_LEROBOT_HOME"
         export PI05_BASE_CKPT="$HOME/models/pi05_torch"        # replace with your base checkpoint dir
         export SFT_REPO_ID="my_org/dual_franka_demo/tcp_rot6d_v1"

         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi05_dualfranka_tcp_rot6d \
             --repo-id "$SFT_REPO_ID"

         mkdir -p "$PI05_BASE_CKPT/$SFT_REPO_ID"
         cp "./assets/pi05_dualfranka_tcp_rot6d/$SFT_REPO_ID/norm_stats.json" \
            "$PI05_BASE_CKPT/$SFT_REPO_ID/norm_stats.json"      # adjust assets dir if customized

         bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_tcp_rot6d

      Checkpoints are saved to ``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``.

   .. tab-item:: 3. Deploy & Evaluate

      **Sync the checkpoint and matching normalization stats** back to ``node 0`` so ``$DEPLOY_CKPT`` contains both ``actor/model_state_dict/full_weights.pt`` and ``<repo_id>/tcp_rot6d_v1/norm_stats.json``:

      .. code-block:: bash

         export TRAINER_IP="192.168.1.50"                       # replace with your training node IP
         export TRAIN_LOG="./logs/sft_run"                      # replace with your SFT log dir on trainer
         export DEPLOY_CKPT="$HOME/deploy/global_step_10000"    # replace with your target step/dir
         export SFT_REPO_ID="my_org/dual_franka_demo/tcp_rot6d_v1"

         mkdir -p "$DEPLOY_CKPT/actor/model_state_dict" "$DEPLOY_CKPT/$SFT_REPO_ID"

         rsync -av \
             "$TRAINER_IP:$TRAIN_LOG/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt" \
             "$DEPLOY_CKPT/actor/model_state_dict/full_weights.pt"
         rsync -av \
             "$TRAINER_IP:$TRAIN_LOG/checkpoints/global_step_10000/$SFT_REPO_ID/norm_stats.json" \
             "$DEPLOY_CKPT/$SFT_REPO_ID/norm_stats.json"

      Set ``rollout.model.model_path`` to ``$DEPLOY_CKPT`` and ``actor.model.openpi_data.repo_id`` to ``<repo_id>/tcp_rot6d_v1`` in ``examples/embodiment/config/realworld_eval_dual_franka.yaml``.

      **Launch evaluation** on ``node 0`` using the active two-node Ray cluster (see the :doc:`real-world evaluation guide <../../evaluations/guides/realworld>` for options):

      .. code-block:: bash

         bash evaluations/run_eval.sh realworld_eval_dual_franka

      Deployment foot-pedal controls: ``a`` starts policy execution from idle, ``b`` marks failure and resets, and ``c`` marks success and resets. After every reset, the wrapper waits in idle for ``a`` so you can reset the scene.

.. dropdown:: Config files and required placeholders

   Edit the repository-provided configs and replace every field marked with ``# Replace:``:

   .. list-table::
      :header-rows: 1
      :widths: 52 48

      * - Config
        - Purpose
      * - ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
        - GELLO joint-space collection
      * - ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
        - π₀.₅ SFT on converted ``tcp_rot6d`` data
      * - ``examples/embodiment/config/realworld_eval_dual_franka.yaml``
        - Real-world eval-only deployment
      * - ``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
        - Shared joint-space hardware defaults
      * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
        - Shared ``tcp_rot6d`` hardware defaults

   Required placeholders:

   * ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``: FCI IP visible from each controller node.
   * ``BASE_CAMERA_SERIAL``, ``LEFT_CAMERA_SERIAL``, ``RIGHT_CAMERA_SERIAL``: RealSense serial and stable ``/dev/v4l/by-id`` paths.
   * ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION``: stable ``/dev/serial/by-id`` paths for the Robotiq adapters.
   * ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT``: stable ``/dev/serial/by-id`` paths for the two GELLO leaders.
   * ``TASK_DESCRIPTION``: natural-language prompt shared across collection, SFT, and deployment.
   * ``SFT_DATASET_REPO_ID``: converted dataset ID (typically ``<repo_id>/tcp_rot6d_v1``).
   * ``MODEL_PATH``: staged deployment checkpoint directory on ``node 0``.

Results and Troubleshooting
---------------------------

Monitor SFT metrics on the training node with ``tensorboard --logdir ./logs`` and track collection progress on ``node 0`` with ``python -m toolkits.realworld_check.collect_monitor logs/collect.log``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Ray worker import failure
     - In the shell that ran ``ray start``, check ``which python`` and ``python -c "import franky, gello, gello_teleop"``. Inspect ``/tmp/ray/session_latest/logs/worker-*.err``.
   * - Foot pedal permission denied
     - Re-run ``sudo chmod 666 /dev/input/eventXX`` and confirm ``RLINF_KEYBOARD_DEVICE`` points to that exact device before ``ray start``.
   * - RealSense shows USB 2 speed
     - Replace the USB cable or port so ``lsusb -t`` reports ``5000M`` instead of ``480M``.
   * - GELLO stops streaming or jumps by ``2π``
     - Power-cycle the leader, replug the FTDI adapter, verify with ``gello_joint --port ...``, and re-run ``align-sequential`` (see *GELLO calibration* above).
   * - One arm stalls during reset
     - Run ``ping -c 100 <robot_ip>`` on that controller node; fix any NIC/FCI packet loss or power-cycle the robot.
   * - Deployment cannot locate ``norm_stats.json``
     - Confirm the file exists at ``<model_path>/<actor.model.openpi_data.repo_id>/norm_stats.json``.
   * - Deployment stays idle after launch
     - Verify foot-pedal permissions and press ``a``; the eval wrapper waits in idle between episodes by design.
