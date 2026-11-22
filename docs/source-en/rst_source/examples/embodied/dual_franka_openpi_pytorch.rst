Dual-Arm Franka OpenPI PyTorch
==============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka-deploy-rlinf.jpg
   :align: center
   :width: 80%
   :alt: RLinf PyTorch dual-Franka deployment

   Fine-tune and deploy a dual-Franka policy with RLinf PyTorch.

Collect dual-arm GELLO demonstrations, clean and merge shards, convert them to ``tcp_rot6d`` data, fine-tune OpenPI π₀.₅ natively in RLinf PyTorch, and deploy with standard two-node evaluation or single-machine Real-Time Chunking (RTC).

Overview
--------

This guide extends the dual-Franka GELLO pipeline (:doc:`dual_franka`) with native OpenPI PyTorch fine-tuning, dataset cleaning and shard merging, ``sft2deploy`` checkpoint conversion, and optional single-workstation RTC deployment. For single-arm setup and GELLO assembly, see :doc:`franka` and :doc:`franka_gello`.

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

      Collect → Clean/Merge → Backfill → PyTorch SFT → Deploy/RTC

   .. grid-item-card:: Hardware
      :text-align: center

      2 robot nodes (or 1 GPU workstation for RTC) · GPU SFT node

.. include:: _dual_franka_setup.rst

For single-workstation RTC deployment, attach both arms, all three cameras, and the GPU to one host.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Install & Check

      **Robot nodes** (``node 0`` and ``node 1``): prebuilt Franky wheels on x86_64 bundle libfranka ``0.15.0`` or ``0.19.0`` (default). Match ``LIBFRANKA_VERSION`` to the `Franka compatibility matrix <https://frankarobotics.github.io/docs/compatibility.html>`_ (or pass a custom wheel in ``FRANKY_WHEEL``). Install GELLO dependencies on ``node 0`` following :doc:`franka_gello`.

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **Training node** (remote GPU server): install OpenPI PyTorch dependencies.

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **Hardware checks on** ``node 0`` (before starting Ray): configure foot-pedal permissions (replace ``event12`` with your resolved event device), verify USB-3 camera links (``5000M`` in ``lsusb -t``), and check both GELLO leaders.

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

      **Start Ray** on both robot nodes from the RLinf repository root with ``.venv`` active (export ``RLINF_NODE_RANK`` and ``RLINF_KEYBOARD_DEVICE`` before ``ray start``):

      .. code-block:: bash

         # node 0 (replace IP and event device for your station)
         export RLINF_NODE_RANK=0
         export RLINF_KEYBOARD_DEVICE=/dev/input/event12
         ray start --head --port=6379 --node-ip-address=192.168.1.10

         # node 1 (replace IPs for your station)
         export RLINF_NODE_RANK=1
         ray start --address=192.168.1.10:6379 --node-ip-address=192.168.1.20

      Confirm ``ray status`` on ``node 0`` reports both nodes ``ALIVE``.

      **Collect demonstrations** on ``node 0`` after ``align-sequential`` (see *GELLO calibration* above) passes:

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh \
             realworld_collect_data_gello_joint_dual_franka 2>&1 | tee logs/collect.log

      Monitor progress in another terminal with ``python -m toolkits.realworld_check.collect_monitor logs/collect.log``. Foot-pedal controls: ``a`` starts recording (press again to abort and drop the buffer), ``b`` increments ``segment_id``, and ``c`` marks success and writes the LeRobot shard. Set ``data_collection.resume: true`` with the same ``data_collection.save_dir`` to append new ``id_*`` shards.

      **Clean, merge, backfill tcp_rot6d, and sync to the training node:**

      .. code-block:: bash

         # Optional: dry-run episode removal and multi-shard merge (drop --dry-run to apply)
         python -m toolkits.dual_franka.delete_lerobot \
             --data-dir ./logs/collect_run/lerobot_dataset/rank_0 --delete "id_0:3,5" --dry-run
         python -m toolkits.dual_franka.merge_lerobot \
             --rank-dir ./logs/collect_run/lerobot_dataset --out-dir ./logs/collect_run/lerobot_dataset \
             --log-file ./logs/collect.log --dry-run

         # Backfill joint data to tcp_rot6d and sync to training node
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

      **Run OpenPI PyTorch SFT** on the training node from the RLinf repository root with ``.venv`` active (first convert the base ``openpi-jax`` checkpoint as shown in :doc:`sft_openpi`, and update ``train_data_paths``, ``model_path``, and ``openpi_data.norm_stats_path`` in ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``):

      .. code-block:: bash

         export HF_LEROBOT_HOME="$HOME/lerobot_data"            # replace with your LeRobot data root
         export SFT_REPO_ID="my_org/dual_franka_demo/tcp_rot6d_v1"

         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi05_dualfranka_tcp_rot6d \
             --repo-id "$SFT_REPO_ID"

         bash examples/sft/run_vla_sft.sh realworld_sft_openpi_dual_franka_tcp_rot6d

      Checkpoints are saved under ``<log_path>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``.

   .. tab-item:: 3. Deploy & Evaluate

      **Convert the SFT checkpoint to deployment format** before syncing to ``node 0``:

      .. code-block:: bash

         python -m rlinf.utils.ckpt_convertor.openpi.convert \
             --mode sft2deploy \
             --ckpt ./logs/sft_run/checkpoints/global_step_10000 \
             --output "$HOME/deploy/global_step_10000" \
             --reference-model "$HOME/models/pi05_base_openpi" \
             --dtype-reference "$HOME/models/deploy_ref/full_weights.pt"  # replace paths for your setup

      **Sync the deployment weights and matching normalization stats** to ``node 0`` so ``$DEPLOY_CKPT`` contains both ``actor/model_state_dict/full_weights.pt`` and ``<repo_id>/tcp_rot6d_v1/norm_stats.json``:

      .. code-block:: bash

         export TRAINER_IP="192.168.1.50"                       # replace with your training node IP
         export TRAINER_DEPLOY_CKPT="$HOME/deploy/global_step_10000"  # --output dir from sft2deploy on trainer
         export TRAIN_LOG="./logs/sft_run"                      # replace with your SFT log dir on trainer
         export DEPLOY_CKPT="$HOME/deploy/global_step_10000"    # replace with your target step/dir on node 0
         export SFT_REPO_ID="my_org/dual_franka_demo/tcp_rot6d_v1"

         mkdir -p "$DEPLOY_CKPT/actor/model_state_dict" "$DEPLOY_CKPT/$SFT_REPO_ID"

         rsync -av \
             "$TRAINER_IP:$TRAINER_DEPLOY_CKPT/actor/model_state_dict/full_weights.pt" \
             "$DEPLOY_CKPT/actor/model_state_dict/full_weights.pt"
         rsync -av \
             "$TRAINER_IP:$TRAIN_LOG/checkpoints/global_step_10000/$SFT_REPO_ID/norm_stats.json" \
             "$DEPLOY_CKPT/$SFT_REPO_ID/norm_stats.json"

      Set ``rollout.model.model_path`` to ``$DEPLOY_CKPT`` and ``actor.model.openpi_data.repo_id`` to ``<repo_id>/tcp_rot6d_v1`` in ``examples/embodiment/config/realworld_eval_dual_franka.yaml``.

      **Launch evaluation** on ``node 0`` (see the :doc:`real-world evaluation guide <../../evaluations/guides/realworld>` for options). Use ``realworld_eval_dual_franka`` for two-node deployment, or ``realworld_dual_franka_pi05_RTC`` when the GPU and both arms share one workstation with overlapped inference:

      .. code-block:: bash

         # Two-node deployment
         bash evaluations/run_eval.sh realworld_eval_dual_franka

         # Single-workstation RTC deployment
         bash evaluations/run_eval.sh realworld_dual_franka_pi05_RTC

      After homing, press ``a`` on the keyboard or foot pedal to start policy execution (``b`` marks failure and resets; ``c`` marks success and resets).

.. dropdown:: Config files, norm stats, and checkpoint conversion notes

   Edit the repository-provided configs and replace every field marked with ``# Replace:``:

   .. list-table::
      :header-rows: 1
      :widths: 52 48

      * - Config
        - Purpose
      * - ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
        - GELLO joint-space collection
      * - ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
        - OpenPI PyTorch π₀.₅ SFT on ``tcp_rot6d`` data
      * - ``examples/embodiment/config/realworld_eval_dual_franka.yaml``
        - Two-node real-world deployment
      * - ``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml``
        - Single-workstation RTC deployment
      * - ``examples/embodiment/config/env/realworld_dual_franka_joint.yaml``
        - Shared joint-space hardware defaults
      * - ``examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml``
        - Shared ``tcp_rot6d`` hardware defaults

   Key fields to verify:

   * ``LEFT_ROBOT_IP`` / ``RIGHT_ROBOT_IP``: FCI IPs visible from each controller node.
   * ``BASE_CAMERA_SERIAL``, ``LEFT_CAMERA_SERIAL``, ``RIGHT_CAMERA_SERIAL``: RealSense serial and stable ``/dev/v4l/by-id`` paths.
   * ``LEFT_GRIPPER_CONNECTION`` / ``RIGHT_GRIPPER_CONNECTION`` and ``LEFT_GELLO_PORT`` / ``RIGHT_GELLO_PORT``: stable ``/dev/serial/by-id`` paths.
   * ``TASK_DESCRIPTION``: natural-language task prompt shared across collection, SFT, and deployment.
   * ``openpi_data.norm_stats_path``: path to the ``norm_stats.json`` written by ``calculate_norm_stats.py`` (typically ``./assets/pi05_dualfranka_tcp_rot6d/<repo_id>/norm_stats.json``); omit this key to use the OpenPI TrainConfig default under ``model_path``.
   * Base ``openpi-jax`` conversion follows :doc:`sft_openpi`; to convert RLinf SFT ``full_weights.pt`` checkpoints to OpenPI safetensors layout, see :doc:`../../guides/convertor` or inspect the CLI flags:

   .. code-block:: bash

      python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi --help

Results and Troubleshooting
---------------------------

Monitor SFT loss and learning rate on the training node with ``tensorboard --logdir ./logs`` and track live collection on ``node 0`` with ``python -m toolkits.realworld_check.collect_monitor logs/collect.log``.

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
