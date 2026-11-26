Franka HG-DAgger
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hg-dagger.jpg
   :align: center
   :width: 80%

   Human-Gated DAgger workflow for collecting interventions and training a Franka policy online.

Overview
--------

Train a real-world Franka OpenPI policy online with Human-Gated DAgger (HG-DAgger): collect PICO demonstrations, compute normalization statistics, run SFT for student initialization, and launch online HG-DAgger with live human intervention. Online LeRobot archives complete successful episodes (policy-executed actions on non-intervention frames and human actions with ``intervene_flag`` on intervention frames); with ``only_save_expert: True``, training samples only action chunks whose non-padded frames are all human interventions.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      Franka Emika Panda (Franky)

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Dual RealSense · PICO VR

   .. grid-item-card:: Workflow
      :text-align: center

      PICO Collect → SFT → Online HG-DAgger

   .. grid-item-card:: Hardware
      :text-align: center

      Controller Node + GPU Node

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Stage
     - Config
     - Description
   * - Collection
     - ``realworld_collect_data_pico``
     - Collect real-world pick-and-place demonstrations with PICO teleoperation.
   * - SFT
     - ``realworld_bin_relocation_sft_openpi``
     - Train the OpenPI student initialization checkpoint.
   * - HG-DAgger
     - ``realworld_pnp_dagger_openpi``
     - Archive complete successful trajectories with online LeRobot and train on fully intervened action chunks.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Dual RealSense camera frames and robot state.
   * - Action
     - OpenPI action chunk decoded to Franka Cartesian and gripper control.
   * - Reward
     - Human-gated PICO intervention signal and episode success outcome.
   * - Prompt
     - Task instruction stored in OpenPI dataset and config metadata.

Hardware and Safety
-------------------

This workflow pairs a Franka controller node (Franka arm, two RealSense cameras, PICO VR teleoperation stream) with a GPU training/rollout node running OpenPI π₀ or π₀.₅. Complete base Franka host setup (firmware check, real-time kernel, FCI activation) in :doc:`franka`, PICO VR streaming in :doc:`franka_vr`, and base OpenPI weights in :doc:`sft_openpi`.

.. warning::

   Keep the emergency stop within reach and supervise every online run. Verify PICO stream latency and trigger calibration before enabling online takeover, clear the workspace, and activate each node's Python environment and ``RLINF_NODE_RANK`` before running ``ray start`` (Ray captures the shell environment at startup).

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Robot / Env Node

      **1. Install**

      Install the Franky ``franka`` environment with ``LIBFRANKA_VERSION`` matching your firmware:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         LIBFRANKA_VERSION=0.19.0 bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. Join the Ray cluster**

      After starting Ray head on the GPU node, join as worker node 1 (replace ``HEAD_NODE_IP`` with the GPU node's IP):

      .. code-block:: bash

         export RLINF_NODE_RANK=1
         export HEAD_NODE_IP=192.168.1.10
         ray start --address="${HEAD_NODE_IP}:6379"

      **3. Collect PICO demonstrations**

      Copy ``examples/embodiment/config/realworld_collect_data_pico.yaml`` to ``my_realworld_pnp_collect.yaml``, switch the environment to bin relocation, and fill in your robot IP, target pose, and camera serials:

      .. code-block:: yaml

         defaults:
           - env/realworld_bin_relocation@env.eval
           - override hydra/job_logging: stdout

         cluster:
           node_groups:
             - label: franka
               hardware:
                 configs:
                   - robot_ip: ROBOT_IP
                     node_rank: 0

         env:
           eval:
             override_cfg:
               target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
               camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

      Start and verify the PICO data stream per :doc:`franka_vr` (the default ``ipc:///tmp/vr_data.ipc`` requires the publisher and env worker on the same machine; use ``tcp://<publisher_ip>:<port>`` in ``env.eval.pico.zmq_addr`` across machines), then launch collection:

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh my_realworld_pnp_collect

      Collection writes replay-buffer trajectories under ``logs/{timestamp}/demo_buffer/`` and LeRobot data under ``logs/{timestamp}/lerobot_dataset/`` (see :doc:`../../guides/data_collection`).

      .. dropdown:: Advanced configuration (optional)

         The shipped PICO collection config sets controller bindings, calibration, and LeRobot export defaults:

         .. code-block:: yaml

            env:
              eval:
                teleop: pico
                pico:
                  zmq_addr: "ipc:///tmp/vr_data.ipc"
                  hand: "right"
                  control_trigger: "grip"
                  control_threshold: 0.85
                  gripper_close_button: "A"
                  gripper_open_button: "B"
                  position_scale: 1.0
                  rotation_scale: 1.0
                  max_stale_s: 0.2
                  calibration:
                    enabled: True
                    required: True
                    auto_calibrate_on_start: True
                    button: "trigger"
                override_cfg:
                  success_hold_steps: 1
                data_collection:
                  enabled: True
                  save_dir: ${runner.logger.log_path}
                  export_format: [replay_buffer, lerobot_dataset]
                  only_success: True
                  robot_type: "panda"
                  fps: 10

   .. tab-item:: Training / Rollout Nodes

      **1. Install**

      Install the OpenPI and Franka environment on the GPU node:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

      **2. Start the Ray head node**

      .. code-block:: bash

         export RLINF_NODE_RANK=0
         export HEAD_NODE_IP=192.168.1.10
         ray start --head --port=6379 --node-ip-address="$HEAD_NODE_IP"

      **3. Compute normalization statistics and run SFT**

      Compute ``pi0_realworld`` normalization statistics for the collected LeRobot dataset (use the same dataset root and ``repo_id`` for SFT; see :doc:`sft_openpi`):

      .. code-block:: bash

         export HF_LEROBOT_HOME=/path/to/lerobot_root
         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi0_realworld \
             --repo-id realworld_franka_bin_relocation

      Update ``examples/sft/config/realworld_bin_relocation_sft_openpi.yaml`` and launch SFT to produce the student initialization checkpoint:

      .. code-block:: yaml

         data:
           train_data_paths: "/path/to/realworld-franka-bin-relocation-dataset"

         actor:
           model:
             model_path: "/path/to/pi0-model"
             openpi:
               config_name: "pi0_realworld"

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

      **4. Launch online async HG-DAgger**

      In ``examples/embodiment/config/realworld_pnp_dagger_openpi.yaml``, set your SFT checkpoint, base model path (rollout inherits ``actor.model.model_path`` automatically), ``repo_id``, ``robot_ip``, ``target_ee_pose``, and ``camera_serials``:

      .. code-block:: yaml

         runner:
           ckpt_path: "/path/to/sft_checkpoint/full_weights.pt"

         actor:
           model:
             model_path: "/path/to/pi0-model"
             openpi_data:
               repo_id: "realworld_franka_bin_relocation"

      Launch HG-DAgger from the Ray head node:

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_pnp_dagger_openpi

      .. dropdown:: Advanced configuration (optional)

         The shipped HG-DAgger config defines the 2-node layout, online LeRobot archiving, and PICO takeover behavior:

         .. code-block:: yaml

            cluster:
              num_nodes: 2
              node_groups:
                - label: "train"
                  node_ranks: 0
                - label: franka
                  node_ranks: 1
                  hardware:
                    type: Franka
                    configs:
                      - robot_ip: ROBOT_IP
                        node_rank: 1

            algorithm:
              dagger:
                only_save_expert: True
                online_lerobot:
                  enabled: True
                  only_success: True
                  robot_type: "panda"
                  fps: 10
                  finalize_interval: 1
                  data_path: ${runner.logger.log_path}/online_lerobot
                  rolling_lerobot_window_size: 50000
                  min_frames: 1
                  lerobot_num_workers: 0

            env:
              train:
                smooth_intervene: True
                teleop: pico
                pico:
                  zmq_addr: "ipc:///tmp/vr_data.ipc"
                  hand: "right"
                  control_trigger: "grip"
                  control_threshold: 0.85
                  gripper_close_button: "A"
                  gripper_open_button: "B"
                  position_scale: 1.0
                  rotation_scale: 1.0
                  max_stale_s: 0.2
                  calibration:
                    enabled: True
                    required: True
                    auto_calibrate_on_start: True
                    button: "trigger"
                override_cfg:
                  target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
                  camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
              eval:
                teleop: none
                override_cfg:
                  target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
                  camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

            actor:
              model:
                openpi:
                  config_name: "pi0_realworld"

         Key behaviors:

         - ``online_lerobot.enabled: True`` streams filtered episodes directly to the actor's rolling LeRobot dataset instead of using a trajectory replay buffer.
         - ``smooth_intervene: True`` skips policy inference while PICO ``grip`` intervention stays active at the end of an action chunk and resumes inference once ``grip`` is released or the episode ends (PICO-only; requires one environment per env-worker pipeline stage). ``env.eval.teleop: none`` keeps evaluation policy-only.
         - ``only_success: True`` discards failed episodes; ``only_save_expert: True`` archives full successful episodes on disk while sampling only chunk starts where every non-padded frame has ``intervene_flag=True`` (reduces to per-frame filtering when ``num_action_chunks: 1``).
         - ``finalize_interval: 1`` writes a LeRobot shard after every successful episode, and ``rolling_lerobot_window_size: 50000`` limits online sampling to the latest 50,000 expert-valid chunk starts while keeping older shards on disk.
         - Beta mixing fields are omitted because real-world human intervention comes from the PICO wrapper rather than ``rollout.expert_model``.

Results and Troubleshooting
---------------------------

Monitor online training in TensorBoard (``tensorboard --logdir ./logs``); see :doc:`Training metrics <../../reference/metrics>` for logging options. Successful episodes are written as LeRobot shards under ``logs/<timestamp>-realworld_pnp_dagger_openpi/online_lerobot/rank_0/id_*/`` (failed episodes are excluded).

Key metrics to watch: ``train/dagger/actor_loss``, ``train/lerobot_dataset/total_episodes``, ``train/lerobot_dataset/physical_frames``, ``train/lerobot_dataset/logical_samples``, ``train/lerobot_dataset/num_sub_datasets``, ``train/actor/lr``, and ``train/actor/grad_norm``.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Fix
   * - Env worker receives no PICO pose updates
     - Confirm the PICO publisher is running per :doc:`franka_vr`; if publisher and env worker run on different machines, replace ``ipc:///tmp/vr_data.ipc`` with ``tcp://<publisher_ip>:<port>`` in ``pico.zmq_addr``.
   * - Workers fail with missing imports or wrong node rank
     - Run ``ray stop``, activate the node's virtual environment, export ``RLINF_NODE_RANK`` (``0`` on head, ``1`` on robot node), and restart Ray.
   * - ``smooth_intervene`` validation error at startup
     - Ensure ``env.train.teleop`` is ``pico`` and each env-worker pipeline stage runs a single environment.
   * - ``train/lerobot_dataset/logical_samples`` stays at ``0``
     - With ``only_save_expert: True`` and ``only_success: True``, an episode must succeed and contain at least one action chunk whose non-padded frames are all human-intervened.
