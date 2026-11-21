Using HG-DAgger with Franka
===========================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hg-dagger.jpg
   :align: center
   :width: 80%

   Human-Gated DAgger workflow for collecting interventions and training a Franka policy online.

Train a real-world Franka policy with Human-Gated DAgger. You will collect intervention data, compute OpenPI normalization stats, run SFT, and then launch online HG-DAgger. During the online stage, LeRobot archives complete successful episodes: non-intervention frames retain the actions actually executed by the policy, while intervention frames store the human actions and ``intervene_flag``. With ``only_save_expert: True``, training samples only action chunks whose non-padded frames are all human interventions.

Overview
--------

Use human-gated interventions to improve a real-world Franka policy online.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀ / π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · HG-DAgger

   .. grid-item-card:: Tasks
      :text-align: center

      Real-world PnP

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · PICO

| **You'll do:** collect intervention data → compute norm stats → run SFT → launch HG-DAgger → monitor interventions.
| **Prerequisites:** :doc:`franka` · :doc:`franka_vr` · :doc:`sft_openpi` · Ray cluster · trained or base OpenPI checkpoint.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - Task
     - Config / entry point
     - Description
   * - Collection
     - ``realworld_collect_data_pico``
     - Collect real-world demonstrations with PICO.
   * - SFT
     - ``realworld_bin_relocation_sft_openpi``
     - Train the student initialization.
   * - HG-DAgger
     - ``realworld_pnp_dagger_openpi``
     - Archive complete successful trajectories with online LeRobot and train on fully intervened action chunks.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - Field
     - Description
   * - Observation
     - Franka camera frames and optional robot state.
   * - Action
     - OpenPI action decoded to Franka real-world control.
   * - Reward
     - Human-gated intervention signal and task outcome.
   * - Prompt
     - Task text in OpenPI dataset/config metadata.

Installation
------------

The real-world pipeline uses separate environments on the robot node and the GPU training / rollout node (or pull a prebuilt image from `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__). In mainland China, append ``--use-mirror`` to ``requirements/install.sh`` and set ``export HF_ENDPOINT=https://hf-mirror.com`` before downloading Hugging Face checkpoints.

.. tab-set::

   .. tab-item:: Robot / Env Node

      Install the Franky-based ``franka`` environment with ``LIBFRANKA_VERSION`` matching your firmware (or use the ``franky`` venv in the Franka Docker image; see :doc:`franka`):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

   .. tab-item:: Training / Rollout Nodes

      Install the OpenPI environment on the GPU node (same as simulator Pi0 DAgger in :doc:`dagger`):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

Cluster Setup
-------------

Before launching any collection or training job, finish the Ray setup described in :doc:`franka`. The training / rollout node is typically the Ray head (``RLINF_NODE_RANK=0``), while the Franka controller node is the worker (``RLINF_NODE_RANK=1``). Replace ``HEAD_NODE_IP`` with the head node's IP address:

.. code-block:: bash

   # On the training / rollout node
   export RLINF_NODE_RANK=0
   export HEAD_NODE_IP=192.168.1.10
   ray start --head --port=6379 --node-ip-address="$HEAD_NODE_IP"

   # On the robot / env node
   export RLINF_NODE_RANK=1
   export HEAD_NODE_IP=192.168.1.10
   ray start --address="${HEAD_NODE_IP}:6379"

Ray records the current Python interpreter and environment variables at startup, so make sure each node has sourced the correct environment before ``ray start``.

Run It
------

1. Collect Human-Guided Real-World Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``examples/embodiment/config/realworld_collect_data_pico.yaml`` (for example to ``my_realworld_pnp_collect.yaml``). Switch the env from peg insertion to bin relocation and fill in your robot IP, camera serials, and target pose:

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

Before launching, start and verify the PICO data stream as described in :doc:`franka_vr`. The default ``ipc:///tmp/vr_data.ipc`` address requires the publisher and env worker to run on the same machine; use ``tcp://<publisher_ip>:<port>`` in ``env.eval.pico.zmq_addr`` when they run on different machines.

Launch collection with your copied config:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh my_realworld_pnp_collect

During teleoperation, the same run writes:

- replay-buffer trajectories under ``logs/{timestamp}/demo_buffer/``
- LeRobot data under ``logs/{timestamp}/lerobot_dataset/``

For the collection format, see :doc:`../../guides/data_collection`.

.. dropdown:: Advanced configuration (optional)

   The shipped collection config already sets PICO button bindings, calibration, and LeRobot export defaults:

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

2. Compute Normalization Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before SFT or HG-DAgger, compute OpenPI normalization stats for the collected
LeRobot dataset:

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

Use the same dataset root and dataset id that you will use for SFT. More
OpenPI-specific dataset notes are documented in :doc:`sft_openpi`.

3. Run OpenPI SFT
~~~~~~~~~~~~~~~~~

Edit ``examples/sft/config/realworld_bin_relocation_sft_openpi.yaml`` before launch:

.. code-block:: yaml

   data:
     train_data_paths: "/path/to/realworld-franka-bin-relocation-dataset"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_realworld"

Then run:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

The SFT checkpoint is the student initialization for the online stage. For more
OpenPI SFT details, see :doc:`sft_openpi`.

4. Run Async HG-DAgger on the Real Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``examples/embodiment/config/realworld_pnp_dagger_openpi.yaml``, set your robot IP, cameras, target pose, SFT checkpoint, and base model path (rollout inherits ``actor.model.model_path`` automatically):

.. code-block:: yaml

   runner:
     ckpt_path: "/path/to/sft_checkpoint/full_weights.pt"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi_data:
         repo_id: "realworld_franka_bin_relocation"

Update ``cluster.node_groups`` (``robot_ip``) and ``env.{train,eval}.override_cfg`` (``target_ee_pose``, ``camera_serials``) for your hardware, then launch HG-DAgger from the Ray head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_pnp_dagger_openpi

.. dropdown:: Advanced configuration (optional)

   The shipped config already sets cluster layout, online LeRobot archiving, and PICO intervention defaults:

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

   ``online_lerobot.enabled: True`` enables the online LeRobot data path. The env worker collects rollouts by episode and sends episodes that satisfy the configured filters to the actor; the actor adds them to its rolling LeRobot dataset for training, so online training does not use the trajectory replay buffer.

   ``smooth_intervene: True`` bypasses policy inference when PICO intervention continues through the last frame of an action chunk. The env worker uses a dummy chunk to keep stepping the teleoperation wrapper and resumes normal inference after ``grip`` is released or the episode ends. It is PICO-only: ``env.train.teleop`` must be ``pico``, and each env-worker pipeline stage must run one environment. ``env.eval.teleop: none`` keeps evaluation policy-only.

   ``only_success: True`` discards failed episodes. ``only_save_expert: True`` keeps the complete successful episode in the LeRobot archive, while restricting the online sampler to chunk starts where every non-padded frame in the action chunk has ``intervene_flag=True``. For ``num_action_chunks: 1``, this reduces to filtering individual frames. Within a successful episode:

   * non-intervention frames remain in the physical archive without entering expert-only training samples;
   * human-intervention frames store the actions actually executed by the intervention device and carry ``intervene_flag=True``;
   * ``finalize_interval: 1`` writes a LeRobot shard after every successful episode;
   * ``rolling_lerobot_window_size: 50000`` limits online sampling to the most recent 50,000 expert-valid logical chunk starts, while older shards remain on disk.

   The real-world DAgger config omits beta-related fields because it does not configure ``rollout.expert_model``. Beta only controls action mixing between a model expert and the student; real-world human intervention is controlled by the PICO intervention wrapper under ``env.train``.

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. Online LeRobot Data**

Each successful episode is written under the current run's log directory:

.. code-block:: text

   logs/<timestamp>-realworld_pnp_dagger_openpi/online_lerobot/rank_0/id_0/
   logs/<timestamp>-realworld_pnp_dagger_openpi/online_lerobot/rank_0/id_1/
   ...

Failed episodes do not enter the online dataset.

**3. Useful Monitoring Metrics**

- ``train/dagger/actor_loss``: Supervised loss computed from expert-only action chunks.
- ``train/lerobot_dataset/total_episodes``: Number of successful episodes received by the actor.
- ``train/lerobot_dataset/physical_frames``: Number of received LeRobot physical frames.
- ``train/lerobot_dataset/logical_samples``: Number of expert-valid trainable chunk starts in the rolling window.
- ``train/lerobot_dataset/num_sub_datasets``: Number of currently loaded LeRobot shards.
- ``train/actor/lr``: Learning rate.
- ``train/actor/grad_norm``: Gradient norm.
