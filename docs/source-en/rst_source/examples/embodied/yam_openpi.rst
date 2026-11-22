YAM π₀.₅ SFT & Evaluation
=========================

Merge demonstrations from :doc:`yam`, compute normalization statistics, fine-tune OpenPI Pi0.5 on a GPU node, and evaluate the checkpoint on a dual-arm YAM setup (or deploy with RPent).

Overview
--------

This guide takes merged LeRobot data from :doc:`yam` through ``pi05_yam_joint`` normalization, base checkpoint conversion from JAX to OpenPI PyTorch, supervised fine-tuning, and two-node real-world evaluation.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      2× YAM · ``flexible_4310`` grippers

   .. grid-item-card:: Teleop & Cameras
      :text-align: center

      3× RealSense RGB (top / left / right)

   .. grid-item-card:: Workflow
      :text-align: center

      Merge & Norm Stats → π₀.₅ SFT → Real-World Eval

   .. grid-item-card:: Hardware
      :text-align: center

      GPU node (rank 0) · YAM controller (rank 1)

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - Top, left-wrist, and right-wrist RealSense RGB views plus 14-D state ``[left joints (6), left gripper, right joints (6), right gripper]`` (radians; grippers in ``[0, 1]``).
   * - Action
     - 14-D absolute joint and gripper targets (trained as state-relative deltas for the 12 joints and absolute targets for grippers, normalized and padded to 32-D). The model predicts 30 steps; evaluation executes the first 5 before re-querying.
   * - Reward & Prompt
     - Task text from ``meta/tasks.jsonl`` tokenized alongside normalized state. Evaluation logs execution metrics with zero step reward; the operator judges task success from live motion and video.

Hardware and Safety
-------------------

SFT runs on your GPU node. Evaluation uses a two-node Ray cluster with the GPU node at rank ``0`` and the YAM controller at rank ``1`` (both on the same RLinf revision with identical dependencies; see :doc:`../../guides/multi_node`). Configure CAN interfaces and camera serials following :doc:`yam`, and configure a shared Ray authentication token following the `Ray self-managed cluster instructions <https://docs.ray.io/en/latest/ray-security/token-auth.html#self-managed-clusters>`_ (keep the token private and never stop another user's Ray cluster).

.. warning::

   Evaluation connects to both CAN devices and all three cameras. If the SDK's ``gripper_limits`` are unset, connection automatically opens and closes each gripper for calibration—keep both grippers empty and their travel clear. Keep an operator at the emergency stop to supervise the entire run. Before the episode ends or the process stops, confirm both arms stay safely supported when motor output is released; reset and parking remain operator-controlled by default.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Install & Prepare Data

      **Install on both the GPU node and the YAM controller** (using the same RLinf revision):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env yam && source .venv/bin/activate

      **Merge shards and compute norm stats on the YAM controller** (set ``YAM_COLLECTION`` to the run directory containing ``lerobot_dataset/``):

      .. code-block:: bash

         export YAM_COLLECTION="$PWD/logs/20261121-120000-realworld_dual_yam_collect_data_pico"  # replace with your run dir
         export YAM_DATASET="$HOME/yam-data/merged"
         mkdir -p "$(dirname "$YAM_DATASET")"

         # Merge successful takes (pass --dry-run first to preview)
         python -m toolkits.lerobot.merge_lerobot_datasets \
             --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
             --only-success

         # Check recorded task prompt and compute norm_stats.json
         head -n 5 "$YAM_DATASET/meta/tasks.jsonl"
         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi05_yam_joint --repo-id "$YAM_DATASET"

      Use the exact task string from ``meta/tasks.jsonl`` as your evaluation prompt and keep the same ``norm_stats.json`` for both training and evaluation.

      **Make the merged dataset available on the GPU node:** point ``YAM_DATASET`` to shared storage, or copy over SSH from the RLinf directory on the GPU node (replace ``user``, ``192.168.1.20``, and ``/data/yam_merged`` with ``whoami``, ``hostname -I``, and ``echo "$YAM_DATASET"`` from the controller):

      .. code-block:: bash

         export YAM_DATASET="$HOME/yam-data/merged"
         mkdir -p "$(dirname "$YAM_DATASET")"
         scp -r 'user@192.168.1.20:/data/yam_merged' "$YAM_DATASET"  # replace user, IP, and source path

   .. tab-item:: 2. SFT Training

      **Download and convert the base Pi0.5 checkpoint** on the GPU node (``gsutil`` is bundled by the installer):

      .. code-block:: bash

         export YAM_MODELS="$HOME/yam-models"
         mkdir -p "$YAM_MODELS/jax"
         gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base "$YAM_MODELS/jax/"
         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
             --input-model "$YAM_MODELS/jax/pi05_base" \
             --input-norm-stats "$YAM_DATASET/norm_stats.json" \
             --output-model "$YAM_MODELS/pi05_base_openpi" \
             --output-norm-stats "$YAM_MODELS/pi05_base_openpi/yam/norm_stats.json" \
             --action-dim 32 --action-horizon 30 --max-token-len 200

      **Configure three absolute paths** in ``examples/sft/config/realworld_dual_yam_sft_openpi_pi05.yaml`` (or pass them via Hydra CLI overrides below):

      .. list-table::
         :header-rows: 1
         :widths: 52 48

         * - Field
           - Value
         * - ``data.train_data_paths``
           - Merged ``YAM_DATASET`` directory.
         * - ``actor.model.model_path``
           - ``YAM_MODELS/pi05_base_openpi``.
         * - ``actor.model.openpi_data.norm_stats_path``
           - ``YAM_DATASET/norm_stats.json``.

      The config uses FP32 weights, BF16 compute, and ``no_shard`` (:doc:`sft_openpi`), so each actor GPU holds full weights, gradients, and optimizer states. On a single 48 GB GPU, start with ``actor.micro_batch_size: 1``, ``actor.global_batch_size: 1``, and ``actor.model.precision: bf16`` (verify quality for your task; keep global batch divisible by micro batch times actor workers).

      **Launch SFT and locate saved checkpoints:**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh realworld_dual_yam_sft_openpi_pi05 \
             data.train_data_paths="$YAM_DATASET" \
             actor.model.model_path="$YAM_MODELS/pi05_base_openpi" \
             actor.model.openpi_data.norm_stats_path="$YAM_DATASET/norm_stats.json"

         # In another terminal on the GPU node, list checkpoints containing actor/model_state_dict/full_weights.pt
         find logs -type d -path '*/checkpoints/global_step_*'

   .. tab-item:: 3. Evaluate & Deploy

      **Verify connectivity and start the 2-node Ray cluster** with ``.venv`` active and a shared auth token. Export ``GPU_IP`` and ``CONTROL_IP`` (from ``hostname -I``; verify mutual ``ping -c 1``), and export ``LEFT_CAN_CHANNEL``, ``RIGHT_CAN_CHANNEL``, and ``CAMERA_SERIALS`` on the controller (:doc:`yam`) before starting Ray:

      .. code-block:: bash

         # GPU node (rank 0)
         export RAY_AUTH_MODE=token
         export YAM_TASK_DESCRIPTION='pick the block'
         export RLINF_NODE_RANK=0
         ray start --head --port=6379 --node-ip-address="$GPU_IP"

         # YAM controller (rank 1; use the same YAM_TASK_DESCRIPTION)
         export RAY_AUTH_MODE=token
         export YAM_TASK_DESCRIPTION='pick the block'
         export RLINF_NODE_RANK=1
         ray start --address="$GPU_IP:6379" --node-ip-address="$CONTROL_IP"

      Confirm ``ray status`` on the GPU node lists two alive nodes, position both arms for the task, and verify joint limits, camera views, e-stop, and workspace clearance.

      **Launch real-world evaluation** on the GPU node (set ``YAM_CHECKPOINT`` to the chosen ``global_step_<N>`` directory):

      .. code-block:: bash

         bash evaluations/run_eval.sh realworld_dual_yam_openpi_pi05_eval \
             rollout.model.model_path="$YAM_CHECKPOINT" \
             rollout.model.openpi_data.norm_stats_path="$YAM_DATASET/norm_stats.json"

.. dropdown:: RPent deployment (optional)

   RPent's YAM OpenPI server uses the same ``pi05_yam_joint`` contract (14-D targets, top/left/right cameras, 30-step prediction, 5 executed steps at 30 Hz). Stop native evaluation first so only one process owns the arms, then run the commands below on both RPent nodes from the RLinf repository directory with ``.venv`` active:

   .. code-block:: bash

      export RPENT_RLINF_ROOT="$(pwd)"
      git clone --branch main https://github.com/RLinf/RPent.git "$HOME/RPent-yam"
      uv pip install -e "$HOME/RPent-yam[yam]" "numpy<2"

   Follow the RPC and controller startup sections in the `RPent YAM deployment instructions <https://github.com/RLinf/RPent/blob/main/docs/source-en/rst_source/real_world_robots/yam.rst>`_ using your fine-tuned checkpoint, ``norm_stats.json``, and ``RPENT_RLINF_ROOT``.

Results and Troubleshooting
---------------------------

Monitor SFT curves on the GPU node with ``tensorboard --logdir logs --port 6006``. Evaluation prints its run log directory and saves rollout videos there for visual grading.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Out-of-memory error during SFT on a 48 GB GPU
     - Set ``actor.micro_batch_size: 1``, ``actor.global_batch_size: 1``, and ``actor.model.precision: bf16`` in the SFT YAML, check free memory with ``nvidia-smi``, and increase batch size only after one optimizer step completes.
   * - Ray worker fails to join the head node
     - Verify mutual ``ping -c 1`` between ``GPU_IP`` and ``CONTROL_IP``, confirm ``RAY_AUTH_MODE=token`` uses identical tokens on both hosts, and check ``ray status`` before launching.
   * - Grippers move unexpectedly on eval startup
     - Expected when SDK ``gripper_limits`` are unset; keep grippers empty and clear during startup calibration.
   * - RPent controller conflicts with native eval
     - Stop ``run_eval.sh`` completely before starting RPent so only one process controls the CAN interfaces.
