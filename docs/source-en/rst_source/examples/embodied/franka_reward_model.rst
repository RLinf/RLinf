Franka Reward Models
====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_reward_model.jpg
   :align: center
   :width: 80%

   Franka reward-model loop: collect labeled real-robot data, train a reward model, then use it to produce rewards during real-world RL.

Overview
--------

Replace hand-coded success rules on a Franka arm with a learned reward model during asynchronous RLPD training. Choose between a ResNet frame-level classifier (for visually unambiguous success states) and a Qwen3-VL-4B LoRA motion-trend judge (for dense 5-frame progress shaping) before collecting data, because each model expects a different label format.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      Franka Emika Panda (Franky)

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Single / Dual RealSense · SpaceMouse · Keyboard

   .. grid-item-card:: Workflow
      :text-align: center

      Collect → Train Reward Model → Online RLPD

   .. grid-item-card:: Hardware
      :text-align: center

      Controller Node + NVIDIA GPU Node

Reward Model Comparison
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 18 32

   * - Type
     - Reward Signal
     - Model
     - Pick This When
   * - :ref:`ResNet <franka-resnet-reward-model>`
     - Per-frame success/failure probability that also drives environment resets.
     - ResNet image classifier
     - Task success is visually clear and you can label success/failure frames.
   * - :ref:`Qwen VLM <franka-vlm-reward-model>`
     - Motion-trend score over a 5-frame sliding window, plus a success bonus.
     - Qwen3-VL-4B (LoRA)
     - Sparse success is insufficient and you need dense shaping toward the goal.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Camera frames shared by the policy and reward model (wrist view for ResNet; dual wrist + global views for Qwen VLM).
   * - Action
     - Same 6-D Franka Cartesian action as the base real-world environment.
   * - Reward
     - Learned reward-model prediction (standalone reset trigger for ResNet, or sliding-window trend score plus ``gt_success_bonus`` for Qwen VLM).
   * - Prompt
     - Task text (VLM trend input) or fixed target pose, depending on config.

Hardware and Safety
-------------------

Complete the base Franka host setup (firmware check, real-time kernel, FCI activation, and Ray cluster) in :doc:`franka`, and review the reward-model workflow in :doc:`../../extending/reward_model`. The ResNet workflow uses a single RealSense camera plus a SpaceMouse (and a physical keyboard for Approach 1 labeling). The Qwen VLM workflow uses two RealSense cameras (``wrist_1`` close-up and ``global`` workspace view) and an NVIDIA GPU capable of Qwen3-VL-4B LoRA SFT and bf16 inference (such as an RTX 4090).

.. warning::

   Keep the emergency stop within reach and supervise every collection and RL run. Clear the workspace and verify robot IP, camera serials, and target pose before connecting. With ``standalone_realworld: True`` (ResNet), the reward model directly triggers environment resets—validate its accuracy offline or via live teleoperation inference (see :doc:`../../extending/reward_model`) before launching autonomous RL.

.. _franka-resnet-reward-model:
.. _franka-vlm-reward-model:

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: ResNet Reward Model

      **1. Install**

      Install the Franka environment on both the controller host and GPU node (see :doc:`franka` for host setup):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. Collect expert demos and reward labels**

      First collect expert trajectories for the RLPD demo buffer per :doc:`franka`, confirming ``env.eval.data_collection`` is enabled with ``export_format: [replay_buffer, pickle_episode]`` and ``only_success: True`` in ``examples/embodiment/config/realworld_collect_data.yaml``.

      Then collect reward-model labels using one of two approaches (full options in :doc:`../../extending/reward_model`):

      - **Approach 1 — Keyboard labeling (task-agnostic):** Press ``c`` (success) and ``a`` (fail) during live teleoperation. The script stops once target frame counts are reached and exports ``train.pt`` / ``val.pt`` directly:

        .. code-block:: bash

           bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

      - **Approach 2 — Fixed-pose labeling (target-driven):** Set ``env.eval.override_cfg.success_hold_steps: 20`` during collection (move slowly for diverse failure frames and make small motions while holding the target pose for diverse success frames), then convert ``.pkl`` episodes to ``train.pt`` / ``val.pt``:

        .. code-block:: bash

           python -m examples.reward.preprocess_reward_dataset \
               --raw-data-path logs/xxx/pickle_episode \
               --output-dir logs/xxx/processed_reward_data \
               --fail-success-ratio 3

      **3. Train the ResNet reward model**

      Train the classifier with ``examples/reward/run_reward_training.sh`` following :doc:`../../extending/reward_model` (for real-world datasets, lower ``runner.early_stop.min_delta`` to ``1e-6``). You can also test live reward inference during SpaceMouse teleoperation without RL per :doc:`../../extending/reward_model`.

      **4. Launch real-world RL**

      Configure your cluster and Franka hardware in ``examples/embodiment/config/realworld_charger_sac_cnn_async_standalone_reward.yaml``, point ``reward.model.model_path`` to the trained ResNet checkpoint, and launch on the Ray head node:

      .. code-block:: yaml

         reward:
           use_reward_model: True
           standalone_realworld: True
           reward_threshold: 0.8
           model:
             model_path: /path/to/reward_model_checkpoint
             model_type: "resnet"

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_charger_sac_cnn_async_standalone_reward

      .. dropdown:: Advanced configuration (optional)

         **Expert demo export & early stopping**

         .. code-block:: yaml

            # In realworld_collect_data.yaml
            env:
              eval:
                data_collection:
                  enabled: True
                  save_dir: ${runner.logger.log_path}
                  export_format: [replay_buffer, pickle_episode]
                  only_success: True

            # In reward training config (recommended for real-world data)
            runner:
              early_stop:
                min_delta: 1e-6

         **Keyboard labeling keys (``realworld_collect_dataset``)**

         - ``runner.num_success_frames`` / ``runner.num_fail_frames``: target counts before collection stops.
         - ``runner.val_split``: validation split fraction; ``runner.fail_success_ratio``: failure-frame downsampling ratio.
         - ``env.eval.keyboard_reward_wrapper: single_stage``: enables ``c`` / ``a`` labeling; ``env.eval.teleop`` selects the teleoperation device; ``env.eval.override_cfg.target_ee_pose`` sets the task target pose.

         **Standalone real-world rollout behavior**

         With ``standalone_realworld: True``, the reward model's thresholded success/failure prediction directly replaces the native environment reward and drives environment resets. Probabilities below ``reward_threshold`` are zeroed out, and weighted-sum keys (``reward_mode``, ``reward_weight``, ``env_reward_weight``) have no effect.

   .. tab-item:: Qwen VLM Reward Model

      The Qwen VLM reward model inspects 5-frame dual-view sliding windows with Qwen3-VL-4B (LoRA) and maps motion-trend judgments to scalar rewards: ``positive`` → ``1.0``, ``negative`` → ``-0.2``, ``unclear`` → ``0.0``, and ``invalid`` → ``0.0``. Reaching and holding the target pose records ``success_once`` and adds ``gt_success_bonus`` (``+20.0``).

      **1. Install**

      Install the Franka environment on the robot node (and set up Qwen3-VL SFT dependencies on the GPU node per :doc:`sft_vlm`):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. Collect dual-view episodes**

      In ``examples/embodiment/config/realworld_collect_data.yaml``, configure ``pickle_episode`` export with ``only_success: False``, disable relative frames (``use_relative_frame: False`` so TCP coordinates stay in the absolute base frame), and map your global and wrist camera serials:

      .. code-block:: yaml

         env:
           eval:
             use_relative_frame: False
             override_cfg:
               camera_names:
                 "SERIAL1": "global"
                 "SERIAL2": "wrist_1"
             data_collection:
               enabled: True
               save_dir: /path/to/run
               export_format: pickle_episode
               only_success: False

      Collect 50+ episodes moving slowly and covering both successes and failures (ensure both ``wrist_1`` and ``global`` views clearly show the end effector and target):

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data

      **3. Preprocess and fine-tune Qwen3-VL-4B (LoRA)**

      Slice episodes into 5-frame dual-view windows (writes ``dataset_info.json``, ``train/``, and ``eval/`` with ``segments.jsonl`` and ``pkl/``). Use the second command with ``--target-ee-pose`` (in metres; pass ``X,Y,Z`` for position-only) when raw data has no GAE signal:

      .. code-block:: bash

         # Auto-label from GAE / rewards
         python -m examples.reward.vlm_trend.preprocess_reward_dataset \
             --raw-data-path /path/to/run/pickle_episode \
             --output-dir /path/to/processed_vlm_trend_reward_data \
             --window-size 5 \
             --task-description "Pick up the peg and insert it into the hole."

         # Or auto-label from TCP distance to target pose (--delta-threshold in metres)
         python -m examples.reward.vlm_trend.preprocess_reward_dataset \
             --raw-data-path /path/to/run/pickle_episode \
             --output-dir /path/to/processed_vlm_trend_reward_data \
             --window-size 5 \
             --target-ee-pose "X,Y,Z,RX,RY,RZ" \
             --delta-threshold 0.005

      Update ``actor.model.model_path`` to your local ``Qwen3-VL-4B-Instruct`` directory in ``examples/sft/config/qwen3vl_sft_vlm_trend_reward.yaml``, export ``VLM_TREND_REWARD_DATA_ROOT``, and run LoRA SFT:

      .. code-block:: bash

         export VLM_TREND_REWARD_DATA_ROOT=/path/to/processed_vlm_trend_reward_data
         bash examples/sft/run_vlm_sft.sh qwen3vl_sft_vlm_trend_reward

      Record the exported LoRA checkpoint path (for example ``checkpoints/global_step_3000``).

      **4. Launch real-world RL**

      In ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_vlm_reward.yaml``, set ``reward.model.model_path`` (base Qwen3-VL-4B), ``reward.model.lora_path`` (LoRA checkpoint), and your Franka cluster fields, then launch from the Ray head node:

      .. code-block:: yaml

         reward:
           model:
             model_path: "/path/to/Qwen3-VL-4B-Instruct"
             lora_path: "/path/to/qwen3-vl-lora-checkpoint"

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async_vlm_reward

      .. dropdown:: Advanced configuration (optional)

         **SFT config (``examples/sft/config/qwen3vl_sft_vlm_trend_reward.yaml``)**

         The shipped SFT config leaves ``gradient_checkpointing: false`` and uses ``sharding_strategy: no_shard`` (single RTX 4090); set ``gradient_checkpointing: true`` if you need lower GPU memory usage:

         .. code-block:: yaml

            data:
              type: vlm
              dataset_name: "vlm_trend_reward_sft"
              train_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/train/segments.jsonl"
              val_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/eval/segments.jsonl"
              video_root: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}"

            actor:
              model:
                model_type: qwen3_vl
                model_path: /path/to/Qwen3-VL-4B-Instruct
                is_lora: true
                lora_rank: 16
                attn_implementation: flash_attention_2
              fsdp_config:
                gradient_checkpointing: false
                sharding_strategy: no_shard

         **RL reward config (``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_vlm_reward.yaml``)**

         .. code-block:: yaml

            reward:
              use_reward_model: true
              worker_type: model
              group_name: "RewardGroup"
              standalone_realworld: False
              reward_mode: history_buffer
              history_reward_assign: true
              reward_weight: 1.0
              env_reward_weight: 0.0
              reward_threshold: 0.5
              model:
                model_path: "/path/to/Qwen3-VL-4B-Instruct"
                model_type: "buffered_vlm"
                lora_path: "/path/to/qwen3-vl-lora-checkpoint"
                gt_success_bonus: 20.0
                precision: "bf16"
                input_builder_name: vlm_trend_reward_input_builder
                input_builder_params:
                  default_task_description: "Pick up the peg and insert it into the hole."
                reward_parser_name: vlm_trend_reward_parser
                reward_parser_params:
                  positive_reward: 1.0
                  negative_reward: -0.2
                  unclear_reward: 0.0
                  invalid_reward: 0.0
                history_buffers:
                  history_window:
                    history_size: 5
                    min_history_size: 5
                    input_interval: 1
                    history_keys:
                      - main_images
                      - extra_view_images
                    input_on_done: false
                interval_reward: 0.0
                max_new_tokens: 16
                do_sample: false
                temperature: 0.0

            cluster:
              num_nodes: 2
              component_placement:
                actor:
                  node_group: "4090"
                env:
                  node_group: franka
                reward:
                  node_group: "4090"

         **Key parameters and reward flow**

         - ``reward_mode: history_buffer`` caches the last 5 dual-view frames (``main_images`` and ``extra_view_images``) per environment and triggers local HuggingFace inference (``worker_type: model``) once ``min_history_size`` frames accumulate (incomplete windows return ``0.0``).
         - ``history_reward_assign: true`` back-assigns the parsed VLM trend reward to every step in the window.
         - ``env_reward_weight: 0.0`` relies on VLM trend shaping plus ``gt_success_bonus: 20.0`` whenever the episode records ``success_once`` upon holding the target pose.

Results and Troubleshooting
---------------------------

Monitor reward distributions, VLM trend outputs, and ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); see :doc:`Training metrics <../../reference/metrics>` for logging configuration.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Fix
   * - ``reward_weight`` or ``reward_mode`` has no effect (ResNet)
     - Expected when ``standalone_realworld: True`` is set; the reward model directly replaces the native environment reward and triggers resets.
   * - ResNet reward training stops too early on real data
     - Lower ``runner.early_stop.min_delta`` to ``1e-6`` in the reward training config.
   * - VLM trend preprocessing fails or produces neutral labels without GAE
     - Pass ``--target-ee-pose "X,Y,Z,RX,RY,RZ"`` and tune ``--delta-threshold`` (in metres; start around ``0.005``), and ensure collection used ``use_relative_frame: False``.
   * - Qwen3-VL-4B LoRA SFT runs out of GPU memory
     - Set ``actor.fsdp_config.gradient_checkpointing: true`` in ``qwen3vl_sft_vlm_trend_reward.yaml``.
