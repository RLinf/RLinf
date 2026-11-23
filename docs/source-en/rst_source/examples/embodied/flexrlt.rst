FlexRLT: Online RL from a Frozen VLA
====================================

Train a compact actor and critic from a frozen VLA checkpoint.

Overview
--------

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_flow.png
   :alt: FlexRLT rollout, optional STEAM routing, replay, and actor-critic updates
   :width: 100%

   The VLA stays frozen while the actor and critic learn from replay. This preset uses STEAM to decide when the actor takes control; other presets can use the environment gate.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Environment
     - Task
     - Preset
   * - ManiSkill
     - PegInsertionSideWideClearance-v1
     - ``maniskill_prefix_stage2_td3_mlp_steam``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Observation
     - Action
     - Reward
   * - Main/wrist images, language instruction, joint state
     - Joint-delta control: 10 steps × 8 values per chunk
     - Sparse insertion reward

The preset uses 4 GPUs.

Compared with RL Token
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Choice
     - :doc:`RL Token <rlt>`
     - FlexRLT
   * - Representation
     - RL tokens learned in Stage 1
     - Pooled VLA features, or the same learned RL tokens
   * - Before online RL
     - Train the token transformer with the VLA
     - Pooling can start from a base or SFT VLA checkpoint
   * - Components
     - The RLT recipe's actor-critic combination
     - Choose the feature model, actor, critic, and algorithm separately

This recipe uses pooled image-prefix features and TD3. The RL-token route remains available through the ManiSkill RLT presets.

Installation
------------

.. include:: _setup_common.rst

Use the OpenPI environment in the ``rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`` image (``source switch_env openpi``), or install locally:

.. code:: bash

   # Add --use-mirror for mainland China.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Download the Model
------------------

Download the feature VLA, expert VLA, STEAM value model, and calibrated phase head:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Resource
     - Hugging Face repository
   * - Feature VLA, step 1000
     - `ChikaYokoyama/maniskill_rlt_1000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_1000_new>`_
   * - Expert VLA, step 3000
     - `ChikaYokoyama/maniskill_rlt_3000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_3000_new>`_
   * - STEAM value model, step 5000
     - `ChikaYokoyama/rlt_steam_value_model_sft_step_5000 <https://huggingface.co/ChikaYokoyama/rlt_steam_value_model_sft_step_5000>`_
   * - Phase head and its saved recommendation
     - `ChikaYokoyama/rlt_steam_phase_head <https://huggingface.co/ChikaYokoyama/rlt_steam_phase_head>`_

Download the four sets of weights into ``data/steam_peg``, and also download the SigLIP and Gemma backbones STEAM uses for images and language.

.. code:: bash

   CHECKPOINT_ROOT="$(pwd)/data/steam_peg"
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download ChikaYokoyama/maniskill_rlt_1000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/maniskill_rlt_3000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
     --local-dir "$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
     --exclude 'dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_phase_head \
     --local-dir "$CHECKPOINT_ROOT"

   hf download google/siglip-so400m-patch14-384 \
     --local-dir "$CHECKPOINT_ROOT/siglip-so400m-patch14-384"
   hf download google/gemma-3-270m \
     --local-dir "$CHECKPOINT_ROOT/gemma-3-270m"

Set the Model Paths
-------------------

In ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml``, replace these placeholders with absolute paths. Expand ``<CHECKPOINT_ROOT>`` to the download directory:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Field
     - Local path
   * - ``rollout.prefix_feature_model.model_path``
     - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_1000/actor``
   * - ``rollout.expert_model.model_path``
     - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor``
   * - ``rollout.routing_gate.model.model_path``
     - ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor``
   * - ``rollout.routing_gate.actor_switch.phase_head_path``
     - ``<CHECKPOINT_ROOT>/steam_phase_head.pt``

Keep ``openpi_data.repo_id: maniskill_peginsertionside_joint`` and ``norm_stats_path: null``. Each OpenPI checkpoint then reads its own ``maniskill_peginsertionside_joint/norm_stats.json``.

Then point the STEAM checkpoint at the backbones you downloaded. Edit ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``config.json`` field
     - Local path
   * - ``vision_repo_id``
     - ``<CHECKPOINT_ROOT>/siglip-so400m-patch14-384``
   * - ``language_repo_id``
     - ``<CHECKPOINT_ROOT>/gemma-3-270m``

The tokenizer and image-processor files stay in that same ``actor`` directory. The gate loads them from there.

Run It
------

From the repository root, with the environment activated:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

Logs and TensorBoard events are under ``logs/<timestamp>-maniskill_prefix_stage2_td3_mlp_steam/``. After data collection and warmup updates, STEAM allows actor control in critical phases. The expert gate records takeover recommendations in ``shadow`` mode.

Expected Results
----------------

The curves compare RL Token (TD3) with FlexRLT on ManiSkill peg insertion, using 64 training environments and 256 evaluation environments. RL Token uses 8 GPUs; the FlexRLT runs use 4 GPUs and a frozen π0.5 with STEAM. SR is the fraction of episodes that succeed at least once (``success_once``).

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_maniskill_sr.png
   :alt: RL Token and FlexRLT training and evaluation success rates on ManiSkill peg insertion
   :width: 100%

   Train shows raw SR and a trailing 20-step mean; eval shows the recorded checkpoints.

.. list-table:: ManiSkill Peg Insertion Success Rate
   :header-rows: 1

   * - Method
     - GPUs
     - Train mean, steps 235–254
     - Eval, step 250
     - Best eval through 250
   * - RL Token (TD3)
     - 8
     - 57.19%
     - 57.03%
     - 59.38%
   * - FlexRLT, min-Q (run 1)
     - 4
     - 56.09%
     - 61.72%
     - 61.72%
   * - FlexRLT, min-Q (run 2)
     - 4
     - 57.34%
     - 62.89%
     - 62.89%
   * - FlexRLT, TD3 (Q1)
     - 4
     - 59.61%
     - 56.25%
     - 60.16%

Calibrate STEAM for Another Task
--------------------------------

Collect phase features after changing the task, cameras, or controlling policy:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

The preset runs 20 steps using the environment's geometry gate and a shadow expert gate. It sets ``rollout.routing_gate.actor_switch.collect_phase_features: true`` and ``algorithm.rlt_gate_calibration.enable: true``. Traces appear under ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/trace_*.pt``.

Fit the phase head offline:

.. code:: bash

   TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
   python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
     --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
     --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

Point the training preset's ``phase_head_path`` at the output. Leave gate thresholds ``null`` to use the saved recommendations; the YAML is a report. Disable both collection switches for ordinary training.

For feature, network, and algorithm changes, see :doc:`../../extending/flexrlt`.
