FlexRLT
========

Train a compact actor and critic online from frozen VLA prefix features, using STEAM routing to control when the learned policy takes over.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      Frozen π₀.₅ + MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      TD3 · AC + STEAM

   .. grid-item-card:: Task
      :text-align: center

      ManiSkill Peg Insertion

   .. grid-item-card:: Hardware
      :text-align: center

      4 × NVIDIA GPUs

| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · 4 GPUs (80 GB recommended) · OpenPI + ManiSkill environment.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_flow.png
   :alt: FlexRLT rollout, optional STEAM routing, replay, and actor-critic updates
   :width: 100%

   The VLA stays frozen while the actor and critic learn from replay. This preset uses STEAM to decide when the actor takes control; other presets can use the environment gate.

Task, Observation, and Comparison with RL Token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The preset ``maniskill_prefix_stage2_td3_mlp_steam`` runs ``PegInsertionSideWideClearance-v1`` in ManiSkill with main/wrist images, language instruction, and joint state, outputting 10-step × 8-dim joint-delta chunks under sparse insertion reward.

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

This recipe uses pooled image-prefix features and TD3. For swapping the feature model, policy heads, or update rule, see :doc:`../../extending/flexrlt`.

Setup and Run
-------------

.. include:: _setup_common.rst

You can either use the prebuilt embodied image on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ (``source switch_env openpi``) or install locally below.

.. tab-set::

   .. tab-item:: ManiSkill Peg Insertion (TD3 + STEAM)
      :sync: flexrlt-maniskill

      **1. Install**

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env maniskill_libero
         source .venv/bin/activate

      **2. Download Checkpoints & Backbones**

      Download the feature VLA (`maniskill_rlt_1000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_1000_new>`__), expert VLA (`maniskill_rlt_3000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_3000_new>`__), STEAM value model (`rlt_steam_value_model_sft_step_5000 <https://huggingface.co/ChikaYokoyama/rlt_steam_value_model_sft_step_5000>`__), and calibrated phase head (`rlt_steam_phase_head <https://huggingface.co/ChikaYokoyama/rlt_steam_phase_head>`__) into ``./data/steam_peg``:

      .. code-block:: bash

         CHECKPOINT_ROOT="$(pwd)/data/steam_peg"
         hf download ChikaYokoyama/maniskill_rlt_1000_new \
           --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/maniskill_rlt_3000_new \
           --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
           --local-dir "$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
           --exclude 'dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_phase_head --local-dir "$CHECKPOINT_ROOT"

      Download the SigLIP and Gemma backbones used by STEAM:

      .. code-block:: bash

         hf download google/siglip-so400m-patch14-384 \
           --local-dir "$CHECKPOINT_ROOT/siglip-so400m-patch14-384"
         hf download google/gemma-3-270m \
           --local-dir "$CHECKPOINT_ROOT/gemma-3-270m"

      **3. Configure Paths & Launch**

      Update the four absolute paths in ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml`` (keep ``openpi_data.repo_id: maniskill_peginsertionside_joint`` and ``norm_stats_path: null``):

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

      In ``rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json`` under ``<CHECKPOINT_ROOT>``, point ``vision_repo_id`` and ``language_repo_id`` to the downloaded SigLIP and Gemma directories (keep the tokenizer and image-processor files in that actor directory). Then launch training from the repository root:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

      Logs and TensorBoard events are written under ``logs/<timestamp>-maniskill_prefix_stage2_td3_mlp_steam/``. After data collection and warmup updates, STEAM enables actor control in critical phases while the expert gate logs takeover recommendations in ``shadow`` mode.

      .. dropdown:: Calibrate STEAM for another task (optional)

         Collect 20 steps of phase features whenever the task, cameras, or controlling policy changes:

         .. code-block:: bash

            bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

         Traces appear under ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/trace_*.pt``. Fit the phase head offline:

         .. code-block:: bash

            TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
            python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
              --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
              --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

         Point ``phase_head_path`` at the new ``.pt`` file, leave gate thresholds ``null`` to load the saved recommendations, and disable both calibration collection switches for normal training.

Visualization and Results
-------------------------

Track ``env/success_once`` and ``eval/success_once`` with ``tensorboard --logdir ./logs``; see :doc:`Training Metrics </rst_source/reference/metrics>` for metric definitions and logging backends.

The curves below compare RL Token (TD3, 8 GPUs) with FlexRLT (4 GPUs, frozen π₀.₅ + STEAM) on ManiSkill peg insertion using 64 training environments and 256 evaluation environments.

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
