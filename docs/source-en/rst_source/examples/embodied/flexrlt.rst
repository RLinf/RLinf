FlexRLT
========

Train a compact actor and critic online from frozen VLA prefix features, using STEAM routing to control when the learned policy takes over.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      TD3 · AC + STEAM

   .. grid-item-card:: Models
      :text-align: center

      Frozen π₀.₅ + MLP

   .. grid-item-card:: Environments / Data
      :text-align: center

      ManiSkill Peg Insertion

   .. grid-item-card:: Training
      :text-align: center

      Online Off-Policy RL (4 GPUs)

| **You'll do:** install OpenPI + ManiSkill → download checkpoints & backbones → launch ``maniskill_prefix_stage2_td3_mlp_steam`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · 4 × NVIDIA GPUs (80 GB recommended).

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt_flow.png
   :alt: FlexRLT rollout, optional STEAM routing, replay, and actor-critic updates
   :width: 100%

   The VLA stays frozen while the actor and critic learn from replay. This preset uses STEAM to decide when the actor takes control; other presets can use the environment gate.

Comparison with RL Token
~~~~~~~~~~~~~~~~~~~~~~~~

The preset ``maniskill_prefix_stage2_td3_mlp_steam`` runs ``PegInsertionSideWideClearance-v1`` in ManiSkill with main/wrist RGB images, language instruction, and joint state, outputting 10-step × 8-dim joint-delta chunks under sparse insertion reward.

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

This recipe uses pooled image-prefix features and TD3. To swap the feature model, policy heads, or update rule, see :doc:`../../extending/flexrlt`.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: ManiSkill (TD3 + STEAM)
      :sync: flexrlt-maniskill

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Download Checkpoints & Backbones**

      Download the feature VLA, expert VLA, STEAM value model, calibrated phase head, and SigLIP/Gemma backbones into ``./data/steam_peg``, then point the STEAM value model's ``config.json`` at the local backbones:

      .. code-block:: bash

         CKPT_ROOT="$(pwd)/data/steam_peg"
         hf download ChikaYokoyama/maniskill_rlt_1000_new \
           --local-dir "$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/maniskill_rlt_3000_new \
           --local-dir "$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
           --exclude '*/dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
           --local-dir "$CKPT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
           --exclude 'dcp_checkpoint/*'
         hf download ChikaYokoyama/rlt_steam_phase_head --local-dir "$CKPT_ROOT"
         hf download google/siglip-so400m-patch14-384 --local-dir "$CKPT_ROOT/siglip-so400m-patch14-384"
         hf download google/gemma-3-270m --local-dir "$CKPT_ROOT/gemma-3-270m"

         python -c "import json, pathlib; r = pathlib.Path('$CKPT_ROOT'); p = r / 'rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json'; d = json.loads(p.read_text()); d['vision_repo_id'], d['language_repo_id'] = str(r / 'siglip-so400m-patch14-384'), str(r / 'gemma-3-270m'); p.write_text(json.dumps(d, indent=2))"

      **3. Launch**

      .. code-block:: bash

         CKPT_ROOT="$(pwd)/data/steam_peg"
         bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam \
           rollout.prefix_feature_model.model_path="$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_1000/actor" \
           rollout.expert_model.model_path="$CKPT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor" \
           rollout.routing_gate.model.model_path="$CKPT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
           rollout.routing_gate.actor_switch.phase_head_path="$CKPT_ROOT/steam_phase_head.pt"

      Logs and TensorBoard events are written under ``logs/<timestamp>-maniskill_prefix_stage2_td3_mlp_steam/``. After warmup updates, STEAM activates actor control in critical phases while the expert gate logs takeover recommendations in ``shadow`` mode.

      .. dropdown:: Calibrate STEAM for another task (optional)

         Whenever the task, cameras, or controlling policy changes, collect 20 steps of phase features and fit the phase head offline:

         .. code-block:: bash

            bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration
            TRACE_DIR=$(ls -d "$(pwd)"/logs/*-maniskill_prefix_stage2_td3_mlp_steam_calibration/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration | tail -n 1)
            python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
              --phase-head-output "$(pwd)/data/steam_peg/steam_phase_head.pt" \
              --yaml-output "$(pwd)/data/steam_peg/steam_gate_recommendation.yaml" --device cpu

         Pass the new ``steam_phase_head.pt`` to ``phase_head_path`` and keep gate thresholds ``null`` to load the calibrated recommendations.

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
