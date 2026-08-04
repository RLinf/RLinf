STEAM: Ensemble Advantage Modeling for Offline Policy Optimization
==================================================================

Run the **STEAM** pipeline in RLinf. STEAM is an offline policy-optimization
recipe that scores existing data with a **pair-classification progress critic**
and a **deep ensemble**, turning the conservative worst-of-N ensemble estimate
into per-frame advantage labels. Those labels then drive the same
**Classifier-Free Guidance (CFG) training** used by :doc:`RECAP <recap>`.

Like RECAP, STEAM needs no online environment interaction, so it suits
real-robot settings where large-scale online sampling is impractical. The
difference is the value signal: instead of regressing discounted returns, STEAM
learns a **temporal-progress** critic from frame pairs and aggregates an
ensemble of critics to suppress the advantage over-estimation a single predictor
would assign to out-of-distribution rollouts.

Overview
--------

Improve a policy offline (no new rollouts) by scoring existing data with an
ensemble progress critic and steering with classifier-free guidance.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      STEAM (worst-of-N ensemble)

   .. grid-item-card:: Models
      :text-align: center

      SigLIP + Gemma3 critic · π₀.₅

   .. grid-item-card:: Environments / Data
      :text-align: center

      LeRobot datasets

   .. grid-item-card:: Training
      :text-align: center

      Offline · 3 stages

| **You'll do:** SFT an ensemble progress critic → compute ensemble advantages → CFG-train the policy → evaluate.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · SigLIP + Gemma3 + π₀.₅ checkpoints · LeRobot-format datasets (steps below).

Pipeline
--------

A STEAM run is two STEAM-specific stages followed by a CFG training stage:

.. code-block:: text

   ┌────────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
   │  Step 1                │     │  Step 2                │     │  Step 3              │
   │  STEAM Value Model SFT │────▶│  Compute Ensemble      │────▶│  CFG Training        │
   │                        │     │  Advantages            │     │                      │
   │  Train an ensemble of  │     │  Worst-of-N ensemble   │     │  Train the policy    │
   │  pair-classification   │     │  signed score -> bool  │     │  with classifier-    │
   │  progress critics      │     │  advantage labels      │     │  free guidance       │
   └────────────────────────┘     └────────────────────────┘     └──────────────────────┘

**Core Idea**

1. **Value Model SFT**: Train an ensemble of progress critics (SigLIP + Gemma3
   backbone + classifier head). Each member sees a frame pair
   :math:`(o_t, o_{t+k})` and classifies the signed frame stride into bins, so
   the head predicts *temporal progress* rather than a regressed return.

2. **Compute Ensemble Advantages**: For every frame, run all ensemble members on
   the pair :math:`(o_t, o_{t+k})` and aggregate with the **worst-of-N** rule
   (:math:`A = \min_m A_m`), yielding a signed score ``advantage_continuous``
   :math:`\in [-1, 1]`. Frames are then labelled positive/negative under a
   threshold or quantile rule.

3. **CFG Training**: Hand the advantage labels to the CFG stage — positive
   (high-advantage) samples are conditional inputs and negative samples are
   unconditional inputs, enabling classifier-free guidance for policy
   optimization.

How STEAM Works
---------------

**STEAM Core Components**

1. **Advantage modeling**

   STEAM (*Self-supervised Temporal Ensemble Advantage Modeling*) learns
   advantages from the *temporal order* of expert demonstrations alone — no
   rewards, human labels, or external value model. For a frame pair
   :math:`(f_i, f_j)` from an expert episode, the **temporal offset** is the
   signed frame stride :math:`j - i`: pairing a frame with a future frame
   supervises forward progress, while feeding the pair in reverse gives a
   negative offset, exposing regressive motion from successful demos alone.
   Offsets are **normalized by trajectory length**
   (:math:`\propto L_{\max}/L_\tau`) so the target measures *temporal efficiency*
   rather than raw step count — shorter, more efficient executions score higher,
   slower or suboptimal ones lower.

   Each predictor (a SigLIP vision encoder + Gemma3 language model + a
   task-specific head) maps the frame pair and language instruction to a
   categorical distribution over :math:`N` (``num_bins``) temporal-offset bins,
   trained with a cross-entropy loss against the binned offset target. The
   per-member advantage subtracts a fixed **baseline** offset from the predicted
   expected bin, so it scores progress *relative to the expected pace*:

   .. math::

      A_m = \frac{2}{N}\left( E_{b \sim p_{\theta_m}}[b] - b_{\mathrm{ref}} \right) \in [-1, 1]

   where :math:`E_{b}[b]` is the expected bin index of predictor :math:`m`'s
   distribution and :math:`b_{\mathrm{ref}}` is the deterministic reference — the
   length-normalized ground-truth offset for a fixed lookahead :math:`H` on the
   longest episode. :math:`A_m` is high near efficient progress and low (or
   negative) near stalls and regressions. (``num_bins == 2`` reduces to a binary
   progress classifier.)

2. **Advantage estimation**

   A single predictor can over-estimate on out-of-distribution rollout states.
   Members agree in-distribution but diverge in unfamiliar states, so STEAM
   aggregates the :math:`M` predictors with the conservative **worst-of-N** rule
   — penalizing high variance to suppress false positives:

   .. math::

      A_{\text{STEAM}} = \min_{m \in \{1, \dots, M\}} A_m

   :math:`A_{\text{STEAM}}` is written to ``advantage_continuous``; per-member
   mean / min / variance are recorded for diagnostics. Because different data
   sources have different advantage distributions, ``advantage_continuous`` is
   turned into the boolean ``advantage`` per source under one of two
   ``label_mode`` rules:

   - ``threshold``: ``advantage = advantage_continuous > positive_threshold`` for
     rollout frames (a signed-score threshold in :math:`[-1, 1]`); sft frames are
     always True (success demos by construction).
   - ``quantile``: label the top ``rollout_quantile`` fraction of rollout frames
     True and, when ``expert_quantile`` is set, the top ``expert_quantile``
     fraction of sft frames True — the two pools are scored independently.

3. **Classifier-Free Guidance (CFG) Training**

   STEAM advantage labels drive the CFG stage on the OpenPI (π₀.₅) policy:
   positive (high-advantage) samples serve as conditional inputs and negative
   samples as unconditional inputs, enabling classifier-free guidance for policy
   optimization. See :doc:`the CFG training stage <recap>` for the full CFG mechanism
   (``positive_only_conditional``, ``unconditional_prob``, ``cfgrl_guidance_scale``).

Installation
------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

STEAM shares the OpenPI environment with RECAP.

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

Please switch to the OpenPI virtual environment via the built-in ``switch_env`` utility:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Download the Model
------------------

The STEAM value model is built from two pretrained backbones:

- **SigLIP-so400m** (``google/siglip-so400m-patch14-384``): vision encoder
- **Gemma3-270M** (``google/gemma-3-270m``): language model and tokenizer

.. code:: bash

   # Download models (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/google/siglip-so400m-patch14-384
   git clone https://huggingface.co/google/gemma-3-270m

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
   hf download google/gemma-3-270m --local-dir gemma-3-270m

Set the paths in the model config (``examples/offline_rl/config/model/steam_value_model.yaml``):

.. code:: yaml

   actor:
     model:
       vision_repo_id: /path/to/siglip-so400m-patch14-384
       language_repo_id: /path/to/gemma-3-270m
       tokenizer_path: /path/to/gemma-3-270m

Data Preparation
----------------

STEAM uses datasets in the LeRobot format, categorized into two types:

- **SFT datasets**: Expert-level demonstrations (successful expert trajectories).
- **Rollout datasets**: Trajectories collected from online interaction (containing
  both successes and failures), plus human-intervention data.

Example dataset configuration:

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

.. note::

   Keep ``train_data_paths`` and ``data.k`` consistent between Step 1 and Step 2:
   the advantage computation must score pairs at the same temporal stride the
   critic was trained on.

Pipeline Tag System
~~~~~~~~~~~~~~~~~~~~~

STEAM uses an **advantage tag** for data passing across steps. Unlike RECAP,
STEAM has no compute-returns step, so there is no ``returns_tag`` — the only tag
is the **advantage_tag**: written by Step 2 and read by Step 3. Ensure that
Step 2's ``advantage.tag`` and Step 3's ``data.advantage_tag`` are consistent so
CFG reads ``meta/advantages_{tag}.parquet``.

.. list-table:: **Tag Flow Across Pipeline Steps**
   :header-rows: 1

   * - Step
     - Config Field
     - Description
   * - 2
     - ``advantage.tag``
     - Writes ``meta/advantages_{tag}.parquet``
   * - 3
     - ``data.advantage_tag``
     - Reads ``meta/advantages_{tag}.parquet``

Step 1: Value Model SFT
-----------------------

Train the ensemble progress critic. Each member is a SigLIP + Gemma3 backbone
with a classifier head; members are cloned from a shared backbone and their value
heads are re-seeded so ensemble variance is a meaningful epistemic signal.

**Configuration**

The config is ``examples/offline_rl/config/steam_value_model_sft.yaml``; the model
defaults live in ``examples/offline_rl/config/model/steam_value_model.yaml``. Key fields:

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
     k: 32                       # max signed stride K (pair temporal scale)
     # Image (view) names the critic loads per frame; must match the views the
     # checkpoint was trained on. Missing views become zero-placeholders.
     camera_keys: [face_view, left_wrist_view, right_wrist_view]

   actor:
     micro_batch_size: 32
     global_batch_size: 512
     model:
       num_bins: 32              # 2 = binary progress; >2 = multi-bin (even)
       ensemble_size: 3          # number of critics in the ensemble
       fusion_hidden_dim: 512
       freeze_vision_encoder: false
       freeze_language_model: false
       use_gradient_checkpointing: true
     optim:
       lr: 5.0e-5
       value_lr: 5.0e-5

**Key Parameters**

.. list-table::
   :header-rows: 1
   :widths: 32 16 52

   * - Parameter
     - Default
     - Description
   * - ``data.k``
     - ``required``
     - Max signed stride :math:`K`. In multi-bin mode ``2*K`` must be a multiple of ``num_bins``.
   * - ``actor.model.num_bins``
     - ``2``
     - Bin count. ``2`` is binary progress; ``> 2`` (even) is multi-bin signed-stride classification.
   * - ``actor.model.ensemble_size``
     - ``1``
     - Number of ensemble members. ``> 1`` enables worst-of-N aggregation and uncertainty stats.

**Launch Command**

.. code:: bash

   bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft

   # Override config fields inline:
   bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft data.k=8

**Output**

- Checkpoints under ``logs/steam_sft/{config_name}-{timestamp}/.../checkpoints/global_step_{N}/actor``
- TensorBoard logs

**Key Metrics**

- ``train/actor/loss``: cross-entropy over the signed-stride bins
- ``train/actor/accuracy``: best-bin classification accuracy
- ``train/actor/grad_norm``: gradient norm

Step 2: Compute Ensemble Advantages
-----------------------------------

Run the trained ensemble over every frame and write per-frame advantage labels.

**Configuration**

The config is ``examples/offline_rl/config/steam_compute_advantages_ensemble.yaml``:

.. code:: yaml

   advantage:
     value_checkpoint: /path/to/steam_value_ensemble/checkpoints/global_step_N/actor
     batch_size: 256
     label_mode: quantile        # required: "threshold" or "quantile"
     rollout_quantile: 0.3       # top 30% of rollout frames labelled True
     expert_quantile: 0.8        # optional: top 80% of sft frames labelled True
     tag: steam_k32_ensemble3_q30

   data:
     k: 32                       # must match Step 1 data.k
     camera_keys: [face_view, left_wrist_view, right_wrist_view]
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout

**Key Parameters**

``label_mode`` decides which knobs are active. In ``threshold`` mode only
``advantage.positive_threshold`` applies — a signed-score cut in :math:`[-1, 1]`;
rollout frames scoring above it are positive and sft frames are always positive.
In ``quantile`` mode ``positive_threshold`` is ignored and the
``rollout_quantile`` / ``expert_quantile`` fractions select the top-scoring frames
in each pool independently (omit ``expert_quantile`` to mark every sft frame
positive).

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Parameter
     - Default
     - Description
   * - ``advantage.value_checkpoint``
     - ``required``
     - Path to the Step 1 ensemble checkpoint (``actor`` directory).
   * - ``advantage.label_mode``
     - ``required``
     - ``threshold`` or ``quantile`` (no default — must be set explicitly).
   * - ``advantage.positive_threshold``
     - ``null``
     - Signed-score threshold in :math:`[-1, 1]` (``label_mode=threshold`` only).
   * - ``advantage.rollout_quantile``
     - ``null``
     - Top fraction of rollout frames labelled True (``label_mode=quantile``, required).
   * - ``advantage.expert_quantile``
     - ``null``
     - Top fraction of sft frames labelled True (``label_mode=quantile``, optional).
   * - ``advantage.tag``
     - ``required``
     - Output tag; writes ``meta/advantages_{tag}.parquet``.
   * - ``data.k``
     - ``required``
     - Pair stride; must match the Step 1 training ``data.k``.

**Launch Command**

.. code:: bash

   # Auto-detects #GPUs; single-GPU or torchrun multi-GPU both supported.
   bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble

   # Force a GPU count:
   bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh steam_compute_advantages_ensemble --nproc 4

**Output Files**

- ``meta/advantages_{tag}.parquet``: per-frame ``advantage`` (bool),
  ``advantage_continuous`` (signed score), ``ensemble_signed_score``, per-member
  values, and ensemble entropy / variance diagnostics.
- ``meta/mixture_config.yaml``: a per-tag entry recording ``label_mode``, the
  applied threshold, ``ensemble_size``, ``num_bins``, and positive counts.

Step 3: CFG Training
--------------------

Policy optimization runs the shared CFG stage directly on the STEAM advantage
parquets. Point the CFG config's ``data.advantage_tag`` at the Step 2
``advantage.tag`` and launch:

.. code:: bash

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi \
       data.advantage_tag=steam_k32_ensemble3_q30

See :doc:`the CFG training stage <recap>` for the full CFG configuration and parameters.

STEAM Results
-------------

We evaluate STEAM against behavior cloning (**BC**), **HG-DAgger**, and
:doc:`RECAP <recap>` on four real-robot manipulation tasks. STEAM markedly raises
the task success rate over the BC baseline on every task (absolute gain over BC
shown as ↑):

.. list-table:: Success rate (%) — higher is better
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Task
     - BC
     - HG-DAgger
     - RECAP
     - STEAM
   * - Towel Folding
     - 33.3
     - 40
     - 55.6
     - **92.3** (↑59)
   * - Chips Checkout
     - 39.5
     - 53.3
     - 53.3
     - **93.8** (↑54.3)
   * - Pick-and-Place
     - 63.8
     - —
     - 53.8
     - **80** (↑16.2)
   * - Cola Restocking
     - 52
     - —
     - 52.9
     - **75** (↑23)

.. list-table:: Throughput (successful episodes per hour) — higher is better
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Task
     - BC
     - HG-DAgger
     - RECAP
     - STEAM
   * - Towel Folding
     - 42
     - 48
     - 39
     - **58**
   * - Chips Checkout
     - 16.3
     - 22.0
     - 23.9
     - **47.5**
   * - Pick-and-Place
     - 230
     - —
     - 161
     - **254**
   * - Cola Restocking
     - 71
     - —
     - 46
     - **90**

Across the four tasks STEAM raises success rates to 75–93.8% and delivers the
highest throughput, with the largest success-rate gains on Towel Folding (↑59)
and Chips Checkout (↑54.3). (↑ marks the absolute gain over the BC baseline.)

Advanced Usage
--------------

Merge Ensemble Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Members trained as separate single-model runs (or extracted from existing
ensembles) can be fused into one ensemble inference checkpoint. Each ``--member``
is a checkpoint path, or ``PATH:idx`` to pull member ``idx`` from an ensemble:

.. code:: bash

   python examples/offline_rl/advantage_labeling/steam/process/merge_steam_ensemble.py \
       --member /path/to/seed1/checkpoints/global_step_5000/actor \
       --member /path/to/seed2/checkpoints/global_step_5000/actor \
       --member /path/to/ensemble/checkpoints/global_step_6000/actor:2 \
       --output /path/to/merged/actor

The merge logic lives in
``rlinf.models.embodiment.value_model.steam.checkpoint_merge.merge_ensemble_checkpoints``.

Threshold / Quantile Relabeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To change the labelling threshold without rerunning GPU inference, relabel an
existing advantages parquet (pure CPU — ``advantage_continuous`` is reused):

.. code:: bash

   python examples/offline_rl/advantage_labeling/steam/process/relabel_advantages.py \
       --dataset_paths /path/to/sft_ds /path/to/rollout_ds \
       --source_tag steam_k32_ensemble3_q30 \
       --new_tag steam_k32_ensemble3_q20 \
       --mode quantile --rollout_quantile 0.2

The relabel logic lives in
``examples/offline_rl/advantage_labeling/steam/process/relabel_advantages.py``.

Visualize Advantages
~~~~~~~~~~~~~~~~~~~~~~

Render distribution, per-member, uncertainty, per-episode, and episode-timeline
diagnostics from an advantages parquet:

.. code:: bash

   python examples/offline_rl/advantage_labeling/steam/process/visualize_advantage.py \
       --dataset /path/to/dataset \
       --tag steam_k32_ensemble3_q30 \
       --output outputs/steam_viz

Visualization and Results
-------------------------

For metric definitions, see :doc:`Training metrics <../../reference/metrics>`.

.. code:: bash

   tensorboard --logdir ./logs --port 6006

File Structure
--------------

Like RECAP, STEAM keeps its pipeline scripts self-contained under ``examples/``
(the inference + labelling strategy that is bound to the model), the model /
dataset code under ``rlinf/models`` and ``rlinf/data/datasets``, and shares the
model-agnostic post-processing with RECAP via ``rlinf/data/process/``:

.. code-block:: text

   examples/offline_rl/
   ├── config/                                  # shared production configs
   │   ├── steam_value_model_sft.yaml           # Step 1
   │   ├── steam_compute_advantages_ensemble.yaml   # Step 2
   │   ├── cfg_rl_openpi.yaml                   # Step 3 (CFG, shared with RECAP)
   │   └── model/
   │       └── steam_value_model.yaml           # value model architecture defaults
   ├── advantage_labeling/
   │   └── steam/
   │       ├── train_steam.py                   # Step 1: value model SFT entry
   │       ├── run_steam_sft.sh                 # Step 1 launch script
   │       └── process/                         # Step 2: self-contained entries (like recap)
   │           ├── compute_advantages_ensemble.py     # Step 2: ensemble inference + labelling (Hydra)
   │           ├── relabel_advantages.py              # CLI: relabel advantages (CPU)
   │           ├── merge_steam_ensemble.py            # CLI: merge ensemble checkpoints
   │           ├── visualize_advantage.py             # advantage visualization
   │           └── run_compute_advantages_ensemble.sh # Step 2 launch script
   └── policy_optimization/
       └── cfg_rl/
           ├── train_cfg.py                      # Step 3: CFG policy training
           └── run_cfg_rl.sh                     # Step 3 launch script

   rlinf/
   ├── models/embodiment/value_model/steam/     # critic, ensemble, config, merge
   │   ├── modeling_steam.py / modeling_critic.py
   │   ├── ensemble_modeling_critic.py          # worst-of-N + coerce_to_ensemble
   │   └── checkpoint_merge.py                  # ensemble checkpoint merge
   ├── data/datasets/steam/                     # pair_dataset.py, mixture.py, binning.py
   └── data/process/                            # shared, model-agnostic (RECAP + STEAM)
       ├── advantage.py                         # quantile threshold + boolean label
       ├── distributed.py                       # sharded-inference helpers
       └── mixture_config.py                    # meta/mixture_config.yaml tag I/O
