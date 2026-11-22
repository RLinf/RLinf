RECAP
=====

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/recap.png
   :align: center
   :width: 80%

   The RECAP offline pipeline.

Run the **RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)** pipeline in RLinf.
RECAP is an offline policy optimization method that requires no online environment interaction. It computes returns from existing datasets,
trains a value model, estimates advantages, and finally uses **Classifier-Free Guidance (CFG) training** to optimize the policy.

This pipeline is especially suited for real-robot scenarios where large-scale online sampling is impractical.

Overview
--------

Improve a π₀.₅ policy offline (no new rollouts) by scoring existing data with a value model and steering with classifier-free guidance.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      RECAP (CFG)

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Environments / Data
      :text-align: center

      LeRobot datasets

   .. grid-item-card:: Training
      :text-align: center

      Offline · 4 stages

| **You'll do:** compute returns → SFT a value model → compute advantages → CFG-train the policy → evaluate.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · SigLIP2 + Gemma3 + π₀.₅ checkpoints · LeRobot-format datasets (steps below).

How RECAP Works
---------------

RECAP runs four offline stages without online environment rollouts:

1. **Compute Returns:** Compute reverse discounted cumulative returns per trajectory (default ``gamma: 1.0``; SFT demos get per-step reward ``-1`` and terminal ``0``, while failed rollouts get terminal penalty ``failure_reward``, e.g. ``-300``) and write sidecar return files without modifying raw data.
2. **Value Model SFT:** Train a value model (SigLIP2 vision encoder + Gemma3 language model + learnable Critic Expert) to predict a 201-bin categorical value distribution over normalized returns in ``[-1, 0]``.
3. **Compute Advantages:** Estimate N-step lookahead advantages with the trained value model across multiple GPUs and label the top quantile (default top 30%) as positive.
4. **CFG Training:** Fine-tune the OpenPI π₀.₅ policy with Classifier-Free Guidance — positive samples act as conditional inputs (randomly dropped to unconditional with probability ``unconditional_prob``, default ``0.1``) and negative samples act as unconditional inputs. At inference time, ``cfgrl_guidance_scale`` steers actions toward high-advantage behavior.

Installation
------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI (LIBERO / ManiSkill)

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate


Download the Model
------------------

The RECAP pipeline requires the following pretrained models:

- **SigLIP2-so400m**: Vision encoder for Step 2 value model training
- **Gemma3-270M**: Language model for Step 2 value model training
- **pi0.5 base (PyTorch)**: Policy model for Step 4 CFG training. Refer to `openpi <https://github.com/Physical-Intelligence/openpi>`_ for obtaining model weights and converting to PyTorch format

.. tab-set::

   .. tab-item:: SigLIP2 + Gemma3 (Value Model)

      .. code-block:: bash

         hf download google/siglip2-so400m-patch14-224 --local-dir ./siglip2-so400m-patch14-224
         hf download google/gemma-3-270m --local-dir ./gemma-3-270m

After downloading, set the model paths in the Step 2 and Step 4 configs:

.. code-block:: yaml

   # Step 2 value model configuration
   actor:
     model:
       siglip_path: ./siglip2-so400m-patch14-224
       gemma3_path: ./gemma-3-270m
       tokenizer_path: ./gemma-3-270m

   # Step 4 policy model configuration
   actor:
     model:
       model_path: /path/to/pi05_base_pytorch


Data Preparation
-------------------

RECAP uses datasets in the LeRobot format. Datasets are categorized into two types:

- **SFT datasets**: Successful trajectories from human demonstrations or trained policies
- **Rollout datasets**: Trajectories collected from online interaction, containing both successes and failures

Example dataset configuration:

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft                # all successful trajectories
         weight: 1.0
       - dataset_path: /path/to/rollout_dataset
         type: rollout            # contains failures
         weight: 1.0

.. note::

   ``train_data_paths`` is a list. If you want to mix multiple datasets, you can add more items; if you want to train with only one dataset, you can also keep just a single item.
   The ``train_data_paths`` should remain consistent across all steps to ensure returns, values, and advantages are computed on the same data.

Pipeline Tag System
~~~~~~~~~~~~~~~~~~~~~~~

RECAP uses **tags** for data passing and version management across steps:

- **returns_tag**: Generated by Step 1, read by Steps 2 and 3. Ensure that Step 1's ``data.tag``, Step 2's ``data.tag``, and Step 3's ``advantage.returns_tag`` are consistent.
- **advantage_tag**: Generated by Step 3, read by Step 4. Ensure that Step 3's ``advantage.tag`` and Step 4's ``data.advantage_tag`` are consistent.

.. list-table:: **Tag Flow Across Pipeline Steps**
   :header-rows: 1

   * - Step
     - Config Field
     - Description
   * - 1
     - ``data.tag``
     - Writes ``meta/returns_{tag}.parquet``
   * - 2
     - ``data.tag``
     - Reads ``meta/returns_{tag}.parquet``
   * - 3
     - ``advantage.returns_tag``
     - Reads ``meta/returns_{tag}.parquet``
   * - 3
     - ``advantage.tag``
     - Writes ``meta/advantages_{tag}.parquet``
   * - 4
     - ``data.advantage_tag``
     - Reads ``meta/advantages_{tag}.parquet``


Step 1: Compute Returns
---------------------------

This step computes discounted cumulative returns for each trajectory in reverse order. Results are saved as sidecar files without modifying the original data.

**Configuration**

Set dataset paths and output tag in ``examples/offline_rl/config/recap_compute_returns.yaml``:

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
       - dataset_path: /path/to/rollout_dataset
         type: rollout
     eval_data_paths:          # must cover Step 2's eval_data_paths
       - dataset_path: /path/to/eval_dataset
         type: rollout
     tag: "fail300"

**Launch Command**

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh recap_compute_returns

.. dropdown:: Advanced configuration (optional)

   The shipped config already sets ``gamma: 1.0``, ``failure_reward: -300.0``, and ``num_workers: 128``:

   .. code-block:: yaml

      data:
        gamma: 1.0              # discount factor
        failure_reward: -300.0   # terminal reward for failed trajectories
        tag: "fail300"           # output file tag
        num_workers: 128         # parallel processing threads

   .. list-table::
      :header-rows: 1
      :widths: 30 15 55

      * - Parameter
        - Default
        - Description
      * - ``data.gamma``
        - ``1.0``
        - Discount factor. ``1.0`` means undiscounted (simple sum of future rewards)
      * - ``data.failure_reward``
        - ``-300.0``
        - Penalty for failed trajectory terminal steps. Larger magnitude increases separation between success and failure returns
      * - ``data.eval_data_paths``
        - ``[]``
        - Eval datasets; must include every dataset in Step 2's ``data.eval_data_paths``
      * - ``data.tag``
        - ``null``
        - Output file tag, generates ``meta/returns_{tag}.parquet``
      * - ``data.num_workers``
        - ``128``
        - Number of threads for parallel parquet file processing

**Output Files**

- ``meta/returns_{tag}.parquet``: Each row contains ``episode_index``, ``frame_index``, ``return``, ``reward``, ``prompt``
- ``meta/stats.json``: Updated with return statistics (mean, std, min, max)

**Verification**

.. code-block:: bash

   python3 -c "
   import json
   stats = json.load(open('./datasets/rollout/meta/stats.json'))
   assert 'return' in stats
   print('return stats:', stats['return'])
   "


Step 2: Value Model SFT
---------------------------

Using the returns computed in Step 1 as supervision signals, train a value model to predict normalized returns from observations (images + language instructions).

**Model Architecture**

The value model consists of three components:

- **Vision Encoder**: SigLIP2-so400m (1152-dim) — processes RGB image inputs
- **Language Model**: Gemma3-270M (640-dim) — processes language instructions
- **Critic Expert**: Learnable expert head that maps multimodal representations to value predictions

The output is a Categorical Value Distribution over 201 bins spanning ``[-1, 0]``.

**Configuration**

Set dataset paths, Step 1 ``tag``, and model paths (from Download the Model above) in ``examples/offline_rl/config/recap_value_model_sft.yaml``:

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
         weight: 1.0
         robot_type: "libero"
         model_type: "pi05"
     tag: "fail300"                     # must match Step 1 tag

**Launch Command**

The training script automatically initializes the Ray cluster:

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh recap_value_model_sft

.. dropdown:: Advanced configuration (optional)

   The shipped config already sets evaluation sampling, batch sizes, optimizer hyperparameters, and save intervals:

   .. code-block:: yaml

      data:
        action_horizon: 10
        normalize_to_minus_one_zero: true  # normalize to [-1, 0]
        eval_data_paths:                   # optional, recommended
          - dataset_path: /path/to/eval_dataset
            max_samples: 10000
            robot_type: "libero"
            model_type: "pi05"

      actor:
        micro_batch_size: 32
        global_batch_size: 256
        model:
          freeze_vlm: false         # whether to freeze VLM backbone
          value_dropout: 0.0        # Value Head dropout
        optim:
          lr: 5.0e-5               # VLM backbone learning rate
          value_lr: 1.0e-4          # Value Head learning rate
          weight_decay: 1.0e-10
          lr_warmup_steps: 500

      runner:
        max_epochs: 30000
        save_interval: 3000         # checkpoint save interval

   .. list-table::
      :header-rows: 1
      :widths: 35 15 50

      * - Parameter
        - Default
        - Description
      * - ``data.tag``
        - ``null``
        - Same tag as Step 1, used to read the corresponding ``returns_{tag}.parquet``
      * - ``data.normalize_to_minus_one_zero``
        - ``true``
        - Normalization mode. ``true`` maps to ``[-1, 0]``; ``false`` maps to ``[0, 1]``
      * - ``actor.model.freeze_vlm``
        - ``false``
        - Freeze the vision encoder. When ``true``, also set ``actor.fsdp_config.use_orig_params: True``
      * - ``actor.model.value_dropout``
        - ``0.0``
        - Dropout rate before the Value Head
      * - ``actor.optim.lr``
        - ``5.0e-5``
        - VLM backbone learning rate
      * - ``actor.optim.value_lr``
        - ``1.0e-4``
        - Value Head learning rate

**Output**

- Model checkpoints saved at ``logs/value_sft/{config_name}-{timestamp}/value_sft/checkpoints/``
- TensorBoard logs

**Key Metrics**

- ``train/actor/loss``: Total value model training loss
- ``train/actor/grad_norm``: Gradient norm
- ``eval/spearman_correlation``: Spearman correlation coefficient measuring rank consistency between predictions and true returns

.. note::

   After training, note the checkpoint path for Step 3. Checkpoints are located at:

   .. code-block:: text

      logs/value_sft/{config_name}-{timestamp}/value_sft/checkpoints/global_step_{N}/actor/model_state_dict


Step 3: Compute Advantages
-----------------------------

Using the value model trained in Step 2, compute per-timestep N-step lookahead advantages (configured via ``advantage_lookahead_step`` and ``gamma``) for the dataset and label samples as positive/negative based on a quantile threshold.

**Configuration**

Set the Step 2 checkpoint path, tags, and dataset paths in ``examples/offline_rl/config/recap_compute_advantages.yaml``:

.. code-block:: yaml

   advantage:
     value_checkpoint: /path/to/value_checkpoint
     tag: "fail300_N10_ckpt18000_q30"
     returns_tag: "fail300"       # must match Step 1 tag

   data:
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         robot_type: "libero"
         type: "sft"
         weight: 1.0

**Launch Command**

Supports multi-GPU distributed inference:

.. code-block:: bash

   bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh recap_compute_advantages

.. dropdown:: Advanced configuration (optional)

   The shipped config already sets ``positive_quantile: 0.3``, ``batch_size: 1024``, ``advantage_lookahead_step: 10``, and ``gamma: 1.0``:

   .. code-block:: yaml

      advantage:
        positive_quantile: 0.3      # top 30% labeled as positive
        batch_size: 1024

      data:
        advantage_lookahead_step: 10   # N-step lookahead
        gamma: 1.0

   .. list-table::
      :header-rows: 1
      :widths: 35 15 50

      * - Parameter
        - Default
        - Description
      * - ``advantage.value_checkpoint``
        - required
        - Path to the value model checkpoint from Step 2
      * - ``advantage.positive_quantile``
        - ``0.3``
        - Positive sample ratio. ``0.3`` means the top 30% by advantage value are labeled as positive
      * - ``data.advantage_lookahead_step``
        - ``10``
        - Lookahead steps N, i.e., how many future steps of rewards to consider
      * - ``advantage.returns_tag``
        - ``null``
        - Tag for reading the return file generated by Step 1
      * - ``advantage.tag``
        - ``null``
        - Output advantage file tag, generates ``meta/advantages_{tag}.parquet``

**Output Files**

- ``meta/advantages_{tag}.parquet``: Contains ``advantage`` (boolean), ``advantage_continuous`` (float), and other columns
- Updated ``mixture_config.yaml``: Records global threshold and normalization statistics

**Verification**

.. code-block:: bash

   python3 -c "
   import pandas as pd
   df = pd.read_parquet('./datasets/rollout/meta/advantages_fail300_N10_ckpt18000_q30.parquet')
   print(f'samples={len(df)}, columns={list(df.columns)}')
   print(df[['advantage_continuous']].describe())
   "


Step 4: CFG Training
-----------------------

Using the advantage labels from Step 3, train the OpenPI policy model with classifier-free guidance.

**Training Mechanism**

- **Positive samples** (``advantage=True``): Serve as conditional inputs
- **Negative samples** (``advantage=False``): Always serve as unconditional inputs
- When ``positive_only_conditional`` is enabled, positive samples are randomly dropped to unconditional with probability ``unconditional_prob`` for regularization
- At inference time, the guidance scale ``cfgrl_guidance_scale`` amplifies the difference between conditional and unconditional predictions, steering the model toward high-advantage actions

**Configuration**

Set ``data.advantage_tag``, dataset paths, and ``actor.model.model_path`` in ``examples/offline_rl/config/cfg_rl_openpi.yaml``:

.. code-block:: yaml

   data:
     advantage_tag: "fail300_N10_ckpt18000_q30"   # must match Step 3 advantage.tag
     train_data_paths:
       - dataset_path: /path/to/sft_dataset
         type: sft
         weight: 1.0
       - dataset_path: /path/to/rollout_dataset
         type: rollout
         weight: 1.0

   actor:
     model:
       model_path: /path/to/pi05_base_pytorch

**Launch Command**

.. code-block:: bash

   bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi

.. dropdown:: Advanced configuration (optional)

   The shipped config already sets CFG conditioning, batch size, and optimizer defaults:

   .. code-block:: yaml

      data:
        balance_dataset_weights: true

      actor:
        micro_batch_size: 32
        global_batch_size: 512
        model:
          openpi:
            config_name: "pi05_libero"
            positive_only_conditional: true
            unconditional_prob: 0.1
            cfgrl_guidance_scale: 1.0
        optim:
          lr: 1.0e-5
          lr_scheduler: cosine
          lr_warmup_steps: 5000
          total_training_steps: 30000

   .. list-table::
      :header-rows: 1
      :widths: 40 15 45

      * - Parameter
        - Default
        - Description
      * - ``data.advantage_tag``
        - ``null``
        - Matches Step 3's ``advantage.tag`` for reading advantage labels
      * - ``actor.model.openpi.positive_only_conditional``
        - ``true``
        - Only positive samples serve as conditional inputs. When ``false``, both positive and negative samples are conditional
      * - ``actor.model.openpi.unconditional_prob``
        - ``0.1``
        - Probability of dropping samples to unconditional. When ``positive_only_conditional`` is enabled, applies only to positive samples; otherwise applies to all samples
      * - ``actor.model.openpi.cfgrl_guidance_scale``
        - ``1.0``
        - Guidance scale at inference. Higher values favor high-advantage actions more strongly
      * - ``actor.model.openpi.config_name``
        - ``pi05_libero``
        - Data transform config. Use ``pi05_libero`` for LIBERO, ``pi05_maniskill_sim_real_co_training`` for Franka
      * - ``data.balance_dataset_weights``
        - ``true``
        - Balance sampling weights by dataset size

**Key Metrics**

- ``train/actor/loss``: Policy training loss
- ``train/actor/grad_norm``: Gradient norm


.. dropdown:: Visualize Advantages (optional)

   After Step 3, run ``visualize_advantage_dataset.py`` to generate distribution panels (``advantage_distribution.png``), per-episode summaries (``episode_{N}_summary.png``), and annotated replay videos (``episode_{N}.mp4``). Pass ``--no-video`` for plots only, or ``--episodes 0 5 10`` to select specific episodes:

   .. code-block:: bash

      python examples/offline_rl/advantage_labeling/recap/process/visualize_advantage_dataset.py \
          --dataset ./datasets/rollout \
          --output ./outputs/advantage_viz \
          --tag "fail300_N10_ckpt18000_q30" \
          --num-episodes 10

.. dropdown:: Advanced Usage (Threshold Relabeling & Iterative Optimization)

   **Threshold relabeling**

   Adjust the quantile threshold on CPU without rerunning GPU inference (reuses ``advantage_continuous``). Pass ``--dataset_root`` with ``--advantage_lookahead_step`` instead of ``--dataset_paths`` to recompute advantages across a directory of datasets:

   .. code-block:: bash

      python examples/offline_rl/advantage_labeling/recap/process/relabel_advantages.py \
          --dataset_paths ./datasets/sft ./datasets/rollout \
          --source_tag "fail300_N10_ckpt18000_q30" \
          --new_tag "fail300_N10_ckpt18000_q20" \
          --positive_quantile 0.2

   **Iterative optimization**

   Collect new rollouts with the Step 4 policy and restart from Step 1 using fresh tags per iteration:

   .. code-block:: text

      Iter 1: tag="fail300"          → Train Value Model → tag="fail300_N10_ckpt18000_q30"     → CFG Training
      Iter 2: tag="fail300_iter2"    → Train Value Model → tag="fail300_iter2_N10_ckpt6000_q20" → CFG Training

Visualization and Results
-------------------------

Track Step 2 (``logs/value_sft/``) and Step 4 (``logs/cfg_sft/``) metrics with TensorBoard (``tensorboard --logdir ./logs --port 6006``). For metric definitions and logger backends, see :doc:`Training metrics <../../reference/metrics>`.

We reproduce the RECAP pipeline on `LIBERO-10 <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ Task 0 using the `RECAP-Libero10-Task0-48succ-Data <https://huggingface.co/datasets/RLinf/RECAP-Libero10-Task0-48succ-Data/tree/main>`_ dataset (expert SFT demos, 4,096 mixed rollouts from a few-shot π\ :sub:`0.5` policy, and a held-out eval split). After one RECAP iteration, success rate rises from **48.8%** (few-shot SFT baseline in this compact setup) to **66.5%** (+17.7%).

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/recap_libero10_task0.png
   :align: center
   :width: 55%

   RECAP results on LIBERO-10 Task 0.
