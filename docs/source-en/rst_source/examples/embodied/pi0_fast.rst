π₀-FAST RL
===========

This example integrates LeRobot's ``PI0FastPolicy`` with RLinf for deterministic
LIBERO-Long evaluation and token-level GRPO fine-tuning. The integration keeps
π₀-FAST's native autoregressive action sequence and replays the sampled tokens
with teacher forcing during the actor update.

Overview
--------

Evaluate and GRPO-fine-tune LeRobot π₀-FAST on LIBERO-10 (Long).

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO-10 (Long)

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO (token-level clipping)

   .. grid-item-card:: Tasks
      :text-align: center

      10 long-horizon LIBERO tasks

   .. grid-item-card:: Hardware
      :text-align: center

      2 nodes · 16 GPUs (or 1 node override)

| **You'll do:** install the validated π₀-FAST runtime → download the three pinned checkpoints → launch ``run_embodiment.sh`` (or ``run_eval.sh``) → watch ``eval/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Hugging Face access to ``google/paligemma-3b-pt-224``.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 22 34 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-10 (Long)
     - ``libero_10_grpo_pi0_fast``, ``lerobot/pi0fast-libero``
     - Token-level GRPO with all-linear LoRA (rank 16) on Python 3.12.12 / PyTorch 2.11.0 + CUDA 12.8.
   * - LIBERO
     - LIBERO-10 (Long)
     - ``libero_10_pi0_fast_eval``
     - Deterministic greedy evaluation across 500 fixed episodes.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Third-person (agentview) and wrist RGB images plus 8-dim robot proprioception.
   * - Action
     - 10-step continuous action chunk (7-DoF per step) decoded from native autoregressive FAST action tokens.
   * - Reward
     - Sparse LIBERO task success reward (0/1).
   * - Prompt
     - Natural-language LIBERO task instruction.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: GRPO Fine-Tuning
      :sync: grpo

      **1. Install**

      The following command uses the π₀-FAST combination validated by this example (skipping Flash Attention; omit ``--no-flash-attn`` to install it). Before the first run, accept the PaliGemma access terms on Hugging Face and run ``hf auth login``:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
            --model pi0_fast --env libero \
            --python 3.12.12 --torch 2.11.0 --no-flash-attn && source .venv/bin/activate

      **2. Download**

      Download the three pinned Hugging Face artifacts before launching RLinf:

      .. code-block:: bash

         hf download lerobot/pi0fast-libero \
            --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
            --local-dir ./pi0fast-libero
         hf download google/paligemma-3b-pt-224 \
            --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
            --local-dir ./paligemma-3b-pt-224
         hf download jadechoghari/tokenizer-lib-mean \
            --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
            --local-dir ./tokenizer-lib-mean

      Update ``examples/embodiment/config/model/pi0_fast.yaml`` to point to the downloaded directories (``num_action_chunks`` and ``action_dim`` must match the checkpoint's ``n_action_steps`` and action feature dim, otherwise loading fails):

      .. code-block:: yaml

         model_path: "./pi0fast-libero"
         num_action_chunks: 10
         action_dim: 7
         pi0_fast:
           text_tokenizer_name: "./paligemma-3b-pt-224"
           action_tokenizer_name: "./tokenizer-lib-mean"

      **3. Launch**

      ``libero_10_grpo_pi0_fast.yaml`` defaults to **2 nodes** (``cluster.num_nodes: 2``, 16 GPUs total; 1,024 trajectories per update across 256 parallel envs × 4 rollout epochs, group size 8, micro batch size 16, temperature 0.3, all-linear LoRA, and optional FP32-master AdamW supporting FSDP1 ``NO_SHARD``):

      .. code-block:: bash

         # 2-node (16-GPU) reference run:
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_pi0_fast

         # Single-node quick start (override cluster.num_nodes=1):
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_pi0_fast cluster.num_nodes=1

   .. tab-item:: Baseline Evaluation
      :sync: eval

      **1. Install**

      Install the validated π₀-FAST runtime (accept the PaliGemma access terms on Hugging Face and run ``hf auth login`` before downloading):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
            --model pi0_fast --env libero \
            --python 3.12.12 --torch 2.11.0 --no-flash-attn && source .venv/bin/activate

      **2. Download**

      Download the three pinned Hugging Face artifacts and update ``examples/embodiment/config/model/pi0_fast.yaml`` as in the GRPO tab:

      .. code-block:: bash

         hf download lerobot/pi0fast-libero \
            --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
            --local-dir ./pi0fast-libero
         hf download google/paligemma-3b-pt-224 \
            --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
            --local-dir ./paligemma-3b-pt-224
         hf download jadechoghari/tokenizer-lib-mean \
            --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
            --local-dir ./tokenizer-lib-mean

      **3. Launch**

      Run deterministic baseline evaluation (greedy decoding, seed 0, 500 ordered fixed LIBERO episodes; shares the FAST sampler and detokenize path with training without collecting replay log-probabilities):

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_pi0_fast_eval

.. dropdown:: Policy semantics & advanced options

   **Pinned artifacts**

   .. list-table::
      :header-rows: 1
      :widths: 28 38 34

      * - Artifact
        - Repository
        - Revision
      * - LeRobot source
        - ``huggingface/lerobot``
        - ``3e37269dc60ee6195e44dd2810c95f1df5dfda99``
      * - Policy checkpoint
        - ``lerobot/pi0fast-libero``
        - ``840f4b503f4c09110421c33c810a85b6684fd658``
      * - Text tokenizer
        - ``google/paligemma-3b-pt-224``
        - ``35e4f46485b4d07967e7e9935bc3786aad50687c``
      * - Action tokenizer
        - ``jadechoghari/tokenizer-lib-mean``
        - ``79ae83e3cbd8786dcb84b628569f8d076ca8151e``

   **Policy semantics**

   π₀-FAST generates the complete native action string; RLinf does not inject an ``Action:`` prefix. The policy mask includes generated prefix, action body, and the first complete ``|`` end marker, while excluding padding and tokens after that marker. Every token from one trajectory shares its trajectory-level GRPO advantage. PPO clipping is applied per token before the masked token losses are aggregated.

   Malformed sequences are not resampled. They execute a safe zero action after the action postprocessor, receive the normal environment failure feedback, and remain in the policy objective. This keeps sampling on-policy and is shared by train and eval.

Visualization and Results
-------------------------

In addition to ``env/success_once`` and ``eval/success_once``, monitor the group-success histogram and keep fraction, token entropy, gradient norm, ``approx_kl``, and log-ratio finite/min/max statistics (see :doc:`Training metrics <../../reference/metrics>`). The first actor update should have finite replay log-probabilities and a ratio close to one before the optimizer changes the policy.

**Baseline evaluation (500 episodes)**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - Episodes
     - ``success_once``
     - ``success_at_end``
   * - 500
     - 85.8%
     - 75.8%

**GRPO fine-tuning (seed 1234 development run)**

One seed-1234 run evaluating 256 fixed episodes every 10 steps reported the following ``success_once`` values (demonstrates the integration rather than a multi-seed convergence guarantee):

.. list-table::
   :header-rows: 1
   :widths: 20 20

   * - Step
     - ``success_once``
   * - 290
     - 95.70%
   * - 300
     - 99.22%
   * - 310
     - 95.70%
   * - 320
     - 97.27%
   * - 330
     - 94.92%
   * - Mean
     - 96.56%
