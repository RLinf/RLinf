RL on PI0-FAST with LeRobot
============================

This example integrates LeRobot's ``PI0FastPolicy`` with RLinf for deterministic
LIBERO-Long evaluation and token-level GRPO fine-tuning. The integration keeps
PI0-FAST's native autoregressive action sequence and replays the sampled tokens
with teacher forcing during the actor update.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Item
     - Configuration
   * - Environment
     - LIBERO-10 (Long)
   * - Base policy
     - ``lerobot/pi0fast-libero``
   * - Algorithm
     - GRPO with token-level PPO clipping
   * - Trainable parameters
     - All-linear LoRA, rank 16
   * - Validated runtime
     - Python 3.12.12, PyTorch 2.11.0 + CUDA 12.8, Transformers 5.5.4

Install
-------

The following command uses the PI0-FAST combination validated by this example.
These versions document the tested runtime rather than hard requirements imposed
by the installer:

.. code:: bash

   UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
      --model pi0_fast --env libero \
      --python 3.12.12 --torch 2.11.0 --no-flash-attn
   source .venv/bin/activate

Add ``--use-mirror`` when the GitHub and PyPI mirrors are required. The validated
command skips Flash Attention; omit ``--no-flash-attn`` to install it. Before the
first run, accept the PaliGemma access terms on Hugging Face and run
``hf auth login``.

Pinned artifacts
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Artifact
     - Repository
     - Revision
   * - LeRobot source
     - ``huggingface/lerobot``
     - ``8a74e0ac6d01706d67fddfed682a09d694d9c8c0``
   * - Policy checkpoint
     - ``lerobot/pi0fast-libero``
     - ``840f4b503f4c09110421c33c810a85b6684fd658``
   * - Text tokenizer
     - ``google/paligemma-3b-pt-224``
     - ``35e4f46485b4d07967e7e9935bc3786aad50687c``
   * - Action tokenizer
     - ``jadechoghari/tokenizer-lib-mean``
     - ``79ae83e3cbd8786dcb84b628569f8d076ca8151e``

Download all three artifacts before launching RLinf:

.. code:: bash

   hf download lerobot/pi0fast-libero \
      --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
      --local-dir /path/to/pi0fast-libero
   hf download google/paligemma-3b-pt-224 \
      --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
      --local-dir /path/to/paligemma-3b-pt-224
   hf download jadechoghari/tokenizer-lib-mean \
      --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
      --local-dir /path/to/tokenizer-lib-mean

Then update these fields in
``examples/embodiment/config/model/pi0_fast.yaml``:

.. code:: yaml

   model_path: "/path/to/pi0fast-libero"
   pi0_fast:
     text_tokenizer_name: "/path/to/paligemma-3b-pt-224"
     action_tokenizer_name: "/path/to/tokenizer-lib-mean"

GRPO fine-tuning
----------------

Launch the two-node, 16-GPU reference configuration with:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_grpo_pi0_fast

The reference configuration samples 1,024 trajectories per update (256
parallel environments, four rollout epochs), uses group size 8, actor micro
batch size 16, and evaluates 256 fixed episodes every 10 steps. Sampling uses
temperature 0.3; evaluation is greedy. The actor uses all-linear LoRA and the
optional FP32-master AdamW path. This optimizer supports FSDP1 ``NO_SHARD`` and
``FULL_SHARD``; the reference configuration uses ``NO_SHARD``.

One seed-1234 development run reported the following ``success_once`` values.
These measurements demonstrate the integration but are not a multi-seed
convergence guarantee:

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

Baseline evaluation
-------------------

The evaluation config uses greedy decoding, seed 0, ordered fixed LIBERO reset
states, and 500 episodes:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_eval_pi0_fast

The pinned runtime and artifacts produced the following development result:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - Episodes
     - ``success_once``
     - ``success_at_end``
   * - 500
     - 85.8%
     - 75.8%

Policy semantics
----------------

PI0-FAST generates the complete native action string; RLinf does not inject an
``Action:`` prefix. The policy mask includes generated prefix, action body, and
the first complete ``|`` end marker, while excluding padding and tokens after
that marker. Every token from one trajectory shares its trajectory-level GRPO
advantage. PPO clipping is applied per token before the masked token losses are
aggregated.

Malformed sequences are not resampled. They execute a safe zero action, receive
the normal environment failure feedback, and remain in the policy objective.
This keeps sampling on-policy.

Monitoring
----------

In addition to ``env/success_once`` and ``eval/success_once``, monitor the
group-success histogram and keep fraction, token entropy, gradient norm,
``approx_kl``, and log-ratio finite/min/max statistics.
The first actor update should have finite replay log-probabilities and a ratio
close to one before the optimizer changes the policy.
