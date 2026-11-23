RL on LingbotVLA 2.0 Models
===========================

.. figure:: https://raw.githubusercontent.com/robbyant/lingbot-vla-v2/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/assets/lingbot_vla2_framework.png
   :align: center
   :width: 90%

   LingbotVLA 2.0 framework (image: `Robbyant <https://github.com/robbyant/lingbot-vla-v2>`__).

Fine-tune the action expert of
`LingbotVLA 2.0 <https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin>`__
with PPO on RoboTwin 2.0. RLinf loads the released RoboTwin checkpoint
**natively** and starts training from it. LingbotVLA 2.0 is a
vision-language-action model built on a Qwen3-VL backbone. Its mixture-of-experts
action expert generates 50-step action chunks by flow matching in a 55-dimensional
latent action space, which the robot configuration maps to RoboTwin's 14 joint
commands. PPO updates only the action expert; the Qwen3-VL backbone and the
action expert's MoE routers stay frozen.

Overview
--------

Train LingbotVLA 2.0 with PPO on the RoboTwin 2.0 ``click_bell`` task.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      RoboTwin 2.0

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Click Bell

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4 GPUs (48 GB)

| **You'll do:** install (native) → clone RoboTwin + assets → download checkpoints → fill in the model paths → PPO → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the RoboTwin repo and assets · the LingbotVLA 2.0 and Qwen3-VL checkpoints (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - RoboTwin
     - Click Bell
     - ``robotwin_click_bell_ppo_lingbotvlav2``
     - PPO training of the action expert from the released RoboTwin checkpoint.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - Head camera, both wrist cameras, and the 14-dimensional joint state.
   * - Action
     - 14 joint commands per step, decoded from a 50-step chunk of 55-dimensional latent actions.
   * - Reward
     - RoboTwin task success.
   * - Prompt
     - Natural-language task instruction for the RoboTwin episode.

Installation
------------

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, clone the RLinf repository and enter the main directory:

.. code-block:: bash

    git clone https://github.com/RLinf/RLinf.git
    cd RLinf
    export RLINF_PATH=$(pwd)

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

LingbotVLA 1.0 and 2.0 both install a Python package named ``lingbotvla``, so each
needs its own virtual environment.

**Option 1: Docker Image**

The RoboTwin image includes a ``lingbotvlav2`` environment. Build it from the
repository root:

.. code-block:: bash

    docker buildx build --load -f docker/Dockerfile \
      --build-arg BUILD_TARGET=embodied-robotwin \
      -t rlinf:embodied-robotwin .

Inside the container, switch to the environment with the built-in ``switch_env`` utility:

.. code-block:: bash

    source switch_env lingbotvlav2

The LingbotVLA 2.0 source checkout is at ``/opt/venv/lingbotvlav2/lingbot-vla-v2``.

**Option 2: Custom Environment**

Install the LingbotVLA 2.0 environment and the RoboTwin dependencies in one command:

.. code-block:: bash

    bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin
    source .venv/bin/activate

The installer clones the ``rlinf`` branch of
`RLinf's LingbotVLA 2.0 fork <https://github.com/RLinf/lingbot-vla-v2/tree/rlinf>`__,
which adds the hooks RLinf's PPO needs, into ``.venv/lingbot-vla-v2``. Set
``LINGBOT_VLA_V2_PATH`` before installing to use another location.

RoboTwin Repository Clone and Assets Download
---------------------------------------------

RoboTwin Assets are asset files required by the RoboTwin environment and need to be downloaded from HuggingFace.

.. code-block:: bash

   # 1. Clone RoboTwin repository
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support

   # 2. Download and extract Assets files
   cd RoboTwin
   bash script/_download_assets.sh

Download the Model
------------------

Download the RoboTwin post-training checkpoint and the Qwen3-VL tokenizer from Hugging Face:

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b-robotwin --local-dir lingbot-vla-v2-6b-robotwin
    hf download Qwen/Qwen3-VL-4B-Instruct --local-dir Qwen3-VL-4B-Instruct

Set the RoboTwin release root and tokenizer path. The shared PPO and evaluation
config uses this root for the post-training weights and ``lingbotvla_cli.yaml``:

.. code-block:: bash

    export LINGBOT_VLA_V2_ROBOTWIN_ROOT=/path/to/lingbot-vla-v2-6b-robotwin
    export LINGBOT_VLA_V2_PATH=/path/to/.venv/lingbot-vla-v2
    export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct

Run It
------

Configuration Files
~~~~~~~~~~~~~~~~~~~

* **PPO (Reinforcement Learning)**:
  ``examples/embodiment/config/robotwin_click_bell_ppo_lingbotvlav2.yaml``
* **Model defaults**, shared with standalone evaluation:
  ``examples/embodiment/config/model/lingbotvlav2.yaml``

Key Config Snippets (PPO)
^^^^^^^^^^^^^^^^^^^^^^^^^

A training rollout draws from the flow-SDE at one random denoising step per
action chunk and follows the ODE at the other steps. PPO scores that step on the
14 latent coordinates that the robot transform decodes into RoboTwin controls,
with one ratio per coordinate (``token_level``) and dual-clip (``clip_ratio_c``).
Rewards and values are chunk-level, and ``critic_warmup_steps`` trains the new
value head alone for that many optimizer steps before the policy updates.

The actor stores the trainable action-expert weights in fp32, because bf16
storage would round away optimizer steps at this learning rate. FSDP2 still
computes in bf16, and the rollout keeps a bf16 copy of the weights.

.. code-block:: yaml

    algorithm:
      reward_type: chunk_level
      logprob_type: token_level
      clip_ratio_c: 3.0
      recompute_logprobs: true
      max_recompute_logprob_gap: 1.0e-3

    rollout:
      model:
        precision: bf16

    actor:
      model:
        precision: fp32
      optim:
        critic_warmup_steps: 20
      fsdp_config:
        strategy: "fsdp2"
        mixed_precision:
          param_dtype: bf16
          reduce_dtype: fp32
          cast_forward_inputs: false

Before each update the actor re-scores every SDE step the rollout sampled.
``algorithm.max_recompute_logprob_gap`` stops training when the summed
log-probability of any sample differs between rollout and actor by more than
this value; ``actor/rollout_train_joint_logprob_gap_max`` logs the largest
difference.

Launch Commands
~~~~~~~~~~~~~~~

**Note**: The task uses a dual-arm robot, so declare the robot platform as ALOHA
before launching. Otherwise, the environment fails to load the action space
correctly:

.. code-block:: bash

    export ROBOT_PLATFORM="ALOHA"
    export ROBOTWIN_PATH=/path/to/RoboTwin
    export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH

Then start PPO training on the RoboTwin Click Bell task:

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2

To resume, set ``runner.resume_dir`` in the training config to a saved checkpoint directory.

Standalone Evaluation
---------------------

Run standalone evaluation through the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>`
with the ``robotwin_click_bell_lingbotvlav2_eval`` config. In a new shell, first
export the three variables from `Launch Commands`_. To evaluate a PPO checkpoint
instead of the released weights, pass its full weights file:

.. code-block:: bash

    bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
      runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

Supervised Fine-Tuning
----------------------

Fine-tune the Foundation Base on RoboTwin actions. SFT trains the action expert
and projection layers while freezing Qwen-VL and MoE routers. It uses the same
upstream model source as inference and runs on one node with 8 GPUs. The SFT
FSDP2 policy uses BF16 for the backbone, FP32 for the action expert, and FP32
gradient reduction.

Download the Foundation Base from `Hugging Face <https://huggingface.co/robbyant/lingbot-vla-v2-6b>`__:

This Foundation Base is separate from the RoboTwin post-training checkpoint used by PPO above.

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b \
      --local-dir /path/to/lingbot-vla-v2-6b

Set the Foundation path, the LingBot-VLA V2 source checkout, the tokenizer, and the dataset manifest:

.. code-block:: bash

    export LINGBOT_VLA_V2_PATH=/path/to/.venv/lingbot-vla-v2
    export LINGBOT_VLA_V2_MODEL_ROOT=/path/to/lingbot-vla-v2-6b
    export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
    export LINGBOT_VLA_SFT_DATASET=/path/to/robotwin_train.txt

``LINGBOT_VLA_V2_MODEL_ROOT`` points to the Foundation Base. SFT reads its
RoboTwin model settings, robot config, and normalization statistics from the
installed source checkout. PPO uses the separate
``LINGBOT_VLA_V2_ROBOTWIN_ROOT`` variable for post-training weights.

Convert an extracted RoboTwin dataset. Each input directory must
contain ``data/episode<i>.hdf5`` and ``instructions/episode<i>.json``:

.. code-block:: bash

    python toolkits/lerobot/convert_robotwin_to_lerobot.py \
      --raw-root /path/to/click_bell/aloha-agilex_clean_50 \
      --output-root /path/to/robotwin2_lerobot/click_bell_clean_50
    printf 'robotwin /path/to/robotwin2_lerobot/click_bell_clean_50\n' > "$LINGBOT_VLA_SFT_DATASET"

The manifest contains one ``robot_name dataset_path`` entry per line. Start SFT:

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvlav2

The SFT config places actor ranks on local GPUs 0–7 and uses a global batch size
of 8. Track ``train/l1_loss`` during training. Standalone evaluation reads the
same ``flex_cached`` attention setting from the shared V2 model config. When
loading SFT weights, disable the PPO value head and enable the SFT constructor
alignment fix so evaluation builds the same parameter set:

.. code-block:: bash

    bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
      runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt \
      rollout.model.model_path="${LINGBOT_VLA_V2_MODEL_ROOT}" \
      rollout.model.lingbotvlav2.training_config_path="${LINGBOT_VLA_V2_PATH}/configs/vla/robotwin/robotwin.yaml" \
      rollout.model.add_value_head=false \
      rollout.model.sft_constructor_alignment_fix=true

Foundation Base Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

The plotted runs initialized from the ModelScope Foundation snapshot
``Robbyant/lingbot-vla-v2-6b`` at ModelScope revision
``edfff7638a31a5dbfaae87016477191735bd754e``. The Hugging Face command above
documents the current download workflow; the chart does not claim that the
Hugging Face snapshot has identical revision metadata or file hashes. Official
and RLinf each completed 423 optimizer updates on the same 3,384-occurrence
manifest with global batch size 8.

.. figure:: /_static/images/lingbotvlav2_foundation_action_loss_ma25.png
   :width: 90%

   Raw Action Loss and its 25-update moving average for the Foundation Base comparison.

.. list-table:: Action Loss Statistics
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Official
     - RLinf
   * - Mean
     - 0.102265
     - 0.101771
   * - Last 50 updates mean
     - 0.065720
     - 0.065274

The paired per-update Action Loss has an MAE of 0.002404, Pearson correlation
of 0.9984, and maximum absolute difference of 0.014081. The chart compares
Official ``VLA_Loss`` with RLinf ``train/l1_loss``. Total Loss is tracked
separately because the frameworks may include different auxiliary terms. This
is a controlled comparison; it does not establish strict parity or task success
improvement.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.
