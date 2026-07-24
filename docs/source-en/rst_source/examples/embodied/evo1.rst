RL on Evo-1 Models
==================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

`Evo-1 <https://github.com/MINT-SJTU/Evo-1>`__ is a compact (~1B) vision-language-action
model: an InternVL3-1B vision-language backbone with a flow-matching (DiT) action head.
RLinf integrates it **natively** — embedded in RLinf's Python memory space for
zero-latency, tensor-level interaction — and supports full-parameter SFT and GRPO
fine-tuning on the LIBERO simulator.

Overview
--------

SFT then GRPO-fine-tune Evo-1 on LIBERO manipulation tasks.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      LIBERO-Spatial

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4–8 GPUs

| **You'll do:** install (native) → download the Evo-1 checkpoint → GRPO → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the Evo-1 checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_grpo_evo1``
     - GRPO training with Evo-1 on the LIBERO-Spatial suite.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - LIBERO camera observations and robot state required by Evo-1.
   * - Action
     - Continuous 7-DoF actions (6-DoF delta EE + gripper), decoded by the flow-matching head.
   * - Reward
     - LIBERO task success.
   * - Prompt
     - Natural-language task instruction for the LIBERO episode.

Installation
------------

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/RLinf/RLinf.git
    cd RLinf
    export RLINF_PATH=$(pwd)

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Install the Evo-1 native environment and LIBERO base dependencies in one command (the
script pulls the Evo-1 source into ``.venv/Evo-1`` and handles dependency isolation):

.. code-block:: bash

    bash requirements/install.sh embodied --model evo1 --env libero --use-mirror
    source .venv/bin/activate

Download the Model
------------------

Download the Evo-1 LIBERO checkpoint from HuggingFace (a checkpoint directory containing
``config.json``, ``norm_stats.json``, ``mp_rank_00_model_states.pt``):

.. code-block:: bash

    git lfs install
    git clone https://huggingface.co/MINT-SJTU/Evo1_LIBERO

Then set ``rollout.model.model_path`` and ``actor.model.model_path`` in the config to your
local checkpoint path. ``actor.model.evo1.arm_key`` / ``dataset_key`` must match the
top-level key(s) in the checkpoint's ``norm_stats.json``.

Run It
------

Configuration Files
~~~~~~~~~~~~~~~~~~~

* **SFT (supervised fine-tuning)**:
  ``examples/sft/config/libero_sft_evo1.yaml`` (run via ``examples/sft/run_vla_sft.sh``)
* **GRPO (Reinforcement Learning)**:
  ``examples/embodiment/config/libero_spatial_grpo_evo1.yaml``
* **Standalone eval**:
  ``evaluations/libero/libero_spatial_evo1_eval.yaml`` (run via ``evaluations/run_eval.sh``)

Key Config Snippets (GRPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The top-level file assembles the environment and model via Hydra and overrides the
flow-matching SDE sampling / GRPO parameters under ``actor.model`` and ``algorithm``.

.. code-block:: yaml

    rollout:
      model:
        model_type: "evo1"

    actor:
      model:
        model_type: "evo1"
        model_path: "/path/to/model/Evo1_LIBERO"
        num_action_chunks: 14        # steps executed per inference
        evo1:
          arm_key: "libero_robot"    # must match norm_stats.json
          dataset_key: "libero_robot"
        rl_head_config:
          noise_method: "flow_sde"
          noise_level: 0.5           # SDE noise scale
          denoising_steps: 8         # SDE denoise steps for RL rollout + replay
      # freeze the InternVL3 VLM; train only the flow-matching action head
      model.evo1.rl_trainable_scope: "action_head"

    algorithm:
      adv_type: grpo
      logprob_type: token_level      # per-dim ratio (14x7 executed dims)
      group_size: 8
      update_epoch: 2
      clip_ratio_low: 0.2
      clip_ratio_high: 0.28          # clip-higher: sustains improvement

Launch Commands
~~~~~~~~~~~~~~~

.. code-block:: bash

    export ROBOT_PLATFORM="LIBERO"
    export EVO1_REPO_PATH=$(python -c "import os,evo1; print(os.path.dirname(evo1.__file__))" 2>/dev/null || echo /path/to/Evo-1)
    bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_evo1

To reproduce the RL result quickly on a single task, restrict training to one
LIBERO-Spatial task with ``+env.train.task_id_filter=[0] +env.eval.task_id_filter=[0]``;
``env/success_at_end`` rises from the SFT baseline within tens of GRPO steps (see Results below).

Supervised Fine-Tuning
----------------------

To SFT Evo-1 on a LIBERO-style LeRobot dataset (e.g. to produce the SFT
checkpoint before RL), use the SFT config and launcher:

.. code-block:: bash

    export EVO1_REPO_PATH=/path/to/Evo-1
    bash examples/sft/run_vla_sft.sh libero_sft_evo1

Point ``data.train_data_paths`` at an Evo-1 dataset config YAML and
``actor.model.model_path`` at the checkpoint (or base) to fine-tune. The SFT
recipe (AdamW, cosine schedule with warmup, flow-matching MSE) follows Evo-1's
native training; checkpoints are written under ``runner.logger.log_path``.

Standalone Evaluation
---------------------

Evaluate a checkpoint with the standalone eval harness (config under ``evaluations/libero/``):

.. code-block:: bash

    bash evaluations/run_eval.sh libero libero_spatial_evo1_eval \
      rollout.model.model_path=/path/to/ckpt

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. GRPO from an Evo-1 SFT checkpoint produces a stable
rise in LIBERO-Spatial success rate (n=64 validation, frozen VLM, action-head-only RL):

.. list-table:: Evo-1 GRPO on LIBERO-Spatial (n=64)
   :header-rows: 1

   * - Setting
     - SFT
     - RLinf-GRPO
   * - Single task (``success_at_end``)
     - 0.58
     - **0.86**
   * - LIBERO-Spatial suite (``success_once``, best ckpt)
     - 0.656
     - **0.750**

Videos are saved through the env video config:

.. code:: yaml

   video_cfg:
     save_video: True
     video_base_dir: ${runner.logger.log_path}/video/eval
