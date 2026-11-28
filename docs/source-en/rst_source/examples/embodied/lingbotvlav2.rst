LingbotVLA 2.0 RL
=================

.. figure:: https://raw.githubusercontent.com/robbyant/lingbot-vla-v2/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/assets/lingbot_vla2_framework.png
   :align: center
   :width: 90%

   LingbotVLA 2.0 framework (image: `Robbyant <https://github.com/robbyant/lingbot-vla-v2>`__).

Fine-tune the action expert of
`LingbotVLA 2.0 <https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin>`__
with PPO or GRPO on RoboTwin 2.0. RLinf loads the released RoboTwin checkpoint
**natively** and starts training from it. LingbotVLA 2.0 is a
vision-language-action model built on a Qwen3-VL backbone. Its mixture-of-experts
action expert generates 50-step action chunks by flow matching in a 55-dimensional
latent action space, which the robot configuration maps to RoboTwin's 14 joint
commands. Both algorithms update only the action expert; the Qwen3-VL backbone and the
action expert's MoE routers stay frozen.

Overview
--------

Train LingbotVLA 2.0 with PPO or GRPO on the RoboTwin 2.0 ``click_bell`` task.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      RoboTwin 2.0

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO / GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Click Bell

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4 GPUs (48 GB)

| **You'll do:** install (native) → clone RoboTwin + assets → download checkpoints → fill in the model paths → PPO / GRPO → watch ``env/success_once``.
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
   * - RoboTwin
     - Click Bell
     - ``robotwin_click_bell_grpo_lingbotvlav2``
     - GRPO training with groups of eight rollouts and no value head.

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

Setup and Run
-------------

.. include:: _setup_common.rst

LingbotVLA 1.0 and 2.0 both install a Python package named ``lingbotvla``, so each needs its own virtual environment. Set ``LINGBOT_VLA_V2_PATH`` before installing if you want the `RLinf LingbotVLA 2.0 fork <https://github.com/RLinf/lingbot-vla-v2/tree/rlinf>`__ cloned outside ``.venv/lingbot-vla-v2``.

.. tab-set::

   .. tab-item:: RoboTwin

      **1. Install**

      Install LingbotVLA 2.0, clone RoboTwin, and download the RoboTwin scene assets:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
         (cd RoboTwin && bash script/_download_assets.sh)

      **2. Download**

      Download the LingbotVLA 2.0 RoboTwin checkpoint and the Qwen3-VL backbone:

      .. code-block:: bash

         hf download robbyant/lingbot-vla-v2-6b-robotwin --local-dir ./lingbot-vla-v2-6b-robotwin
         hf download Qwen/Qwen3-VL-4B-Instruct --local-dir ./Qwen3-VL-4B-Instruct

      **3. Launch**

      Shipped LingbotVLA 2.0 configs:

      .. list-table::
         :header-rows: 1
         :widths: 32 68

         * - Purpose
           - Config
         * - PPO (Click Bell)
           - ``examples/embodiment/config/robotwin_click_bell_ppo_lingbotvlav2.yaml``
         * - GRPO (Click Bell)
           - ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvlav2.yaml``
         * - Model defaults (shared with eval)
           - ``examples/embodiment/config/model/lingbotvlav2.yaml``

      Set the asset paths under ``actor.model`` in the PPO or GRPO experiment config and under ``rollout.model`` in the standalone evaluation config:

      .. code-block:: yaml

         model_path: ./lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
         tokenizer_path: ./Qwen3-VL-4B-Instruct
         lingbotvlav2:
           training_config_path: ./lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
           robot_config_path: ./.venv/lingbot-vla-v2/configs/robot_configs/robotwin.yaml
           stats_path: ./.venv/lingbot-vla-v2/assets/norm_stats/robotwin.json

      Export the RoboTwin paths, then start PPO training:

      .. code-block:: bash

         export ROBOT_PLATFORM=ALOHA
         export ROBOTWIN_PATH=$PWD/RoboTwin
         export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2

      To train with GRPO, use the same environment and run

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvlav2

      When evaluating a GRPO checkpoint, pass ``rollout.model.add_value_head=false``.

      .. dropdown:: Advanced configuration (optional)

         - **Resume training**: set ``runner.resume_dir`` in the training config to a saved checkpoint directory.
         - **Flow-SDE PPO & critic warmup**: rollout samples one random denoising step per chunk from the flow-SDE and integrates the rest with the ODE. PPO scores that step over the 14 decoded latent coordinates with ``logprob_type: token_level``, dual-clip ``clip_ratio_c: 3.0``, chunk-level rewards, and ``actor.optim.critic_warmup_steps: 20``.
         - **Mixed precision**: the actor stores trainable action-expert weights in ``fp32`` so small updates do not round away, while FSDP2 computes in ``bf16`` and rollout keeps a ``bf16`` copy.
         - **Log-prob drift guard**: before each update, the actor re-scores every sampled SDE step. ``algorithm.max_recompute_logprob_gap: 1.0e-3`` halts training if the summed log-probability differs between rollout and actor by more than the threshold (logged under ``actor/rollout_train_joint_logprob_gap_max``).
         - **Standalone evaluation**: follow the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>` with ``robotwin_click_bell_lingbotvlav2_eval`` (in a new shell, export the three variables above first). To evaluate a PPO checkpoint instead of the released weights, pass its full weights file:

           .. code-block:: bash

              bash evaluations/run_eval.sh robotwin robotwin_click_bell_lingbotvlav2_eval \
                runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

Training curves
~~~~~~~~~~~~~~~

On RoboTwin ``click_bell``, PPO and GRPO each ran for 100 training steps on four H20 GPUs.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvlav2_robotwin_click_bell_ppo_grpo_success_once.png
   :align: center

   Training success rate (``env/success_once``) by TensorBoard step for GRPO (green)
   and PPO (blue). Light lines show the recorded values and dark lines show the smoothed curves.

Checkpoint evaluation
~~~~~~~~~~~~~~~~~~~~~

On four A100 80 GB GPUs, every checkpoint was evaluated on the same 64 fixed
environment seeds with a limit of 400 steps per trajectory and
``rollout.model.eval_seed=0``:

.. list-table:: LingbotVLA 2.0 Evaluation Results on RoboTwin (n=64)
   :header-rows: 1

   * - Task
     - Released Weights
     - RLinf-GRPO (Step 100)
     - RLinf-PPO (Step 100)
   * - ``click_bell``
     - 76.56%
     - **100.00%**
     - **100.00%**

To compare checkpoints, keep environment seeds, worker placement, batch sizes, and
``rollout.model.eval_seed`` identical. ``eval_seed`` sets the initial evaluation
noise independently of model initialization.
