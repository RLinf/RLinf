LingbotVLA 2.0 RL
=================

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
         * - Model defaults (shared with eval)
           - ``examples/embodiment/config/model/lingbotvlav2.yaml``

      Fill in the empty paths in ``examples/embodiment/config/model/lingbotvlav2.yaml`` (shared by training and standalone evaluation; startup reports any path left empty):

      .. code-block:: yaml

         model_path: ./lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
         tokenizer_path: ./Qwen3-VL-4B-Instruct
         lingbotvlav2:
           training_config_path: ./lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
           robot_config_path: ./.venv/lingbot-vla-v2/configs/robot_configs/robotwin.yaml
           stats_path: ./.venv/lingbot-vla-v2/assets/norm_stats/robotwin.json

      Export the RoboTwin paths, then start PPO training:

      .. code-block:: bash

         export ROBOTWIN_PATH=$PWD/RoboTwin
         export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2

      .. dropdown:: Advanced configuration (optional)

         - **Resume training**: set ``runner.resume_dir`` in the training config to a saved checkpoint directory.
         - **Flow-SDE PPO & critic warmup**: rollout samples one random denoising step per chunk from the flow-SDE and integrates the rest with the ODE. PPO scores that step over the 14 decoded latent coordinates with ``logprob_type: token_level``, dual-clip ``clip_ratio_c: 3.0``, chunk-level rewards, and ``actor.optim.critic_warmup_steps: 20``.
         - **Mixed precision**: the actor stores trainable action-expert weights in ``fp32`` so small updates do not round away, while FSDP2 computes in ``bf16`` and rollout keeps a ``bf16`` copy.
         - **Log-prob drift guard**: before each update, the actor re-scores every sampled SDE step. ``algorithm.max_recompute_logprob_gap: 1.0e-3`` halts training if the summed log-probability differs between rollout and actor by more than the threshold (logged under ``actor/rollout_train_joint_logprob_gap_max``).
         - **Standalone evaluation**: follow the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>` with ``robotwin_click_bell_lingbotvlav2_eval`` (in a new shell, export the two variables above first). To evaluate a PPO checkpoint instead of the released weights, pass its full weights file:

           .. code-block:: bash

              bash evaluations/run_eval.sh robotwin robotwin_click_bell_lingbotvlav2_eval \
                runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.
