LingbotVLA 2.0 SFT
==================

.. figure:: https://raw.githubusercontent.com/robbyant/lingbot-vla-v2/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/assets/lingbot_vla2_framework.png
   :align: center
   :width: 90%

   LingbotVLA 2.0 framework (image: `Robbyant <https://github.com/robbyant/lingbot-vla-v2>`__).

Run RoboTwin supervised fine-tuning (SFT) for LingbotVLA 2.0 in RLinf. Convert RoboTwin HDF5 demonstrations to LeRobot format, launch the SFT recipe, monitor its checkpoints, and evaluate an SFT checkpoint with the RoboTwin evaluation entry point.

Overview
--------

Use this page when you want to adapt the released LingbotVLA 2.0 RoboTwin checkpoint with action-expert SFT.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      LingbotVLA 2.0

   .. grid-item-card:: Methods
      :text-align: center

      Action-expert SFT

   .. grid-item-card:: Data
      :text-align: center

      RoboTwin 2.0 → LeRobot

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install the model environment → convert RoboTwin demonstrations → run ``robotwin_sft_lingbotvlav2`` → inspect the loss and saved checkpoints → evaluate a checkpoint.
| **Prerequisites:** :doc:`RLinf installation </rst_source/start/installation>` · a RoboTwin RLinf_support checkout and assets · the LingbotVLA 2.0 RoboTwin checkpoint · a Qwen3-VL tokenizer checkpoint.

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
     - ``robotwin_sft_lingbotvlav2``
     - SFT on the action expert from the released RoboTwin checkpoint.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The SFT dataset uses the same RoboTwin observation and action contract as the released model.

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - Head camera, both wrist cameras, and the 14-dimensional joint state.
   * - Action
     - 14 joint commands decoded from a 50-step chunk of 55-dimensional latent actions.
   * - Prompt
     - Natural-language task instruction from the RoboTwin episode.

Prepare the Dataset
-------------------

Convert each extracted RoboTwin release into one LeRobot dataset before starting SFT. The input directory must contain ``data/episode<i>.hdf5`` and ``instructions/episode<i>.json``.

Set the output directory and convert a Click Bell release:

.. code-block:: bash

    export LINGBOT_VLA_SFT_DATASET=/path/to/robotwin2_lerobot/click_bell_clean_50
    python toolkits/lerobot/convert_robotwin_to_lerobot.py \
      --raw-root /path/to/click_bell/aloha-agilex_clean_50 \
      --output-root "$LINGBOT_VLA_SFT_DATASET"

The converter writes a LeRobot v3 dataset with the three RoboTwin camera streams, the 14-dimensional state, the next-step action, and the episode instructions.

Download the Model
------------------

Download the released RoboTwin checkpoint and the Qwen3-VL tokenizer checkpoint, then set the paths used by the SFT model template:

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b-robotwin \
      --local-dir /path/to/lingbot-vla-v2-6b-robotwin
    hf download Qwen/Qwen3-VL-4B-Instruct \
      --local-dir /path/to/Qwen3-VL-4B-Instruct

    export LINGBOT_VLA_V2_PATH=/path/to/.venv/lingbot-vla-v2
    export LINGBOT_VLA_V2_MODEL_ROOT=/path/to/lingbot-vla-v2-6b-robotwin
    export LINGBOT_VLA_V2_CHECKPOINT=$LINGBOT_VLA_V2_MODEL_ROOT/checkpoints/global_step_50000/hf_ckpt
    export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct

The model template reads ``lingbotvla_cli.yaml`` from ``LINGBOT_VLA_V2_MODEL_ROOT`` and uses the shared upstream robot configuration and normalization statistics under ``LINGBOT_VLA_V2_PATH``.

Installation
------------

Use the LingbotVLA 2.0 RoboTwin environment before running the SFT script:

.. code-block:: bash

    bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin
    source .venv/bin/activate

For the Docker option and the RoboTwin repository setup, see :doc:`LingbotVLA 2.0 PPO <lingbotvlav2>` and the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>`.

Run It
------

Start the SFT recipe after the dataset, model paths, and environment are ready:

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvlav2

The recipe uses the SFT model configuration in ``examples/sft/config/model/lingbotvlav2.yaml``. It trains for 30,000 steps, saves checkpoints every 1,000 steps, and uses ``flex_cached`` for the SFT model configuration. The launcher writes run logs under ``logs/<timestamp>-robotwin_sft_lingbotvlav2``.

Before evaluating a checkpoint, set the RoboTwin environment variables used by the evaluation config:

.. code-block:: bash

    export ROBOT_PLATFORM=ALOHA
    export ROBOTWIN_PATH=/path/to/RoboTwin
    export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH

Evaluate an SFT checkpoint with the existing RoboTwin evaluation entry point. The CLI override keeps evaluation on the same ``flex_cached`` attention implementation used by SFT; the shared eval YAML retains its ``eager`` default.

.. code-block:: bash

    bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
      rollout.model.lingbotvlav2.attention_implementation=flex_cached \
      runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

Use the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>` for the evaluation protocol and result interpretation.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repository root and point it at the SFT logs:

.. code-block:: bash

    tensorboard --logdir ./logs --port 6006

Monitor the training loss and the saved ``global_step_N`` checkpoints. RoboTwin evaluation reports ``eval/success_once``; use the same evaluator and seed protocol when comparing the base checkpoint with an SFT checkpoint.
