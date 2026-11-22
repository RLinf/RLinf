LingbotVLA 1.0 RL
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png
   :align: center
   :width: 90%

   LingbotVLA 1.0 on RoboTwin (image: `RLinf <https://github.com/RLinf>`__).

`LingbotVLA 1.0 <https://huggingface.co/robbyant/lingbot-vla-4b>`__ is a Qwen2.5-VL-based
vision-language-action model that autoregressively generates continuous action chunks.
RLinf integrates it **natively** — embedded in RLinf's Python memory space for
zero-latency, tensor-level interaction — and supports full-parameter SFT and GRPO
fine-tuning on the RoboTwin 2.0 simulator.

Overview
--------

SFT then GRPO-fine-tune LingbotVLA 1.0 on RoboTwin 2.0 dual-arm manipulation tasks.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      RoboTwin 2.0

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Click Bell · Place Shoe

   .. grid-item-card:: Hardware
      :text-align: center

      1–2 nodes · 8–16 GPUs

| **You'll do:** install (native) → clone RoboTwin + assets → download checkpoints → SFT / GRPO → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the RoboTwin repo and assets · the LingbotVLA 1.0 and Qwen backbone checkpoints (steps below).

Tasks
~~~~~

Select the model page by matching the environment, task family, and config or checkpoint artifact.

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - RoboTwin
     - Click Bell
     - ``robotwin_click_bell_grpo_lingbotvla``
     - GRPO training with LingbotVLA on a RoboTwin manipulation task.
   * - RoboTwin
     - Place Shoe
     - ``robotwin_place_shoe_grpo_lingbotvla``
     - GRPO training on a second RoboTwin task variant.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - RoboTwin camera observations and robot state required by LingbotVLA.
   * - Action
     - Continuous robot actions decoded by the LingbotVLA policy.
   * - Reward
     - RoboTwin task success or shaped task reward.
   * - Prompt
     - Natural-language task instruction for the RoboTwin episode.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: GRPO Training

      **1. Install**

      Install the LingbotVLA 1.0 environment, clone RoboTwin, and download the RoboTwin scene assets:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvla --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
         (cd RoboTwin && bash script/_download_assets.sh)

      **2. Download**

      Download the LingbotVLA 1.0 base weights, the pinned RoboTwin SFT checkpoint (use this revision instead of ``main``), and the Qwen2.5-VL backbone:

      .. code-block:: bash

         hf download robbyant/lingbot-vla-4b --local-dir ./lingbot-vla-4b
         hf download robbyant/lingbot-vla-4b-posttrain-robotwin \
           --revision 3e0c7c476bde3daaac00f79f3741a292a299f60a \
           --local-dir ./lingbot-vla-4b-posttrain-robotwin
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      **3. Launch**

      Shipped GRPO configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - Task
           - Config
         * - Click Bell
           - ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``
         * - Place Shoe
           - ``examples/embodiment/config/robotwin_place_shoe_grpo_lingbotvla.yaml``

      Set the local model paths in your config (rollout inherits ``actor.model.model_path`` and ``tokenizer_path`` automatically), and point ``env.train.assets_path`` and ``env.eval.assets_path`` at your RoboTwin assets directory:

      .. code-block:: yaml

         actor:
           model:
             model_path: ./lingbot-vla-4b-posttrain-robotwin
             tokenizer_path: ./Qwen2.5-VL-3B-Instruct
             lingbotvla:
               config_path: ./lingbot-vla-4b

      Export the repository paths, then launch GRPO training:

      .. code-block:: bash

         export ROBOTWIN_PATH=$PWD/RoboTwin
         export LINGBOT_VLA_PATH=$(python -c "import lingbotvla, os; print(os.path.dirname(lingbotvla.__path__[0]))")
         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla

      .. dropdown:: Advanced configuration (optional)

         - **Unified normalization**: LingbotVLA 1.0 uses global normalization keys from ``robotwin_50.json`` (such as ``action.arm.position``), so switching RoboTwin tasks needs no ``unnorm_key`` override.
         - **Host memory trimming (Linux/glibc)**: GRPO configs default to ``env.train.malloc_trim_reset_interval: 8`` to reclaim unused glibc heap memory every 8 automatic resets per environment instance (targets host RSS, not GPU memory). Set to ``0`` to disable; evaluation keeps it disabled unless configured.
         - **Standalone evaluation**: follow the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>` with configs such as ``robotwin_click_bell_lingbotvla_eval`` and ``robotwin_place_shoe_lingbotvla_eval``.

   .. tab-item:: SFT Training

      **1. Install**

      Install the LingbotVLA 1.0 environment, clone RoboTwin, and download the RoboTwin scene assets:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvla --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
         (cd RoboTwin && bash script/_download_assets.sh)

      **2. Download**

      Download the LingbotVLA 1.0 base weights, the pinned RoboTwin SFT checkpoint (use this revision instead of ``main``), and the Qwen2.5-VL backbone:

      .. code-block:: bash

         hf download robbyant/lingbot-vla-4b --local-dir ./lingbot-vla-4b
         hf download robbyant/lingbot-vla-4b-posttrain-robotwin \
           --revision 3e0c7c476bde3daaac00f79f3741a292a299f60a \
           --local-dir ./lingbot-vla-4b-posttrain-robotwin
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      **3. Launch**

      Set the local model paths in ``examples/sft/config/robotwin_sft_lingbotvla.yaml`` and point ``data.train_data_paths`` at your converted LeRobot dataset directory:

      .. code-block:: yaml

         actor:
           model:
             model_path: ./lingbot-vla-4b-posttrain-robotwin
             tokenizer_path: ./Qwen2.5-VL-3B-Instruct
             lingbotvla:
               config_path: ./lingbot-vla-4b

      Export the repository paths, then launch SFT:

      .. code-block:: bash

         export ROBOTWIN_PATH=$PWD/RoboTwin
         export LINGBOT_VLA_PATH=$(python -c "import lingbotvla, os; print(os.path.dirname(lingbotvla.__path__[0]))")
         bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvla

      .. dropdown:: Advanced configuration (optional)

         - **Unified normalization**: LingbotVLA 1.0 uses global normalization keys from ``robotwin_50.json`` (such as ``action.arm.position``), so switching RoboTwin tasks needs no ``unnorm_key`` override.
         - **Standalone evaluation**: follow the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>` with configs such as ``robotwin_click_bell_lingbotvla_eval`` and ``robotwin_place_shoe_lingbotvla_eval``.

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

.. list-table:: LingbotVLA 1.0 evaluation results on RoboTwin tasks
   :header-rows: 1

   * - Task
     - SFT
     - RLinf-GRPO
   * - ``click_bell``
     - |huggingface| `96.88% <https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin/tree/3e0c7c476bde3daaac00f79f3741a292a299f60a>`__
     - |huggingface| `98.75% <https://huggingface.co/RLinf/RLinf-lingbotvla-click-bell-grpo>`__
   * - ``place_shoe``
     - |huggingface| `93.75% <https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin/tree/3e0c7c476bde3daaac00f79f3741a292a299f60a>`__
     - |huggingface| `98.44% <https://huggingface.co/RLinf/RLinf-lingbotvla-place-shoe-grpo>`__
   * - Average
     - 95.31%
     - **98.60%**
   * - Δ Avg.
     - ---
     - **+3.29%**

.. note::

   LingbotVLA 1.0 results use the ``demo_randomized`` setting. For task-level simulator
   options, see the
   `RoboTwin configuration documentation <https://robotwin-platform.github.io/doc/usage/configurations.html>`__.
