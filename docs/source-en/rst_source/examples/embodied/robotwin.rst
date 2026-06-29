RL with RoboTwin Benchmark
===========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 dual-arm manipulation tasks (image: `RoboTwin <https://robotwin-platform.github.io>`__).

`RoboTwin 2.0 <https://robotwin-platform.github.io>`__ is a dual-arm manipulation
benchmark with a large task suite. You'll use RLinf to RL-fine-tune VLA policies on
RoboTwin tasks such as ``place_empty_cup`` and ``adjust_bottle``.

Overview
--------

Fine-tune a VLA on RoboTwin 2.0; OpenVLA-OFT + GRPO lifts average task success by +57%.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅ · Lingbot-VLA

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO · DAgger

   .. grid-item-card:: Tasks
      :text-align: center

      46 supported · 10 configured

   .. grid-item-card:: Hardware
      :text-align: center

      1–2 nodes · 8–16 GPUs

| **You'll do:** install → clone RoboTwin + assets → download an SFT model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · RoboTwin repo and assets · an SFT checkpoint.

Tasks
~~~~~

RoboTwin supports 46 manipulation tasks. RLinf ships ready-to-run env configs for these tasks:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Task
     - Description
   * - ``adjust_bottle``
     - Pick up the bottle on the table head-up with the correct arm.
   * - ``place_empty_cup``
     - Place the empty cup on the coaster.
   * - ``place_container_plate``
     - Place the container onto the plate.
   * - ``pick_dual_bottles``
     - Pick up one bottle with each arm.
   * - ``move_can_pot``
     - Move a can to beside the pot.
   * - ``lift_pot``
     - Lift the pot with both arms.
   * - ``handover_block``
     - Handover a red block from the left arm to the right arm, then place it on the blue pad.
   * - ``beat_block_hammer``
     - Grab the hammer and hit the block.
   * - ``click_bell``
     - Click the bell's top center.
   * - ``place_shoe``
     - Pick up a shoe and place it on the mat.

.. note::

   Four RoboTwin tasks are not yet supported in RLinf: ``place_fan``, ``open_laptop``,
   ``place_object_scale``, and ``put_object_cabinet``. Dense reward functions are
   still being expanded across tasks.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Specification
   * - ``images``
     - Head-camera RGB, ``[B, 224, 224, 3]`` uint8, center-cropped when enabled.
   * - ``wrist_images``
     - Optional left/right wrist-camera RGB, ``[B, n, 224, 224, 3]`` uint8, or ``None``.
   * - ``states``
     - Proprioception, ``[B, 14]`` float32.
   * - ``task_descriptions``
     - Natural-language task descriptions.
   * - ``actions``
     - VLA-dependent continuous action chunks for ALOHA-style dual-arm control.

Installation
------------

.. include:: _setup_common.rst

**Docker image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-robotwin

   # For mainland China users:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-robotwin

Switch to the matching virtual environment inside the image:

.. code:: bash

   # OpenVLA-OFT
   source switch_env openvla-oft

   # OpenPI π₀ / π₀.₅
   # source switch_env openpi

   # Lingbot-VLA, if available in your image
   # source switch_env lingbotvla

**Custom environment**

Install dependencies for the model you want to run:

.. code:: bash

   # Mainland China users can add --use-mirror.

   # OpenVLA-OFT
   bash requirements/install.sh embodied --model openvla-oft --env robotwin

   # OpenPI π₀ / π₀.₅
   # bash requirements/install.sh embodied --model openpi --env robotwin

   # Lingbot-VLA
   # bash requirements/install.sh embodied --model lingbotvla --env robotwin

   source .venv/bin/activate

Clone RoboTwin and download its assets:

.. code:: bash

   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   cd RoboTwin
   bash script/_download_assets.sh

   export PYTHONPATH=/path/to/RoboTwin:$PYTHONPATH
   export ROBOT_PLATFORM=ALOHA

Download the Model
------------------

Download the SFT checkpoint that matches your config. Examples:

**OpenVLA-OFT**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

**OpenPI π₀ / π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
   git clone https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir RLinf-Pi0-RoboTwin-SFT-adjust_bottle
   hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir RLinf-Pi05-RoboTwin-SFT-adjust_bottle

.. include:: _model_path.rst

For Lingbot-VLA recipes, point ``actor.model.model_path`` and
``rollout.model.model_path`` at your Lingbot-VLA SFT checkpoint.

.. note::

   The action-normalization key ``unnorm_key`` in the config (e.g.
   ``unnorm_key: "place_empty_cup"``) must match the ``unnorm_key`` used when the SFT
   checkpoint was trained, otherwise actions will be denormalized incorrectly.

Run It
------

Pick one recipe and launch training:

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - Recipe
     - Config
     - Command suffix
   * - OpenVLA-OFT + GRPO
     - ``examples/embodiment/config/robotwin_place_empty_cup_grpo_openvlaoft.yaml``
     - ``robotwin_place_empty_cup_grpo_openvlaoft``
   * - OpenVLA-OFT + PPO
     - ``examples/embodiment/config/robotwin_place_empty_cup_ppo_openvlaoft.yaml``
     - ``robotwin_place_empty_cup_ppo_openvlaoft``
   * - π₀ + PPO
     - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml``
     - ``robotwin_adjust_bottle_ppo_openpi``
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml``
     - ``robotwin_adjust_bottle_ppo_openpi_pi05``
   * - OpenPI + DAgger
     - ``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml``
     - ``robotwin_adjust_bottle_dagger_openpi``
   * - Lingbot-VLA + GRPO
     - ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``
     - ``robotwin_click_bell_grpo_lingbotvla``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_place_empty_cup_grpo_openvlaoft
   bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_ppo_openpi_pi05

What this does:

1. Starts the embodied training entrypoint with the selected RoboTwin Hydra config.
2. Creates Ray workers for actor, rollout, and RoboTwin env components.
3. Runs rollouts, computes task rewards, and updates the selected VLA policy.

Run standalone evaluation through the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>`.
It owns ``ROBOTWIN_PATH`` / ``assets_path`` setup, available eval configs such as
``robotwin_place_empty_cup_openvlaoft_eval`` and ``robotwin_adjust_bottle_openpi_pi05_eval``,
and result interpretation.

.. note::

   Replace ``assets_path`` in the config with your downloaded RoboTwin assets path.
   The provided configs use train/eval seed files under ``rlinf/envs/robotwin/seeds/``.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Videos are saved through the env video config:

.. code:: yaml

   video_cfg:
     save_video: True
     video_base_dir: ${runner.logger.log_path}/video/eval

.. list-table:: OpenVLA-OFT evaluation results on seven RoboTwin tasks
   :header-rows: 1

   * - Task
     - SFT
     - RLinf-GRPO
     - RLinf-PPO
   * - ``beat_block_hammer``
     - |huggingface| `10.15% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer>`__
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer>`__
     - ---
   * - ``pick_dual_bottles``
     - |huggingface| `20.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles>`__
     - |huggingface| `92.96% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles>`__
     - ---
   * - ``place_empty_cup``
     - |huggingface| `75.78% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`__
     - |huggingface| `94.53% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup>`__
     - |huggingface| `92.97% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-PPO-place_empty_cup>`__
   * - ``place_container_plate``
     - |huggingface| `54.69% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_container_plate>`__
     - |huggingface| `95.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_container_plate>`__
     - ---
   * - ``move_can_pot``
     - |huggingface| `9.37% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot>`__
     - |huggingface| `83.59% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot>`__
     - ---
   * - ``lift_pot``
     - |huggingface| `3.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot>`__
     - ---
   * - ``handover_block``
     - |huggingface| `28.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block>`__
     - ---
   * - Average
     - 28.79%
     - **86.16%**
     - ---
   * - Δ Avg.
     - ---
     - **+57.37%**
     - ---

.. list-table:: OpenPI evaluation results on RoboTwin ``adjust_bottle``
   :header-rows: 1

   * - Method
     - SFT
     - RLinf-PPO
   * - π₀
     - |huggingface| `76.56% <https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle>`__
     - |huggingface| `98.44% <https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-PPO-adjust_bottle>`__
   * - π₀.₅
     - |huggingface| `85.94% <https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle>`__
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-PPO-adjust_bottle>`__

.. note::

   OpenVLA-OFT results use the ``demo_randomized`` setting. OpenPI results use
   ``demo_clean``. For task-level simulator options, see the
   `RoboTwin configuration documentation <https://robotwin-platform.github.io/doc/usage/configurations.html>`__.
