RoboTwin RL
===========

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

      NVIDIA CUDA · :ref:`AMD ROCm <robotwin-hardware>` (1–2 nodes · 8–16 GPUs)

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

.. _robotwin-hardware:

Setup and Run
-------------

.. include:: _setup_common.rst

Select a model tab for NVIDIA CUDA, or switch to **AMD ROCm** for AMD GPUs.

.. tab-set::

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. Install**

      Install RLinf, clone the ``RLinf_support`` RoboTwin branch alongside it, and download RoboTwin assets:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA

      **2. Download**

      Download the OpenVLA-OFT SFT checkpoint (example: ``place_empty_cup``):

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir ./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

      **3. Launch**

      Available configs under ``examples/embodiment/config/``:

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

      Launch training with ``run_embodiment.sh``, overriding ``actor.model.model_path`` (rollout inherits it automatically) and ``env.train.assets_path`` / ``env.eval.assets_path`` (pointing to the ``RoboTwin`` directory, parent of ``assets/``) on the command line (or set the same keys in the selected YAML):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_place_empty_cup_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
           env.train.assets_path=../RoboTwin \
           env.eval.assets_path=../RoboTwin

   .. tab-item:: OpenPI
      :sync: openpi

      **1. Install**

      Install RLinf, clone the ``RLinf_support`` RoboTwin branch alongside it, and download RoboTwin assets:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"

      **2. Download**

      Download the OpenPI π₀.₅ SFT checkpoint for ``adjust_bottle`` (for π₀, use ``RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`` instead):

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle

      **3. Launch**

      Available configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 24 48 28

         * - Recipe
           - Config
           - Command suffix
         * - π₀ + PPO
           - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml``
           - ``robotwin_adjust_bottle_ppo_openpi``
         * - π₀.₅ + PPO
           - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml``
           - ``robotwin_adjust_bottle_ppo_openpi_pi05``
         * - OpenPI + DAgger
           - ``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml``
           - ``robotwin_adjust_bottle_dagger_openpi``

      Set ``actor.model.model_path`` (rollout inherits it automatically) and point ``env.train.assets_path`` / ``env.eval.assets_path`` to the ``RoboTwin`` directory:

      .. code-block:: yaml

         actor:
           model:
             model_path: ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle
         env:
           train:
             assets_path: ../RoboTwin
           eval:
             assets_path: ../RoboTwin

      Launch training with ``run_embodiment.sh``:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_ppo_openpi_pi05

   .. tab-item:: LingBotVLA
      :sync: lingbotvla

      **1. Install**

      Install RLinf, clone the ``RLinf_support`` RoboTwin branch alongside it, and download RoboTwin assets:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvla --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"

      **2. Download**

      Prepare your Lingbot-VLA SFT checkpoint locally (for example under ``./LingBotVLA-RoboTwin-SFT``).

      **3. Launch**

      Available config under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 24 48 28

         * - Recipe
           - Config
           - Command suffix
         * - Lingbot-VLA + GRPO
           - ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``
           - ``robotwin_click_bell_grpo_lingbotvla``

      Set ``actor.model.model_path`` to your Lingbot-VLA SFT checkpoint (rollout inherits it automatically) and point ``env.train.assets_path`` / ``env.eval.assets_path`` to the ``RoboTwin`` directory:

      .. code-block:: yaml

         actor:
           model:
             model_path: ./LingBotVLA-RoboTwin-SFT
         env:
           train:
             assets_path: ../RoboTwin
           eval:
             assets_path: ../RoboTwin

      Launch training with ``run_embodiment.sh``:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla

   .. tab-item:: AMD ROCm
      :sync: rocm

      RoboTwin installs on AMD ROCm with one change to the task config: ``planner_backend: mplib``.

      **1. Install**

      Install on a host with ROCm available, then clone the ``RLinf_support`` RoboTwin branch and download assets (omit ``--rocm`` to detect the installed version, and replace ``--model`` with ``openvla-oft`` or ``lingbotvla`` if needed):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"

      .. dropdown:: Build and run the ROCm Docker image (optional)

         To build the ROCm RoboTwin image from your checkout, run this command on the host with ``ROCM_ARCHS`` set to the ``gfx`` architecture of your GPUs (``gfx1100`` for the Radeon PRO W7900):

         .. code-block:: bash

            DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
               --build-arg PLATFORM=amd \
               --build-arg ROCM_VER=7.2 \
               --build-arg ROCM_ARCHS=gfx1100 \
               --build-arg BUILD_TARGET=embodied-robotwin \
               -t rlinf:embodied-robotwin-rocm7.2 .

         Start the container with the AMD kernel and render devices:

         .. code-block:: bash

            docker run -it --rm \
               --device=/dev/kfd --device=/dev/dri --group-add video \
               --ipc=host --shm-size 32g --network host \
               -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
               rlinf:embodied-robotwin-rocm7.2 bash

         .. warning::

            Match ``ROCM_ARCHS`` to the target GPUs. During Docker builds the devices may be invisible, so extensions such as ``flash-attn`` need explicit architecture values. On GPUs other than ``gfx90a`` and ``gfx942``, the installer also switches ``flash-attn`` to its Triton backend.

      **2. Download**

      Download the matching SFT checkpoint (example: OpenPI π₀.₅ on ``adjust_bottle``):

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle

      **3. Launch**

      ``pytorch3d``, ``warp-lang``, and ``curobo`` are CUDA-only, so the installer skips them. They exist for the ``curobo`` motion planner, which means RoboTwin needs the other planner: set ``planner_backend: mplib`` in the task config (the ``curobo`` default fails while the environment is being constructed). Set ``actor.model.model_path`` and ``assets_path`` in ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml``, then launch with the same command:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_ppo_openpi_pi05

.. dropdown:: Advanced configuration (optional)

   - **Camera near plane (``camera_near``):** RLinf creates RoboTwin's cameras with a 0.01 m near plane instead of RoboTwin's 0.1 m: SAPIEN 3.0.1 clips everything closer than the near plane, and 0.1 m hides the gripper fingers from the wrist cameras. Set ``camera_near: null`` in the RoboTwin env config to keep RoboTwin's own value.

   - **Action normalization (``unnorm_key``):** ``unnorm_key`` in the config (e.g. ``unnorm_key: "place_empty_cup"``) must match the key used when the SFT checkpoint was trained, otherwise actions are denormalized incorrectly.

   - **Seed files:** The provided configs use RLinf's bundled RoboTwin train/eval seed files.

   - **Standalone evaluation:** Run standalone evaluation through the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>`, which covers ``ROBOTWIN_PATH`` / ``assets_path`` setup, eval configs such as ``robotwin_place_empty_cup_openvlaoft_eval`` and ``robotwin_adjust_bottle_openpi_pi05_eval``, and result interpretation.

Visualization and Results
-------------------------

Watch ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); set ``env.eval.video_cfg.save_video=True`` to record videos, or see :doc:`Training metrics <../../reference/metrics>` for W&B/SwanLab and all metrics.

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
