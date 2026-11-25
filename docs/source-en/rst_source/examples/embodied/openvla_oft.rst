OpenVLA-OFT RL
==============

.. figure:: https://openvla-oft.github.io/static/images/libero_task_performance_results.png
   :align: center
   :width: 90%

   LIBERO results from the original OFT fine-tuning study (image: `OpenVLA-OFT project <https://openvla-oft.github.io/>`__).

Fine-tune OpenVLA-OFT with reinforcement learning in RLinf. This recipe trains
OpenVLA-OFT on LIBERO with GRPO on NVIDIA CUDA, and covers LIBERO and ManiSkill
on AMD ROCm, Huawei Ascend CANN, and Moore Threads MUSA. For the original
OpenVLA model, see :doc:`maniskill`.

Overview
--------

Use a task-specific checkpoint and config to train OpenVLA-OFT on LIBERO.
The linked simulator pages cover its other environments.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO · ManiSkill · RoboTwin · MetaWorld · BEHAVIOR · OpenSora · Wan

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Language-conditioned manipulation

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA · :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <openvla-oft-hardware>` (LIBERO · ManiSkill)

| **You'll do:** install → download a LIBERO-Goal checkpoint → set model paths → launch GRPO → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · hardware and drivers for your selected backend.

Tasks
~~~~~

Start with LIBERO-Goal. Other simulator workflows remain on their own pages.

.. list-table::
   :header-rows: 1
   :widths: 18 22 36 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - :doc:`LIBERO <libero>`
     - Goal
     - ``libero_goal_grpo_openvlaoft``
     - Goal-conditioned manipulation.
   * - :doc:`ManiSkill <maniskill>`
     - Plate-25
     - ``maniskill_ppo_openvlaoft``
     - Tabletop manipulation.
   * - :doc:`RoboTwin <robotwin>`
     - Place empty cup
     - ``robotwin_place_empty_cup_grpo_openvlaoft``
     - Dual-arm manipulation.
   * - :doc:`MetaWorld <metaworld>`
     - MT50
     - ``metaworld_50_grpo_openvlaoft``
     - Multiple manipulation tasks.
   * - :doc:`BEHAVIOR <behavior>`
     - Household tasks
     - ``behavior_ppo_openvlaoft``
     - Long-horizon activities.
   * - :doc:`OpenSora <opensora>` / :doc:`Wan <wan>`
     - LIBERO Spatial
     - ``opensora_libero_spatial_grpo_openvlaoft`` / ``wan_libero_spatial_grpo_openvlaoft``
     - Training with a world model.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The LIBERO recipe uses images and a task prompt to produce action chunks.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - RGB images selected by the model config.
   * - Action
     - Chunks of 7-D end-effector and gripper actions.
   * - Reward
     - LIBERO task success, scaled by the training config.
   * - Prompt
     - Natural-language instruction for the current task.

.. _openvla-oft-hardware:

Setup and Run
-------------

.. include:: _setup_common.rst

NVIDIA CUDA supports all environments listed in the Overview. AMD ROCm, Huawei Ascend CANN, and Moore Threads MUSA support OpenVLA-OFT on LIBERO and ManiSkill through the shared platform installer and scheduler device API; other environments have separate hardware requirements.

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env libero && source .venv/bin/activate

      **2. Download**

      Download the LIBERO-Goal SFT checkpoint:

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 --local-dir ./Openvla-oft-SFT-libero-goal-traj1

      .. include:: _model_path.rst

      Set the path in ``examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml``. Keep ``actor.model.unnorm_key: libero_goal_no_noops`` to match the checkpoint's action statistics. Other suites require their matching checkpoints and keys; see :doc:`libero`.

      **3. Launch**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

      The script loads the named YAML, creates actor, rollout, and environment workers according to ``cluster.component_placement``, and starts GRPO. Adjust placement and batch sizes for your devices; see :doc:`../../concepts/placement` and :doc:`../../resources/faq`.

   .. tab-item:: AMD ROCm
      :sync: rocm

      **1. Install (LIBERO)**

      Start a ROCm container or install natively on a host with ROCm available.

      .. include:: _amd_libero.rst

      The published image already contains this model environment:

      .. code-block:: bash

         source switch_env openvla-oft

      For a native installation on a host with ROCm installed (omit ``--rocm`` to detect the installed version):

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 --local-dir ./Openvla-oft-SFT-libero-goal-traj1

      .. include:: _model_path.rst

      Set the path in ``examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml`` and keep ``actor.model.unnorm_key: libero_goal_no_noops``.

      **3. Launch LIBERO**

      Enable software rendering (or ``egl`` on supported Radeon GPUs) in the active shell:

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

      **ManiSkill on AMD ROCm**

      Install the combined ManiSkill and LIBERO environment:

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      Configure CPU simulation:

      .. include:: _maniskill_non_cuda.rst

      Download ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` and its LoRA adapter, then set ``model_path`` and ``lora_path`` in ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``. Adjust placement and batch sizes for your devices, then launch PPO:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft

   .. tab-item:: Huawei Ascend CANN
      :sync: ascend

      **1. Install (LIBERO)**

      Start the Ascend container or install natively on a host with CANN and its NPU driver available.

      .. include:: _ascend_libero.rst

      Inside the published RLinf container, activate the model environment:

      .. code-block:: bash

         source switch_env openvla-oft

      For a native installation, install CPU PyTorch and its matching ``torch-npu`` (skips the CUDA flash-attention build):

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model openvla-oft --env libero && source .venv/bin/activate

      .. include:: _ascend_torch.rst

      **2. Download**

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 --local-dir ./Openvla-oft-SFT-libero-goal-traj1

      .. include:: _model_path.rst

      Set the path in ``examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml`` and keep ``actor.model.unnorm_key: libero_goal_no_noops``.

      **3. Launch LIBERO**

      Enable software rendering in the active shell:

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

      **ManiSkill on Huawei Ascend CANN**

      Install the combined ManiSkill and LIBERO environment:

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      Configure CPU simulation:

      .. include:: _maniskill_non_cuda.rst

      Download ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` and its LoRA adapter, then set ``model_path`` and ``lora_path`` in ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``. Adjust placement and batch sizes for your devices, then launch PPO:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft

   .. tab-item:: Moore Threads MUSA
      :sync: musa

      **1. Install (LIBERO)**

      Run inside a Moore Threads container. The installer reuses the image's MUSA builds of PyTorch and ``torch_musa`` instead of replacing them.

      .. include:: _musa_libero.rst

      Inside the container, install OpenVLA-OFT with the MUSA platform selected:

      .. code-block:: bash

         bash requirements/install.sh --platform musa embodied --model openvla-oft --env libero && source .venv/bin/activate

      RLinf detects MUSA devices and assigns them through the same placement configuration used on other GPU backends.

      **2. Download**

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 --local-dir ./Openvla-oft-SFT-libero-goal-traj1

      .. include:: _model_path.rst

      Set the path in ``examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml`` and keep ``actor.model.unnorm_key: libero_goal_no_noops``.

      **3. Launch LIBERO**

      Enable software rendering in the active shell:

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

      **ManiSkill on Moore Threads MUSA**

      Keep the vendor simulator packages and add the OpenVLA-OFT model environment inside the vendor image:

      .. include:: _musa_maniskill.rst

      .. code-block:: bash

         bash requirements/install.sh --platform musa embodied --venv openvla-oft --model openvla-oft --env libero && source openvla-oft/bin/activate

      Configure CPU simulation:

      .. include:: _maniskill_non_cuda.rst

      Download ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` and its LoRA adapter, then set ``model_path`` and ``lora_path`` in ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``. Adjust placement and batch sizes for your devices, then launch PPO:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft

Visualization and Results
-------------------------

Watch ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); set ``env.eval.video_cfg.save_video=True`` to record videos. See :doc:`Training metrics <../../reference/metrics>` for metric definitions, :doc:`LIBERO results <libero>` for published training results, and :doc:`the LIBERO evaluation guide <../../evaluations/guides/libero>` for standalone evaluation.
