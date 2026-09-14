RL on OpenVLA-OFT
=================

.. figure:: https://openvla-oft.github.io/static/images/libero_task_performance_results.png
   :align: center
   :width: 90%

   LIBERO results from the original OFT fine-tuning study (image: `OpenVLA-OFT project <https://openvla-oft.github.io/>`__).

Fine-tune OpenVLA-OFT with reinforcement learning in RLinf. This recipe starts
with LIBERO and GRPO on NVIDIA, then shows how to install and launch the same
model on LIBERO or ManiSkill with AMD ROCm, Huawei Ascend CANN, and Moore
Threads MUSA. For the original OpenVLA model, see :doc:`maniskill`.

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

Installation
------------

Use the NVIDIA setup below for the default recipe. For AMD, Ascend, or MUSA, use
:ref:`the backend setup <openvla-oft-hardware>` before downloading the model.

.. include:: _setup_common.rst

Start the NVIDIA container from the repository root:

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g --network host \
      -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero bash
   source switch_env openvla-oft

Alternatively, install only the OpenVLA-OFT and LIBERO dependencies locally:

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env libero
   source .venv/bin/activate

Download the Model
------------------

Download the LIBERO-Goal SFT checkpoint used by this config:

.. code-block:: bash

   hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 \
      --local-dir checkpoints/Openvla-oft-SFT-libero-goal-traj1

.. include:: _model_path.rst

Set these paths in ``examples/embodiment/config/libero_goal_grpo_openvlaoft.yaml``.
Keep ``actor.model.unnorm_key: libero_goal_no_noops`` to match the checkpoint's
action statistics. Other suites require their matching checkpoints and keys;
see :doc:`libero`.

Run It
------

With the model environment active and paths configured, launch LIBERO-Goal:

.. code-block:: bash

   ROBOT_PLATFORM=LIBERO bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

The script loads the named YAML, creates actor, rollout, and environment workers
according to ``cluster.component_placement``, and starts GRPO. Adjust placement
and batch sizes for your devices; see :doc:`../../concepts/placement` and
:doc:`../../resources/faq`.

.. _openvla-oft-hardware:

Run on Different Hardware Backends
----------------------------------

NVIDIA uses the installation and launch above. AMD ROCm, Huawei Ascend CANN,
and Moore Threads MUSA support OpenVLA-OFT on LIBERO and ManiSkill through the
shared platform installer, scheduler device API, and model path. The other
environments in the Overview have separate hardware requirements.

.. _openvla-oft-amd:

AMD ROCm
~~~~~~~~

Start a ROCm container or install on a host with ROCm available.

.. include:: _amd_libero.rst

The published image already contains this model environment:

.. code-block:: bash

   source switch_env openvla-oft

For a native installation on a host with ROCm already installed:

.. code-block:: bash

   bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env libero
   source .venv/bin/activate

Omit ``--rocm`` to detect the installed version, or add ``--use-mirror`` for
downloads from mainland China.

.. _openvla-oft-ascend:

Huawei Ascend CANN
~~~~~~~~~~~~~~~~~~

Choose a container or install on a host with CANN and its NPU driver available.

.. include:: _ascend_libero.rst

Inside the published RLinf container, activate the model environment:

.. code-block:: bash

   source switch_env openvla-oft

For a native installation, install CPU PyTorch and its matching ``torch-npu``
through the Ascend option:

.. code-block:: bash

   bash requirements/install.sh --platform ascend embodied --model openvla-oft --env libero
   source .venv/bin/activate

Add ``--use-mirror`` for downloads from mainland China. The installer skips the
CUDA flash-attention build on Ascend.

.. _openvla-oft-musa:

Moore Threads MUSA
~~~~~~~~~~~~~~~~~~

Run inside a Moore Threads container. The installer reuses the image's MUSA
builds of PyTorch and ``torch_musa`` instead of replacing them.

.. include:: _musa_libero.rst

Inside the container, install OpenVLA-OFT with the MUSA platform selected:

.. code-block:: bash

   bash requirements/install.sh --platform musa embodied --model openvla-oft --env libero
   source .venv/bin/activate

Add ``--use-mirror`` for downloads from mainland China. RLinf detects the MUSA
devices and assigns them through the same placement configuration used on the
other GPU backends.

.. _openvla-oft-backend-launch:

Launch LIBERO on AMD, Ascend, or MUSA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After completing one of the backend setups, download the LIBERO-Goal checkpoint
and set the model paths as described above. Enable software rendering in that
same shell.

.. include:: _libero_osmesa.rst

Start the configured GRPO run:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openvlaoft

ManiSkill on AMD, Ascend, or MUSA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On AMD or Ascend, install the combined ManiSkill and LIBERO environment for
OpenVLA-OFT. Use the command for the selected model accelerator:

.. code-block:: bash

   # AMD ROCm
   bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openvla-oft --env maniskill_libero

   # Huawei Ascend CANN
   bash requirements/install.sh --platform ascend embodied --model openvla-oft --env maniskill_libero

   source .venv/bin/activate

For MUSA, keep the vendor simulator packages and add the OpenVLA-OFT model
environment inside the vendor image:

.. include:: _musa_maniskill.rst

.. code-block:: bash

   bash requirements/install.sh --platform musa embodied --venv openvla-oft --model openvla-oft --env libero
   source openvla-oft/bin/activate

Configure CPU simulation for all three non-CUDA backends:

.. include:: _maniskill_non_cuda.rst

Download ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` and its LoRA adapter,
then set ``model_path`` and ``lora_path`` in
``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``. Launch the PPO
recipe after adjusting placement and batch sizes for the available devices:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft

Visualization and Results
-------------------------

Watch ``env/success_once`` in the training logs. Use
:doc:`Training metrics <../../reference/metrics>` for metric definitions and
:doc:`LIBERO results <libero>` for the published training results. Standalone
evaluation follows :doc:`the LIBERO evaluation guide <../../evaluations/guides/libero>`.
