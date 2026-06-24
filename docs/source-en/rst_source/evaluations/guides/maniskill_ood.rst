ManiSkill OOD Evaluation
========================

ManiSkill OOD evaluation measures how well VLA policies generalize to out-of-distribution ManiSkill scenes. It is built on the **Put-on-Plate** task family (placing a carrot on a plate) and follows the OOD test protocol from `rl4vla <https://arxiv.org/abs/2505.19789>`_, with scenes grouped into **Vision**, **Semantic**, and **Execution** categories.

Related training doc: :doc:`../../examples/embodied/maniskill`

Environment Setup
-----------------

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

``evaluations/maniskill/`` currently ships only an OpenVLA-OFT example config. OpenVLA, OpenPI, and other models have ManiSkill training configs but no dedicated eval YAML yet — see :ref:`maniskill-derive-from-train` below.

Download ManiSkill assets if not already present:

.. code-block:: bash

   cd rlinf/envs/maniskill
   # For faster downloads in China, you can set:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

Models and Checkpoints
----------------------

OpenVLA-OFT evaluation typically requires two weight sources:

1. **Base model** ``rollout.model.model_path``: e.g. `RLinf/Openvla-oft-SFT-libero10-trajall <https://huggingface.co/RLinf/Openvla-oft-SFT-libero10-trajall>`_
2. **ManiSkill LoRA** ``rollout.model.lora_path``: e.g. `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_

To evaluate an RL-trained policy, pass the ``.pt`` checkpoint via ``runner.ckpt_path`` or ``CKPT_PATH``; it overrides model initialization.

Example Configs
---------------

Available under ``evaluations/maniskill/``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config file
     - Description
     - Model
   * - ``maniskill_ood_openvlaoft_eval.yaml``
     - OOD generalization template (default: training scene)
     - OpenVLA-OFT

End-to-End Workflow
-------------------

**Step 1: Activate the environment**

.. code-block:: bash

   source .venv/bin/activate

**Step 2: Edit the config**

Copy or edit the target YAML and set at least ``rollout.model.model_path`` and ``rollout.model.lora_path``. See :doc:`../reference/configuration` (:ref:`env-eval-fields`) for general ``env.eval`` fields; see :ref:`maniskill-eval-config` below for ManiSkill scene selection and protocol.

**Step 3: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     rollout.model.model_path=/path/to/model \
     rollout.model.lora_path=/path/to/lora

**Step 4: Check results**

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs and videos.

.. _maniskill-eval-config:

Evaluation Configuration
------------------------

ManiSkill evaluation selects scenes via ``id`` (environment ID) and ``obj_set`` (object split) under ``env.eval.init_params``, and reports ``eval/success_once`` (fraction of trajectories with at least one success).

Protocol Overview
~~~~~~~~~~~~~~~~~

RLinf's ManiSkill OOD protocol matches `rl4vla <https://arxiv.org/abs/2505.19789>`_ for fair comparison with published results.

- **In-distribution**: ``PutOnPlateInScene25Main-v3`` + ``obj_set=train`` (the plate-25-main training task);
- **Out-of-distribution**: 13 variant environments + ``obj_set=test``, split into Vision / Semantic / Execution;
- **Supplementary runs**: the ``mani-ood`` mode also runs 3 Semantic tasks with ``obj_set=train``.

Each trajectory is identified by ``episode_id`` (i.e. ``reset_state_id``), which fixes the object, plate, pose, and (for some scenes) visual perturbation. With ``use_fixed_reset_state_ids=True``, the env loads deterministic initial conditions from ``episode_id``; with ``auto_reset=True``, the next ``episode_id`` is assigned sequentially after each episode.

OOD Scene List
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Category
     - Environment ID (``env.eval.init_params.id``)
     - Description
   * - Vision
     - ``PutOnPlateInScene25VisionImage-v1``
     - Background image perturbation
   * -
     - ``PutOnPlateInScene25VisionTexture03-v1`` / ``PutOnPlateInScene25VisionTexture05-v1``
     - Texture perturbation (strength 0.3 / 0.5)
   * -
     - ``PutOnPlateInScene25VisionWhole03-v1`` / ``PutOnPlateInScene25VisionWhole05-v1``
     - Whole-scene visual perturbation (strength 0.3 / 0.5)
   * - Semantic
     - ``PutOnPlateInScene25Carrot-v1``
     - Unseen carrot objects
   * -
     - ``PutOnPlateInScene25Plate-v1``
     - Unseen plates
   * -
     - ``PutOnPlateInScene25Instruct-v1``
     - Changed language instructions
   * -
     - ``PutOnPlateInScene25MultiCarrot-v1`` / ``PutOnPlateInScene25MultiPlate-v1``
     - Multiple carrots / multiple plates
   * - Execution
     - ``PutOnPlateInScene25Position-v1``
     - Changed object initial positions
   * -
     - ``PutOnPlateInScene25EEPose-v1``
     - Changed robot initial pose
   * -
     - ``PutOnPlateInScene25PositionChangeTo-v1``
     - Dynamic target position changes

Key Environment Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

These fields live under ``env.eval.init_params``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Purpose
   * - ``id``
     - ManiSkill registered env name; selects the OOD variant (see table above). The default template uses ``PutOnPlateInScene25Main-v3`` (training scene).
   * - ``obj_set``
     - Object split: ``train``, ``test``, or ``all``. OOD scenes typically use ``test``; in-distribution eval uses ``train``.
   * - ``obs_mode``
     - Observation mode; VLA eval uses ``rgb+segmentation``.
   * - ``sim_backend``
     - Simulation backend; default ``gpu`` requires an NVIDIA GPU.
   * - ``policy_setup``
     - Action-space setup; OpenVLA-OFT uses ``widowx_bridge`` (set in ``maniskill_ood_template``).

See :ref:`env-eval-fields` in :doc:`../reference/configuration` for general ``env.eval`` fields (``total_num_envs``, ``max_episode_steps``, ``auto_reset``, etc.). ManiSkill eval examples typically use ``max_episode_steps=80``, ``max_steps_per_rollout_epoch=80``, and ``ignore_terminations=True``.

OpenVLA-OFT Model Fields
~~~~~~~~~~~~~~~~~~~~~~~~

In addition to ``model_path``, ``maniskill_ood_openvlaoft_eval.yaml`` requires:

.. code-block:: yaml

   rollout:
     model:
       model_type: openvla_oft
       unnorm_key: bridge_orig
       is_lora: True
       lora_path: /path/to/RLinf-OpenVLAOFT-ManiSkill-Base-Lora
       add_value_head: True
       max_prompt_length: 30

Single-Scene Evaluation
-----------------------

Override ``init_params`` via Hydra to evaluate the default training scene or any OOD scene.

**In-distribution (plate-25-main)**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     env.eval.init_params.id=PutOnPlateInScene25Main-v3 \
     env.eval.init_params.obj_set=train \
     rollout.model.model_path=/path/to/model \
     rollout.model.lora_path=/path/to/lora

**Single OOD scene (Vision example)**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     env.eval.init_params.id=PutOnPlateInScene25VisionImage-v1 \
     env.eval.init_params.obj_set=test \
     rollout.model.model_path=/path/to/model \
     runner.ckpt_path=/path/to/checkpoint.pt

Covering the Full Test Set
~~~~~~~~~~~~~~~~~~~~~~~~~~

``total_num_trials`` per scene depends on object count, plate count, and pose combinations. With sufficient resources, increase ``total_num_envs``; when resources are limited, set ``max_steps_per_rollout_epoch`` to a multiple of ``max_episode_steps`` under ``auto_reset=True`` so each ``rollout_epoch`` serially covers more ``episode_id`` values:

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 16
       max_episode_steps: 80
       max_steps_per_rollout_epoch: 320   # 4 * 80; ~4 * total_num_envs trajectories per epoch
       auto_reset: True
       ignore_terminations: True
       use_fixed_reset_state_ids: True
       rollout_epoch: 1

Batch OOD Evaluation (``mani-ood`` mode)
----------------------------------------

The ``mani-ood`` mode runs evaluation on all 13 OOD scenes (``obj_set=test``) plus 3 Semantic scenes (``obj_set=train``), for **16** runs total — matching the full OOD protocol in the training docs.

**Required environment variables**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Variable
     - Description
   * - ``EVAL_NAME``
     - Batch eval name; logs go to ``logs/eval/<EVAL_NAME>/``
   * - ``CKPT_PATH``
     - Path to RL-trained ``.pt`` checkpoint; overrides ``runner.ckpt_path``
   * - ``TOTAL_NUM_ENVS``
     - Parallel env count; maps to ``env.eval.total_num_envs``
   * - ``EVAL_ROLLOUT_EPOCH``
     - Eval epochs; maps to ``env.eval.rollout_epoch``

.. code-block:: bash

   export EVAL_NAME=my_ood_eval
   export CKPT_PATH=/path/to/checkpoint.pt
   export TOTAL_NUM_ENVS=16
   export EVAL_ROLLOUT_EPOCH=1
   bash evaluations/run_eval.sh mani-ood maniskill_ood_openvlaoft_eval

Batch logs: ``logs/eval/<EVAL_NAME>/<timestamp>-<env_id>-<obj_set>/run_ppo.log``

The ``mani-ood`` mode sets ``HF_ENDPOINT`` automatically (default ``https://hf-mirror.com``); override it before running if needed.

.. _maniskill-derive-from-train:

Advanced Usage
--------------

**Deriving eval configs from training**

ManiSkill also supports training tasks such as ``PickCube-v1`` and ``PutCarrotOnPlateInScene-v2`` (see ``examples/embodiment/config/env/``), but there are no dedicated eval YAMLs yet. Copy the training config and set:

- ``runner.task_type: embodied_eval``
- ``runner.only_eval: True``
- Remove training sections (``algorithm``, ``actor``, etc.) and keep ``env.eval`` and ``rollout``

**Adjust parallelism**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     env.eval.total_num_envs=32 \
     rollout.model.model_path=/path/to/model

**Load RL checkpoint**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     runner.ckpt_path=/path/to/checkpoint.pt \
     rollout.model.model_path=/path/to/model

FAQ
---

- **Asset path:** Ensure ManiSkill assets are downloaded to ``rlinf/envs/maniskill/assets``.
- **GPU simulation:** ``sim_backend: gpu`` requires an NVIDIA GPU; ``run_eval.sh`` sets ``MUJOCO_GL=osmesa`` etc. for headless environments.
- **LoRA path:** OpenVLA-OFT eval requires ``lora_path``; without it the ManiSkill policy cannot load correctly.
- **Checkpoint:** Batch mode passes ``.pt`` weights via ``CKPT_PATH``; single runs use ``runner.ckpt_path``.
- **Scene selection:** The default YAML points to the training scene ``PutOnPlateInScene25Main-v3``; for OOD scenes, explicitly override ``env.eval.init_params.id`` and ``obj_set``, or use ``mani-ood`` mode.
