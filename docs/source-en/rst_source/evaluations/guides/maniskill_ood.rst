ManiSkill OOD Evaluation
========================

Overview
--------

ManiSkill OOD evaluation measures VLA policy generalization on out-of-distribution ManiSkill scenes built on the **Put-on-Plate** task family. Following the `rl4vla <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c0fabe372177d2aded596be2d3b4544-Abstract-Conference.html>`_ protocol, it evaluates in-distribution performance plus 13 OOD variants grouped into **Vision**, **Semantic**, and **Execution** categories.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      ManiSkill OOD (13 OOD + in-distribution scenes)

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT (example) · OpenVLA / OpenPI (derived)

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/maniskill/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`ManiSkill Training <../../examples/embodied/maniskill>`

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: Single-Scene Eval

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download RLinf/Openvla-oft-SFT-libero10-trajall --local-dir ./Openvla-oft-SFT-libero10-trajall
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      **3. Evaluate (in-distribution default)**

      .. code-block:: bash

         bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
           rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall \
           rollout.model.lora_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      Override ``env.eval.init_params.id`` and ``env.eval.init_params.obj_set`` to test any OOD scene, or pass ``runner.ckpt_path`` to evaluate an RL-trained ``.pt`` checkpoint.

   .. tab-item:: Batch 16-Scene OOD (mani-ood)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download RLinf/Openvla-oft-SFT-libero10-trajall --local-dir ./Openvla-oft-SFT-libero10-trajall
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      Set ``rollout.model.model_path: ./Openvla-oft-SFT-libero10-trajall`` and ``rollout.model.lora_path: ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora`` in ``evaluations/maniskill/maniskill_ood_openvlaoft_eval.yaml``.

      **3. Evaluate (all 16 OOD runs)**

      .. code-block:: bash

         export EVAL_NAME=my_ood_eval
         export CKPT_PATH=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt
         export TOTAL_NUM_ENVS=16
         export EVAL_ROLLOUT_EPOCH=1
         bash evaluations/run_eval.sh mani-ood maniskill_ood_openvlaoft_eval

      Batch logs go to ``logs/eval/<EVAL_NAME>/<timestamp>-<env_id>-<obj_set>/run_ppo.log``; ``mani-ood`` defaults ``HF_ENDPOINT`` to ``https://hf-mirror.com`` unless already exported.

``evaluations/maniskill/`` ships an OpenVLA-OFT example config (`RLinf/Openvla-oft-SFT-libero10-trajall <https://huggingface.co/RLinf/Openvla-oft-SFT-libero10-trajall>`_ base + `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_ LoRA). For OpenVLA, OpenPI, or other ManiSkill tasks, see :ref:`Derive Eval Configs from Training <maniskill-derive-from-train>`.

.. _maniskill-eval-config:

.. dropdown:: Example Configs and Evaluation Configuration

   Available under ``evaluations/maniskill/``:

   .. list-table::
      :header-rows: 1
      :widths: 40 35 25

      * - Config file
        - Description
        - Model
      * - ``maniskill_ood_openvlaoft_eval.yaml``
        - OOD generalization template (default: training scene)
        - OpenVLA-OFT

   **Protocol overview**

   ManiSkill OOD selects scenes via ``id`` and ``obj_set`` under ``env.eval.init_params`` and reports ``eval/success_once``:

   - **In-distribution:** ``PutOnPlateInScene25Main-v3`` + ``obj_set=train``.
   - **Out-of-distribution:** 13 variant environments + ``obj_set=test`` across Vision, Semantic, and Execution.
   - **Supplementary runs:** ``mani-ood`` mode also evaluates 3 Semantic tasks with ``obj_set=train`` (16 runs total).

   Each trajectory is fixed by ``episode_id`` (``reset_state_id``). With ``use_fixed_reset_state_ids=True``, initial conditions are deterministic; with ``auto_reset=True``, completed envs advance sequentially.

   **OOD scene list**

   .. list-table::
      :header-rows: 1
      :widths: 18 44 38

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

   **Key environment parameters (``env.eval.init_params``)**

   .. list-table::
      :header-rows: 1
      :widths: 26 74

      * - Field
        - Purpose
      * - ``id``
        - ManiSkill registered env ID; default is ``PutOnPlateInScene25Main-v3``.
      * - ``obj_set``
        - Object split: ``train``, ``test``, or ``all``. OOD scenes use ``test``; in-distribution uses ``train``.
      * - ``obs_mode``
        - Observation mode; VLA eval uses ``rgb+segmentation``.
      * - ``sim_backend``
        - Simulation backend; default ``gpu`` requires an NVIDIA GPU.
      * - ``policy_setup``
        - Action-space setup; OpenVLA-OFT uses ``widowx_bridge``.

   **Batch ``mani-ood`` environment variables**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Variable
        - Description
      * - ``EVAL_NAME``
        - Batch run name; logs go to ``logs/eval/<EVAL_NAME>/``.
      * - ``CKPT_PATH``
        - RL-trained ``.pt`` checkpoint path; overrides ``runner.ckpt_path``.
      * - ``TOTAL_NUM_ENVS``
        - Parallel env count (``env.eval.total_num_envs``).
      * - ``EVAL_ROLLOUT_EPOCH``
        - Eval epochs (``env.eval.rollout_epoch``).

   See :ref:`env-eval-fields` in :doc:`../reference/configuration` for generic ``env.eval`` fields (examples typically set ``max_episode_steps=80``, ``max_steps_per_rollout_epoch=80``, and ``ignore_terminations=True``). OpenVLA-OFT also sets ``unnorm_key: bridge_orig``, ``is_lora: True``, ``lora_path``, ``add_value_head: True``, and ``max_prompt_length: 30`` under ``rollout.model``.

.. _maniskill-derive-from-train:

.. dropdown:: Advanced Usage (Single OOD Scene, Parallelism, Derive from Training)

   **Single OOD scene and RL checkpoint**

   .. code-block:: bash

      bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
        env.eval.init_params.id=PutOnPlateInScene25VisionImage-v1 \
        env.eval.init_params.obj_set=test \
        env.eval.total_num_envs=32 \
        rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall \
        rollout.model.lora_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Lora \
        runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt

   To cover more ``episode_id`` values with fewer parallel envs, set ``max_steps_per_rollout_epoch`` to a multiple of ``max_episode_steps`` under ``auto_reset: True`` (for example, ``total_num_envs: 16``, ``max_episode_steps: 80``, ``max_steps_per_rollout_epoch: 320``).

   **Derive eval configs from training**

   ManiSkill also supports training tasks such as ``PickCube-v1`` and ``PutCarrotOnPlateInScene-v2`` under ``examples/embodiment/config/env/``. Copy the matching training config, set ``runner.task_type: embodied_eval`` and ``runner.only_eval: True``, remove training sections (``algorithm``, ``actor``), and keep ``env.eval`` and ``rollout``.

Results and Troubleshooting
---------------------------

The terminal prints ``eval/success_once``; single-run logs and videos go to ``logs/<timestamp>-<config>/`` and batch logs to ``logs/eval/<EVAL_NAME>/`` (see :doc:`../reference/results`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Missing ManiSkill assets
     - Download ``RLinf/maniskill_assets`` using the asset download command in Setup and Run.
   * - GPU simulation errors
     - ``sim_backend: gpu`` requires an NVIDIA GPU.
   * - OpenVLA-OFT policy fails to load
     - Provide ``rollout.model.lora_path`` pointing to ``./RLinf-OpenVLAOFT-ManiSkill-Base-Lora``.
   * - RL checkpoint ignored
     - Pass ``.pt`` weights via ``CKPT_PATH`` in batch ``mani-ood`` mode, or ``runner.ckpt_path`` in single-scene runs.
   * - Still evaluating training scene instead of OOD
     - The default YAML targets ``PutOnPlateInScene25Main-v3``; override ``env.eval.init_params.id`` and ``obj_set``, or launch ``mani-ood`` mode.
