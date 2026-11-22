RoboDojo Evaluation
===================

`RoboDojo <https://robodojo-benchmark.github.io/>`__ is an Isaac Sim-based
benchmark for dual-arm manipulation. It includes tabletop rearrangement,
pick-and-place, and tool-use tasks for robots such as ARX-X5. RLinf exposes
RoboDojo as a Gym environment for evaluating VLA policies; RL training is not
supported yet.

Overview
--------

Run RoboDojo on Isaac Sim and drive a dual-arm robot through RLinf's unified
environment interface.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ (Pi_05)

   .. grid-item-card:: Algorithms
      :text-align: center

      Evaluation (π₀.₅)

   .. grid-item-card:: Tasks
      :text-align: center

      Put Bottles Into Dustbin

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1+ GPU

| **You'll do:** install → download scene assets and the Pi_05 checkpoint →
  convert the checkpoint → run the evaluation.
| **Prerequisites:** :doc:`Base installation </rst_source/start/installation>` ·
  an NVIDIA CUDA GPU · at least 120 GB of free disk space.

Tasks
~~~~~

RLinf provides a ``put_bottles_into_dustbin`` environment configuration as the integration and verification example.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``put_bottles_into_dustbin``
     - Grab bottles from the table with both arms and place them in a dustbin.

The configuration is located at ``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml``.
Other RoboDojo tasks use the same environment interface; change ``task_config.task_name`` and the corresponding task parameters.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Head-camera RGB in ``main_images``, optional left/right wrist-camera RGB
       in ``wrist_images``, and a ``[B, 14]`` float32 proprioceptive state in
       ``states`` that holds the left/right arm joints and gripper states.
   * - Action
     - 14 continuous values in ``actions`` by default: left arm 6, right arm 6,
       left gripper 1, right gripper 1.
   * - Reward
     - Sparse task-completion reward by default.
   * - Prompt
     - Natural-language instruction generated from the current scene and exposed
       as ``task_descriptions``.

Setup and Run
-------------

.. include:: _setup_common.rst

RoboDojo supports Linux x86_64 with an NVIDIA CUDA GPU (CPU-only, macOS, and AMD ROCm are unsupported); check ``nvidia-smi``, ``uname -m``, and ``df -h .`` before installing. Export ``ROBODOJO_ASSETS_ROOT`` before installing to record it in ``.venv/bin/activate`` (defaults to ``data/robodojo`` under the repo root). The installer sets ``OMNI_KIT_ACCEPT_EULA=YES``, which accepts the applicable NVIDIA Omniverse license terms; do not install ``rlinf-robodojo-runtime`` separately.

.. tab-set::

   .. tab-item:: OpenPI Evaluation
      :sync: openpi

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         bash requirements/install.sh embodied --model openpi --env robodojo && source .venv/bin/activate

      **2. Download**

      Download the scene assets (~39 GB under ``Assets/``) and official Pi_05 checkpoint from ``RoboDojo-Benchmark/RoboDojo`` into ``$ROBODOJO_ASSETS_ROOT``, then convert the raw JAX/orbax snapshot under ``59999/`` to OpenPI format (``model.safetensors`` and ``assets/norm_stats.json``):

      .. code-block:: bash

         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         hf download RoboDojo-Benchmark/RoboDojo \
           --repo-type dataset \
           --local-dir "$ROBODOJO_ASSETS_ROOT" \
           --include "Assets/**" \
           --include "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/**"

         CKPT=$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999
         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model "$CKPT" \
           --input-norm-stats "$CKPT/assets/arx_x5_sim/norm_stats.json" \
           --output-model "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5" \
           --output-norm-stats "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5/assets/norm_stats.json" \
           --action-dim 32 \
           --action-horizon 50

      ``ROBODOJO_ASSETS_ROOT`` must point to the parent of ``Assets/``, not ``Assets/`` itself. Keep the ``--include`` filters and download only one run: the full repository contains multiple terabytes of data, and the three ``Pi_05`` runs (``joint-0``/``1``/``2``) total ~70 GB. ``ckpt/RoboDojo/Pi_0/`` holds the corresponding π₀ checkpoints.

      **3. Launch**

      ``evaluations/robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml`` extends ``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml`` (14-dim actions ``[6, 6, 1, 1]``, ``state_action_order: aloha``) and derives ``norm_stats_path`` from ``rollout.model.model_path`` automatically. Launch evaluation:

      .. code-block:: bash

         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         bash evaluations/run_eval.sh robodojo \
           robodojo_put_bottles_into_dustbin_pi05_eval \
           rollout.model.model_path="$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5"

      Logs go to ``logs/<timestamp>-<config_name>/`` and results to ``runner.logger.log_path``.

   .. tab-item:: Environment Only
      :sync: env-only

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         bash requirements/install.sh embodied --env robodojo && source .venv/bin/activate

      **2. Download**

      Download the scene assets (~39 GB under ``Assets/``) into ``$ROBODOJO_ASSETS_ROOT`` (keep ``--include`` so the multi-terabyte repository is not pulled in full):

      .. code-block:: bash

         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         hf download RoboDojo-Benchmark/RoboDojo \
           --repo-type dataset \
           --local-dir "$ROBODOJO_ASSETS_ROOT" \
           --include "Assets/**"

      **3. Launch**

      Run the following Python snippet to start one environment and verify Isaac Sim, cameras, and the downloaded assets:

      .. code-block:: python

         import numpy as np
         from robodojo_runtime.bridge import VectorEnv

         env = VectorEnv(
             {
                 "task_name": "put_bottles_into_dustbin",
                 "env_cfg_type": "arx_x5",
                 "headless": True,
                 "cuda_device": 0,
                 "max_episode_steps": 700,
             },
             n_envs=1,
             env_seeds=[0],
         )
         try:
             env.reset()
             obs = env.get_obs()[0]
             print(obs["full_image"].shape, obs["state"].shape)
             env.step(np.zeros((1, 1, 14), dtype=np.float32))
         finally:
             env.close()

.. dropdown:: Advanced configuration (optional)

   - **Pinned runtime and compatibility overrides**: ``requirements/install.sh`` and ``requirements/embodied/envs/robodojo-override.txt`` pin Python 3.11, RoboDojo runtime 0.3.0 (``6d76a0c``), PyTorch 2.7.0 (cu126, or cu128 on CUDA 12.8+), TorchVision 0.22.0 / TorchAudio 2.7.0, Isaac Sim 5.1.0, and IsaacLab 0.54.3 (``afca7b09``). Do not upgrade overlapping pins manually.

   - **Scale up parallel environments**: the default runs one environment; once verified, scale up with Hydra overrides such as ``env.eval.total_num_envs=4``, ``cluster.component_placement.env=0-3``, and ``cluster.component_placement.rollout=0-3``.

   - **ModelScope mirror download**: install ``modelscope>=1.40.0`` with ``uv pip install "modelscope>=1.40.0"`` and run ``modelscope download`` with the same ``--repo-type dataset``, ``--local-dir``, and ``--include`` flags.

   - **Verify package and CUDA imports** (without starting Isaac Sim):

     .. code-block:: bash

        python - <<'PY'
        import torch
        from robodojo_runtime import assets_root, source_root
        from robodojo_runtime.bridge import VectorEnv

        print("torch:", torch.__version__)
        print("cuda:", torch.cuda.is_available())
        print("source:", source_root())
        print("assets:", assets_root())
        PY

   - **Single-environment check**: verify Isaac Sim, cameras, and assets directly before launching multi-environment rollout (same snippet as step 3 of the **Environment Only** tab).

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``eval/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to save evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for metric definitions.

.. dropdown:: Troubleshooting

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Symptom
        - Fix
      * - ``robodojo_runtime`` is not found
        - Activate the virtual environment created by the installer (``source .venv/bin/activate``, or the custom ``--venv`` path).
      * - Dependency resolution fails
        - Do not install Isaac Sim, IsaacLab, or the RoboDojo runtime manually. Remove the failed environment and rerun ``install.sh`` so the pinned versions and compatibility overrides apply together.
      * - ``pip check`` reports Isaac Sim/IsaacLab version differences
        - These packages use overlapping exact pins. Keep the installer's runtime-tested combination and do not manually upgrade ``starlette``, ``uvicorn``, or ``typing-extensions``.
      * - ``torch.cuda.is_available()`` returns ``False``
        - Run ``nvidia-smi`` to check the driver and GPU visibility. RoboDojo does not support CPU-only, macOS, or AMD ROCm execution.
      * - ``Assets``, ``cam_head``, or task layouts are missing
        - Verify that ``ROBODOJO_ASSETS_ROOT`` points to the parent of ``Assets/`` and that all four required subdirectories exist. Remove incomplete downloads and download them again.
      * - GPU memory is insufficient
        - Set the environment count to one and stop other evaluation or inference processes. RoboDojo starts a separate Isaac Sim process per parallel environment, so multi-environment counts depend on measured per-scene memory use.
      * - GPU memory remains allocated after exit
        - Call ``env.close()`` from a ``try/finally`` block. The RoboDojo adapter supports a complete close and recreate, not suspend and resume.
