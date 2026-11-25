LIBERO Evaluation
=================

Overview
--------

LIBERO is a robotic manipulation simulation benchmark built on robosuite (MuJoCo), covering the Spatial, Object, Goal, and Long (``libero_10``) suites plus LIBERO-PRO and LIBERO-PLUS perturbation variants. RLinf runs parallel VLA policy evaluation on LIBERO and reports task-level metrics such as ``eval/success_once``.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      LIBERO (Spatial · Object · Goal · Long · Mem · Pro · Plus)

   .. grid-item-card:: Models
      :text-align: center

      OpenPI · OpenVLA-OFT · FastWAM · MolmoAct2 · StarVLA · Evo-1 · π₀-FAST · DreamZero · Cosmos3

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/libero/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`LIBERO <../../examples/embodied/libero>` · :doc:`FastWAM <../../examples/embodied/sft_fastwam>` · :doc:`DreamZero <../../examples/embodied/sft_dreamzero>`

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: OpenPI π₀.₅

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      **3. Evaluate (LIBERO-Spatial)**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT

      Swap the config name for ``libero_object_openpi_pi05_eval``, ``libero_goal_openpi_eval`` (π₀), or ``libero_10_openpi_pi05_eval`` to evaluate other suites.

   .. tab-item:: OpenVLA-OFT

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download RLinf/Openvla-oft-SFT-libero10-trajall --local-dir ./Openvla-oft-SFT-libero10-trajall

      **3. Evaluate (LIBERO-10)**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
           rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall

      Swap the config name for ``libero_object_openvlaoft_eval`` or ``libero_goal_openvlaoft_eval`` for other suites.

   .. tab-item:: FastWAM

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model fastwam --env libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download yuanty/fastwam libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json --local-dir ./fastwam

      **3. Evaluate (LIBERO-Spatial)**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval \
           rollout.model.model_path=./fastwam/libero_uncond_2cam224.pt \
           rollout.model.dataset_stats_path=./fastwam/libero_uncond_2cam224_dataset_stats.json

      Default placement uses 4 GPUs (20 slots × 25 episodes = 500 states). Swap the config name for ``libero_object_fastwam_eval``, ``libero_goal_fastwam_eval``, or ``libero_10_fastwam_eval``.

   .. tab-item:: MolmoAct2

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model molmoact2 --env libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download allenai/MolmoAct2-LIBERO --local-dir ./MolmoAct2-LIBERO

      **3. Evaluate (LIBERO-10)**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
           rollout.model.model_path=./MolmoAct2-LIBERO

      Swap the config name for ``libero_spatial_molmoact2_eval``, ``libero_object_molmoact2_eval``, or ``libero_goal_molmoact2_eval`` (see :doc:`../../examples/embodied/molmoact2`).

   .. tab-item:: StarVLA

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model starvla --env libero && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      Set ``framework.qwenvl.base_vlm`` in ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` to ``./Qwen2.5-VL-3B-Instruct`` (see :doc:`../../examples/embodied/starvla`).

      **3. Evaluate (LIBERO-Spatial)**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_starvla_eval \
           rollout.model.model_path=./Qwen2.5-VL-OFT-LIBERO-4in1

   .. tab-item:: π₀-FAST

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
           --model pi0_fast --env libero \
           --python 3.12.12 --torch 2.11.0 --no-flash-attn && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download lerobot/pi0fast-libero \
           --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
           --local-dir ./pi0fast-libero
         hf download google/paligemma-3b-pt-224 \
           --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
           --local-dir ./paligemma-3b-pt-224
         hf download jadechoghari/tokenizer-lib-mean \
           --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
           --local-dir ./tokenizer-lib-mean

      **3. Evaluate (LIBERO-10)**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_pi0_fast_eval \
           rollout.model.model_path=./pi0fast-libero \
           rollout.model.pi0_fast.text_tokenizer_name=./paligemma-3b-pt-224 \
           rollout.model.pi0_fast.action_tokenizer_name=./tokenizer-lib-mean

      Accept the PaliGemma terms on Hugging Face and run ``hf auth login`` before downloading. The default config uses 2 nodes / 10 GPUs (``cluster.num_nodes=2``); see :doc:`../../examples/embodied/pi0_fast`.

   .. tab-item:: DreamZero

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dreamzero --env libero && source .venv/bin/activate
         git clone https://github.com/RLinf/dreamzero.git ../dreamzero

      **2. Download Checkpoints, WAN2.2 Components, and Generate LIBERO Metadata**

      .. code-block:: bash

         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 --local-dir ./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000
         hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
         hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./Wan2.1-CLIP
         hf download google/umt5-xxl --local-dir ./umt5-xxl
         hf download physical-intelligence/libero --repo-type dataset --local-dir ./libero
         python toolkits/lerobot/generate_dreamzero_metadata.py \
           --preset libero_sim \
           --dataset-root ./libero \
           --output-metadata ./libero_metadata.json

      Set ``tokenizer_path`` (``./umt5-xxl``), ``diffusion_model_pretrained_path`` (``./Wan2.2-TI2V-5B``), ``image_encoder_pretrained_path`` (``./Wan2.1-CLIP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth``), ``text_encoder_pretrained_path`` (``./Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth``), and ``vae_pretrained_path`` (``./Wan2.2-TI2V-5B/Wan2.2_VAE.pth``) under ``rollout.model`` in ``evaluations/libero/libero_spatial_dreamzero_eval.yaml``.

      **3. Evaluate (LIBERO-Spatial)**

      .. code-block:: bash

         export DREAMZERO_PATH="$(realpath ../dreamzero)"
         bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval \
           runner.ckpt_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000/model.pt \
           rollout.model.metadata_json_path=./libero_metadata.json

      Point ``runner.ckpt_path`` at your ``.pt`` weights (such as ``full_weights.pt`` from SFT). For SGLang serving with the Diffusers checkpoint, see :doc:`dreamzero_sglang`.

   .. tab-item:: ApxInf

      **1. Install RLinf and Build APXinf-robo CUDA Binding**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate
         git clone --recursive https://github.com/RLinf/APXinf-robo.git ../APXinf-robo
         (cd ../APXinf-robo && pip install maturin && \
           CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda \
             --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml && \
           pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl && \
           pip install -e ./apxinf/python/apxinf --config-settings editable_mode=strict && \
           pip install -e ".[rlinf]")

      **2. Download π₀.₅ Checkpoint and Normalization Stats**

      .. code-block:: bash

         hf download lerobot/pi05_libero_base --local-dir ./pi05_libero_base
         curl -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
           -o ./pi05_libero_base/norm_stats.json

      **3. Evaluate (LIBERO-10)**

      .. code-block:: bash

         export APXINF_PI05_MODEL_DIR="$(pwd)/pi05_libero_base"
         bash evaluations/run_eval.sh libero libero_10_apxinf_pi05_eval

      Requires Linux with an NVIDIA GPU, CUDA toolkit, and Rust toolchain; see :ref:`ApxInf Backend <apxinf-backend>` and :doc:`../../guides/apxinf`.

.. note::

   **Host RAM & full-suite coverage:** Each LIBERO env holds ~2 GiB of host RAM, so configs defaulting to ``total_num_envs=500`` need ~1 TiB RAM. On standard hosts, lower ``env.eval.total_num_envs`` and raise ``env.eval.max_steps_per_rollout_epoch`` together (for example ``env.eval.total_num_envs=128 env.eval.max_steps_per_rollout_epoch=960`` on ``libero_spatial``) so ``auto_reset`` still covers all 500 states (see :ref:`Evaluation Configuration <libero-eval-config>`).

With ``--env libero``, ``install.sh`` clones LIBERO into ``.venv/libero`` (or reuses ``LIBERO_PATH``) and adds it to ``PYTHONPATH`` in ``.venv/bin/activate``. For SGLang serving, see :doc:`dreamzero_sglang` and :doc:`cosmos3_sglang`; to overlap inference latency with action-chunk execution, see :doc:`RTC <../../guides/rtc>`.

.. dropdown:: Example Configs

   Available under ``evaluations/libero/``:

   .. list-table::
      :header-rows: 1
      :widths: 38 28 34

      * - Config file
        - Task suite
        - Model
      * - ``libero_spatial_openpi_pi05_eval.yaml``
        - Spatial
        - π₀.₅
      * - ``libero_spatial_eval_pi05_RTC.yaml``
        - Spatial
        - π₀.₅ (RTC)
      * - ``libero_spatial_evo1_eval.yaml``
        - Spatial
        - Evo-1
      * - ``libero_spatial_starvla_eval.yaml``
        - Spatial
        - StarVLA
      * - ``libero_spatial_dreamzero_eval.yaml``
        - Spatial
        - DreamZero
      * - ``libero_spatial_dreamzero_eval_sglang.yaml``
        - Spatial
        - DreamZero (SGLang backend)
      * - ``libero_spatial_cosmos3_eval_sglang.yaml``
        - Spatial
        - Cosmos3 (SGLang backend)
      * - ``libero_spatial_fastwam_eval.yaml``
        - Spatial
        - FastWAM
      * - ``libero_spatial_molmoact2_eval.yaml``
        - Spatial
        - MolmoAct2
      * - ``libero_object_openpi_pi05_eval.yaml``
        - Object
        - π₀.₅
      * - ``libero_object_openvlaoft_eval.yaml``
        - Object
        - OpenVLA-OFT
      * - ``libero_object_fastwam_eval.yaml``
        - Object
        - FastWAM
      * - ``libero_object_molmoact2_eval.yaml``
        - Object
        - MolmoAct2
      * - ``libero_goal_openpi_eval.yaml``
        - Goal
        - π₀
      * - ``libero_goal_openvlaoft_eval.yaml``
        - Goal
        - OpenVLA-OFT
      * - ``libero_goal_fastwam_eval.yaml``
        - Goal
        - FastWAM
      * - ``libero_goal_molmoact2_eval.yaml``
        - Goal
        - MolmoAct2
      * - ``libero_10_openpi_pi05_eval.yaml``
        - Long (``libero_10``)
        - π₀.₅
      * - ``libero_10_apxinf_pi05_eval.yaml``
        - Long (``libero_10``)
        - π₀.₅ (ApxInf backend)
      * - ``libero_10_openvlaoft_eval.yaml``
        - Long (``libero_10``)
        - OpenVLA-OFT
      * - ``libero_10_fastwam_eval.yaml``
        - Long (``libero_10``)
        - FastWAM
      * - ``libero_10_molmoact2_eval.yaml``
        - Long (``libero_10``)
        - MolmoAct2
      * - ``libero_10_pi0_fast_eval.yaml``
        - Long (``libero_10``)
        - π₀-FAST
      * - ``libero_mem_openpi_pi05_eval.yaml``
        - LIBERO-Mem (``libero_mem``)
        - π₀.₅ (Short Memory)

.. _apxinf-backend:

.. dropdown:: ApxInf Backend

   The ApxInf path is evaluation-only. RLinf calls `APXinf-robo <https://github.com/RLinf/APXinf-robo>`_, wrapping the L1 interface of `ApxInf <https://github.com/infinigence/ApxInf>`_. Point ``APXINF_PI05_MODEL_DIR`` at a checkpoint directory containing ``config.json``, ``model.safetensors``, ``tokenizer.model``, and ``norm_stats.json``:

   .. code-block:: bash

      export APXINF_PI05_MODEL_DIR="$(pwd)/pi05_libero_base"
      bash evaluations/run_eval.sh libero libero_10_apxinf_pi05_eval

   This config keeps the native OpenPI contract: two oriented LIBERO images, an 8-D state, a state-free π₀.₅ prompt, 5 flow steps, a 10-action horizon, and execution of the first 5 actions. Set ``rollout.model.apxinf.noise_source=observation`` and pass an explicit ``[B,10,32]`` noise tensor for paired parity tests. The default 10 envs × 10 rollout epochs evaluate 10 reset states per LIBERO-10 task (100 trajectories total). For full build and backend details, see :doc:`../../guides/apxinf`.

.. _libero-eval-config:

.. dropdown:: Evaluation Configuration

   LIBERO evaluation runs one trajectory per ``(task_id, trial_id)`` pair and reports ``eval/success_once``. Each of the four standard suites (``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``) from the `official repo <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ contains 10 tasks and ~50 initial states per task (~500 trajectories).

   **Evaluation protocol**

   Trials across tasks are concatenated into a global ``reset_state_id`` and traversed in interleaved order in eval mode (``is_eval=True``) — ``(task0, trial0), (task1, trial0), …, (task0, trial1), …`` — so parallel envs advance evenly across tasks. Cover every ``(task_id, trial_id)`` pair in one ``rollout_epoch`` using either high parallelism (``total_num_envs >= S``) or auto-reset (``auto_reset=True`` with ``max_steps_per_rollout_epoch = N * max_episode_steps``).

   **Per-suite episode step limits**

   The longest training demo sets a lower bound for ``max_episode_steps``; match your training config:

   .. list-table::
      :header-rows: 1
      :widths: 22 20 24 34

      * - Suite
        - Lower bound
        - RLinf example values
        - Example config
      * - ``libero_spatial``
        - 220
        - 240 / 480
        - ``libero_spatial_openpi_pi05_eval`` / ``libero_spatial_dreamzero_eval``
      * - ``libero_object``
        - 280
        - 280 / 512
        - ``libero_object_openpi_pi05_eval`` / ``libero_object_openvlaoft_eval``
      * - ``libero_goal``
        - 300
        - 320 / 512
        - ``libero_goal_openpi_eval`` / ``libero_goal_openvlaoft_eval``
      * - ``libero_10``
        - 520
        - 520
        - ``libero_10_openpi_pi05_eval`` / ``libero_10_openvlaoft_eval``

   See :ref:`env-eval-fields` in :doc:`../reference/configuration` for generic ``env.eval`` fields.

   **Covering the full test set**

   Let ``S`` = total init states, ``E`` = ``total_num_envs``, and ``T`` = ``max_episode_steps``.

   *Option 1 — High parallelism:*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 500                # S for Spatial / Object / Goal / Long
          max_episode_steps: 240
          max_steps_per_rollout_epoch: 240   # equals max_episode_steps
          auto_reset: True                   # optional when E >= S
          rollout_epoch: 1

   Each LIBERO env runs in its own process and holds about 2 GiB of unshared host RAM after reset (about 1 TiB total for ``total_num_envs: 500``).

   .. warning::

      Env processes are subprocesses of the env worker, not Ray workers. Exhausting host RAM can hang the node without Ray's memory monitor killing a worker.

   *Option 2 — Auto-reset (recommended unless each env has >2 GiB host RAM plus headroom for the rollout model):*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 128
          max_episode_steps: 480
          # N = ceil(S / E); Spatial: ceil(500/128) = 4
          max_steps_per_rollout_epoch: 1920  # N * max_episode_steps = 4 * 480
          auto_reset: True
          ignore_terminations: True
          use_fixed_reset_state_ids: True
          use_ordered_reset_state_ids: True
          rollout_epoch: 1

   Set ``env.eval.rollout_epoch: 2`` to average metrics across two passes. Keep ``max_steps_per_rollout_epoch`` divisible by ``rollout.model.num_action_chunks``. Env workers offset seeds by ``seed + rank × stage_num + stage_id`` so each receives a distinct init-state subset; with ``auto_reset``, ``eval/success_once`` records each completed episode once.

.. dropdown:: Advanced Usage (LIBERO-PRO, LIBERO-PLUS, Parallelism)

   For LIBERO-PRO and LIBERO-PLUS environment installation, see :ref:`LIBERO-Pro & LIBERO-Plus <liberopro-plus-benchmark>`.

   **LIBERO-PRO**

   .. code-block:: bash

      export LIBERO_TYPE=pro
      export LIBERO_PERTURBATION=all
      bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
        rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall

   **LIBERO-PLUS**

   .. code-block:: bash

      export LIBERO_TYPE=plus
      export LIBERO_SUFFIX=all
      bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
        rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall

   **Adjust parallel scale**

   Lowering ``total_num_envs`` alone lowers coverage (64 envs with a 240-step budget evaluate only 64 of 500 init states). Raise ``max_steps_per_rollout_epoch`` together with it, and keep ``total_num_envs`` divisible by the number of env workers (``128`` works on 1, 2, 4, or 8 GPUs):

   .. code-block:: bash

      # max_steps_per_rollout_epoch = ceil(500 / 128) * 240 = 960
      bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
        env.eval.total_num_envs=128 \
        env.eval.max_steps_per_rollout_epoch=960 \
        rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT

Results and Troubleshooting
---------------------------

The terminal prints ``eval/success_once``; logs and videos go to ``logs/<timestamp>-<config>/`` (see :doc:`../reference/results` and :ref:`env-eval-fields` in :doc:`../reference/configuration`).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - EGL rendering errors on headless hosts
     - ``run_eval.sh`` uses EGL by default; on hosts without EGL, export ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
   * - Incomplete test-set coverage
     - Coordinate ``total_num_envs``, ``auto_reset``, and ``max_steps_per_rollout_epoch`` per :ref:`Evaluation Configuration <libero-eval-config>`.
   * - Host RAM exhaustion with 500 envs
     - Lower ``total_num_envs`` (such as ``128``) and enable ``auto_reset: True`` with a larger ``max_steps_per_rollout_epoch``; each LIBERO env uses ~2 GiB of host RAM.
   * - Step count validation error
     - Set ``max_steps_per_rollout_epoch`` to a multiple of ``rollout.model.num_action_chunks``.
