RPent RL on LIBERO
==================

Train a Qwen3.5-4B visual planner with RLinf's multi-turn agentic RL loop and `RPent <https://github.com/RLinf/RPent>`__ on LIBERO-Pro and LIBERO. RPent runs camera rendering, SAM3 segmentation, and end-effector tool calls inside isolated simulator processes, and RLinf scores each episode with the simulator's native ``0`` / ``1`` success check — raising admitted rollout success from **26.88%** to **75.78%** over 96 synchronous GRPO updates from an SFT checkpoint.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen3.5-4B (Megatron-Bridge · SGLang)

   .. grid-item-card:: Algorithm
      :text-align: center

      Sync & Async GRPO

   .. grid-item-card:: Tools
      :text-align: center

      RPent Episode · SAM3 · LIBERO-Pro

   .. grid-item-card:: Hardware
      :text-align: center

      4×A100 80 GB (single node)

| **What you'll do:** install the learner and RPent simulator → download Qwen3.5-4B, SAM3 weights, and LIBERO-Pro assets → launch SAM3 and ``run_train.sh``.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · Hugging Face account (``hf auth login``) with approved access to `facebook/sam3 <https://huggingface.co/facebook/sam3>`__ · `RPent simulator guide <https://github.com/RLinf/RPent/blob/762a57c2b78278aff8d5c2c8f752ee42501bd5dc/docs/source-en/rst_source/simulators/rl.rst>`__.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Mode
     - Suite
     - Config (``examples/agent/rpent/config/``)
     - Focus
   * - Synchronous GRPO
     - LIBERO-Pro (``goal``, ``object``, ``spatial``)
     - ``train_qwen3.5_4b_grpo.yaml``
     - Samples 8 episodes per state, admits first 4 of 5 complete groups, recomputes actor logprobs (TIS clip 2.0)
   * - Asynchronous GRPO
     - LIBERO-Pro (``goal``, ``object``, ``spatial``)
     - ``train_qwen3.5_4b_async.yaml``
     - Overlaps rollout generation with actor updates and supports buffer checkpoint recovery
   * - Evaluation
     - LIBERO-Pro held-out initial states
     - ``eval_qwen3.5_4b.yaml``
     - Evaluates served checkpoints and writes per-suite success rates to ``eval_results.json``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Observation
     - Adaptive camera views rendered at ``1024 × 1024`` and SAM3 perception masks
   * - Action
     - Perception tool calls, up to three ordered end-effector pose/gripper commands per motion call, and ``finish`` to end the episode
   * - Reward
     - Native simulator task check: ``1`` only when the environment reports task success (``terminated``); ``0`` if the planner calls ``finish`` without completing the goal or hits the turn/step limit
   * - Prompt
     - Task language instruction with explicit JSON suite, task index, and initial-state seed

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Qwen3.5-4B (Megatron · 4×A100)

      **1. Install**

      Install the RLinf agentic environment from the repository root, then create an isolated environment for the tested RPent revision (``762a57c2``), which installs the matching ``rpent-libero`` and ``rpent-liberopro`` packages:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UNINSTALL_FA4=1 bash requirements/install.sh agentic --sglang 0.5.12.post1 && source .venv/bin/activate

      .. code-block:: bash

         export RLINF_ROOT="$PWD"
         git clone --no-checkout https://github.com/RLinf/RPent.git ../RPent-episode
         git -C ../RPent-episode fetch --depth 1 origin 762a57c2b78278aff8d5c2c8f752ee42501bd5dc
         git -C ../RPent-episode checkout --detach FETCH_HEAD
         (cd ../RPent-episode && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --prerelease=allow --torch-backend cu128 -r requirements/episode-ppo-env.txt)

      **2. Download and Prepare**

      Log in once with ``hf auth login`` if needed, download ``Qwen/Qwen3.5-4B`` (or point ``RPENT_MODEL`` to an SFT-initialized checkpoint), ``sam3.pt``, and the LIBERO-Pro simulator assets:

      .. code-block:: bash

         hf download Qwen/Qwen3.5-4B --local-dir ./models/Qwen3.5-4B
         hf download facebook/sam3 sam3.pt --local-dir ./models
         env -u LIBERO_PRO_ASSET_PATH ../RPent-episode/.venv/bin/liberopro-download-assets --assets-dir "$PWD/libero_pro_assets" --skip-existing --no-cache

      Export the runtime variables, register the asset directory in ``libero_config/config.yaml``, and generate the training and evaluation JSONL manifests:

      .. code-block:: bash

         export RPENT_PYTHON="$PWD/../RPent-episode/.venv/bin/python"
         export RPENT_MODEL="$PWD/models/Qwen3.5-4B"
         export LIBERO_CONFIG_PATH="$PWD/libero_config"
         export RPENT_RUN_DIR="$PWD/logs"
         export RPENT_TRAIN_DATA="$PWD/data/rpent/train.jsonl"
         export RPENT_EVAL_DATA="$PWD/data/rpent/eval.jsonl"
         "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --assets-dir ./libero_pro_assets --seeds 0-31 --output-dir data/rpent
         "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --seeds 32-39 --split eval --output-dir data/rpent

      In a second terminal from the RLinf root directory, start the SAM3 perception service on GPU 3:

      .. code-block:: bash

         SAM3_CHECKPOINT_PATH="$PWD/models/sam3.pt" "$PWD/../RPent-episode/.venv/bin/python" -m rpent.robots.components.sam3_server --cuda-device 3 --port 8101

      **3. Launch**

      Run synchronous GRPO on four GPUs (GPUs 0–1 for the TP2 actor, GPU 2 for SGLang, GPU 3 for SAM3 and EGL environments; defaults to ``runner.max_steps=150`` with checkpoints every 10 steps, or append ``runner.max_steps=2 runner.save_interval=2`` for a quick sanity check):

      .. code-block:: bash

         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_grpo

      To overlap rollout collection with learner updates using asynchronous GRPO, launch:

      .. code-block:: bash

         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_async

      To evaluate an exported Hugging Face checkpoint on the held-out split, point ``RPENT_MODEL`` to the exported directory and run:

      .. code-block:: bash

         export RPENT_MODEL="$RPENT_RUN_DIR/rpent-grpo/checkpoints/global_step_150/actor/hf_checkpoint"
         bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b

      .. dropdown:: Advanced configuration (optional)

         - If the simulator environment runs an older ``transformers`` release that cannot load the tokenizer class inside an exported checkpoint, pass ``tools.rpent.tokenizer_path=./models/Qwen3.5-4B`` (pointing to a compatible tokenizer and image processor directory) while keeping ``RPENT_MODEL`` on the exported weights.
         - ``actor.optim.optimizer_cpu_offload`` defaults to ``true`` to fit on 4×80 GB GPUs; pass ``actor.optim.optimizer_cpu_offload=false`` to keep optimizer states on GPU and verify checkpoint resume compatibility with your Megatron-Core build before long runs.
         - To train or evaluate on standard LIBERO instead of LIBERO-Pro, pass ``--libero-type standard`` when running ``prepare_tasks.py`` and override ``tools.rpent.libero_type=standard`` at launch.

Visualization and Results
-------------------------

Track training rewards, importance-sampling ratios, and rollout staleness in TensorBoard (``tensorboard --logdir "$RPENT_RUN_DIR"``); see :doc:`Training metrics </rst_source/reference/metrics>` for metric definitions.

Each episode saves its task request, interaction trace, and simulator outcome under ``$RPENT_RUN_DIR/<experiment_name>/episodes/``, and evaluation writes overall and per-suite success rates to ``$RPENT_RUN_DIR/rpent-eval/eval_results.json``. Training batches mask intermediate tool observations in ``response_mask`` so loss and importance-sampling weights apply only to sampled policy tokens.

.. list-table:: **LIBERO-Pro planner training result**
   :header-rows: 1
   :widths: 22 20 28 30

   * - Benchmark
     - Setup
     - Workload
     - Observed Result
   * - LIBERO-Pro (30 tasks)
     - 32×H100 · Sync GRPO from SFT checkpoint
     - ``goal``, ``object``, ``spatial`` (50 states/task, 32 of 40 groups × 8 episodes, 96 updates)
     - Admitted rollout success rose from ``688/2,560`` (**26.88%**) in steps 1–10 to ``1,940/2,560`` (**75.78%**) in steps 87–96
