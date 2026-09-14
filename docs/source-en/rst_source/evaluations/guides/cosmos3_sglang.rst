Cosmos3 SGLang Evaluation
==========================

Evaluate Cosmos3 on the LIBERO simulator using the SGLang backend: the model runs in a standalone SGLang server process, and rollout workers act as clients that send observations, receive action commands, and feed them into the simulation. Suitable for inference-only evaluation scenarios.

How It Works
----------------------------------------

Each GPU runs one SGLang server (``server_type: embodied``) executing ``Cosmos3OmniDiffusersPipeline``, exposing the action policy as the HTTP endpoint ``POST /v1/actions/generations``. The eval driver hands each server URL to a rollout worker; the worker sends all N environments' observations at once, the server runs a single batched forward returning ``[N, horizon, 10]`` normalized rot6d, and ``sglang_adapter`` de-normalizes and converts to 7-D axis-angle for LIBERO.

.. code:: text

   EnvWorker(libero) --obs(images+task)--> Cosmos3SGLangAdapter builds request
        --POST /v1/actions/generations-->
   SGLang server (Cosmos3OmniDiffusersPipeline, diffusion num_inference_steps steps)
        --response [N, horizon, 10] (normalized rot6d)-->
   Cosmos3SGLangAdapter parses:
     take first 10 channels → quantile de-normalize → rot6d(6) to axis-angle(3) → assemble [N, 16, 7]
        --[N, 16, 7]-->
   EnvWorker.chunk_step advances the simulation

Installation
----------------------------------------

Install embodied + LIBERO dependencies:

.. code-block:: bash

   bash requirements/install.sh embodied --env libero
   source .venv/bin/activate

.. note::

   Do not use ``--model cosmos3`` here: The model dependencies for cosmos3 (such as natten and cuDNN pin) conflict too much with the SGLang stack described below. Simply install the RLinf core and the LIBERO environment dependencies; install SGLang separately following the steps below.

Cosmos3 SGLang serving requires SGLang with the ``diffusion`` extra (batched Cosmos3 action support is merged to ``main`` at commit ``924aeee59cb68ea99bf7c7c183512958ed3d6272``):

.. code-block:: bash

   git clone https://github.com/FxxxxU/sglang.git /path/to/sglang && cd /path/to/sglang && git checkout 924aeee59cb68ea99bf7c7c183512958ed3d6272 && pip install -e "python[diffusion]"

Prepare Checkpoint
----------------------------------------

The eval input is a diffusers component directory ``model_diffusers``, produced by converting the SFT checkpoint via cosmos-framework. The full four-step conversion is described in the "Checkpoint Conversion" section of :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>`.

.. note::

   Evaluation **does not** require network access to HuggingFace or the Qwen3-VL cache: the tokenizer is copied into ``model_diffusers/text_tokenizer/`` during conversion, and the server reads it directly from there.

Run LIBERO-Spatial
----------------------------------------

The default config is ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``. Before running, point the YAML at your local ``model_diffusers``:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model_diffusers          # eval input diffusers directory
       action_stats_path: /path/to/cosmos3_framework/libero_native_frame_wise_relative_rot6d.json  # rot6d stats file from cosmos3_framework

   env:
     eval:
       total_num_envs: 128   # adjust by GPU count / memory (example: 128 for 8 GPUs)

Then run:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang

**What this command does:** launches one Cosmos3 SGLang server per GPU, starts the LIBERO environment for evaluation, prints per-episode success/failure, and summarizes the success rate at the end. Logs are written to ``logs/<timestamp>-<config>/eval_embodiment.log``.

Key Configuration
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Description
   * - ``rollout.model.model_path``
     - diffusers checkpoint directory (eval input).
   * - ``rollout.model.action_stats_path``
     - Quantile stats file for action de-normalization; must be from the same source as SFT (``*_rot6d.json``).
   * - ``rollout.model.action_normalization``
     - ``quantile_rot``; must match training.
   * - ``rollout.model.raw_action_dim`` / ``action_dim``
     - Model side 10 (rot6d) / env side 7 (axis-angle); **both required** (see FAQ).
   * - ``rollout.model.num_action_chunks``
     - Number of action steps returned per request (example 16); ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this.
   * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
     - Diffusion steps and input video specs; must match training.
   * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
     - GPUs per server and TP; both 1 for single-GPU deployment (one server per GPU).
   * - ``rollout.sglang.http_timeout_s``
     - HTTP timeout; diffusion inference is slow, recommend ``600``.
   * - ``env.eval.total_num_envs``
     - Number of parallel environments; adjust by GPU / memory.

Verification
----------------------------------------

Check ``eval_embodiment.log`` and confirm these milestones appear in order:

1. Server launch: ``Launching sglang server (server_type=embodied) ...``
2. Weight loading complete: ``[RunAI Streamer] Overall time to stream 28.3 GiB ... to cpu: <seconds>`` (usually within tens of seconds on local disk; **missing this line** means loading is stuck — see FAQ)
3. Server ready: ``sglang server assigned: rank=i -> http://...``
4. Per-episode results: ``[libero eval] task_id=.., trial_id=.., success=..``
5. Summary: ``success_once`` / ``success_at_end`` / ``num_trajectories``

LIBERO trajectory counting rules: see :ref:`libero-eval-config`.

FAQ
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Fix
   * - SGLang cannot find model components
     - Confirm both ``model_path`` entries point to the ``model_diffusers`` directory containing ``model_index.json``.
   * - Actions are wrong / nothing succeeds
     - Verify ``action_normalization`` / ``action_stats_path`` / ``num_inference_steps`` / ``num_frames`` / ``size`` match SFT.
   * - First batch HTTP timeout
     - Increase ``rollout.sglang.http_timeout_s`` and ``http_max_retries``.
   * - Local requests blocked by proxy
     - Set ``NO_PROXY=127.0.0.1,localhost`` before launch.
   * - LIBERO rendering errors
     - Set ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl`` when GPU is available.
   * - GPU not released before re-run
     - Confirm the previous ``ray stop`` completed; ``nvidia-smi`` shows all GPUs free; no residual ``ray::SGLangServerGroup`` processes.
