Cosmos3 SGLang Evaluation
==========================

Evaluate Cosmos3 on the LIBERO simulator using the SGLang backend: the model runs in a standalone SGLang server process, and rollout workers act as clients that send observations, receive action commands, and feed them into the simulation. Suitable for inference-only evaluation scenarios.

How It Works
----------------------------------------

Each GPU runs one SGLang server (``server_type: embodied``) executing ``Cosmos3OmniDiffusersPipeline``, exposing the action policy as the HTTP endpoint ``POST /v1/actions/generations``. The eval driver hands each server URL to a rollout worker, which turns one step of N environments into one or more requests.

The server only batches prompts of equal token length, so ``Cosmos3SGLangAdapter.request_groups`` first groups the environments by their augmented prompt, then splits each group into chunks of at most ``rollout.sglang.server.batching_max_size`` environments. The example config sets it to ``8``; left unset, the adapter follows SGLang's own default of ``1``, which sends every environment in its own request. The worker posts each group separately and reassembles the results in the original environment order.

Each response carries one record per environment: ``data[i].action.values`` holds that environment's ``[horizon, raw_action_dim]`` normalized rot6d chunk, and the adapter sorts the records by ``input_index`` before de-normalizing and converting them to 7-D axis-angle for LIBERO.

.. code:: text

   EnvWorker(libero) --obs(images+task)--> Cosmos3SGLangAdapter.request_groups
     groups the N envs by prompt, each group at most batching_max_size envs
        --one POST /v1/actions/generations per group-->
   SGLang server (Cosmos3OmniDiffusersPipeline, diffusion num_inference_steps steps)
        --response data[i].action.values = [horizon, 10] per env (normalized rot6d)-->
   Cosmos3SGLangAdapter parses each record, ordered by input_index:
     take first 10 channels → quantile de-normalize → rot6d(6) to axis-angle(3)
   the worker reassembles the groups into [N, 16, 7] in env order
        --[N, 16, 7]-->
   EnvWorker.chunk_step advances the simulation

Installation
----------------------------------------

Install the LIBERO environment and SGLang 0.5.19 with its ``diffusion`` extra. The ``sglang`` model pins torch 2.13.0 and picks the CUDA 12 or CUDA 13 build that the driver supports:

.. code-block:: bash

   bash requirements/install.sh embodied --model sglang --env libero
   source .venv/bin/activate

Prepare Checkpoint
----------------------------------------

The eval input is a diffusers component directory ``model_diffusers``. Both Nano and Edge can export SFT weights directly using ``--base-model-path`` and a matching complete model directory, or use the four-step cosmos-framework conversion. Both methods are described in the "Checkpoint Conversion" section of :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>`.

.. note::

   Evaluation **does not** require network access to HuggingFace or the Qwen3-VL cache: the tokenizer is copied into ``model_diffusers/text_tokenizer/`` during conversion, and the server reads it directly from there.

Run LIBERO-Spatial
----------------------------------------

The default config is ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``. Before running, point the YAML at your local ``model_diffusers``:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model_diffusers          # eval input diffusers directory
       action_stats_path: /path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json  # rot6d stats file from cosmos-framework

   env:
     eval:
       total_num_envs: 128   # adjust by GPU count / memory (example: 128 for 8 GPUs)

Then run:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang

**What this command does:** launches one Cosmos3 SGLang server per GPU, starts the LIBERO environment for evaluation, prints per-episode success/failure, and summarizes the success rate at the end. Logs are written to ``logs/<timestamp>-<config>/eval_embodiment.log``.

Cosmos3-Edge Evaluation
----------------------------------------

Edge uses ``evaluations/libero/libero_spatial_cosmos3_edge_eval_sglang.yaml``. In a SGLang environment with RLinf and LIBERO installed, export the RLinf SFT weights using a complete Edge HF snapshot with a matching architecture:

.. code-block:: bash

   python -m toolkits.cosmos3.convert_checkpoint \
     --src /path/to/model_path/actor/model_state_dict/full_weights.pt \
     --base-model-path /path/to/Cosmos3-Edge \
     --out /path/to/hf_model_path

The exporter checks every transformer key and shape, replaces all transformer tensors with SFT weights, and retains the snapshot's tokenizer, VAE, and configuration. It selects the online weights (``omni.net``) by default, matching the Edge HF eval config; use ``--use-ema-weights`` to select EMA instead. The output directory must be new.

Set ``rollout.model.model_path`` in the YAML to the exported ``model_diffusers`` directory and set ``action_stats_path`` to the statistics used for training, then run:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_edge_eval_sglang

For a batch smoke test, append ``cluster.component_placement.env=0 cluster.component_placement.rollout=0 env.eval.total_num_envs=4 env.eval.specific_reset_id=0 env.eval.max_episode_steps=32 env.eval.max_steps_per_rollout_epoch=32``. This sends a four-environment batch for the same task on one GPU; 32 steps only check execution. Keep the YAML's episode length and task distribution for normal evaluation.

RLinf uses SGLang's pipeline registration API to preserve explicit sampling parameters, preventing SGLang 0.5.19 from replacing an Edge action request's ``guidance_scale=1.5`` with the video default of ``5.0``. Model computation uses SGLang's native Cosmos3 pipeline; the installed package source requires no changes.

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
   * - ``rollout.sglang.server.batching_max_size``
     - Largest number of environments the server batches in one request (example 8). The adapter splits each prompt group into chunks of this size; left unset it follows SGLang's default of 1, one request per environment.
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
     - ``run_eval.sh`` renders with EGL by default; on hosts without EGL, set ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
   * - GPU not released before re-run
     - Confirm the previous ``ray stop`` completed; ``nvidia-smi`` shows all GPUs free; no residual ``ray::SGLangServerGroup`` processes.
