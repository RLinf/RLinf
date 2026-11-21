Cosmos3 SGLang Evaluation
=========================

Overview
--------

Evaluate Cosmos3 on LIBERO using the SGLang embodied backend. Each GPU runs a standalone SGLang server (``server_type: embodied``) serving ``POST /v1/actions/generations``, while rollout workers act as thin HTTP clients that batch observations by prompt, request action chunks, and step LIBERO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      LIBERO-Spatial (``libero``)

   .. grid-item-card:: Models
      :text-align: center

      Cosmos3 (SGLang diffusers backend)

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/libero/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>`

Setup and Run
-------------

The ``sglang`` target installs LIBERO with SGLang 0.5.19 (``diffusion`` extra), pins PyTorch 2.13.0, and picks the CUDA 12 or CUDA 13 wheel matching your driver. Before launching, convert your SFT checkpoint into a ``model_diffusers`` directory following the Checkpoint Conversion section of :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>`.

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: Cosmos3 SGLang + LIBERO

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model sglang --env libero && source .venv/bin/activate

         # 2. Evaluate LIBERO-Spatial (replace model_path and action_stats_path with your local files)
         bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang \
           rollout.model.model_path=/path/to/model_diffusers \
           rollout.model.action_stats_path=/path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json \
           env.eval.total_num_envs=128

The default config is ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml`` (one SGLang server per GPU). Adjust ``env.eval.total_num_envs`` to match your GPU count and memory (for example, ``128`` on 8 GPUs). Evaluation runs offline without Hugging Face or Qwen3-VL cache access because conversion copies the tokenizer into ``model_diffusers/text_tokenizer/``.

.. dropdown:: How SGLang Serving Works

   Each GPU runs one SGLang server executing ``Cosmos3OmniDiffusersPipeline``. The eval driver passes each server URL to a rollout worker. Because the server only batches prompts of equal token length, the adapter groups the N environments by augmented prompt and splits each group into chunks of at most ``rollout.sglang.server.batching_max_size`` environments (example: ``8``; default: ``1``). Each response carries ``data[i].action.values`` as a ``[horizon, raw_action_dim]`` normalized rot6d chunk; the adapter sorts records by ``input_index``, quantile de-normalizes them, converts rot6d to 7-D axis-angle for LIBERO, and reassembles them in environment order.

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

.. dropdown:: Key Configuration

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Field
        - Description
      * - ``rollout.model.model_path``
        - Diffusers checkpoint directory containing ``model_index.json``.
      * - ``rollout.model.action_stats_path``
        - Quantile stats file for action de-normalization; must match SFT (``*_rot6d.json``).
      * - ``rollout.model.action_normalization``
        - ``quantile_rot``; must match training.
      * - ``rollout.model.raw_action_dim`` / ``action_dim``
        - Model side ``10`` (rot6d) / env side ``7`` (axis-angle); both required.
      * - ``rollout.model.num_action_chunks``
        - Action steps returned per request (example ``16``); ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this.
      * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
        - Diffusion steps and input video specs; must match training.
      * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
        - GPUs per server and TP; both ``1`` for single-GPU deployment (one server per GPU).
      * - ``rollout.sglang.server.batching_max_size``
        - Max environments batched per request (example ``8``; unset defaults to ``1``).
      * - ``rollout.sglang.http_timeout_s``
        - HTTP timeout; diffusion inference takes longer, so ``600`` is recommended.
      * - ``env.eval.total_num_envs``
        - Number of parallel environments; scale with GPU count and memory.

Results and Troubleshooting
---------------------------

Logs go to ``logs/<timestamp>-<config>/eval_embodiment.log`` (see :doc:`../reference/results` for metrics and videos, and :ref:`LIBERO evaluation configuration <libero-eval-config>` for LIBERO trajectory counting rules). Confirm these milestones appear in order in ``eval_embodiment.log``:

1. ``Launching sglang server (server_type=embodied) ...``
2. ``[RunAI Streamer] Overall time to stream 28.3 GiB ... to cpu: <seconds>`` (missing this line means weight loading stalled)
3. ``sglang server assigned: rank=i -> http://...``
4. ``[libero eval] task_id=.., trial_id=.., success=..``
5. Summary with ``success_once``, ``success_at_end``, and ``num_trajectories``

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Symptom
     - Fix
   * - SGLang cannot find model components
     - Confirm both ``model_path`` entries point to the ``model_diffusers`` directory containing ``model_index.json``.
   * - Actions are wrong or success rate is zero
     - Verify ``action_normalization``, ``action_stats_path``, ``num_inference_steps``, ``num_frames``, and ``size`` match SFT.
   * - First batch HTTP timeout
     - Increase ``rollout.sglang.http_timeout_s`` and ``http_max_retries``.
   * - Local requests blocked by proxy
     - Set ``NO_PROXY=127.0.0.1,localhost`` before launch.
   * - LIBERO rendering errors
     - ``run_eval.sh`` uses EGL by default; on hosts without EGL, set ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
   * - GPU not released before re-run
     - Confirm the previous ``ray stop`` completed, ``nvidia-smi`` shows all GPUs free, and no ``ray::SGLangServerGroup`` process remains.
