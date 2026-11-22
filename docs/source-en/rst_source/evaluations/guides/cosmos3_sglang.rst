Cosmos3 SGLang Evaluation
=========================

Overview
--------

Evaluate Cosmos3 on LIBERO through RLinf's SGLang embodied backend. Each GPU runs an SGLang server (``server_type: embodied``) serving ``POST /v1/actions/generations``, while rollout workers batch observations by prompt over HTTP and step LIBERO with the returned action chunks.

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

The ``sglang`` target installs LIBERO with SGLang 0.5.19 (``diffusion`` extra), pins PyTorch 2.13.0, and picks the CUDA 12 or CUDA 13 wheel for your driver. Convert your SFT checkpoint into ``./model_diffusers`` following :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>` and place ``libero_native_frame_wise_relative_rot6d.json`` from NVIDIA ``cosmos-framework`` in the working directory before launching.

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: LIBERO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model sglang --env libero && source .venv/bin/activate

      **2. Evaluate LIBERO-Spatial**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang \
           rollout.model.model_path=./model_diffusers \
           rollout.model.action_stats_path=./libero_native_frame_wise_relative_rot6d.json \
           env.eval.total_num_envs=128

The default config is ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml`` (one SGLang server per GPU). Scale ``env.eval.total_num_envs`` with your GPU count and memory (for example, ``128`` on 8 GPUs). Evaluation runs offline because checkpoint conversion copies the tokenizer into ``model_diffusers/text_tokenizer/``.

.. dropdown:: How SGLang Serving Works

   Each GPU hosts one SGLang server assigned to a rollout worker. Each inference step runs in four stages:

   1. **Group by prompt:** Rollout workers receive LIBERO images and task instructions, group the parallel environments by augmented prompt (the server batches prompts of equal token length), and split each group into batches of at most ``rollout.sglang.server.batching_max_size`` environments (example ``8``; default ``1``).

   2. **Diffusion inference:** Each batch is sent via ``POST /v1/actions/generations`` to the SGLang server, which runs ``num_inference_steps`` diffusion steps and returns ``[horizon, 10]`` normalized rot6d chunks per environment.

   3. **Post-process actions:** Rollout workers sort responses by ``input_index``, quantile de-normalize the 10-D output, and convert 6-D rot6d rotations to 3-D axis-angle (producing 7-D LIBERO actions).

   4. **Step simulation:** Rollout workers reassemble the ``[N, 16, 7]`` action chunks in environment order and send them to the LIBERO env workers to advance simulation.

.. dropdown:: Key Configuration

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Field
        - Description
      * - ``rollout.model.model_path``
        - Diffusers checkpoint directory containing ``model_index.json``.
      * - ``rollout.model.action_stats_path``
        - Quantile stats JSON for action de-normalization; must match SFT (``*_rot6d.json``).
      * - ``rollout.model.action_normalization``
        - ``quantile_rot``; must match training.
      * - ``rollout.model.raw_action_dim`` / ``action_dim``
        - Model side ``10`` (rot6d) and env side ``7`` (axis-angle); both required.
      * - ``rollout.model.num_action_chunks``
        - Action steps returned per request (example ``16``); ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this value.
      * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
        - Diffusion steps and input video specs; must match training.
      * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
        - GPUs per server and tensor parallel size; both ``1`` for one server per GPU.
      * - ``rollout.sglang.server.batching_max_size``
        - Max environments batched per request (example ``8``; unset defaults to ``1``).
      * - ``rollout.sglang.http_timeout_s``
        - HTTP timeout in seconds; ``600`` is recommended for diffusion inference.
      * - ``env.eval.total_num_envs``
        - Parallel environment count; scale with GPU count and memory.

Results and Troubleshooting
---------------------------

Logs go to ``logs/<timestamp>-<config>/eval_embodiment.log`` (see :doc:`../reference/results` for metrics and videos, and :ref:`LIBERO evaluation configuration <libero-eval-config>` for trajectory counting). Confirm these milestones appear in order:

1. ``Launching sglang server (server_type=embodied) ...``
2. ``[RunAI Streamer] Overall time to stream 28.3 GiB ... to cpu: <seconds>`` (missing means weight loading stalled)
3. ``sglang server assigned: rank=i -> http://...``
4. ``[libero eval] task_id=.., trial_id=.., success=..``
5. Summary with ``success_once``, ``success_at_end``, and ``num_trajectories``

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Symptom
     - Fix
   * - SGLang cannot find model components
     - Confirm ``rollout.model.model_path`` points to the ``model_diffusers`` directory containing ``model_index.json``.
   * - Wrong actions or zero success rate
     - Verify ``action_normalization``, ``action_stats_path``, ``num_inference_steps``, ``num_frames``, and ``size`` match SFT.
   * - First batch HTTP timeout
     - Increase ``rollout.sglang.http_timeout_s`` and ``rollout.sglang.http_max_retries``.
   * - Local requests blocked by proxy
     - Export ``NO_PROXY=127.0.0.1,localhost`` before launch.
   * - LIBERO rendering errors
     - ``run_eval.sh`` uses EGL by default; on hosts without EGL, set ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
   * - GPU not released before re-run
     - Check ``nvidia-smi`` for leftover SGLang server processes; on a dedicated host, stop your Ray session with ``ray stop`` (check ``ray status`` first on shared machines because ``ray stop`` is machine-wide).
