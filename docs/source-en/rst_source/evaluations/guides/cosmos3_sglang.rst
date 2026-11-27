Cosmos3 SGLang Evaluation
=========================

Evaluate Cosmos3-Nano or Cosmos3-Edge on LIBERO using SGLang Diffusion servers or the native Hugging Face backend.

How It Works
------------

Each rollout worker launches and owns one SGLang Diffusion server on its assigned GPUs, exposing the policy on ``POST /v1/actions/generations``. On every environment step, the worker groups parallel environments that share the same prompt into chunks of at most ``rollout.sglang.server.batching_max_size`` environments, sends one request per chunk, and reassembles the returned actions in environment order.

Each response record holds a normalized 10-D rot6d action chunk. The rollout worker sorts records by ``input_index``, denormalizes the first 10 channels with the LIBERO quantile statistics, converts rot6d to 3-D axis-angle, and passes the 7-D action chunk to LIBERO.

.. code-block:: text

   LIBERO env workers --observations (images + prompt)--> Rollout workers
     group environments by prompt (up to batching_max_size per request)
        --POST /v1/actions/generations-->
   Worker-owned SGLang Diffusion server
        --normalized [horizon, 10] rot6d action chunks-->
   Rollout workers (sort by input_index -> denormalize -> rot6d to axis-angle)
        --[N, 16, 7] env actions-->
   LIBERO env workers advance simulation

Installation
------------

Follow :ref:`Install SGLang Serving Environment <cosmos3-sglang-install>` in the Cosmos3 guide to set up ``.venv-sglang`` and the pinned Cosmos3 SGLang checkout.

Prepare Checkpoint
------------------

SGLang evaluation reads a Diffusers component directory (``model_diffusers``). Export an RLinf SFT checkpoint into ``model_diffusers`` following the "Checkpoint Conversion" section of :doc:`Cosmos3 SFT and NFT Training <../../examples/embodied/sft_cosmos3>`.

.. note::

   Evaluation runs fully offline once ``model_diffusers`` is exported because the tokenizer and processor files live inside that directory.

Run Cosmos3-Edge on LIBERO-Spatial
----------------------------------

Edit ``evaluations/libero/libero_spatial_cosmos3_edge_eval_sglang.yaml`` to set your exported ``model_diffusers`` directory and LIBERO statistics file:

.. code-block:: yaml

   rollout:
     model:
       model_path: ./assets/cosmos3-edge-policy-liberoall-10k/model_diffusers
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

Launch evaluation from the repository root:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_edge_eval_sglang

For a single-GPU quick check (4 parallel environments, 32 steps, 1 epoch):

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_edge_eval_sglang \
     cluster.component_placement.env=0 cluster.component_placement.rollout=0 \
     env.eval.total_num_envs=4 env.eval.specific_reset_id=0 \
     env.eval.max_episode_steps=32 env.eval.max_steps_per_rollout_epoch=32

Run Cosmos3-Nano on LIBERO-Spatial
----------------------------------

For SGLang evaluation of Cosmos3-Nano, set ``rollout.model.model_path`` and ``rollout.model.action_stats_path`` in ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml`` and run:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang

Key Configuration
-----------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Field
     - Description
   * - ``rollout.model.model_path``
     - Exported ``model_diffusers`` checkpoint directory.
   * - ``rollout.model.action_stats_path``
     - Quantile statistics JSON matching the SFT run (``*_rot6d.json``).
   * - ``rollout.model.action_normalization``
     - ``quantile_rot``; must match training.
   * - ``rollout.model.raw_action_dim`` / ``action_dim``
     - Model width 10 (rot6d) and environment width 7 (axis-angle).
   * - ``rollout.model.num_action_chunks``
     - Action steps per request (16); ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this value.
   * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
     - Diffusion steps and video geometry; must match training.
   * - ``rollout.sglang.server.tp_size`` / ``sp_degree`` / ``dp_size``
     - Per-server parallelism. GPUs per server are derived automatically from ``dp_size * tp_size * sp_degree * cfg_parallel_degree``.
   * - ``rollout.sglang.server.batching_max_size``
     - Maximum environments batched in one HTTP request (default 8 in the example configs).
   * - ``rollout.sglang.http_timeout_s``
     - Per-request HTTP timeout in seconds (recommended ``600``).
   * - ``env.eval.total_num_envs``
     - Total parallel LIBERO environments across workers.

Verification
------------

Inspect ``logs/<timestamp>-<config>/eval_embodiment.log`` and confirm these milestones appear in order:

1. Server launch: ``Launching sglang server (server_type=embodied) ...``
2. Server readiness: ``sglang server ready at http://...``
3. Per-episode logs: ``[libero eval] task_id=.., trial_id=.., success=..``
4. Final summary table: ``success_once``, ``success_at_end``, and ``num_trajectories``.

See :ref:`libero-eval-config` for LIBERO trajectory counting rules.

FAQ
---

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Fix
   * - SGLang cannot find model components
     - Confirm ``rollout.model.model_path`` points to the ``model_diffusers`` directory containing ``model_index.json``.
   * - Actions look wrong or success rate is zero
     - Verify ``action_normalization``, ``action_stats_path``, ``num_inference_steps``, ``num_frames``, and ``size`` match SFT.
   * - First batch hits an HTTP timeout
     - Increase ``rollout.sglang.http_timeout_s`` and ``rollout.sglang.http_max_retries``.
   * - Local requests are intercepted by a proxy
     - Export ``NO_PROXY=127.0.0.1,localhost`` before launching.
   * - LIBERO rendering fails
     - ``run_eval.sh`` uses EGL by default; on hosts without EGL, export ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
   * - GPUs remain occupied after an aborted run
     - Run ``ray stop --force`` and verify ``nvidia-smi`` shows no leftover rollout worker or server processes.
