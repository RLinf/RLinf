VLM-as-Policy: LIBERO + Qwen3.8-27B
===================================

This recipe trains ``Qwen3.8-27B`` directly as a multi-turn robot policy on LIBERO using RLinf's :doc:`embodied agent <index>` architecture. The model reads multi-view camera frames and base-frame arm pose, reasons inside ``<think>``, and outputs a JSON pose delta executed by a closed-loop controller.

Overview
--------

Train a reasoning VLM end-to-end on multi-turn tabletop manipulation tasks from sparse 0/1 rewards.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO (``libero_gso``, ``libero_all``) · RoboDojo schema

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO (``grpo_dynamic`` + ``pear_reinforce``) · DAPO filtering

   .. grid-item-card:: Tasks
      :text-align: center

      Multi-turn tabletop manipulation

   .. grid-item-card:: Hardware
      :text-align: center

      2 nodes (16 × 80 GB GPU) · 4 nodes (32 H200) paper scale

| **You'll do:** install → download ``Qwen3.8-27B`` → start multi-node Ray → launch → watch ``rollout/rewards``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · :doc:`Multi-node Ray cluster </rst_source/guides/multi_node>` · shared storage visible from every node.

In each episode, the model observes camera frames and rounded base-frame arm coordinates, writes a chain of thought inside ``<think>``, and emits one JSON pose delta. The simulator runs closed-loop PD and operational-space control until the target is reached, the arm stalls, the task finishes, or the native step budget runs out.

.. figure:: ../../../_static/svg/vlm_episode_loop.svg
   :width: 100%
   :align: center

   One multi-turn episode: observe camera frames and arm pose, reason in ``<think>``, emit a JSON pose delta, step the simulator, and receive a sparse 0/1 reward at completion.

For the disaggregated rollout and training architecture behind this recipe, see :doc:`index`.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - ``libero_gso`` (Goal, Spatial, Object)
     - ``libero_gso_grpo_pear_sglang.yaml``
     - Stage 1 default (2 nodes, 16 GPUs, 8 groups/step, 160k budget, top-K score centering, ``max_steps: 2`` quick check)
   * - LIBERO
     - ``libero_spatial`` (Spatial)
     - ``libero_spatial_grpo_pear_sglang_a100_4node.yaml``
     - Stage 1 80 GB A100 convergence setup (4 nodes, 1 actor + 3 rollout nodes with 6 TP4 replicas, 32 groups/step, 32k context, ``reasoning_effort: low``)
   * - LIBERO
     - ``libero_gso`` (Goal, Spatial, Object)
     - ``libero_gso_grpo_pear_sglang_paper.yaml``
     - Stage 1 paper scale (4 nodes, 32 GPUs, 128 groups/step, ``max_turns: 300``, ``max_steps: 1000``)
   * - LIBERO
     - ``libero_all`` (Goal, Spatial, Object, Long)
     - ``libero_10_grpo_pear_stage2.yaml``
     - Stage 2 long-horizon training (262,144 token budget, ``max_turns: 520``)
   * - LIBERO / RoboDojo
     - Mixed multi-embodiment
     - ``pear_stage3_libero_robodojo.yaml``
     - Stage 3 off-policy training with top-K score centering (``topk_log_probs: 32``, ``seq_length: 8192``)

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - Observation
     - Two 512x512 RGB camera frames rotated 180 degrees (``[agentview]`` and ``[wrist]`` for LIBERO) plus rounded base-frame arm pose (cm and rad) and gripper width.
   * - Action
     - One fenced ``json`` block giving relative deltas for ``arm`` (``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``) and ``gripper`` (``width``). Unparseable output ends the episode with reward ``0.0``.
   * - Reward
     - Sparse ``1.0`` on task success and ``0.0`` otherwise, with no critic, KL penalty, or entropy bonus.
   * - Prompt
     - System prompt defining the coordinate frame and JSON schema, followed by incremental per-turn state JSON and tagged camera frames. Left and right directions in LIBERO instructions are mirrored to match the rotated cameras.

Setup and Run
-------------

.. include:: ../embodied/_setup_common.rst

Start a multi-node Ray cluster before launching (2 nodes for the default and Stage 2 configs, 4 nodes for the paper-scale config), exporting a unique ``RLINF_NODE_RANK`` on each node (``0`` on the head node, ``1`` on the next node, and so on) before ``ray start``.

.. tab-set::

   .. tab-item:: Stage 1 (LIBERO-GSO)
      :sync: pear-stage1

      **1. Install**

      Install the embodied Qwen3.8 environment with LIBERO and SGLang ``0.5.22`` (CUDA 13 wheels), and activate it on every node:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model qwen3_8 --env libero && source .venv/bin/activate

      **2. Download**

      Download ``Qwen/Qwen3.8-27B`` into shared storage visible from every node (the launch script runs with ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``):

      .. code-block:: bash

         hf download Qwen/Qwen3.8-27B --local-dir ./Qwen3.8-27B

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 26 40 34

         * - Setting
           - Config (``examples/embodiment/config/``)
           - Notes
         * - 2-node default (16 GPUs)
           - ``libero_gso_grpo_pear_sglang.yaml``
           - ``cluster.num_nodes: 2``, ``max_steps: 2`` quick check
         * - 4-node A100 convergence (32 × 80 GB)
           - ``libero_spatial_grpo_pear_sglang_a100_4node.yaml``
           - ``cluster.num_nodes: 4``, 6 TP4 replicas, 32 groups/step, ``reasoning_effort: low``
         * - 4-node paper scale (32 GPUs)
           - ``libero_gso_grpo_pear_sglang_paper.yaml``
           - ``cluster.num_nodes: 4``, 128 groups/step, ``max_steps: 1000``

      Run the 2-node (16-GPU) default quick check from the head node, passing your local model path:

      .. code-block:: bash

         bash examples/embodiment/run_vlm_policy.sh libero_gso_grpo_pear_sglang \
           actor.model.model_path=$PWD/Qwen3.8-27B \
           rollout.model.model_path=$PWD/Qwen3.8-27B \
           actor.tokenizer.tokenizer_model=$PWD/Qwen3.8-27B

      For full training, raise ``runner.max_steps`` (for example ``runner.max_steps=1000``), or launch the 4-node (32-GPU) paper-scale config:

      .. code-block:: bash

         bash examples/embodiment/run_vlm_policy.sh libero_gso_grpo_pear_sglang_paper \
           actor.model.model_path=$PWD/Qwen3.8-27B \
           rollout.model.model_path=$PWD/Qwen3.8-27B \
           actor.tokenizer.tokenizer_model=$PWD/Qwen3.8-27B

      .. dropdown:: Advanced configuration (optional)

         Edit these YAML keys when adapting placement, concurrency, or memory headroom to your cluster:

         .. list-table::
            :header-rows: 1
            :widths: 24 38 38

            * - Area
              - Config Keys
              - Guidance
            * - Placement and length
              - ``cluster.num_nodes``, ``cluster.component_placement``, ``runner.max_steps``, ``runner.seq_length``
              - Keep ``actor`` and ``rollout`` on disjoint GPU ranks; assign ``reward`` to the same ranks as ``rollout``.
            * - Rollout concurrency
              - ``data.task_suite_name``, ``data.rollout_batch_size``, ``agentloop.max_turns``, ``tools.embodied_env.max_concurrent_envs``
              - Scale parallel LIBERO simulators and turn limits with rollout GPU count.
            * - Buffer and staleness
              - ``rollout_buffer.max_inflight_groups``, ``rollout_buffer.max_buffered_groups``, ``agentloop.max_actor_accept_staleness``
              - Keep engines saturated while bounding policy lag.
            * - Streaming sessions
              - ``session_server.engine_streaming_session``, ``session_server.merge_continuous_turns``, ``session_server.payload_shards``
              - Preserve cross-turn KV/GDN caches and store heavy frames locally on rollout nodes.
            * - SGLang and FSDP2 memory
              - ``rollout.sglang.attention_backend``, ``rollout.sglang.max_mamba_cache_size``, ``actor.fsdp_config.enable_gradient_accumulation``
              - Keep ``attention_backend: null``, ``max_mamba_cache_size: 256``, and ``enable_gradient_accumulation: False`` on 80 GB GPUs.
            * - Filtering and score centering
              - ``algorithm.down_sampling.down_sampling_config.reject_equal_reward``, ``algorithm.pear.score_centering``, ``session_server.topk_log_probs``
              - Drop zero-variance groups under DAPO; keep ``score_centering: True`` and ``topk_log_probs: 32`` enabled across all stages.

   .. tab-item:: Stage 2 & Stage 3
      :sync: pear-stage23

      **1. Install and Download**

      Reuse the same Qwen3.8 environment and ``./Qwen3.8-27B`` checkpoint from the Stage 1 tab. Stage 3 RoboDojo execution additionally requires Isaac Sim and cuRobo.

      **2. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 26 40 34

         * - Setting
           - Config (``examples/embodiment/config/``)
           - Notes
         * - Stage 2 (LIBERO-All)
           - ``libero_10_grpo_pear_stage2.yaml``
           - ``cluster.num_nodes: 2``, 262,144 token budget, ``max_turns: 520``
         * - Stage 3 (Mixed RoboDojo)
           - ``pear_stage3_libero_robodojo.yaml``
           - ``cluster.num_nodes: 2``, ``score_centering: True``, ``topk_log_probs: 32``

      Launch Stage 2 on ``libero_all`` (2 nodes, 16 GPUs) from the head node:

      .. code-block:: bash

         bash examples/embodiment/run_vlm_policy.sh libero_10_grpo_pear_stage2 \
           actor.model.model_path=$PWD/Qwen3.8-27B \
           rollout.model.model_path=$PWD/Qwen3.8-27B \
           actor.tokenizer.tokenizer_model=$PWD/Qwen3.8-27B

      Launch Stage 3 with top-K score centering (2 nodes, 16 GPUs):

      .. code-block:: bash

         bash examples/embodiment/run_vlm_policy.sh pear_stage3_libero_robodojo \
           actor.model.model_path=$PWD/Qwen3.8-27B \
           rollout.model.model_path=$PWD/Qwen3.8-27B \
           actor.tokenizer.tokenizer_model=$PWD/Qwen3.8-27B

Visualization and Results
-------------------------

Track ``rollout/rewards``, ``rollout/buffer/dropped_equal_reward``, ``rollout/session_server/payload_queued_groups``, and ``train/actor/policy_loss`` in TensorBoard (``tensorboard --logdir logs/``; see :doc:`Training metrics <../../reference/metrics>`). A steady queued-group count close to the rollout buffer size confirms that discarded multimodal groups release rollout-node memory promptly.

.. list-table::
   :header-rows: 1
   :widths: 34 32 34

   * - Config
     - Hardware
     - Verified Runtime Reference
   * - ``libero_gso_grpo_pear_sglang.yaml``
     - 2 nodes (16 × 80 GB A100)
     - FSDP2 update ~8–10 s/step; 27B cross-node NCCL weight sync ~5 s; multi-turn prefix cache hits on turn 2+

Troubleshooting
---------------

**``InvalidAuthToken`` when a worker node joins Ray**
   On Ray 2.60+ without a shared cluster token, export ``RAY_AUTH_MODE=disabled`` on every node before running ``ray start``.

**Out of memory on 80 GB actor GPUs**
   Keep ``actor.fsdp_config.enable_gradient_accumulation: False`` and ``actor.fsdp_config.gradient_checkpointing_use_reentrant: False`` so FSDP2 reduce-scatters gradients per layer instead of retaining full unsharded 27B gradients.

**Rollout stalls before filling a batch**
   Increase ``rollout.sglang.max_mamba_cache_size`` (default ``256``) if you raise ``rollout.max_running_requests`` or ``rollout_buffer.max_inflight_groups``, because each active streaming session holds multiple recurrent-state slots.

Next Steps
----------

- :doc:`index`: review the embodied agent architecture, async training flow, and multi-turn group lifecycle.
- :doc:`../../guides/async_agentic`: tune admission caps, staleness bounds, and weight-sync pause modes.
- :doc:`../../guides/session_server`: configure session servers, payload shards, and router policies.
- :doc:`../../guides/multi_node`: set up multi-node Ray clusters and ``RLINF_NODE_RANK``.
- :doc:`../../reference/metrics`: inspect rollout, staleness, and actor training metrics.
