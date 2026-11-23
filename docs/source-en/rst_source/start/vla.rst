First Training Run
==================

Pick either the **Embodied AI** or **Agent & LLM** single-GPU track to complete your first RLinf training run and inspect metrics.

Launch Training
---------------

Prepare and activate the matching virtual environment from :doc:`installation` first (``--env maniskill_libero`` for Embodied AI, ``agentic --engine sglang`` for Agent & LLM).

.. tab-set::

   .. tab-item:: Embodied AI (ManiSkill3)
      :sync: embodied

      Runs PPO training on `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__'s built-in ``PickCube-v1`` task (~4 GB VRAM on 1 GPU, no extra asset or checkpoint downloads):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

   .. tab-item:: Agent & LLM (GRPO)
      :sync: agentic

      Runs single-GPU GRPO math reasoning training on ``DeepSeek-R1-Distill-Qwen-1.5B`` with SGLang and Megatron (config ``examples/reasoning/config/math/qwen2.5-1.5b-single-gpu.yaml``):

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data
         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-single-gpu \
           rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-1.5B \
           actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-1.5B \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

Each config name matches a YAML file in the repository (``examples/embodiment/config/`` for embodied, ``examples/reasoning/config/math/`` for reasoning) — see :doc:`../reference/configuration` for all config directories and tuning presets.

.. dropdown:: Common Overrides and Operations (Optional)

   .. code-block:: bash

      # Resume from a saved checkpoint directory
      bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
        runner.resume_dir=./logs/<run>/checkpoints/global_step_50

      # Pin a run to GPU 0
      bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
        cluster.component_placement.actor,rollout,env=0

      # Clear a stale local Ray session (check `ray status` first on shared hosts)
      ray stop

Results and Troubleshooting
---------------------------

Logs, TensorBoard events, and checkpoints are written to the configured output directory (``logs/<timestamp>-<config>/`` for embodied, ``../results/grpo-1.5b/`` for reasoning).

Start ``tensorboard --host 0.0.0.0 --logdir ./logs`` to inspect curves: watch ``rollout/env_info/return`` and ``rollout/env_info/success_once`` for embodied runs, or ``rollout/rewards`` and response length for Agent & LLM runs.

The embodied run prints a metric table at each step, and ``success_once`` climbs above ``0.90`` within a few hundred epochs:

.. code-block:: text

   ╭───────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step: 10/1000 │ Progress: ████░░░░░░░░░░░░░░ │ 1.0%              │
   │ Elapsed: 00:51       │ ETA: 01:24:09                │ Step Time: 5.122s │
   ├───────────────────────────── Environment ───────────────────────────────┤
   │ episode_len=50.0   │ num_trajectories=1024          │ return=7.187      │
   ├───────────────────────────── Training/Actor ─────────────────────────────┤
   │ actor/policy_loss=-0.013 │ actor/clip_fraction=0.179 │ critic/explained_variance=0.646 │
   ╰─────────────────────────────────────────────────────────────────────────╯

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - Out of GPU memory (OOM)
     - Lower ``actor.micro_batch_size`` and ``env.train.total_num_envs``; see :ref:`Out of Memory <faq-oom>`.
   * - EGL or Vulkan rendering error
     - Check graphics driver and EGL config in :ref:`Rendering Issues <faq-rendering>`.
   * - Stale Ray / NCCL error on rerun
     - Run ``ray stop`` before relaunching (:ref:`CUDA and Memory Issues <faq-cuda>`).

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: Tour of RLinf
      :link: walkthrough
      :link-type: doc

      Visual tour of how RLinf runs, scales across GPUs, and organizes configs and tools.

   .. grid-item-card:: Embodied AI
      :link: ../examples/index
      :link-type: doc

      Explore OpenVLA, OpenPI, LIBERO, RoboTwin, SFT, and world models.

   .. grid-item-card:: Agent & LLM
      :link: ../examples/agentic/index
      :link-type: doc

      Explore Search-R1, WideSeek-R1, rStar2, AgentLightning, and math reasoning.
