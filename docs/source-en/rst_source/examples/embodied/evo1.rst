Evo-1 RL
========

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

`Evo-1 <https://github.com/MINT-SJTU/Evo-1>`__ is a compact (~1B) vision-language-action
model: an InternVL3-1B vision-language backbone with a flow-matching (DiT) action head.
RLinf integrates it **natively** — embedded in RLinf's Python memory space for
zero-latency, tensor-level interaction — and supports full-parameter SFT and GRPO
fine-tuning on the LIBERO simulator.

Overview
--------

SFT then GRPO-fine-tune Evo-1 on LIBERO manipulation tasks.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO

   .. grid-item-card:: Algorithms
      :text-align: center

      SFT · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      LIBERO-Spatial

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4–8 GPUs

| **You'll do:** install (native) → download the Evo-1 checkpoint → GRPO → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the Evo-1 checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_grpo_evo1``
     - GRPO training with Evo-1 on the LIBERO-Spatial suite.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - LIBERO camera observations and robot state required by Evo-1.
   * - Action
     - Continuous 7-DoF actions (6-DoF delta EE + gripper), decoded by the flow-matching head.
   * - Reward
     - LIBERO task success.
   * - Prompt
     - Natural-language task instruction for the LIBERO episode.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model evo1 --env libero && source .venv/bin/activate

      **2. Download**

      Download the Evo-1 LIBERO checkpoint (contains ``config.json``, ``norm_stats.json``, and ``mp_rank_00_model_states.pt``):

      .. code-block:: bash

         hf download MINT-SJTU/Evo1_LIBERO --local-dir ./Evo1_LIBERO

      **3. Launch**

      Shipped Evo-1 configs:

      .. list-table::
         :header-rows: 1
         :widths: 22 50 28

         * - Workflow
           - Config
           - Launcher
         * - GRPO
           - ``examples/embodiment/config/libero_spatial_grpo_evo1.yaml``
           - ``examples/embodiment/run_embodiment.sh``
         * - SFT
           - ``examples/sft/config/libero_sft_evo1.yaml``
           - ``examples/sft/run_vla_sft.sh``
         * - Standalone Eval
           - ``evaluations/libero/libero_spatial_evo1_eval.yaml``
           - ``evaluations/run_eval.sh``

      .. include:: _model_path.rst

      Launch GRPO training on LIBERO-Spatial:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_evo1 actor.model.model_path=$PWD/Evo1_LIBERO

      .. dropdown:: Advanced configuration (optional)

         - **Offline base VLM**: on machines without internet access at runtime, also download ``OpenGVLab/InternVL3-1B`` and point ``actor.model.evo1.vlm_name`` at the local directory:

           .. code-block:: bash

              hf download OpenGVLab/InternVL3-1B --local-dir ./InternVL3-1B

         - **Custom datasets and checkout**: match ``actor.model.evo1.arm_key`` and ``actor.model.evo1.dataset_key`` to the top-level keys in ``norm_stats.json``. If you cloned Evo-1 separately, set ``actor.model.evo1.repo_path`` (or ``EVO1_REPO_PATH``) to its root. To test quickly on a single task, append ``+env.train.task_id_filter=[0] +env.eval.task_id_filter=[0]``; ``env/success_at_end`` rises within tens of GRPO steps.
         - **Supervised Fine-Tuning (SFT)**: set ``data.train_data_paths`` to your Evo-1 dataset config YAML and ``actor.model.model_path`` to the base checkpoint, then launch (checkpoints are written under ``runner.logger.log_path``):

           .. code-block:: bash

              bash examples/sft/run_vla_sft.sh libero_sft_evo1

         - **Standalone evaluation**: evaluate a checkpoint with the standalone eval harness:

           .. code-block:: bash

              bash evaluations/run_eval.sh libero libero_spatial_evo1_eval \
                rollout.model.model_path=$PWD/Evo1_LIBERO

         - **CUDA / Ascend graph capture**: when action-head kernel launch overhead bottlenecks rollout, enable graph capture for the deterministic rollout path and keep the rollout model resident so later calls reuse captured graphs (graphs capture lazily per input shape and dtype, and in-place weight updates stay visible to replay; offloading releases captured graphs):

           .. code-block:: yaml

              rollout:
                enable_cuda_graph: True
                enable_offload: False

         - **Ascend vision-attention batching**: large batches on Ascend can exhaust memory in the frozen InternVL vision attention. Set ``actor.model.evo1.vision_attention_batch_size: 8`` (default ``0``, native attention) to compute attention for at most eight camera frames at once on Ascend without dropout, and measure throughput with your target rollout batch.

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

GRPO from an Evo-1 SFT checkpoint produces a steady rise in LIBERO-Spatial success rate (n=64 validation, frozen VLM, action-head-only RL):

.. list-table:: Evo-1 GRPO on LIBERO-Spatial (n=64)
   :header-rows: 1

   * - Setting
     - SFT
     - RLinf-GRPO
   * - Single task (``success_at_end``)
     - 0.58
     - **0.86**
   * - LIBERO-Spatial suite (``success_once``, best ckpt)
     - 0.656
     - **0.750**
