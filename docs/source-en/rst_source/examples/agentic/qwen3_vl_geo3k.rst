Qwen3-VL Visual Reasoning (Geo3K)
=================================

Train Qwen3-VL-2B-Instruct on the Geo3K geometric reasoning dataset with :doc:`GRPO </rst_source/reference/algorithms/grpo>` and Truncated Importance Sampling (TIS) using FSDP2 and SGLang.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen3-VL-2B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO + TIS (FSDP2 + SGLang)

   .. grid-item-card:: Task
      :text-align: center

      Geometric reasoning (Geo3K)

   .. grid-item-card:: Hardware
      :text-align: center

      Single node (1–8 GPUs)

| **What you'll do:** install Qwen3-VL deps → download Qwen3-VL-2B-Instruct and Geo3K → launch ``run_main_grpo_vqa.sh`` → track reward and entropy curves.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · ``torch >= 2.8.0``, ``sglang == 0.5.4``, and ``transformers == 4.57.1`` (installed below).

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Qwen3-VL-2B-Instruct

      **1. Install**

      Install the pinned Qwen3-VL dependency versions (``--no-apex`` skips Apex compilation for FSDP2):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic \
             --torch 2.8.0 \
             --sglang 0.5.4 \
             --transformers 4.57.1 \
             --no-apex && source .venv/bin/activate

      **2. Download**

      Download the model and the `Geo3K dataset <https://huggingface.co/datasets/CAIR-HKISI/geo3k>`__:

      .. code-block:: bash

         hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen3-VL-2B-Instruct
         hf download --repo-type dataset CAIR-HKISI/geo3k --local-dir ./geo3k

      **3. Launch**

      Launch single-node training with ``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml`` (rollout, actor, and reward are collocated on all GPUs):

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_vqa.sh qwen3-vl-2b-grpo-fsdp-geo3k \
           rollout.model.model_path=./Qwen3-VL-2B-Instruct \
           data.train_data_paths='["./geo3k/data/train-00000-of-00001.parquet"]' \
           data.val_data_paths='["./geo3k/data/validation-00000-of-00001.parquet"]'

.. dropdown:: Dataset format & advanced configuration (optional)

   Geo3K samples use ``problem`` (with ``<image>`` placeholders), ``images`` (image byte arrays), and ``answer`` (``\boxed{x}``). If a prompt omits the image placeholder, images are prepended before text.

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Option
        - Purpose
      * - ``data.prompt_key`` / ``data.answer_key`` / ``data.image_keys``
        - Map custom VQA fields (images from alternate mirrors may need conversion from base64 or PIL).
      * - ``data.system_prompt``
        - Prepend a shared instruction such as ``Answer: \boxed{$Answer}``.
      * - ``algorithm.importance_sampling_fix`` / ``importance_sampling_clip``
        - Enable TIS (clip = 2) to prevent entropy blow-up and unbounded response length.
      * - ``reward.reward_max_val`` / ``reward.reward_min_val``
        - Adjust correct / incorrect math reward values (defaults: +1 / 0).
      * - ``runner.enable_dynamic_batch_size`` / ``runner.max_tokens_per_mbs``
        - Control dynamic sequence packing under FSDP2 (``actor.fsdp_config.strategy: fsdp2``).

   ``actor/rollout_train_kl`` logs automatically when both ``recompute_logprobs`` and ``return_logprobs`` are enabled. For multi-node scaling, follow :doc:`/rst_source/guides/multi_node` and run the launch command on the head node.

Visualization and Results
-------------------------

Watch training progress with ``tensorboard --logdir ./logs --port 6006``; see :doc:`Training metrics </rst_source/reference/metrics>` for metric definitions and W&B/SwanLab logging.

With TIS enabled on Qwen3-VL-2B-Instruct, reward continues climbing past 750 steps while entropy converges:

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_reward.jpeg" style="width: 100%;"/>
       <p><em>Reward with TIS</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_entropy.jpeg" style="width: 100%;"/>
       <p><em>Entropy with TIS</em></p>
     </div>
   </div>

Without TIS, reward often collapses and entropy becomes unstable:

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_reward_wo_tis.jpeg" style="width: 100%;"/>
       <p><em>Reward without TIS</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_entropy_wo_tis.jpeg" style="width: 100%;"/>
       <p><em>Entropy without TIS</em></p>
     </div>
   </div>
