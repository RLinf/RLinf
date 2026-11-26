Moonlight-16B (MoE GRPO)
========================

Train Moonlight-16B-A3B-Instruct (DeepSeek-V3 MLA + 64-expert MoE, ~3B active) with :doc:`GRPO </rst_source/reference/algorithms/grpo>` on AReaL-boba math data, running on a single 8-GPU node with Megatron-Bridge (TP=8, EP=8) and SGLang.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Moonlight-16B-A3B-Instruct (MLA + MoE)

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO (token-level loss, Megatron-Bridge + SGLang)

   .. grid-item-card:: Task
      :text-align: center

      Math reasoning (AReaL-boba)

   .. grid-item-card:: Hardware
      :text-align: center

      8×A100 (TP=8, EP=8)

| **What you'll do:** install ``agentic`` deps → download Moonlight-16B and AReaL-boba data → launch ``run_main_grpo_math.sh`` → monitor reward in TensorBoard.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · 8×A100 GPUs · NVIDIA driver ≥ 580 and CUDA 13.3 toolkit (``nvcc``).

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Moonlight-16B-A3B-Instruct

      **1. Install**

      The default ``agentic`` environment installs SGLang 0.5.12.post1, PyTorch 2.11, Megatron-Core 0.18, and ``megatron-bridge`` 0.5.0:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the model and `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__:

      .. code-block:: bash

         hf download moonshotai/Moonlight-16B-A3B-Instruct --local-dir ./Moonlight-16B-A3B-Instruct
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      Launch single-node 8-GPU training with ``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``:

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh moonlight-16b-grpo-megatron \
           rollout.model.model_path=./Moonlight-16B-A3B-Instruct \
           actor.tokenizer.tokenizer_model=./Moonlight-16B-A3B-Instruct \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

.. dropdown:: Advanced configuration & multi-node (optional)

   The recipe config (``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``) enables DeepSeek-V3 MLA + MoE with TP=8, EP=8, Megatron-Bridge weight conversion, SGLang expert-parallel rollout, a precision-aware optimizer, and math reward (+5 / -5).

   - **Dataset keys:** see :doc:`Qwen2.5 Math GRPO <reasoning>` for overriding prompt, answer, and chat-template keys.

   - **Multi-node scaling:** start a Ray cluster following :doc:`/rst_source/guides/multi_node`, increase ``cluster.num_nodes``, and run the launch command on the head node.

Visualization and Results
-------------------------

Watch training progress with ``tensorboard --logdir ./logs --port 6006``; see :doc:`Training metrics </rst_source/reference/metrics>` for metric definitions and W&B/SwanLab logging.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/moonlight/moonlight_grpo_reward.jpeg" style="width: 100%;"/>
       <p><em>Moonlight-16B GRPO reward curve</em></p>
     </div>
   </div>
