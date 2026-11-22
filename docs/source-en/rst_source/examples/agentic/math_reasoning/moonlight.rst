Moonlight-16B Math GRPO
=======================

Train the Moonlight-16B-A3B-Instruct model (DeepSeek-V3 architecture: Multi-Latent Attention + MoE) with GRPO on math data. Compared with supervised fine-tuning, RL encourages diverse reasoning paths while optimizing final-answer correctness.

Overview
--------

Use this recipe to train Moonlight-16B-A3B-Instruct with GRPO on AReaL-boba math reasoning data, on a single 8-GPU node with the Megatron-Bridge actor (TP=8, EP=8).

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Moonlight-16B-A3B-Instruct (DeepSeek-V3 MLA + MoE, 64 experts / top-6, ~3B active)

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO with token-level loss and minibatch early-stop

   .. grid-item-card:: Data
      :text-align: center

      AReaL-boba math reasoning data

   .. grid-item-card:: Hardware
      :text-align: center

      8×A100, Megatron training (TP=8, EP=8)

Model Architecture
-------------------

Moonlight-16B-A3B-Instruct is a DeepSeek-V3 architecture Mixture-of-Experts model:

- **Multi-Latent Attention (MLA)**: low-rank compressed KV (``q_lora_rank`` / ``kv_lora_rank``), reducing the KV cache footprint.
- **MoE**: 64 routed experts (top-6 routing) + 1 shared expert; ``first_k_dense_replace=1`` (layer 0 is dense, layers 1–46 are MoE).
- **MTP**: a Multi-Token-Prediction head (``num_nextn_predict_layers=1``); not trained in RL.
- Trained with the **precision-aware optimizer** (Float8 master weights) for memory efficiency.

The actor uses Megatron-Bridge (``megatron-bridge``) to convert HuggingFace
weights into Megatron-Core sharded checkpoints; the rollout uses SGLang (DeepSeek-V2
backend, ``model_type: deepseek_v3``) with expert parallelism (``ep_size == tp_size``).

How GRPO Works
---------------

We adopt GRPO (Group Relative Policy Optimization) with the following modifications:

- **Token-level loss**: instead of averaging loss over the whole response, we average
  over tokens (as in DAPO), so long responses do not dominate training.
- **Minibatch early-stop**: if the importance ratio within a minibatch becomes too large,
  that minibatch is discarded to stabilize training.
- **Reward**: +5 if the final boxed/numeric answer is correct; -5 if incorrect.

Setup and Run
-------------

**Prerequisites**: NVIDIA driver ≥580 and CUDA 13.3 toolkit (``nvcc``) must be installed. In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Moonlight-16B-A3B-Instruct (Megatron · 8×A100)

      **1. Install**

      The default ``agentic`` environment installs SGLang 0.5.12.post1, PyTorch 2.11, Megatron-Core 0.18, and ``megatron-bridge`` 0.5.0:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the model and `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`_ (see :doc:`math reasoning <reasoning>` for ``prompt_key`` / ``answer_key`` / ``apply_chat_template`` customization):

      .. code-block:: bash

         hf download moonshotai/Moonlight-16B-A3B-Instruct --local-dir ./Moonlight-16B-A3B-Instruct
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      Update the model and data paths in ``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml`` (``model_type: deepseek_v3``, TP=8, EP=8, ``multi_latent_attention: True``, precision-aware optimizer, ``reward_type: math``), then launch training:

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh moonlight-16b-grpo-megatron

      For multi-node scaling, follow :doc:`/rst_source/guides/multi_node` and run the command above on the head node.

Visualization and Results
-------------------------

Monitor training with:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

For common metric meanings, see :doc:`Training metrics <../../../reference/metrics>`.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/moonlight/moonlight_grpo_reward.jpeg" style="width: 100%;"/>
       <p><em>Moonlight-16B GRPO reward curve</em></p>
     </div>
   </div>
