GRPO training for Moonlight-16B
================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

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

The actor uses Megatron-Bridge (``rlinf-megatron-bridge``) to convert HuggingFace
weights into Megatron-Core sharded checkpoints; the rollout uses SGLang (DeepSeek-V2
backend, ``model_type: deepseek_v3``) with expert parallelism (``ep_size == tp_size``).

Dataset
-------

Same as the general :doc:`math reasoning <reasoning>` recipe: `AReaL-boba-Data
<https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`_, with ``prompt`` /
``solutions`` keys. See that page for prompt-key / answer-key / ``apply_chat_template``
customization if your dataset uses a different format.

How GRPO Works
---------------

We adopt GRPO (Group Relative Policy Optimization) with the following modifications:

- **Token-level loss**: instead of averaging loss over the whole response, we average
  over tokens (as in DAPO), so long responses do not dominate training.
- **Minibatch early-stop**: if the importance ratio within a minibatch becomes too large,
  that minibatch is discarded to stabilize training.
- **Reward**: +5 if the final boxed/numeric answer is correct; -5 if incorrect.

Dependency Installation
-----------------------

**Prerequisites**: NVIDIA driver ≥580 and CUDA 13.3 toolkit (``nvcc``) must be installed.

.. code-block:: bash

   cd /path/to/RLinf
   MEGATRON_PATH=/path/to/Megatron-LM-core0.17 \
   bash requirements/install.sh agentic \
     --sglang 0.5.12 --torch 2.11.0 --transformers 5.6.0 \
     --no-apex --platform nvidia --use-mirror \
     --venv /opt/venv/reason_0512_v3

This installs the agentic stack — sglang 0.5.12 + torch 2.11+cu130 + TransformerEngine 2.17 +
transformers 5.6 + the py3.11-compatible ``rlinf-megatron-bridge`` 0.4.2 wheel + flash-attn —
and reuses the Megatron-Core 0.17 clone via ``MEGATRON_PATH``.

Run It
------

**1. Configuration file**

``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``
(``model_type: deepseek_v3``, TP=8, EP=8, ``multi_latent_attention: True``,
precision-aware optimizer, ``reward_type: math``).

**2. Launch**

.. code-block:: bash

   cd /path/to/RLinf/ray_utils
   rm -f ray_head_ip.txt
   export TOKENIZERS_PARALLELISM=false
   bash start_ray.sh
   if [ "$RANK" -eq 0 ]; then
       bash check_ray.sh 8          # number of GPUs in the cluster
       cd /path/to/RLinf
       bash examples/reasoning/run_main_grpo_math.sh moonlight-16b-grpo-megatron
   else
       if [ "$RANK" -eq 1 ]; then sleep 3m; fi
       sleep 10d
   fi
   sleep 10d

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
