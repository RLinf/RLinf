PPO Training
============

Train Qwen2.5-3B-Instruct on Search-R1 with PPO and an independent critic.
Prepare the RLinf environment and local wiki retrieval server in :doc:`index`,
then configure the paths below and launch training. See
:doc:`/rst_source/reference/algorithms/ppo` for the multi-turn PPO training flow
and implementation details.

Overview
--------

This recipe trains the model to retrieve wiki passages before answering a
question, using one final answer reward for each complete episode.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO with GAE and an independent critic

   .. grid-item-card:: Tools
      :text-align: center

      FAISS or Qdrant local wiki server

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run on 8×A800 80 GB

Training on 8×A800
----------------------

Set the model directory and Search-R1-format training JSONL in
``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml``. The
`Search-R1 training dataset <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
uses this format:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
   data:
     train_data_paths: ["/path/to/train.jsonl"]
   tools:
     search:
       server_addr: 127.0.0.1:8000

Point ``tools.search.server_addr`` at the retrieval service prepared in
:doc:`index`.

The recipe uses rollout logprobs directly. It keeps every generated token within
the response budget, including any suffix after the closing search tag:

.. code-block:: yaml

   agentloop:
     truncate_response: false
   algorithm:
     recompute_logprobs: false
   rollout:
     return_logprobs: true

Set ``agentloop.truncate_response=true`` to exclude that suffix from training and
subsequent prompts. Both settings keep token IDs aligned with their logprobs.

Launch the recipe with its configuration name:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo

The default batch contains 1024 prompts with one trajectory per prompt and one
minibatch. Each batch produces one actor update and one critic update.

Visualization and Results
-------------------------

Monitor ``rollout/reward_scores_traj`` in TensorBoard; see
:doc:`/rst_source/reference/metrics` for metric definitions. The figure compares
GRPO and PPO training rewards for Qwen2.5-3B-Instruct:

.. raw:: html

   <div style="text-align: center;">
     <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_ppo_grpo_reward.png" alt="Search-R1 training reward: GRPO in orange and PPO in blue" style="width: 100%; max-width: 410px;"/>
     <p><em>Orange: GRPO; blue: PPO. Horizontal axis: training step; vertical axis: trajectory reward. TensorBoard smoothing: 0.6.</em></p>
   </div>

Both runs used 8 A800 80 GB GPUs, 20,000 WideSeek training examples, wiki retrieval,
200 training steps, one minibatch, and effective batch size 1024. GRPO used
``256 × 4`` prompts/trajectories; PPO used ``1024 × 1``. Both used seed 1234,
actor learning rate ``1e-6`` without warmup, two turns, sequence length 7096,
temperature 1.0, zero KL coefficient, SGLang server rollouts, and
``agentloop.truncate_response=false``.
