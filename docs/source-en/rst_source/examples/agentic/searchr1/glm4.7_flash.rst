Search-R1 Training for GLM-4.7-Flash
====================================

Train GLM-4.7-Flash (an MoE model with about 3B active parameters) to answer
questions by calling a search tool, using
`examples/agent/searchr1/config/train_glm4.7_flash.yaml`. Set up the RLinf
environment and the Qdrant wiki retrieval server first, as described in
:doc:`index`.

Overview
--------

Use this recipe to train GLM-4.7-Flash on a single 8-GPU node with the
Megatron-Bridge actor and SGLang rollout.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      GLM-4.7-Flash (MoE, about 3B active parameters)

   .. grid-item-card:: Algorithm
      :text-align: center

      Multi-turn RL with search-tool calls

   .. grid-item-card:: Tools
      :text-align: center

      Qdrant local wiki server

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run on 8×A100-80G

Data
----

GLM needs samples whose `prompt` field is a message list
(`[{"role": "system", ...}, {"role": "user", ...}]`), rendered with GLM's own chat
template (`data.apply_chat_template: True`). The released `RLinf/Search-R1-Data`
is pre-rendered with the Qwen chat template and cannot be used for GLM as is:
GLM's vocabulary has no `<|im_start|>` token, so the Qwen markers would reach the
model as plain text without any error. Convert the prompts into message lists
first and keep the other fields (`solutions`, ...) unchanged.

The reference run disables thinking (see below), so the user message also drops
the sentence "You must conduct reasoning inside <think> and </think> first every
time you get new information." from the Search-R1 instruction.

Configuration
-------------

Fill in the data path, the model path, and the retrieval server address:

.. code-block:: yaml

   data:
     train_data_paths: ["/path/to/data/searchr1_train_glm.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/GLM-4.7-Flash
   tools:
     search:
       server_addr: <server_ip>:8000

The model-specific settings are already set in the configuration:

.. code-block:: yaml

   data:
     apply_chat_template: True
     apply_chat_template_kwargs:
       enable_thinking: false
       clear_thinking: false
   agentloop:
     agentic_tokenizer: "glm47"
   rollout:
     sglang:
       tool_call_parser: glm47
   actor:
     megatron:
       mbridge: True

- `agentloop.agentic_tokenizer` selects the model-specific tokenization used by
  the agent loop (`default`, `qwen` or `glm47`; `default` if unset). It locates
  the end of the reasoning block, so that the response is truncated at the first
  `</search>` of the answer rather than at one written while reasoning.
- `rollout.sglang.tool_call_parser` must match the parser bound to the selected
  tokenizer type; a mismatch fails at startup.
- `enable_thinking` is passed to GLM's chat template and only affects the
  generation prompt of the first turn. Later turns append the tool output to the
  conversation directly.
- `actor.megatron.mbridge: True` loads the HuggingFace weights in process through
  Megatron-Bridge, so no offline HuggingFace-to-Megatron conversion is needed.

Training on 8×A100
------------------

Parallelism on 8×A100-80G (actor and rollout collocated):

- Actor (Megatron): tensor parallel 4, expert parallel 8, data parallel 2.
- Rollout (SGLang): one engine across the 8 GPUs with attention data parallel 2
  and expert parallel 8.
- `rollout.max_running_requests: 512` is split across the two attention data
  parallel ranks (256 each). Keep `rollout.cuda_graph_max_bs` (256) at least as
  large as the per-rank value; otherwise decoding falls out of CUDA graphs and
  becomes slower.

The retrieval server can share a GPU with training. The Qdrant server starts one
e5 query encoder on every visible GPU, so launch it with `CUDA_VISIBLE_DEVICES=0`
(about 1 GB of GPU memory) and start it before training.

Run the training with:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_glm4.7_flash

Visualization and Results
-------------------------

The following shows the training curves of GLM-4.7-Flash with thinking disabled,
using 512 prompts × 4 samples per step on 8×A100-80G and the Qdrant retrieval server
from :doc:`index`. The x-axis counts parameter updates: each rollout step performs four of
them (`algorithm.n_minibatches: 4`), so the 100 rollout steps span about 400
updates. Over these 100 rollout steps the trajectory reward rises from 0.28 to 0.54
(mean of the first and of the last five steps), the average number of turns per
trajectory grows from 1.7 to 2.0, and one rollout step takes about 870 s.

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png" style="width: 100%;"/>
       <p><em>Trajectory reward</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_length.png" style="width: 100%;"/>
       <p><em>Total sequence length</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_time.png" style="width: 100%;"/>
       <p><em>Time per rollout step (s)</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_turn.png" style="width: 100%;"/>
       <p><em>Turns per trajectory</em></p>
     </div>
   </div>
   <p style="text-align: center;"><em>GLM-4.7-Flash in RLinf</em></p>
