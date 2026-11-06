Synchronous PPO for Search-R1
=================================

Train Qwen2.5-3B-Instruct with synchronous PPO and an independent critic. Prepare
the retrieval service in :doc:`index`, download the model and question dataset,
then launch the configuration below. The GRPO recipes are described in :doc:`qwen2.5`.

Overview
------------

PPO reuses the multi-turn agent and search-tool interface. Each training step
finishes a complete rollout batch, computes fixed predictions and targets, and
updates the actor and critic before generating the next batch.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      Synchronous PPO with GAE

   .. grid-item-card:: Tools
      :text-align: center

      FAISS or Qdrant wiki retrieval

   .. grid-item-card:: Hardware
      :text-align: center

      8 NVIDIA A100 80 GB GPUs; Megatron and SGLang

Tasks
~~~~~~~~~

The model can retrieve wiki passages before answering a question. Training uses
one terminal answer reward per complete episode.

.. list-table:: Search-R1 Task
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Recipe
   * - Dataset
     - `RLinf/Search-R1-Data <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
   * - Training unit
     - A complete multi-turn episode
   * - PPO batch
     - 1024 prompts, one trajectory per prompt, one minibatch

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each turn appends generated tokens and any retrieved passages to the conversation.
Only generated action tokens contribute to PPO losses.

.. list-table:: Agent Interface
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Meaning
   * - Observation
     - Question, previous turns, and retrieved passages
   * - Action
     - Reasoning, a ``<search>...</search>`` query, or an ``<answer>...</answer>`` response
   * - Limits
     - Two turns, 7096 total tokens, and at most 500 tokens per tool response

Installation
----------------

Install the Megatron training stack with SGLang following
:doc:`/rst_source/start/installation`:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   source .venv/bin/activate

PPO currently requires Megatron TP=PP=CP=1 and fully shared actor, critic, and
rollout GPUs. The configuration uses SGLang's HTTP server backend with TP=PP=1.
Manage retrieval resources separately from the eight training GPUs; see
:doc:`index` for retrieval setup.

Download the Model
----------------------

Download the actor/critic initialization model and training dataset:

.. code-block:: bash

   hf download Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/model/Qwen2.5-3B-Instruct
   hf download RLinf/Search-R1-Data --repo-type dataset --local-dir /path/to/data/searchr1

Set ``rollout.model.model_path`` and ``data.train_data_paths`` in
``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` to the model directory and
training JSONL file. Set
``tools.search.server_addr`` to the retrieval service's reachable address.

Run It
----------

Set ``runner.max_steps: 200`` and ``runner.max_epochs: 200`` in the configuration
to train for up to 200 batches. The epoch setting allows the dataloader to provide
enough batches. The configuration shares GPUs 0–7 between rollout, actor, and
critic and uses ``rollout.sglang.backend_type: server``.

Pass the configuration name without ``.yaml`` to the launcher:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo

The default configuration trains for one dataset epoch. It saves checkpoints
every 25 batches and at the final batch.

Rollout and Training Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At each step, the runner publishes actor weights and waits for all episodes in the
batch to finish, then offloads the rollout engine before critic inference. The critic predicts old values; the registered ``gae_dynamic`` algorithm computes fixed
returns and normalizes advantages over all action tokens in the rollout batch.
The critic worker computes these targets before updating its model and passes them
to the actor. Normalization reduces action statistics across critic DP ranks.
Tool text and padding are excluded, and the terminal answer reward is applied once.
Turn and token limits also terminate the episode with zero future value.

The runner trains the critic first and offloads its state before actor inference
and training. The configuration enables ``offload_weight``, ``offload_grad``, and
``offload_optimizer`` on both roles. These flags control which state moves to CPU
between roles. The runner stops the rollout engines when training ends.

The actor uses PPO clipping and the critic uses clipped Huber loss. With
``algorithm.n_minibatches=1``, each rollout batch produces one actor optimizer
update and one critic optimizer update. Megatron microbatches accumulate gradients
within each update. Actor learning rate is ``1e-6``; critic learning rate is
``5e-6`` with 10 warmup updates. ``critic.value_head_init`` selects ``normal`` or
``zero`` initialization for the independent value head.

``actor.pack_traj=true`` packs compatible turns while preserving every action
token's targets and behavior probability. ``algorithm.use_valid_token_scale=true``
compensates for microbatch and data-parallel averaging to keep a global action-token
mean loss. ``runner.max_tokens_per_mbs=8192`` bounds each training microbatch's token
budget. Set ``actor.pack_traj=false`` to train without packing.

Preserve Generated Tokens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``agentloop.truncate_response`` defaults to ``true`` and trims the generated token
prefix at the closing search tag using the model-aware tokenizer. The PPO configuration
selects ``false`` to retain the generated sequence and its behavior logprobs:

.. code-block:: yaml

   agentloop:
     truncate_response: false
   algorithm:
     recompute_logprobs: false
   rollout:
     return_logprobs: true

The remaining response-token budget always applies, regardless of the truncation
flag. Tokens beyond that cap and their logprobs are removed together. The retained
tokens are never decoded and re-encoded; decoding supplies text for tool parsing
and rewards. Using rollout logprobs skips the extra actor inference pass.
To recompute them instead, set ``algorithm.recompute_logprobs=true`` and
``rollout.return_logprobs=false``.

Resume and Evaluate
~~~~~~~~~~~~~~~~~~~~~~~

Set ``runner.resume_dir`` to a complete ``checkpoints/global_step_N`` directory to
restore the actor, critic, optimizers, and dataloader state. Keep the original
actor and critic checkpoint-conversion caches. When changing the experiment name,
point both ``megatron.ckpt_convertor.save_path`` fields to their original locations.

Export the actor checkpoint and run evaluation using the Megatron instructions in
:doc:`qwen2.5`. The critic is used only during training.

Training Metrics and Validation
-----------------------------------

Monitor ``rollout/reward_scores_traj`` in TensorBoard. ``Global Step`` advances once
per completed rollout batch; training metrics use optimizer-step indices. With one
minibatch these counts match. See :doc:`/rst_source/reference/metrics` for the shared
metric conventions.
