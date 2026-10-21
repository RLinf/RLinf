Search-R1 Training
======================

Train Qwen2.5 to answer questions through search-tool calls with GRPO, synchronous
PPO, or asynchronous PPO. Start a retrieval service, configure the model and
question dataset, then choose a GPU layout and training recipe. This page also
covers the PPO update schedule, checkpoint evaluation, and training validation
for `Search-R1 <https://arxiv.org/abs/2503.09516>`__.

Overview
------------

The recipes use the same question-answering task and retrieval interface. PPO adds
an independent critic to the existing multi-turn agent workflow; asynchronous PPO
consumes completed episodes while agent workers continue producing them.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO, synchronous PPO, asynchronous PPO with SAO

   .. grid-item-card:: Tools
      :text-align: center

      FAISS or Qdrant wiki retrieval

   .. grid-item-card:: Hardware
      :text-align: center

      PPO: 8 NVIDIA A100 80 GB GPUs; Megatron and SGLang

Tasks
~~~~~~~~~

Train on questions paired with reference answers. The model can retrieve wiki
passages before producing its final answer.

.. list-table:: Search-R1 Task
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Recipe
   * - Dataset
     - `RLinf/Search-R1-Data <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
   * - Retrieval corpus
     - `ASearcher Local Knowledge <https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge>`__
   * - Training unit
     - A complete multi-turn episode, with one terminal answer reward
   * - PPO batch
     - 1024 prompts, one trajectory per prompt, one actor minibatch

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each turn extends the conversation with generated tokens and, after a search call,
the retrieved passages. Only generated action tokens contribute to PPO losses.

.. list-table:: Agent Interface
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Meaning
   * - Observation
     - Question, previous turns, and retrieved passages
   * - Action
     - Generated reasoning, a ``<search>...</search>`` query, or an ``<answer>...</answer>`` response
   * - Reward
     - Search-R1 answer reward, applied once at episode completion
   * - Limits
     - Two turns, 7096 total tokens, and at most 500 tokens per tool response

Installation
----------------

Install the agentic environment following :doc:`/rst_source/start/installation`.
For the PPO recipes, use the Megatron training stack with SGLang:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang --sglang 0.5.12.post1
   source .venv/bin/activate

The async engine adapter targets ``sglang==0.5.12.post1``. It adds acknowledged
pause/resume, request retraction, and token-level weight-version tracking. PPO
recipes use Megatron TP=PP=CP=1 and SGLang TP=PP=1.

Run retrieval in a separate environment so its dependencies and GPU allocation can
be managed independently from the eight training GPUs:

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub uvicorn fastapi

Download the Model
----------------------

Download `Qwen2.5-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>`__
and the training dataset. PPO initializes the actor and critic from this model;
the critic gets an independent value head and optimizer.

.. code-block:: bash

   hf download Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/model/Qwen2.5-3B-Instruct
   hf download RLinf/Search-R1-Data --repo-type dataset --local-dir /path/to/data/searchr1

Set ``rollout.model.model_path`` and ``data.train_data_paths`` in
``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` to the downloaded model
and training JSONL file. The shared, split, and SAO configurations inherit these
paths. For GRPO, set the corresponding fields in ``train_qwen2.5.yaml`` or
``train_qwen2.5_fsdp.yaml``.

Run It
----------

Start retrieval before launching training, then select one of the recipes below.
The model receives the same tool interface in all execution modes.

Start Wiki Retrieval
~~~~~~~~~~~~~~~~~~~~~~~~

Download the wiki corpus and the
`e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ retrieval model. The corpus
files require approximately 50–60 GB of disk space.

.. code-block:: bash

   conda activate retriever
   python examples/agent/searchr1/download.py --save_path /path/to/wiki
   hf download intfloat/e5-base-v2 --local-dir /path/to/model/e5-base-v2

For FAISS, set the corpus and model paths in
``examples/agent/tools/search_local_server_faiss/build_index.sh`` and run it:

.. code-block:: bash

   bash examples/agent/tools/search_local_server_faiss/build_index.sh
   python examples/agent/tools/search_local_server_faiss/local_retrieval_server.py \
     --index_path /path/to/wiki/e5.index/e5_Flat.index \
     --corpus_path /path/to/wiki/wiki_corpus.jsonl \
     --pages_path /path/to/wiki/wiki_webpages.jsonl \
     --retriever_name e5 --retriever_model /path/to/model/e5-base-v2 \
     --topk 3 --faiss_gpu --port 8000

The first command builds the retrieval index; the second serves searches on port
8000. Set ``tools.search.server_addr`` in the training YAML to this service's
reachable host and port.

For Qdrant, download the `Qdrant binary <https://github.com/qdrant/qdrant/releases>`__
and configure the corpus, retriever, and service paths in the scripts under
``examples/agent/tools/search_local_server_qdrant``:

.. code-block:: bash

   /path/to/qdrant/qdrant &
   bash examples/agent/tools/search_local_server_qdrant/build_index.sh
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

Keep Qdrant running after indexing. The last command exposes the same retrieval
interface used by training. Configure index/search parameters using the
`Qdrant optimization guide <https://qdrant.tech/documentation/guides/optimize/>`__.

Choose a Training Recipe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activate the agentic environment again. Run a configuration by passing its name
without ``.yaml`` to ``run_train.sh``; subsequent arguments are Hydra overrides.

.. list-table:: PPO Placement and Execution
   :header-rows: 1
   :widths: 22 34 22 22

   * - Mode
     - Configuration
     - Rollout GPUs
     - Actor and critic GPUs
   * - Sync, shared
     - ``train_qwen2.5_ppo_shared``
     - 0–7
     - 0–7
   * - Async, shared
     - ``train_qwen2.5_sao``
     - 0–7
     - 0–7
   * - Sync, split
     - ``train_qwen2.5_ppo_split``
     - 0–3
     - 4–7
   * - Async, split
     - ``train_qwen2.5_sao_split``
     - 0–3
     - 4–7

For example, launch shared synchronous PPO or split asynchronous PPO:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_shared
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao_split

Run these commands separately when they use the same machine. The recipes run one
dataset epoch and save every 25 learner batches and at the final batch. A fixed
200-batch run needs enough dataset epochs:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao \
     runner.max_steps=200 runner.max_epochs=200

Synchronous PPO additionally supports ``train_qwen2.5_ppo_separate``, with rollout
on GPUs 0–3, actor on 4–5, and critic on 6–7. Async PPO requires actor and critic to
share their GPU set. Its rollout set must either match that set or be disjoint.

The existing GRPO workflows remain available:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp

The first selects the Megatron multi-agent actor; the second selects FSDP2. The
PPO recipes use an independent Megatron critic.

Understand the PPO Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Synchronous PPO completes the rollout batch, computes fixed old values and GAE
targets, then updates actor and critic. Inference and training reuse the existing
multi-agent workers. Tools and padding stay outside the action-token loss mask;
natural termination and turn/token limits both settle the episode with zero future
value. The recipes disable KL reward and reference-policy loss terms.

``actor.pack_traj=true`` folds compatible turns from one trajectory into fewer
training rows. Packing preserves each action token's targets and behavior
probability. ``algorithm.use_valid_token_scale=true`` compensates for Megatron's
microbatch and data-parallel averaging so the loss remains a global action-token
mean. ``runner.max_tokens_per_mbs=8192`` bounds each training microbatch's token
budget.

The actor uses learning rate ``1e-6`` without warmup; the critic uses ``5e-6`` and
10 optimizer updates of warmup. ``critic.value_head_init`` selects ``normal`` or
``zero`` initialization; ``normal`` uses ``critic.model.init_method_std`` with a
default standard deviation of 0.02. Checkpoint loading restores the saved head.

Preserve Generated Tokens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``agentloop.truncate_response`` controls application-side processing at tool
boundaries. Its default is ``true``: trim at the closing search tag and re-encode
the retained text, using actor inference to recompute training logprobs.

Both PPO recipes instead preserve generated tokens and use their rollout behavior
probabilities directly:

.. code-block:: yaml

   agentloop:
     truncate_response: false
   algorithm:
     recompute_logprobs: false
   rollout:
     return_logprobs: true

The remaining response-token budget is enforced in both modes. Tokens beyond that
cap and their corresponding logprobs are removed together. With truncation disabled,
all retained token IDs and rollout logprobs stay unchanged; decoding is used for
tool parsing and rewards. This path skips the extra actor logprob inference pass.
Synchronous PPO can still enable recomputation with
``algorithm.recompute_logprobs=true``. Async PPO requires the preserved-token path.

Run Asynchronously
~~~~~~~~~~~~~~~~~~~~~~

The async runner consumes complete episodes from a bounded Channel instead of
waiting for a fixed cohort of prompts. It reuses synchronous PPO's worker groups,
packing, checkpoint utilities, and weight transport. GPU role transitions and
request retraction follow `Miles <https://github.com/radixark/miles>`__; the losses
and update ordering follow `SAO <https://arxiv.org/abs/2607.07508>`__, Sections 3.1,
3.2, and Appendix A.

In shared placement, SGLang pauses, retracts unfinished requests, and releases its
model/cache memory before training. After critic and actor training, the runner
publishes actor weights, restores rollout memory, and resumes those requests.
Generated prefixes and their original behavior logprobs survive the transition.

In split placement, rollout continues while the learner trains on its own GPUs.
Generation pauses briefly for weight publication. If actor and critic cannot both
remain resident, enable offload at role transitions:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao_split \
     actor.offload_weight=true actor.offload_grad=true actor.offload_optimizer=true \
     critic.offload_weight=true critic.offload_grad=true critic.offload_optimizer=true

The same six overrides apply to synchronous split PPO. Shared recipes enable them
by default. SGLang drains outstanding GPU work before acknowledging a pause and
rebuilds KV state after publication. Existing token logprobs are retained instead
of being rescored with the new weights. Each generated token carries the actor
weight version that produced it.

For each async learner batch, the runner performs these operations:

1. Receive completed trajectories, check their age, and compute fixed Monte Carlo
   critic targets over generated action tokens.
2. Update the critic on that batch ``critic.update_epochs`` times, then infer new
   token values.
3. Compute globally normalized policy GAE using
   ``lambda = 1 - 1 / (gae_lambda_alpha * L)``, where ``L`` counts generated tokens
   across the episode and ``gae_lambda_alpha=1.5``.
4. Update the actor once with DIS and publish the successful actor update.

DIS uses the recorded behavior logprobs. With the supplied bounds, only tokens
whose current-to-behavior probability ratio lies strictly between 0.7 and 6.0
contribute a policy gradient. The ratio is detached; the mean includes all action
tokens in its denominator.

``critic.loss_type=sao`` selects unnormalized, unclipped squared-error regression.
Set ``critic.loss_type=ppo`` for the synchronous clipped Huber loss, using
``algorithm.value_cliprange`` and Huber delta 10000. The first critic training
forward pass caches detached old values for every subsequent update of that batch.
Both losses use fixed returns; actor advantages use values inferred after training.

``critic.update_epochs`` accepts any positive integer. The recipe uses ``1``;
set it to ``2`` for SAO's two-critic-update schedule. With
``algorithm.n_minibatches=1``, this is exactly that many full-batch critic optimizer
steps. Microbatching and packing do not add optimizer steps. The paper's MoE
parameter-freezing and value-pretraining stages are outside this dense Qwen recipe;
the task retains Search-R1's two-turn, 7096-token limits and terminal reward.

Control Buffered Data
~~~~~~~~~~~~~~~~~~~~~~~~~

Producer slots and completion-buffer capacity bound how far rollout can run ahead
of training. Defaults are derived in configuration validation:

.. list-table:: Async Controls
   :header-rows: 1
   :widths: 37 23 40

   * - Configuration
     - Default
     - Meaning
   * - ``rollout.max_in_flight_rollouts``
     - ``2 * data.rollout_batch_size``
     - Concurrent episode producers, including tool calls and result transfer
   * - ``rollout.rollout_buffer_capacity``
     - ``2 * data.rollout_batch_size``
     - Completed episodes retained by the Channel
   * - ``rollout.max_weight_staleness``
     - ``null``
     - Disable age filtering; a nonnegative integer bounds allowed version lag

Leave either capacity ``null`` to derive it; an explicit positive integer takes
precedence. Buffer capacity must hold at least one learner batch. At batch size
1024, the two defaults are 2048 each, allowing up to 4096 outstanding prompts.

Staleness is checked when data is consumed, against the latest published rollout
weight version. The oldest generated token determines a trajectory's age. A rejected
trajectory is regenerated from its original prompt and retains its outstanding
prompt slot. A consumed trajectory frees a slot for a new prompt.

Rollout GPUs can become idle when the bounded buffer is full or producers are
waiting for tool responses. Inspect ``async/buffer_size``, ``async/in_flight``, and
``async/waiting_for_buffer`` before increasing capacity. Larger buffers can increase
staleness and do not improve throughput once training is the bottleneck.

Resume and Evaluate
~~~~~~~~~~~~~~~~~~~~~~~

Set ``runner.resume_dir`` to a complete ``checkpoints/global_step_N`` directory.
Both optimizers are restored. Async checkpoints also save buffered trajectories,
behavior probabilities, and prompt ownership. Completed buffered data is restored;
unfinished episodes restart from their original prompts after a process restart.
Live pause/resume during weight publication retains generated prefixes.

Retain the original actor and critic conversion caches. If changing the experiment
name on resume, set both ``megatron.ckpt_convertor.save_path`` fields to their
original directories.

For checkpoint export, follow :doc:`/rst_source/resources/faq` or use the Megatron
conversion utilities with the saved actor checkpoint:

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
     --load-path /path/to/checkpoints/global_step_N/actor \
     --save-path /path/to/intermediate --model qwen_2.5_3b \
     --tp-size 1 --ep-size 1 --pp-size 1 \
     --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
     --te-extra-state-check-none true --use-gpu-num 0 --process-num 16
   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
     --load-path /path/to/intermediate --save-path /path/to/exported-model \
     --model qwen_2.5_3b --use-gpu-num 0 --process-num 16

Copy the original model's configuration and tokenizer files into the export
directory without overwriting the converted weights. For FSDP GRPO, use
``rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf`` with the saved
training configuration and ``model_state_dict/full_weights.pt``.

Set the exported model, evaluation dataset, and retrieval-service address in
``examples/agent/searchr1/config/eval_qwen2.5.yaml``, then run:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh

Training Metrics and Validation
-----------------------------------

Monitor ``rollout/reward_scores_traj`` in TensorBoard. Both PPO runners advance
``Global Step`` once per learner batch. Individual async critic updates use the
``critic_update/`` namespace; ``async_trace.jsonl`` records sample ownership and
weight versions. See :doc:`/rst_source/reference/metrics` for shared metric
conventions.

To run a controlled comparison after configuring the model and dataset paths:

.. code-block:: bash

   python tests/e2e_tests/agent/searchr1/run_ppo_benchmarks.py \
     --steps 200 --batch-size 1024 --critic-updates 2 --name searchr1-ppo-comparison
   python tests/e2e_tests/agent/searchr1/plot_ppo_benchmarks.py \
     --steps 200 --name searchr1-ppo-comparison

The launcher runs all four modes sequentially with matching prompts, learning
rates, sampling, seeds, packing, offload flags, and checkpoint cadence. Async uses
DIS, adaptive GAE, and the selected critic update count. The report requires NumPy,
Matplotlib, and TensorBoard, and checks reward improvement, complete-run elapsed
time, optimizer counts, and sample ownership. Model initialization is timed
separately. Add ``--modes sync-split async-split`` to launch only split runs and
``--layouts split`` to report them.

To check SGLang prefix preservation independently, use the configured Megatron
path from your installation:

.. code-block:: bash

   PYTHONPATH="$PWD:$PWD/examples:$MEGATRON_PATH" python \
     tests/e2e_tests/agent/searchr1/verify_async_engine.py \
     --config-path "$PWD/examples/agent/searchr1/config" \
     --config-name train_qwen2.5_sao

This GPU test publishes weights three times during generation and verifies the
retained token IDs, behavior logprobs, and producing weight versions.

References
--------------

`Search-R1 <https://github.com/PeterGriffinJin/Search-R1>`__,
`SAO <https://arxiv.org/abs/2607.07508>`__,
`Miles <https://github.com/radixark/miles>`__, and
`ASearcher <https://github.com/inclusionAI/ASearcher>`__ provide the task, async
algorithm, engine scheduling reference, and retrieval corpus used by this recipe.
