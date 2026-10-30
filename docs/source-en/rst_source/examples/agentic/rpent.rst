Synchronous PPO for the RPent LIBERO Planner
==============================================

Train Qwen3.5-4B as the planner of `RPent <https://github.com/RLinf/RPent>`__ on
LIBERO-PRO tasks with synchronous PPO and an independent critic. RPent runs each
episode: it asks the planner for the next tool call, executes it with the frozen
Pi0.5 policy and SAM3, and reports the task outcome. RLinf samples every planner
turn with SGLang and trains the actor and critic with Megatron.

Overview
------------

A second machine hosts the LIBERO simulators and the Pi0.5 and SAM3 services,
and RLinf places its agent workers there. Each worker runs every episode as one
RPent process, in which the planner runs unchanged and calls the worker's OpenAI
chat-completions endpoint for every turn. Each training step finishes a batch of
complete episodes, computes fixed predictions and targets, and updates the actor
and critic before the next batch.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen3.5-4B

   .. grid-item-card:: Algorithm
      :text-align: center

      Synchronous PPO with GAE

   .. grid-item-card:: Environment
      :text-align: center

      RPent on LIBERO-PRO with Pi0.5 and SAM3

   .. grid-item-card:: Hardware
      :text-align: center

      8 NVIDIA A100 80 GB GPUs for training; a second GPU machine for the simulators

Tasks
~~~~~~~~~

Each episode is one LIBERO-PRO task from a fixed initial state. The episode earns
reward 1 only when the task succeeds, and every turn of the episode shares that
outcome through GAE.

.. list-table:: RPent Task
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Recipe
   * - Training unit
     - A complete multi-turn episode
   * - PPO batch
     - 128 episodes, one trajectory per task, two minibatches
   * - Task file
     - JSONL; ``prompt`` holds the task language and ``answer`` a string with the task as JSON: ``suite``, ``task`` and ``seed``, and optionally ``vla_seed`` and the frozen ``memory_entries``

``answer`` is a JSON-encoded string, not a nested object. A row without memory
looks like this; ``prepare_tasks.py`` (below) writes complete rows:

.. code-block:: json

   {"prompt": "open the bottom drawer of the cabinet", "answer": "{\"suite\": \"libero_goal_task\", \"task\": 0, \"seed\": 1}"}

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~

The planner sees tool results as text; the SAM3 ``segment`` tool locates objects
instead of images. Each turn's prompt is the previous prompt plus the sampled
reply plus the new messages, so the turns of an episode share a token prefix and
are trained as one row. Tool-result tokens in the row get no advantage, value
target, or loss.

.. list-table:: Agent Interface
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Meaning
   * - Observation
     - Task, frozen memory, and the text results of earlier tool calls
   * - Action
     - A reply with one Qwen3.5 tool call (``<tool_call>`` XML), or ``finish``
   * - Limits
     - 24 turns, 65,536 total tokens, and 2,048 tokens per reply

Installation
----------------

Install the Megatron training stack with SGLang following
:doc:`/rst_source/start/installation`:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   source .venv/bin/activate

Install the same RLinf environment on the simulator machine, which runs the
agent workers; both machines need the same Python and Ray versions. On that
machine, also install RPent with its LIBERO-PRO extra in a separate environment,
following the
`RPent quick start <https://rpent.readthedocs.io/en/latest/rst_source/get_started/quickstart.html>`__:

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git && cd RPent
   pip install -e ".[libero-pro]"

The two machines form one Ray cluster (:doc:`/rst_source/guides/multi_node`).
The agent workers load the tokenizer from ``rollout.model.model_path``, so that
path must also exist on the simulator machine. Set ``tools.rpent.python`` in the
config to the RPent environment's interpreter.

Download the Model
----------------------

Download the planner, the Pi0.5 LIBERO policy, and SAM3, then extract the
language model that Megatron trains:

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir /path/to/Qwen3.5-4B
   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT --exclude optimizer.pt \
     --local-dir /path/to/RLinf-Pi05-LIBERO-130-fullshot-SFT
   modelscope download facebook/sam3 --local-dir /path/to/sam3
   python examples/agent/rpent/checkpoint.py extract \
     --vlm /path/to/Qwen3.5-4B --output /path/to/Qwen3.5-4B-text

``extract`` writes the language model of the served checkpoint as the text-only
model the actor and critic start from. In
``examples/agent/rpent/config/train_qwen3.5_4b_ppo.yaml``, set
``rollout.model.model_path`` and both ``tokenizer_model`` fields to the served
model and both ``hf_model_path`` fields to the extracted one.

Original Qwen3.5 checkpoints may also contain MTP auxiliary weights. ``extract``
omits them along with the vision tower; PPO trains the language model alone.

Prepare the Tasks
---------------------

Each task row names its frozen memory notes. Download the memory corpus, then
write the task files in the RPent environment, which has LIBERO-PRO:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --revision 47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1 \
     --include 'libero/GPT_5.5_xhigh/*' --local-dir /path/to/rpent-memory
   python examples/agent/rpent/prepare_tasks.py \
     --memory-dir /path/to/rpent-memory/libero/GPT_5.5_xhigh \
     --output-dir /path/to/rpent

The script keeps the 72 tasks that have all 50 initial states and their memory
notes, and holds out one task per suite. ``train.jsonl``, the file the training
config reads, runs the other 64 tasks on seeds 1-32, 16 steps per pass.
``panel_train.jsonl``, ``dev_quick.jsonl``, and ``holdout.jsonl`` evaluate
training seeds, new seeds, and held-out tasks. ``manifest.json`` lists the
held-out and excluded tasks.

Run It
----------

On the simulator machine, start the Pi0.5 and SAM3 services in the RPent
environment:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/RLinf-Pi05-LIBERO-130-fullshot-SFT
   python -m rpent.robots.components.pi05_vla_server --embodiment libero \
     --cuda-device 6 --transport http --host 127.0.0.1 --port 8100 &
   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
   python -m rpent.robots.components.sam3_server \
     --cuda-device 7 --transport http --host 127.0.0.1 --port 8101 &

Start Ray with the training machine as node 0 and the simulator machine as node 1,
then launch training on node 0:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_ppo

The config places eight agent workers on node 1 (``cluster.component_placement``,
``agentloop``). Each worker runs up to ``tools.rpent.episodes_per_worker``
episodes at once, rendering their simulators on ``tools.rpent.env_gpus``; each
episode starts a fresh LIBERO environment in its own process, which ends with the
episode or with its worker. ``tools.rpent.frozen_memory_dir`` must be the
directory the task files were built from, and ``tools.rpent.keep_episodes``
(``failures``) selects which episodes keep their artifacts under
``tools.rpent.output_dir``. The recipe saves checkpoints every 10 steps.

Rollout and Training Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep ``rollout.sglang.enable_weights_cpu_backup=true`` in this example. SGLang
backs up its full model weights in host memory before releasing GPU memory, so
the vision tower survives while the language-only actor supplies updated weights.
This uses additional host memory and transfer time, and is separate from
Megatron optimizer CPU offload.

The actor, critic, and SGLang share all eight GPUs and offload between phases. At
each step the runner syncs the actor's weights to SGLang in process, waits for
every episode of the batch, then infers old logprobs and values. GAE uses
``algorithm.gae_lambda`` (1.0) for both the actor's advantages and the critic's
returns, so every value target is the episode's outcome. Advantages are
normalized over the batch's generated tokens.

Token-level truncated importance sampling (cap 2) corrects for the difference
between SGLang and Megatron logprobs; ``rollout/tis/*`` reports it. With
``algorithm.n_minibatches=2``, each batch gives two actor and two critic
optimizer steps. Turns merge in the agent loop (``agentloop.merge_prefix_turns``),
so the actor keeps ``actor.pack_traj=false``.

Recipe Settings
~~~~~~~~~~~~~~~~~~

The recipe sets these keys. The ``agentloop`` and ``tools.rpent`` keys belong to
RPent; the others are general training options that the recipe turns on.

.. list-table:: RPent Recipe Settings
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Meaning
   * - ``agentloop.minimum_output_tokens``
     - Output tokens every request must still have within ``runner.seq_length``.
       A longer history ends the episode as ``context_limit``, a failure.
   * - ``agentloop.history_compaction``
     - The planner's history compaction: past ``max_messages`` messages or
       ``max_fraction`` of ``context_window`` it keeps the first user message and
       the last ``keep_messages``. A compacted request starts a new training row.
   * - ``agentloop.max_scored_tokens``
     - Generated tokens one merged row may score; defaults to
       ``algorithm.sampling_params.max_new_tokens``. A larger value gives fewer,
       longer rows and more logits memory per row.
   * - ``agentloop.model_request_timeout_s``, ``agentloop.episode_timeout_s``
     - Time limits the planner applies to each model request and each episode.
       An episode that reaches its time limit ends as a failure.
   * - ``agentloop.max_infra_retries``, ``agentloop.max_infra_retry_rate``
     - Retries of an episode whose services failed, and the fraction of retried
       episodes that stops the update.
   * - ``tools.rpent.process_timeout_s``
     - Stops an episode process that runs past this time; the episode is retried.
   * - ``rollout.replica_affinity``
     - Sends each agent worker's requests to one SGLang replica, so the turns of
       an episode reuse its prefix cache.
   * - ``actor.response_only_logits``
     - Computes vocabulary logits only at positions that score generated tokens.
   * - ``actor.megatron.empty_cache_after_backward``,
       ``critic.megatron.empty_cache_after_backward``
     - Returns cached activation memory before the gradient reduction.

Resume and Evaluate
~~~~~~~~~~~~~~~~~~~~~~~

Set ``runner.resume_dir`` to a ``checkpoints/global_step_N`` directory to restore
the actor, critic, optimizers, and dataloader. Keep ``rollout.model.model_path``
unchanged: with ``rollout.sglang.enable_weights_cpu_backup=true``, SGLang
loads the deployment model and preserves its unsynchronized vision weights across
offload. The first weight sync replaces the language weights with the restored
actor's weights. ``rollout.validate_weight_first_sync`` only checks a fresh start.

To serve a trained policy, for example for evaluation, write it into a copy of
the served model:

.. code-block:: bash

   python examples/agent/rpent/checkpoint.py export \
     --checkpoint logs/rpent-qwen3.5-4b-ppo/checkpoints/global_step_10 \
     --vlm /path/to/Qwen3.5-4B --output /path/to/exports/step10

``export`` retains the original model's vision and MTP weights. Use the exported
policy with speculative decoding disabled: its unchanged MTP weights have not
been aligned with the trained language model.

Then evaluate it on a fixed task file with the same services:

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b \
     rollout.model.model_path=/path/to/exports/step10 \
     "data.val_data_paths=[/path/to/rpent/holdout.jsonl]"

Each task runs ``algorithm.group_size`` times. The success rate, overall and per
suite, and every episode's outcome are written to
``logs/rpent-qwen3.5-4b-eval/eval_results.json``.

Training Metrics and Validation
-----------------------------------

Monitor ``rpent/success_rate`` in TensorBoard, with ``rpent/turns`` and
``rpent/rows`` for episode length, ``rollout/tis/*`` for the sampler mismatch, and
``rpent/infra_retry_rate`` for service failures. The update stops when more than
``agentloop.max_infra_retry_rate`` of the episodes needed a retry. See
:doc:`/rst_source/reference/metrics` for the shared metric conventions.

Limitations
-----------

- Training requires rollout TP1, and the actor and critic require pipeline and
  context parallel sizes of 1.
- CI runs the unit tests only; training needs GPUs, RPent, and LIBERO.
