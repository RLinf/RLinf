Reinforcement Learning Training of Search-R1
================================================

Train a search-augmented Qwen model with GRPO, synchronous PPO, or fully asynchronous
PPO. This page covers the retrieval server, training configurations, and checkpoint
evaluation for
`Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/abs/2503.09516>`__
in RLinf.

Overview
--------

Use this recipe to train a search-augmented reasoning model with a local wiki
retrieval server.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      Multi-turn RL with search-tool calls

   .. grid-item-card:: Tools
      :text-align: center

      FAISS or Qdrant local wiki server

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run on 8×H100

Environment
-----------

RLinf Environment
~~~~~~~~~~~~~~~~~

RLinf environment setup follows :doc:`RLinf Installation </rst_source/start/installation>`.

Local Wiki Server Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the local retrieval server from the Search-R1 example.
Install faiss via conda; details in
`SearchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__
and installation reference in
`Search-R1 & veRL-SGLang <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__
The environment is also configured via conda.

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever

   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub

   # Install GPU version of faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   pip install uvicorn fastapi

Wiki Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~

We use the local retrieval files provided by Asearcher.
The downloaded files are approximately 50–60 GB in size.

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/agent/searchr1/download.py --save_path $save_path

Download the `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding model from HuggingFace,
and build the index

.. code-block:: bash

   bash examples/agent/tools/search_local_server_faiss/build_index.sh

Write the paths to the previously downloaded wiki files and the index into examples/agent/searchr1/launch_local_server.sh

.. code-block:: bash

   #!/bin/bash

   set -ex

   WIKI2018_WORK_DIR=$save_path

   index_file=$WIKI2018_WORK_DIR/e5.index/e5_Flat.index
   corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
   pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
   retriever_name=e5
   retriever_path=path/to/intfloat/e5-base-v2

   python3  ./local_retrieval_server.py --index_path $index_file \
                                               --corpus_path $corpus_file \
                                               --pages_path $pages_file \
                                               --topk 3 \
                                               --retriever_name $retriever_name \
                                               --retriever_model $retriever_path \
                                               --faiss_gpu --port 8000

Run `launch_local_server.sh` to start the Local Wiki Server.
Wait until server IP information is printed — indicating successful startup.

(Optional) Using Qdrant as Local Wiki Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also support qdrant as the wiki server as well. If you don't want to use the qdrant, move on to the Training section.

Download the local retrieval wiki corpus files provided by ASearcher using the method mentioned in the previous section.

Download the `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding model from HuggingFace.

Download `qdrant <https://github.com/qdrant/qdrant/releases>`__ binary file and build a qdrant collection with follwing steps. First, Create a new folder and put the qdrant binary into this folder, to facilitate the subsequent storage of qdrant binary and constructed collection files.

In `examples/agent/tools/search_local_server_qdrant/build_index.sh` and `examples/agent/tools/search_local_server_qdrant/launch_local_server.sh`, update the file paths for `WIKI2018_DIR`, `retriever_path`, and `qdrant_path` according to your downloaded wiki corpus, e5-base-v2, and qdrant paths.

Use the following commands to build the qdrant wiki server collection:

.. code-block:: bash

   # Create folder for qdrant
   mkdir -p /path/to/qdrant
   # Copy the binary
   cp qdrant /path/to/qdrant

   # Launch qdrant server
   /path/to/qdrant/qdrant &

   # Build qdrant collection
   bash examples/agent/tools/search_local_server_qdrant/build_index.sh

Run launch_local_server.sh to start the Local Qdrant Wiki Server. Wait until server IP information is printed — indicating successful startup.

.. code-block:: bash

   # Launch qdrant server
   /path/to/qdrant/qdrant &

   # Launch qdrant-based wiki server
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

Qdrant uses the HNSW graph index algorithm by default. For details on optimizing the HNSW graph index, please refer to the `Qdrant documentation <https://qdrant.tech/documentation/guides/optimize/>`__.


Training on 8×H100
------------------

Download the `training dataset <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__ from HuggingFace
and write its path into `examples/agent/searchr1/config/train_qwen2.5.yaml`:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/train.jsonl"]
     val_data_paths: ["/path/to/train.jsonl"]

Modify `rollout.model.model_path` in `train_qwen2.5.yaml`:

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
       model_type: qwen2.5

If you use `sampling_params.stop` to control model stop and save training time, detokenize should be set to True.

.. code-block:: yaml

   rollout:
      ……
      distributed_executor_backend: mp   # ray or mp
      disable_log_stats: False
      detokenize: True

By default, ``agentloop.truncate_response: true`` trims generated responses at
tool boundaries and re-encodes their text. Keep ``recompute_logprobs: true`` for
this mode so training scores the retained tokens.

.. code-block:: yaml

   algorithm:
      ……
      recompute_logprobs: True
      shuffle_rollout: False

To train directly with rollout probabilities, preserve the generated tokens:

.. code-block:: yaml

   agentloop:
      truncate_response: false
   algorithm:
      recompute_logprobs: false
   rollout:
      return_logprobs: true

The response-length cap applies in both modes: generated tokens beyond the remaining
budget are removed along with their logprobs. With ``truncate_response: false``,
tokens within that cap and their rollout logprobs stay unchanged, including tokens
at stop boundaries; manual tool-boundary trimming and re-encoding are skipped.
Tool parsing and reward calculation decode text without rewriting those tokens.
PPO skips the extra actor logprob inference pass. Set ``recompute_logprobs: true``
to retain that pass while preserving the original response tokens.

Run `bash examples/agent/searchr1/run_train.sh` to start training.

MA-FSDP Training
~~~~~~~~~~~~~~~~

RLinf also provides an FSDP2 implementation of the Search-R1 multi-agent actor.
Use `examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml`, set the same
model and dataset paths described above, and start it with:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp

With `agentloop.is_dynamic_rollout_batch: True`, the entrypoint selects
`MAFSDPActor` for `actor.training_backend: fsdp`. The configuration sets
`actor.pack_traj: True`, which packs compatible turns from the same trajectory
before FSDP training to avoid recomputing shared prefixes. Keep it enabled for
the provided Search-R1 FSDP recipe.

Synchronous PPO on 8×A100
------------------------------------------------

Use the PPO configurations to train an independent Megatron actor and critic
with the same SearchR1 model, dataset, search tool, and reward. Each global step
finishes every rollout, freezes its old predictions and GAE targets, updates both
models, and installs the new actor weights in SGLang before admitting another
batch. Reference models and KL rewards/losses are disabled.

Choose the configuration that matches how you want to allocate eight GPUs:

.. list-table:: PPO Placement
   :header-rows: 1
   :widths: 24 20 20 20

   * - Configuration suffix
     - Rollout GPUs
     - Actor GPUs
     - Critic GPUs
   * - ``shared``
     - 0–7
     - 0–7
     - 0–7
   * - ``split``
     - 0–3
     - 4–7
     - 4–7
   * - ``separate``
     - 0–3
     - 4–5
     - 6–7

The shared configuration releases rollout memory once per batch, runs the
complete critic phase followed by the complete actor phase, and restores
rollout KV/graph memory after weight installation and actor offload. The split
configuration keeps all models resident and serializes the two training phases.
The separate configuration keeps all models resident and runs actor/critic
inference and training in parallel, with a shared GAE barrier in between.

Set the existing model, dataset, and search-server paths in
``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml``, then run one of:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_shared
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_split
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_separate

All three commands use the same initial model, 1024 prompts with one trajectory
each (``algorithm.group_size: 1``), and one global minibatch. Only placement
and memory residency differ. The critic uses independent optimizer state and
a normally initialized value head by default (mean 0, standard deviation 0.02).
Set ``critic.value_head_init=zero`` for zero initialization; the default is
``critic.value_head_init=normal``. Normal initialization uses
``critic.model.init_method_std`` when set. These settings initialize a new head;
checkpoint loading restores the saved weights.
``runner.synchronous: true`` selects synchronous
scheduling; ``critic.use_critic_model: true`` and
``algorithm.adv_type: actor_gae`` select PPO. The original ``train_qwen2.5``
command continues to select GRPO.

The provided PPO configurations run one epoch and save both models every 25 global
steps and at the final step. Set ``runner.max_steps=20 runner.save_interval=20``
for a 20-step run.

The configurations limit each training microbatch to 8192 tokens with
``runner.max_tokens_per_mbs``. This bounds activation memory when actor and critic
share training GPUs. Changing this budget changes microbatch scheduling while
preserving the global batch and global valid-token loss weighting.

The runner schedules the rollout and model phases. The existing multi-agent
Megatron actor and Megatron critic execute inference and training; each worker
owns minibatch updates, microbatch scheduling, and checkpoint saving.

PPO and SAO inherit ``agentloop.truncate_response: false`` and
``algorithm.recompute_logprobs: false`` from the PPO base YAML, which also resolves
``rollout.return_logprobs`` to ``true``. Both train directly with rollout behavior
logprobs and preserve generated tokens within the response-length cap. The base
also sets the actor learning rate to ``1e-6`` with no warmup and the critic learning
rate to ``5e-6`` with 10 optimizer updates of warmup. PPO updates its critic once
per batch; asynchronous training uses ``critic.update_epochs`` updates per batch.
With one update, critic warmup spans 10 batches; with two, it spans five batches.

For synchronous PPO, you can choose the response-processing mode described above.
With recomputation enabled,
both placements with shared training GPUs retain actor weights between inference
and training. Tool observations stay outside policy/value losses, and
the final reward is applied once per trajectory. SearchR1 settles a complete
episode at natural termination or its turn/token budget; both boundaries use
zero future value.

Use ``actor.pack_traj: true`` to fold compatible turns from the same trajectory
for both actor and critic training. This avoids recomputing shared prefixes.
Old predictions and globally normalized GAE targets are fixed before packing;
packing merges their action positions and keeps tool observations masked.
Both losses use the global valid-token mean, so packing preserves their weighting.
Set ``algorithm.loss_agg_func: token-mean`` and
``algorithm.use_valid_token_scale: true`` to compensate Megatron's microbatch
and data-parallel averaging with each update's global valid-token count.
Workers pack the full rollout batch, balance its rows across DP replicas, then split optimizer
minibatches, following the GRPO execution order. Packing preserves the full
batch's action tokens and targets, but can change which tokens enter each
optimizer update when there are multiple minibatches; the two settings can
therefore produce different training curves.
Append ``actor.pack_traj=false`` to train with separate turn rows.

To alternate actor/critic offload on the split training GPUs, enable all three
state components for both roles:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo_split \
     actor.offload_weight=true actor.offload_grad=true actor.offload_optimizer=true \
     critic.offload_weight=true critic.offload_grad=true critic.offload_optimizer=true

Inference and training then load and release each role's state once per batch.
Default split and separate runs keep their state resident, including after
checkpoint saves. Saving temporarily loads offloaded state and then applies each
role's ``offload_weight``, ``offload_grad``, and ``offload_optimizer`` settings.
With a positive ``runner.save_interval``, save both models, optimizers, and
dataloader state at that interval and at the final step.

TensorBoard uses the same minibatch step axis as ``AgentRunner``. Global step
``g`` starts at ``(g - 1) * algorithm.n_minibatches``: reward and timings are
recorded once there, while actor and critic training metrics are recorded for
each minibatch.

Fully Asynchronous PPO on 8×A100
--------------------------------

To train as trajectories finish, use the asynchronous configurations. They implement
the token-level design in `SAO <https://arxiv.org/abs/2607.07508>`__, Sections 3.1,
3.2 and Appendix A, using continuous rollout production and a bounded completion
buffer. The runner extends the synchronous PPO runner and reuses its multi-agent
actor, critic, trajectory packing, token weighting, checkpointing, and weight
transport. Its role transitions and request retraction follow
`Miles <https://github.com/radixark/miles>`__.

Set the model, data, and retrieval-server paths in the base PPO configuration, then
choose one of the two supported layouts:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao
   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao_split

The first command shares GPUs 0–7 across all three models. Before training, SGLang
retracts unfinished requests and releases rollout memory. After actor training,
the runner installs weights, restores KV/graph memory, and resumes those requests.
Their generated token prefixes and original behavior logprobs remain unchanged.
The split command places rollout on GPUs 0–3 and both training models on GPUs 4–7.
Generation continues during critic and actor training; it pauses briefly for weight
publication. Enable the six actor/critic offload settings shown above to alternate
their residency when training memory is limited.

Both configurations use batches of 1024 prompts, one trajectory per prompt, and raw
rollout logprobs. They use stochastic full-vocabulary sampling at temperature 1,
without repetition penalties, Megatron TP=PP=CP=1, and SGLang TP=PP=1. The targeted
adapter supports ``sglang==0.5.12.post1``: it acknowledges pause/resume, drains
overlapped GPU work, parks waiting and partially prefetched requests outside cache
ownership, and flushes old KV state before resumption. Every generated token carries
the weight version that produced it, including tokens in requests resumed after
multiple publications. The published version advances only after all rollout
workers acknowledge installation; a failed publication stops the run.
Resumed requests rebuild their KV state while keeping recorded output logprobs;
output-only requests skip logprob recomputation for the retained prefix.

The runner owns the dataloader and fills the prompt Channel at startup. After
consuming each batch, it replenishes that Channel before training. Agent workers
generate continuously and send completed trajectories to a bounded Channel.
The runner calls batch construction directly, consuming the global completion
order, checking staleness, and sending targets to the critic training ranks.
After critic updates and inference, the runner reads the new values through a
Channel, computes globally normalized advantages, and sends the targets to actor
training. These controls bound pending work and optionally filter stale data:

.. list-table:: Async Controls
   :header-rows: 1
   :widths: 36 14 50

   * - Configuration
     - Default
     - Meaning
   * - ``rollout.max_in_flight_rollouts``
     - 2048
     - Trajectory producer slots, including tool calls and result transfer.
   * - ``rollout.rollout_buffer_capacity``
     - 2048
     - Completed trajectories retained by the Channel.
   * - ``rollout.max_weight_staleness``
     - ``null``
     - No staleness filtering; otherwise require a nonnegative integer.

Leave either capacity as ``null`` or omit it to derive its value as
``2 * data.rollout_batch_size`` during configuration validation. Explicit integer
values take precedence. The base PPO configuration sets ``data.rollout_batch_size=1024``
for both synchronous and asynchronous runs.
Keeping two learner batches in flight lets generation fill the next batch while
slower trajectories remain unfinished. Staleness is checked when consuming data,
against the latest published rollout version. A trajectory's age uses its oldest
generated action-token version. Rejected
trajectories are regenerated from their original prompts. The producer's finite
prompt budget is ``runner.max_steps * data.rollout_batch_size``, so benchmark runs
consume the same prompt set despite different completion orders.

The sum of the two capacities bounds all outstanding prompts, including generation,
transfer queues, and buffered results. A consumed trajectory releases its prompt
slot; a stale trajectory keeps its slot while it is regenerated. With both capacities
set to 2048, at most 4096 prompts are outstanding. When rollout finishes those
trajectories before training consumes them, the split rollout GPUs become idle
until the learner frees space. Check
``async/buffer_size`` against ``async/buffer_capacity`` and inspect
``async/waiting_for_buffer``. ``async/in_flight`` counts unfinished trajectories,
including tool calls, and excludes completed trajectories waiting for buffer space.
A larger buffer lets rollout run farther ahead and increases weight staleness; it
does not increase steady-state throughput when training is the bottleneck.

Each learner batch follows this order:

1. Compute fixed Monte Carlo critic targets with ``gamma=1`` and critic lambda 1.
   The default critic loss is an unclipped, globally token-averaged squared error.
2. Update the critic ``critic.update_epochs`` times over the same full batch, then
   infer fresh token values.
3. Compute policy GAE with ``lambda = 1 - 1 / (1.5 * L)``, where ``L`` counts
   generated action tokens across the whole episode. Tool observations and padding
   do not advance GAE. Normalize advantages over the complete learner batch.
4. Update the actor once with DIS, then publish its successful update.

Select the critic loss with ``critic.loss_type``: ``sao`` uses the SAO loss, while
``ppo`` reuses synchronous PPO's clipped Huber loss. To train with the latter, run:

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_sao critic.loss_type=ppo

The first critic training forward pass supplies detached value predictions as
the clipping reference for every update of that batch, without a separate inference
pass. Both losses use fixed Monte Carlo targets; actor GAE uses values inferred
after all critic updates. The Huber path keeps the synchronous
loss's ``huber_delta=10000`` and ``algorithm.value_cliprange``. Use
``train_qwen2.5_sao_split`` in the command for the split layout.

DIS uses the current-to-behavior ratio from the recorded rollout logprobs. Only
tokens with ``0.7 < ratio < 6.0`` contribute a policy gradient. The ratio is detached
in the score-function loss ``-mean(mask * ratio * advantage * logprob)``; the mean's
denominator includes all action tokens. This interprets the paper's importance
weight as a constant for differentiation, avoiding an extra derivative of
``ratio * logprob``. No old-policy inference pass is needed.

``algorithm.n_minibatches=1`` means one actor optimizer step on all 1024 trajectories.
``critic.update_epochs`` accepts any positive integer and controls the number of
critic optimizer steps, each using the same full batch. The supplied recipe uses
``1``; set ``critic.update_epochs=2`` to reproduce SAO's two-update schedule.
Setting ``n_minibatches=2`` would divide the batch and is a different update schedule.
Packing and gradient-accumulation microbatches do not add optimizer steps.
The actor learning rate is ``1e-6``; the critic uses ``5e-6`` and a 10-optimizer-step
warmup, covering 10 learner batches with one update or five with two updates.
Fractional warmup is disabled explicitly.

The dense Qwen2.5 recipe trains all value-model parameters and uses no value
pretraining corpus. It keeps the token-level method that outperforms the turn-level
ablations in Appendix A; step-average and last-token ablations are not implemented.
Search-R1 uses its existing two-turn, 7096-token limits and terminal reward, rather
than the paper's Python/coding environments and 128k context.

Checkpoints include both optimizers, buffered trajectories with their behavior
probabilities, and prompt ownership. Set ``runner.resume_dir`` to a complete
``checkpoints/global_step_N`` directory to resume. Completed buffered data is
restored exactly; unfinished requests restart from their original prompts after a
process restart. Live pause/resume during weight publication retains their prefixes.
Keep the original actor/critic conversion caches available: existing Megatron
workers initialize from them before restoring checkpoints. When changing the
experiment name, point both ``megatron.ckpt_convertor.save_path`` fields to those
original directories.

Both PPO runners advance the ``Global Step`` progress bar once per learner batch.
The synchronous and asynchronous recipes both set ``algorithm.n_minibatches=1``,
so their common TensorBoard metrics use matching step indices. Each SAO learner
batch contains ``critic.update_epochs`` full-batch critic updates; their individual
metrics use the critic optimizer-call index under ``critic_update/``.
``async_trace.jsonl`` records consumed sample IDs, weight versions, successful
optimizer counts, retractions, and tokens generated during training.

To verify real retraction and prefix retention, run:

.. code-block:: bash

   PYTHONPATH="$PWD:$PWD/examples:/opt/venv/reason/Megatron-LM" python \
     tests/e2e_tests/agent/searchr1/verify_async_engine.py \
     --config-path "$PWD/examples/agent/searchr1/config" \
     --config-name train_qwen2.5_sao \
     runner.experiment_name=searchr1-engine-check

The test publishes weights three times during generation and checks exact prefix
IDs, logprobs, and token versions after completion.

For the controlled four-run comparison, use:

.. code-block:: bash

   python tests/e2e_tests/agent/searchr1/run_ppo_benchmarks.py \
     --steps 200 --name searchr1-ppo-bsz1024-200

This runs sync/async training in both layouts with batch size 1024, matching model,
data, data/training seeds, sampling, packing, learning rates, role offloading, and checkpoint cadence. SAO changes
DIS, adaptive GAE, value loss, and critic update frequency. The launcher allows
multiple dataset epochs to reach the requested step count. ``benchmark.json``
measures initial buffer fill, training, publication, and checkpoint time, including
terminal weight publication for both runners; model initialization is recorded
separately. All six actor/critic offload flags are enabled in both layouts for memory
headroom. Rewards remain in each run's TensorBoard directory.

After all four runs finish, generate the reward curves and elapsed-time comparison:

.. code-block:: bash

   python tests/e2e_tests/agent/searchr1/plot_ppo_benchmarks.py \
     --steps 200 --name searchr1-ppo-bsz1024-200

The reporting script requires Matplotlib, NumPy, and TensorBoard. It writes plots,
``report.md``, and ``results.json`` under the comparison's log directory, and checks
reward improvement, elapsed speed, optimizer counts, and async sample ownership.
To compare only split placement, add ``--modes sync-split async-split`` to the
launcher and ``--layouts split`` to the reporting command. The report reads the
batch size from each run and requires matching values across the comparison.

Checkpoint Export and Evaluation
--------------------------------

Checkpoint export depends on the training backend.

For Megatron, run the following commands to convert a checkpoint into a
HuggingFace model:

.. code-block:: bash

   CKPT_PATH_MG={your_output_dir}/{exp_name}/checkpoints/global_step_xxx/actor
   CKPT_PATH_HF={path/to/save/huggingface/model}
   CKPT_PATH_ORIGINAL_HF={path/to/model/Qwen2.5-3B-Instruct}
   CKPT_PATH_MF="${CKPT_PATH_HF}_middle_file"

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
       --load-path "${CKPT_PATH_MG}" \
       --save-path "${CKPT_PATH_MF}" \
       --model qwen_2.5_3b \
       --tp-size 1 --ep-size 1 --pp-size 1 \
       --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
       --te-extra-state-check-none true --use-gpu-num 0 --process-num 16

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
       --load-path "${CKPT_PATH_MF}" \
       --save-path "${CKPT_PATH_HF}" \
       --model qwen_2.5_3b \
       --use-gpu-num 0 --process-num 16

   rm -rf "${CKPT_PATH_MF}"
   rm -f "${CKPT_PATH_HF}"/*.done
   shopt -s extglob
   cp "${CKPT_PATH_ORIGINAL_HF}"/!(*model.safetensors.index.json) "${CKPT_PATH_HF}"

For FSDP, use the built-in FSDP converter rather than the Megatron
commands above. Point it at the saved FSDP checkpoint and the matching resolved
training configuration, then write the exported HuggingFace model to a new
directory:

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
       convertor.train_config_path=/path/to/config.yaml \
       convertor.ckpt_path=/path/to/checkpoints/global_step_xxx/actor/model_state_dict/full_weights.pt \
       convertor.save_path=/path/to/hf_model

`config.yaml` is written with the training run. The converter can also locate
it automatically when it is stored near `convertor.ckpt_path`. Use the exported
HuggingFace model for standalone evaluation.

Fill the converted HuggingFace model path into
`examples/agent/searchr1/config/eval_qwen2.5.yaml`:

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/eval/model
       model_type: qwen2.5

Modify the evaluation dataset path:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/eval.jsonl"]
     val_data_paths: ["/path/to/eval.jsonl"]

Run `bash examples/agent/searchr1/run_eval.sh` to start evaluation.

Visualization and Results
-------------------------

The following shows the reward curves and training time curves.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png" style="width: 100%;"/>
       <p><em>Qwen2.5-3B-Instruct in RLinf</em></p>
     </div>
   </div>

Compared to the original performance (133s per step after response length stabilizes), we achieved a 55% speedup while maintaining consistent reward curves and evaluation results.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1_orig_impl_time.png" style="width: 35%;"/>
       <p><em>Qwen2.5-3B-Instruct in original implementation at PeterGriffinJin/Search-R1</em></p>
     </div>
   </div>

References
----------

search-r1: https://github.com/PeterGriffinJin/Search-R1

Search-R1 & veRL-SGLang:
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md

Asearcher: https://github.com/inclusionAI/ASearcher
