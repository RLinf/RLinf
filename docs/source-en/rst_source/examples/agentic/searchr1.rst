Reinforcement Learning Training of Search-R1
================================================

Multi-turn RL with tool calls extends the interaction boundary of large language models (LLMs) to the real world.
Reproduce the experiments from
`Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/abs/2503.09516>`__
in RLinf, training LLMs to answer questions by invoking search tools.

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

SearchR1 re-tokenizes the model output, so set ``recompute_logprobs`` to True.

.. code-block:: yaml

   algorithm:
      ……
      recompute_logprobs: True
      shuffle_rollout: False

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

All three commands use the same initial model, 2048 prompts with one trajectory
each (``algorithm.group_size: 1``), and two global minibatches. Only placement
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

Keep ``algorithm.recompute_logprobs: true``: SearchR1 re-tokenizes generated text
when preparing tool calls, so the actor must recompute old logprobs on the retained
tokens. Both placements with shared training GPUs retain actor weights between
inference and training. Tool observations stay outside policy/value losses, and
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
