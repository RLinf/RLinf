Search-R1 Training for Qwen2.5-3B-Instruct
==========================================

Train Qwen2.5-3B-Instruct to answer questions by calling a search tool. Set up
the RLinf environment and the local wiki retrieval server first, as described
in :doc:`index`.

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

The response is truncated at `</search>` in token space, so it stays a prefix of
the generated tokens and the rollout log-probabilities remain aligned.
`recompute_logprobs` can be True or False; the provided configurations use True.
`rollout.return_logprobs` must equal `not algorithm.recompute_logprobs`, which is
checked at startup.

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
