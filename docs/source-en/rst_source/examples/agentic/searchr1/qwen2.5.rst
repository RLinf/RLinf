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

      GRPO or PPO

   .. grid-item-card:: Tools
      :text-align: center

      FAISS or Qdrant local wiki server

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run on 8×H100

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run
``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt
images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.
Start the local wiki retrieval server from :doc:`index` before launching training.

.. tab-set::

   .. tab-item:: Megatron GRPO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the base model and the `training dataset <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__:

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. Launch**

      Update ``data.train_data_paths`` and ``rollout.model.model_path`` in ``examples/agent/searchr1/config/train_qwen2.5.yaml`` (set ``rollout.detokenize: True`` if using ``sampling_params.stop``, and keep ``rollout.return_logprobs`` equal to ``not algorithm.recompute_logprobs``):

      .. code-block:: yaml

         data:
           train_data_paths: ["/path/to/train.jsonl"]

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-3B-Instruct

      Start training:

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5

   .. tab-item:: Megatron PPO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. Launch**

      Update ``data.train_data_paths`` and ``rollout.model.model_path`` in ``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` (PPO uses rollout logprobs directly, keeping ``rollout.return_logprobs`` equal to ``not algorithm.recompute_logprobs``):

      .. code-block:: yaml

         data:
           train_data_paths: ["/path/to/train.jsonl"]

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-3B-Instruct

      Start training:

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo

   .. tab-item:: MA-FSDP GRPO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. Launch**

      Update ``data.train_data_paths`` and ``rollout.model.model_path`` in ``examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml`` (with ``agentloop.is_dynamic_rollout_batch: True`` and ``actor.training_backend: fsdp``, the entrypoint uses ``MAFSDPActor`` and keeps ``actor.pack_traj: True`` to pack compatible turns from the same trajectory and avoid recomputing shared prefixes):

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp

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
   cp "${CKPT_PATH_ORIGINAL_HF}"/!(*model.safetensors.index).json "${CKPT_PATH_HF}"

A PPO run saves an ``actor`` and a ``critic`` directory under each
``global_step_xxx``. Export the ``actor`` directory with the commands above.
To resume training, set ``runner.resume_dir`` to the ``global_step_xxx``
directory; RLinf restores both the actor and the critic.

For FSDP, use the built-in FSDP converter rather than the Megatron
commands above. Point it at the saved FSDP checkpoint and the matching resolved
training configuration, then write the exported HuggingFace model to a new
directory:

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
       convertor.train_config_path=/path/to/config.yaml \
       convertor.ckpt_path=/path/to/checkpoints/global_step_xxx/actor/model_state_dict/full_weights.pt \
       convertor.save_path=/path/to/hf_model

``config.yaml`` is written with the training run. The converter can also locate
it automatically when it is stored near ``convertor.ckpt_path``. Use the exported
HuggingFace model for standalone evaluation.

Fill the converted HuggingFace model path into
``examples/agent/searchr1/config/eval_qwen2.5.yaml``:

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
     val_data_paths: ["/path/to/eval.jsonl"]

Run ``bash examples/agent/searchr1/run_eval.sh`` to start evaluation.

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

The following compares GRPO and PPO training rewards:

.. raw:: html

   <div style="text-align: center;">
     <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_ppo_grpo_reward.png" alt="Qwen2.5-3B Search-R1 reward: GRPO in orange and PPO in blue" style="width: 100%; max-width: 410px;"/>
     <p><em>GRPO (orange) and PPO (blue) on 8×A800 80 GB, both with effective batch size 1024. TensorBoard smoothing: 0.6.</em></p>
   </div>
