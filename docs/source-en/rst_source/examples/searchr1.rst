Reinforcement Learning Training of Search-R1
===========================================

Multi-turn RL with tool calls has been proven to extend the interaction boundary of large language models (LLMs) to the real world.  
This document describes how to reproduce the experiments from  
`Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/abs/2503.09516>`__  
under the RLinf framework, using reinforcement learning (RL) to train LLMs to answer questions by invoking search tools.

Environment
-----------

RLinf Environment
~~~~~~~~~~~~~~~~~

RLinf environment setup follows:  
`RLinf Installation <https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html>`__

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

The local retrieval files are large; prepare sufficient disk space.  
Download size is around 60–70GB, and about 132GB after decompression.

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/searchr1/download.py --save_path $save_path
   cat $save_path/part_* > $save_path/e5_Flat.index
   gzip -d $save_path/wiki-18.jsonl.gz

Download the `flat e5 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding model from HuggingFace,  
and write the downloaded wiki file paths into `examples/searchr1/launch_local_server.sh`:

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

Training on 8×H100
------------------

Download the `training dataset <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__ from HuggingFace  
and write its path into `examples/searchr1/config/qwen2.5-3b-tool-1node.yaml`:

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8

     model_dir: /path/to/model/Qwen2.5-3B-Instruct
     model_arch: qwen2.5
     precision: ${actor.model.precision}

Modify `rollout.model.model_path` in `qwen2.5-3b-tool-1node.yaml`:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/train.jsonl"]
     val_data_paths: ["/path/to/train.jsonl"]

Run `examples/searchr1/run_main_searchr1_single.sh` to start training.

Evaluation
----------

Run `toolkits/ckpt_convertor/mg2hf_3b.sh` to convert a Megatron checkpoint into a HuggingFace model:

.. code-block:: bash

   sh toolkits/ckpt_convertor/mg2hf_3b.sh {your_output_dir}/{exp_name}/checkpoints/global_step_xxx/actor {path/to/save/huggingface/model} {path/to/model/Qwen2.5-3B-Instruct}

Fill the converted HuggingFace model path into  
`examples/searchr1/config/qwen2.5-3b-tool-1node-eval.yaml`:

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8

     model_dir: /path/to/eval/model
     model_arch: qwen2.5
     precision: ${actor.model.precision}

Modify the evaluation dataset path:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/eval.jsonl"]
     val_data_paths: ["/path/to/eval.jsonl"]

Run `examples/searchr1/run_main_searchr1_single_eval.sh` to start evaluation.

References
----------

search-r1: https://github.com/PeterGriffinJin/Search-R1

Search-R1 & veRL-SGLang:  
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md
