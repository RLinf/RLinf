Search-R1的强化学习训练
=======================

结合工具调用的Multi-turn
RL被证明能够将大语言模型（LLM）的交互边界扩展到真实世界。本文档介绍了如何在
RLinf 框架下复现论文\ `Search-R1: Training LLMs to Reason and Leverage
Search Engines with Reinforcement
Learning <https://arxiv.org/abs/2503.09516>`__\ 中的实验，使用强化学习（RL）来训练大语言模型（LLM）通过调用搜索工具回答问题。

环境
----

RLinf环境
~~~~~~~~~

RLinf 环境配置参照 `RLinf
Installation <https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html>`__

Local Wiki Server运行环境
~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用search-R1示例中的local retrieve
server，通过conda安装faiss，详细文档见\ `SearchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__\ ，安装过程参考\ `Search-R1 &
veRL-SGLang <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__\ ，同样使用conda来配置环境

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever

   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub

   #  安装 GPU 版 faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   pip install uvicorn fastapi

Wiki配置文件
~~~~~~~~~~~~

我们使用Asearcher提供的本地检索文件，下载文件大约 50~60GB

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/searchr1/download.py --save_path $save_path

从huggingface上下载\ `flat e5 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding模型，并生成index

.. code-block:: bash

   bash examples/searchr1/build_index.sh

将之前下载好的wiki文件路径和index路径等写入examples/searchr1/launch_local_server.sh

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

运行launch_local_server.sh启动Local Wiki Server，等待直至输出server ip等信息，代表server启动完成

(Optional) 使用Qdrant作为Wiki Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们也支持使用 qdrant 作为 wiki 服务器。如果你不打算使用 qdrant，可以直接跳到 训练 部分。

下载 `qdrant <https://github.com/qdrant/qdrant/releases>`__ 并按照以下步骤构建 qdrant collection。

在 `examples/searchr1/qdrant_scripts/build_qdrant_coll.sh` 和 `examples/searchr1/qdrant_scripts/launch_local_server_qdrant.sh` 中，更新 `qdrant_path`、`WIKI2018_DIR` 和 `retriever_path` 的文件路径。

使用以下指令构建并启动 qdrant wiki 服务器：

.. code-block:: bash

   cd examples/searchr1/qdrant_scripts
   # 构建 qdrant collection
   bash ./build_qdrant_coll.sh
   # 启动 qdrant 服务器
   bash ./launch_local_server_qdrant.sh

Qdrant 默认使用 HNSW 图索引算法。关于 HNSW 图索引的优化,请参考 `Qdrant 文档 <https://qdrant.tech/documentation/guides/optimize/>`__。

在8*H100上训练
--------------

从huggingface上下载\ `训练集 <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
，并将路径写入examples/searchr1/config/qwen2.5-3b-tool-1node.yaml

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/train.jsonl"]
     val_data_paths: ["/path/to/train.jsonl"]

修改examples/searchr1/config/qwen2.5-3b-tool-1node.yaml中rollout.model.model_path的路径

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
       model_type: qwen2.5

如果使用sampling_params.stop来控制模型停止节省训练时间，detokenize应当设置为True

.. code-block:: yaml

   rollout:
      ……
      distributed_executor_backend: mp   # ray or mp
      disable_log_stats: False
      detokenize: True  

由于search-R1会re-tokenize模型输出，recompute_logprobs应当设置为True

.. code-block:: yaml

   algorithm:
      ……
      recompute_logprobs: True
      shuffle_rollout: False

运行examples/searchr1/run_main_searchr1_single.sh启动训练。

测试
----

运行以下命令将 Megatron checkpoint 转换为 HuggingFace model

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

将转换得到的huggingface
model路径填入examples/searchr1/config/qwen2.5-3b-tool-1node-eval.yaml

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/eval/model
       model_type: qwen2.5

修改测试数据集路径

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/eval.jsonl"]
     val_data_paths: ["/path/to/eval.jsonl"]

运行examples/searchr1/run_main_searchr1_single_eval.sh启动测试。

训练曲线
--------

下面展示 reward 曲线和训练时间曲线。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/searchr1.png" style="width: 100%;"/>
       <p><em>Qwen2.5-3B-Instruct in RLinf</em></p>
     </div>
   </div>

相较于原版性能( response length 稳定后，单 step 133s)，我们加速了 55%，同时 reward 曲线和 eval 结果保持一致。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/searchr1_orig_impl_time.png" style="width: 35%;"/>
       <p><em>Qwen2.5-3B-Instruct in original implementation at PeterGriffinJin/Search-R1</em></p>
     </div>
   </div>

References
----------

search-r1: https://github.com/PeterGriffinJin/Search-R1

Search-R1 &
veRL-SGLang:
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md

Asearcher: https://github.com/inclusionAI/ASearcher
