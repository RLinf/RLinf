Rstar2的强化学习训练
=======================

结合工具调用的Multi-turn
RL被证明能够将大语言模型（LLM）的交互边界扩展到真实世界。本文档介绍了如何在
RLinf 框架下复现论文\ `rStar2-Agent: Agentic Reasoning Technical Report <https://arxiv.org/abs/2508.20722>`__\ 的实验，使用强化学习（RL）来训练大语言模型（LLM）通过调用代码运行工具回答问题。

环境
----

RLinf环境
~~~~~~~~~

RLinf 环境配置参照 `RLinf
Installation <https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html>`__

Code judge运行环境
~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用Rstar2示例中的code judge工具，安装过程参考\ `Rstar2 &
veRL-SGLang <https://github.com/volcengine/verl/blob/c12e3cbce8dceb70e9c9b16252bfd5675ec3129c/recipe/rstar2_agent/README.md>`__\ 

.. code-block:: bash
   cd toolkits/rstar2

   #install code judge
   sudo apt-get update -y && sudo apt-get install redis -y
   git clone https://github.com/0xWJ/code-judge
   pip install -r code-judge/requirements.txt
   pip install -e code-judge

   # install rstar2_agent requirements
   pip install -r requirements.txt

   cd ../..

Reward计算工具
~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用Math-Verify辅助进行reward计算，需通过pip安装
.. code-block:: bash

   pip install math-verify


在8*H100上训练
--------------

通过examples/rstar2/data_process/process_train_dataset.py下载训练集，并将路径写入examples/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/train.jsonl"]
     val_data_paths: ["/path/to/train.jsonl"]

修改examples/rstar2/config/rstar2-qwen2.5-7b-megatron.yaml中rollout.model.model_path的路径

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.5
     model:
       model_path: /path/to/model/Qwen2.5-7B-Instruct
       model_type: qwen2.5

由于down sample逻辑不适配目前inference逻辑，recompute_logprobs应当设置为False

.. code-block:: yaml

   algorithm:
      ……
      recompute_logprobs: False
      shuffle_rollout: False

运行examples/rstar2/run_rstar2.sh启动训练。


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

References
----------

Rstar2 & veRL-SGLang:
veRL-SGLang:
https://github.com/volcengine/verl/pull/3397

