AgentLightning 的强化学习训练（calc_x）
======================================

``calc_x`` 是 RLinf 中的 AgentLightning 示例，用于训练一个会做数学题的 agent。  
agent 会读取题目，生成推理过程与答案，并根据反馈做强化学习更新。

环境
----

RLinf 基础环境请参考 `RLinf Installation <https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html>`__。

安装本示例依赖：

.. code-block:: bash

   pip install "agentlightning==0.3.0" "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" "mcp-server-calculator"

硬件建议：

- 这个例子需要一个节点，至少有一个40GB的显卡。

数据准备
--------

下载并解压 ``calc_x`` 数据集（Google Drive），下载链接见 `这里 <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_。


----------------

进入示例目录：

.. code-block:: bash

   cd /path/to/rlinf/examples/agentlightning/calc_x

先修改 ``config/qwen2.5-1.5b-trajectory.yaml``：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-1.5B-Instruct

   data:
     train_data_paths: ["/path/to/train.parquet"]
     val_data_paths: ["/path/to/test.parquet"]

启动训练：

.. code-block:: bash

   bash run_calc_x.sh qwen2.5-1.5b-multiturn

测试
----

评测时可直接指定 HuggingFace 模型目录：

.. code-block:: bash

   bash run_calc_x.sh eval=true rollout.model.model_path=/path/to/eval/hf_model



