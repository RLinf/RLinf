AgentLightning 的强化学习训练（calc_x）
============================================

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

训练
----

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

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn

训练曲线
--------

以下为一次 ``calc_x`` 训练运行的指标曲线示例（具体曲线会因配置与随机种子而有所不同）：

.. figure:: https://github.com/RLinf/misc/raw/main/pic/agentlightning_calcx.png
   :width: 90%
   :align: center
   :alt: AgentLightning calc_x 训练曲线

   AgentLightning ``calc_x`` 训练曲线

测试
----

HF 评测时在对应的 ``*_eval.yaml`` 里设置 ``rollout.model.model_path``。例如：

.. code-block:: bash

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn_eval
   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory_eval



