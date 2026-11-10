PPO 训练
========

使用 PPO 和独立 critic 训练 Qwen2.5-3B-Instruct 完成 Search-R1 搜索问答。先按照 :doc:`index` 准备 RLinf 环境和本地 wiki 检索服务，再配置路径并启动训练。多轮 PPO 的通用训练流程与实现说明见 :doc:`/rst_source/reference/algorithms/ppo`。

概览
----

该配置训练模型在回答问题前检索 wiki 文本，每个完整 episode 使用一次最终答案 reward。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      PPO、GAE 与独立 critic

   .. grid-item-card:: 工具
      :text-align: center

      FAISS 或 Qdrant 本地 wiki 服务

   .. grid-item-card:: 硬件
      :text-align: center

      参考实验使用 8×A800 80 GB

在 8×A800 上训练
----------------

在 ``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` 中填写模型目录和 Search-R1 格式的训练 JSONL。`Search-R1 训练数据集 <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__ 使用这一格式：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
   data:
     train_data_paths: ["/path/to/train.jsonl"]
   tools:
     search:
       server_addr: 127.0.0.1:8000

将 ``tools.search.server_addr`` 指向 :doc:`index` 中启动的检索服务。

该配置直接使用 rollout logprobs，并保留响应预算内的全部生成 token，包括搜索结束标签之后生成的后缀：

.. code-block:: yaml

   agentloop:
     truncate_response: false
   algorithm:
     recompute_logprobs: false
   rollout:
     return_logprobs: true

设置 ``agentloop.truncate_response=true`` 可将后缀排除在训练和后续 prompt 之外。两种设置都会保持 token ID 与 logprob 对齐。

向启动脚本传入配置名：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo

默认 batch 包含 1024 个 prompt，每个 prompt 生成一条轨迹，使用一个 minibatch；每个 batch 分别更新一次 actor 和 critic。

可视化与结果
------------

在 TensorBoard 中查看 ``rollout/reward_scores_traj``，指标定义见 :doc:`/rst_source/reference/metrics`。下图比较 Qwen2.5-3B-Instruct 的 GRPO 和 PPO 训练 reward：

.. raw:: html

   <div style="text-align: center;">
     <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_ppo_grpo_reward.png" alt="Search-R1 训练 reward：橙色为 GRPO，蓝色为 PPO" style="width: 100%; max-width: 410px;"/>
     <p><em>橙色：GRPO；蓝色：PPO。横轴为训练 step，纵轴为轨迹 reward。TensorBoard 平滑系数为 0.6。</em></p>
   </div>

两组实验均使用 8 张 A800 80 GB、20,000 条 WideSeek 训练数据、wiki 检索服务、200 个训练 step 和一个 minibatch，有效 batch size 均为 1024。GRPO 使用 ``256 × 4`` 个 prompt/轨迹，PPO 使用 ``1024 × 1``。两组实验的 seed 均为 1234，actor 学习率均为 ``1e-6`` 且无 warmup，最多两轮，序列长度为 7096，temperature 为 1.0，KL 系数为零，使用 SGLang server 和 ``agentloop.truncate_response=false``。
