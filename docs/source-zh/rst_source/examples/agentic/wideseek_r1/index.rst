.. _wideseek-r1-example:

WideSeek-R1
===========

训练并评测 WideSeek-R1——基于共享大语言模型的多智能体强化学习（MARL）框架，由主智能体编排并行子智能体完成广域信息检索。在 WideSearch 基准上，WideSeek-R1-4B 达到 ``40.0%`` item F1，与单智能体 DeepSeek-R1-671B 相当，且随并行子智能体数量增加持续提升（详见 :doc:`论文页面 </rst_source/resources/publications/wideseek_r1>`、`项目主页 <https://wideseek-r1.github.io>`__、`arXiv <https://arxiv.org/abs/2602.04634>`__ 与 `示例代码 <https://github.com/RLinf/RLinf/tree/main/examples/agent/wideseek_r1>`__）。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3-4B · Qwen3 稠密模型 · WideSeek-R1-4B

   .. grid-item-card:: 算法
      :text-align: center

      混合多智能体强化学习（MARL）

   .. grid-item-card:: 工具
      :text-align: center

      离线 Qdrant · 在线 Serper / Jina

   .. grid-item-card:: 硬件
      :text-align: center

      8×H100 + 8×H100 评判（或单机本地评判）

| **你将完成：** 配置搜索后端与 LLM 评判服务 → 启动混合多智能体 RL 训练 → 在 WideSearch 或标准 QA 上评测。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · :doc:`多机训练指南 </rst_source/guides/multi_node>`\ （可选扩展）。

.. _wideseek-r1-setup:
.. _wideseek-r1-eval:

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像（含 Hopper+ 的 FA4 版本）见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__；``install.sh agentic`` 会按 GPU 架构自动选择 FA2 或 FA4。

.. tab-set::

   .. tab-item:: 1. 工具与评判服务
      :sync: wideseek-tools

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 配置搜索后端**

      训练与标准 QA 评测使用 **离线 Qdrant** 检索，WideSearch 评测使用 **在线 Serper + Jina** 网页搜索。

      **离线 Qdrant：** 共用 :doc:`Search-R1 </rst_source/examples/agentic/searchr1/index>` 中的预构建 Wiki-2018 检索服务（安装 ``qdrant-client==1.16.2``，下载 ``RLinf/Wiki-2018-Corpus`` 与 ``intfloat/e5-base-v2``，启动 Qdrant 后运行 ``examples/agent/tools/search_local_server_qdrant/launch_local_server.sh``，监听 8000 端口）。

      **在线 Serper / Jina：** 从 `Serper <https://serper.dev>`__ 与 `Jina AI <https://jina.ai>`__ 获取密钥并在运行前导出：

      .. code-block:: bash

         export SERPER_API_KEY=your_serper_api_key
         export JINA_API_KEY=your_jina_api_key

      **3. 启动 LLM 评判服务**

      用 SGLang 启动 `Qwen3-30B-A3B-Instruct-2507 <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__\ （默认端口 ``30000``，按卡数调整 ``--dp``）；也可跳过此步，直接使用内置本地评判配置（``train_qwen3_hybrid_local_judge`` / ``eval_qwen3_widesearch_local_judge``）：

      .. code-block:: bash

         hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir ./Qwen3-30B-A3B-Instruct-2507
         python3 -m sglang.launch_server --model-path ./Qwen3-30B-A3B-Instruct-2507 \
           --host 0.0.0.0 --log-level info --context-length 32768 --dp 8

      **4. 快速验证**

      检查工具后端（``--is_online true`` 或 ``false``）与外部评判服务连通性（将 ``127.0.0.1`` 替换为 ``hostname -I`` 查到的评判主机 IP）：

      .. code-block:: bash

         python rlinf/agents/wideseek_r1/tools.py --is_online false
         python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip 127.0.0.1

      .. dropdown:: 在线/离线 YAML 配置、内置本地评判器与多节点说明

         **在线配置**\ （``examples/agent/wideseek_r1/config/``）：设置 ``tools.online: True``、``tools.use_jina: True``、``tools.enable_cache: True`` 和 ``tools.cache_file: "./webpage_cache.json"``。

         **离线配置：** 设置 ``tools.online: False``；若检索服务部署在远程主机，设置 ``tools.search.server_addr: "HOST:8000"``\ （默认 ``localhost:8000``）。

         **内置本地评判器：** 设置 ``agentloop.use_local_judge: true`` 并配置 ``rollout_judge``\ （示例见 ``examples/agent/wideseek_r1/config/train_qwen3_hybrid_local_judge.yaml`` 与 ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch_local_judge.yaml``），无需单独启动外部评判服务。

         **多节点扩展：** 多智能体生成开销较大，可扩展至多个 8 卡节点，详见 :doc:`多机训练指南 </rst_source/guides/multi_node>`。

   .. tab-item:: 2. 训练
      :sync: wideseek-train

      确认离线 Qdrant 检索服务与 LLM 评判服务均已运行（单机 8 卡可直接改用 ``train_qwen3_hybrid_local_judge``）。

      **1. 下载模型与数据**

      下载 `Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>`__ 与 `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__\ （使用其中的 ``hybrid_20k.jsonl``；另有 `WideSeek-R1-Corpus <https://huggingface.co/datasets/RLinf/WideSeek-R1-Corpus>`__ 供查阅语料）：

      .. code-block:: bash

         hf download Qwen/Qwen3-4B --local-dir ./Qwen3-4B
         hf download --repo-type dataset RLinf/WideSeek-R1-train-data --local-dir ./WideSeek-R1-train-data

      **2. 启动训练**

      使用 ``examples/agent/wideseek_r1/config/train_qwen3_hybrid.yaml`` 启动混合多智能体 RL 训练（将 ``127.0.0.1`` 替换为评判主机 IP）：

      .. code-block:: bash

         bash examples/agent/wideseek_r1/run_train.sh train_qwen3_hybrid \
           rollout.model.model_path=./Qwen3-4B \
           data.train_data_paths=./WideSeek-R1-train-data/hybrid_20k.jsonl \
           agentloop.llm_ip=127.0.0.1

      .. dropdown:: 单一来源数据集与单智能体工作流

         **混合与单一来源数据：** ``data.is_hybrid: True`` 混合 WideSearch 风格与标准 QA 数据（``hybrid_20k.jsonl``）。若改用 width_20k 或 depth_20k 等单一来源数据集，将 ``data.is_hybrid`` 设为 ``False``，并匹配 ``data.is_markdown``\ （width_20k 为 True，depth_20k 为 False）。

         **工作流切换：** 通过 ``agentloop.workflow`` 选择 ``mas``\ （多智能体）或 ``sa``\ （单智能体，与 `ASearcher <https://github.com/inclusionAI/ASearcher>`__ 可比）。

   .. tab-item:: 3. 评测
      :sync: wideseek-eval

      .. note::

         `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__ 权重以 fp16 存储，但 ``config.json`` 声明为 ``bfloat16``。RLinf 自带配置已设置 ``rollout.model.precision: fp16``；在 RLinf 之外服务该权重时请显式传入 ``--dtype float16``，否则 WideSearch 上约损失 3 分 avg@4（55.9 对 59.0，见 `issue #1662 <https://github.com/RLinf/RLinf/issues/1662>`__）。

      **1. 下载模型与评测集**

      下载已发布的 checkpoint，并按需下载 `WideSeek-R1-test-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-test-data>`__\ （WideSearch）或 `ASearcher-test-data <https://huggingface.co/datasets/inclusionAI/ASearcher-test-data>`__\ （标准 QA）：

      .. code-block:: bash

         hf download RLinf/WideSeek-R1-4b --local-dir ./WideSeek-R1-4b
         hf download --repo-type dataset RLinf/WideSeek-R1-test-data --local-dir ./WideSeek-R1-test-data
         hf download --repo-type dataset inclusionAI/ASearcher-test-data --local-dir ./ASearcher-test-data

      **2. 启动评测**

      全量 WideSearch（200 条样本，``data.data_size=-1``）在 8 卡生成 + 8 卡评判上约需 7 小时；可先调小 ``data.data_size`` 快速验证流程。

      .. code-block:: bash

         # WideSearch（在线 Serper/Jina 工具，data.is_markdown=True）
         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_widesearch \
           rollout.model.model_path=./WideSeek-R1-4b \
           data.val_data_paths=./WideSeek-R1-test-data \
           agentloop.llm_ip=127.0.0.1

         # 标准 QA（离线 Qdrant 工具，data.is_markdown=False）
         bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_qa \
           rollout.model.model_path=./WideSeek-R1-4b \
           data.val_data_paths=./ASearcher-test-data \
           agentloop.llm_ip=127.0.0.1

      **3. WideSearch 打分**

      标准 QA 会直接将 LLM 评判分数写入 ``metric.json``。WideSearch 最终基准打分请使用 `WideSeek-R1-Eval <https://github.com/RLinf/WideSeek-R1-Eval>`__ 对保存的 ``responses/`` 目录评分。

      .. dropdown:: 输出文件、双引擎评测与工作流切换

         **输出文件**\ （位于 ``${runner.output_dir}/${runner.experiment_name}``）：``metric.json``\ （聚合指标与 QA 评判结果）、``allresult.json``\ （完整多轮交互日志）与 ``responses/``\ （最终回答）。

         **双引擎评测**\ （``examples/agent/wideseek_r1/config/eval_qwen3_qa_2eng.yaml``）：设置 ``rollout.use_fixed_worker: True`` 与 ``agentloop.fixed_role: worker``\ （或 planner），通过 ``rollout.model.model_path`` 和 ``rollout_fixed_worker.model.model_path`` 为两个角色分别指定模型。

         **工作流切换：** 通过 ``agentloop.workflow`` 选择 ``mas``\ （多智能体）或 ``sa``\ （单智能体）。

可视化与结果
------------

训练与评测产物写入 ``${runner.output_dir}/${runner.experiment_name}``；运行 ``tensorboard --logdir ./logs`` 查看指标，日志后端与指标说明见 :doc:`训练指标 </rst_source/reference/metrics>`。

.. list-table:: **WideSeek-R1 核心基准结果**
   :header-rows: 1
   :widths: 30 22 20 28

   * - 模型
     - 基准
     - 指标
     - 说明
   * - WideSeek-R1-4B（MARL）
     - WideSearch
     - **40.0% item F1**
     - 与单智能体 DeepSeek-R1-671B 相当，随并行子智能体数量增加持续提升。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wideseek_r1/scaling.png
   :align: center
   :width: 85%

   WideSeek-R1-4B 在 WideSearch 上随并行子智能体数量扩展的表现（完整消融实验见 :doc:`WideSeek-R1 论文页面 </rst_source/resources/publications/wideseek_r1>`）。
