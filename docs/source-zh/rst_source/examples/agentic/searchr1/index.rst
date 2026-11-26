Search-R1 强化学习训练
======================

在 RLinf 中复现 `Search-R1 <https://arxiv.org/abs/2503.09516>`__ 的多轮搜索强化学习实验，训练大语言模型调用本地 Wiki-2018 检索工具回答问题。在 8×H100 上训练 Qwen2.5-3B 比原版实现快 55%，两节点异步训练可将 GLM-4.7-Flash 单步耗时减半。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-3B-Instruct · GLM-4.7-Flash

   .. grid-item-card:: 算法
      :text-align: center

      GRPO · PPO · 异步多轮 RL

   .. grid-item-card:: 工具
      :text-align: center

      Qdrant 或 FAISS 本地 Wiki-2018 检索服务

   .. grid-item-card:: 硬件
      :text-align: center

      8×H100 · 8×A100 · 2×8 A100

| **你将完成：** 启动本地 Wiki-2018 检索服务 → 安装并下载模型与数据 → 运行 ``run_train.sh`` → 观察 reward 与步耗时。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · :doc:`多机训练指南 </rst_source/guides/multi_node>`\ （两节点运行） · :doc:`异步 agentic RL </rst_source/guides/async_agentic>`\ （异步调优）。

.. _searchr1-setup:
.. _searchr1-retriever:
.. _searchr1-glm:
.. _searchr1-async:

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。启动任意训练或评测前，先按标签页 1 部署本地检索服务。

.. tab-set::

   .. tab-item:: 1. 本地检索服务
      :sync: searchr1-retriever

      Qdrant 与 FAISS 后端均使用 `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ 编码，在 ``8000`` 端口暴露 ``POST /retrieve`` 与 ``POST /access`` 接口。推荐使用预构建 Qdrant 方案：检索在 CPU 上运行，仅编码器占约 1 GB 显存。

      **1. 安装与下载（Qdrant，推荐）**

      .. code-block:: bash

         uv pip install qdrant-client==1.16.2
         hf download --repo-type dataset RLinf/Wiki-2018-Corpus --local-dir ./Wiki-2018-Corpus
         hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2

      **2. 启动 Qdrant 与检索服务**

      在 tmux 中启动 Qdrant（``cd ./Wiki-2018-Corpus/qdrant && ./qdrant``），更新 ``examples/agent/tools/search_local_server_qdrant/launch_local_server.sh`` 中的 ``WIKI2018_DIR``、``retriever_path`` 与 ``qdrant_url``\ （集合名 wiki_collection_m32_cef512，HNSW ef 256），再启动服务：

      .. code-block:: bash

         bash examples/agent/tools/search_local_server_qdrant/launch_local_server.sh

      尽量将检索服务部署在训练主机上以降低网络延迟；若在独立主机运行，用 ``hostname -I`` 获取 IP，并在启动时覆盖 ``tools.search.server_addr=<server_ip>:8000``。

      .. dropdown:: 自建 Qdrant 索引与 FAISS GPU（ASearcher）方案

         **自建 Qdrant 索引：** 从 `Qdrant Releases <https://github.com/qdrant/qdrant/releases>`__ 下载二进制并启动，配置并运行 ``examples/agent/tools/search_local_server_qdrant/build_index.sh``\ （HNSW 调优见 `Qdrant 优化文档 <https://qdrant.tech/documentation/guides/optimize/>`__）。

         **FAISS GPU（ASearcher）：** 创建包含 PyTorch 2.4.0、``faiss-gpu=1.8.0``、transformers、datasets、pyserini、uvicorn 和 fastapi 的 ``retriever`` conda 环境，在 ``examples/agent/tools/search_local_server_faiss/`` 下的两个脚本中填好路径后下载、建索引并启动服务：

         .. code-block:: bash

            python examples/agent/searchr1/download.py --save_path ./wiki2018
            hf download intfloat/e5-base-v2 --local-dir ./e5-base-v2
            bash examples/agent/tools/search_local_server_faiss/build_index.sh
            bash examples/agent/tools/search_local_server_faiss/launch_local_server.sh

         参考：`Search-R1 <https://github.com/PeterGriffinJin/Search-R1>`__、`ASearcher <https://github.com/inclusionAI/ASearcher>`__。

   .. tab-item:: 2. Qwen2.5-3B（GRPO / PPO / FSDP）
      :sync: searchr1-qwen

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载 `Qwen2.5-3B-Instruct <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>`__ 和 `RLinf/Search-R1-Data <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. 启动训练（Megatron GRPO）**

      运行 ``examples/agent/searchr1/config/train_qwen2.5.yaml``\ （使用 ``sampling_params.stop`` 时设置 ``rollout.detokenize=True``，并保持 ``rollout.return_logprobs`` 等于 ``not algorithm.recompute_logprobs``）：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5 \
           rollout.model.model_path=./Qwen2.5-3B-Instruct \
           'data.train_data_paths=["./Search-R1-Data/train.jsonl"]'

      .. dropdown:: PPO、FSDP GRPO 及 checkpoint 导出与评测

         **Megatron PPO**\ （``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml``）：在每个 ``global_step_xxx`` 下同时保存 ``actor`` 和 ``critic``，通过 ``runner.resume_dir`` 一并恢复。

         **FSDP GRPO**\ （``examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml``）：启用动态 rollout batch、``actor.training_backend: fsdp`` 与 ``actor.pack_traj: True``，打包同一轨迹的多轮样本以避免重复计算共享前缀。

         .. code-block:: bash

            bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo \
              rollout.model.model_path=./Qwen2.5-3B-Instruct 'data.train_data_paths=["./Search-R1-Data/train.jsonl"]'
            bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp \
              rollout.model.model_path=./Qwen2.5-3B-Instruct 'data.train_data_paths=["./Search-R1-Data/train.jsonl"]'

         **导出 Hugging Face checkpoint 并评测**\ （评测配置为 ``examples/agent/searchr1/config/eval_qwen2.5.yaml``）：

         .. code-block:: bash

            # Megatron（转换 actor 目录；完成后从原始模型目录复制 tokenizer JSON）
            python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
              --load-path "${CKPT_PATH_MG}" --save-path "${CKPT_PATH_MF}" --model qwen_2.5_3b \
              --tp-size 1 --ep-size 1 --pp-size 1 --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
              --te-extra-state-check-none true --use-gpu-num 0 --process-num 16
            python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
              --load-path "${CKPT_PATH_MF}" --save-path "${CKPT_PATH_HF}" --model qwen_2.5_3b --use-gpu-num 0 --process-num 16
            # FSDP
            python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
              convertor.train_config_path=/path/to/config.yaml \
              convertor.ckpt_path=/path/to/checkpoints/global_step_xxx/actor/model_state_dict/full_weights.pt \
              convertor.save_path=/path/to/hf_model
            # 评测导出的 HF 模型
            bash examples/agent/searchr1/run_eval.sh eval_qwen2.5 \
              rollout.model.model_path=/path/to/hf_model 'data.val_data_paths=["./Search-R1-Data/eval.jsonl"]'

   .. tab-item:: 3. GLM-4.7-Flash
      :sync: searchr1-glm

      GLM-4.7-Flash（MoE，激活参数约 3B）通过 GLM chat template 渲染消息列表（``data.apply_chat_template: True``）。请先将 Search-R1-Data 中按 Qwen 模板预渲染的 ``prompt`` 转为 ``[{"role": "system", ...}, {"role": "user", ...}]`` 格式（保留 ``solutions`` 等字段；关闭思考模式时去掉 ``<think>`` 提示句）。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      .. code-block:: bash

         hf download zai-org/GLM-4.7-Flash --local-dir ./GLM-4.7-Flash
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. 启动训练（单机 8×A100-80G）**

      运行 ``examples/agent/searchr1/config/train_glm4.7_flash.yaml``\ （``actor.megatron.mbridge: True`` 直接在进程内加载 Hugging Face 权重）：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_glm4.7_flash \
           rollout.model.model_path=./GLM-4.7-Flash \
           'data.train_data_paths=["./Search-R1-Data/searchr1_train_glm.jsonl"]' \
           tools.search.server_addr=localhost:8000

      .. dropdown:: 两节点 server 模式（2×8 A100）与并行说明

         按 :doc:`多机训练指南 </rst_source/guides/multi_node>` 搭建 2 个节点的 Ray 集群（每台节点设置不同的 ``RLINF_NODE_RANK``），在 head 节点启动 ``examples/agent/searchr1/config/train_glm4.7_flash_2node_server.yaml``\ （``cluster.num_nodes: 2``，搜索工具运行在 0 号节点）：

         .. code-block:: bash

            bash examples/agent/searchr1/run_train.sh train_glm4.7_flash_2node_server \
              rollout.model.model_path=./GLM-4.7-Flash \
              'data.train_data_paths=["./Search-R1-Data/searchr1_train_glm.jsonl"]' \
              tools.search.server_addr=localhost:8000

         **Tokenizer 与 parser：** ``agentloop.agentic_tokenizer: "glm47"`` 须与 ``rollout.sglang.tool_call_parser: glm47`` 保持一致。

         **单机并行：** Megatron actor 采用 TP4 / EP8 / DP2，SGLang rollout 采用 ADP2 / EP8；``rollout.cuda_graph_max_bs``\ （256）不能小于单 rank 并发数（512 / 2）。

         **两机并行：** 跨机 rollout 须设置 ``rollout.sglang.backend_type: server``，每机 rollout 卡数一致且 ``tp×pp`` 能被单机卡数整除，``dp_size > 1`` 时开启 ``enable_dp_attention: True``\ （本配置为 DP4 / EP16 rollout 与 TP4 / EP8 / DP4 actor，EP 保持在单机内）。若 RoCE/IB 跨机卡住，按子网限定 ``NCCL_IB_HCA``。

   .. tab-item:: 4. 异步训练
      :sync: searchr1-async

      在 **2 个 8×A100-80G 节点**\ （``cluster.num_nodes: 2``）上重叠运行 rollout 与训练：node 0 运行 Megatron actor，node 1 运行 SGLang server rollout，规则奖励在 agent loop 内直接计算。在两台机器完成 **GLM-4.7-Flash** 标签页中的安装与下载，在 node 0 启动 Qdrant 检索服务，并按 :doc:`多机训练指南 </rst_source/guides/multi_node>` 搭建 Ray 集群。

      **启动训练（2×8 A100 异步）**

      在 head 节点启动 ``examples/agent/searchr1/config/train_glm4.7_flash_2node_async.yaml``\ （已开启 ``runner.async_mode: True``、``rollout.sglang.backend_type: server`` 与 ``agentloop.task_name``）：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_glm4.7_flash_2node_async \
           rollout.model.model_path=./GLM-4.7-Flash \
           'data.train_data_paths=["./Search-R1-Data/searchr1_train_glm.jsonl"]' \
           tools.search.server_addr=localhost:8000

      ``max_inflight_groups``、``max_buffered_groups``、``max_actor_accept_staleness`` 与 ``rollout.pause_mode``\ （retract、in_place、drain）的调优原理见 :doc:`异步 Agent RL </rst_source/guides/async_agentic>`。如需启用 Session Server，追加 ``session_server.enable=true``\ （要求见 :doc:`Session Server </rst_source/guides/session_server>`\ ）。

      .. dropdown:: 异步指标、断点续训与限制

         **核心指标：** 关注 ``rollout/buffer/inflight_groups``、``rollout/buffer/size``、``rollout/buffer/admission_wait_*``\ （在途上限与缓冲满等待比例）、``rollout/data_staleness/mean`` 与 ``rollout/buffer/dropped_groups``。

         **断点续训：** checkpoint 将 dataloader 进度保存在 ``data/data.pt``，恢复时跳过存档前已投放的 prompt（Qwen 默认配置实测跳过 192 个：在途与缓冲 128 个加预取 64 个）。

         **限制：** 需配合 Megatron actor、多轮 agent loop、SGLang server 后端与分离式 placement 使用；直接使用 rollout logprob（``algorithm.recompute_logprobs: False``）；开启 ``reject_equal_reward`` 时需监控 ``dropped_groups``\ （``max_dropped_groups_per_step`` 默认为 10 × batch size）。

可视化与结果
------------

运行 ``tensorboard --logdir ./logs`` 查看 reward 与步耗时曲线，W&B / SwanLab 配置与指标定义见 :doc:`训练指标 </rst_source/reference/metrics>`。

.. list-table:: **Search-R1 参考实验结果**
   :header-rows: 1
   :widths: 24 16 14 14 32

   * - 配置
     - 硬件
     - 平均步时
     - 吞吐
     - 主要结果
   * - Qwen2.5-3B（GRPO）
     - 8×H100
     - 提速 55%
     - —
     - reward 曲线与原版（133 s/步）一致；8×A800（batch 1024）上 PPO 与 GRPO 表现相当。
   * - GLM-4.7-Flash（同步）
     - 8×A100-80G
     - 约 870 s
     - —
     - 100 步（400 次更新，512×4）内 reward 从 0.28 升至 0.54，平均轮数 1.7 → 2.0。
   * - GLM-4.7-Flash（同步）
     - 2×8 A100-80G
     - 278.6 s
     - 1159 tok/s
     - reward 0.25–0.47，``ratio_abs`` 0.052–0.062（第 2–19 步，128×4）。
   * - GLM-4.7-Flash（异步）
     - 2×8 A100-80G
     - **139.0 s**
     - **2024 tok/s**
     - **吞吐提升 1.75 倍**，staleness 1.3–1.4，reward 0.26–0.42，``ratio_abs`` 0.053–0.063。

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png" style="width: 100%;"/><p><em>Qwen2.5-3B-Instruct 在 RLinf 中的 reward 与步时</em></p></div>
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_ppo_grpo_reward.png" style="width: 100%; max-width: 410px;"/><p><em>Qwen2.5-3B GRPO（橙）与 PPO（蓝）对比（8×A800）</em></p></div>
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png" style="width: 100%;"/><p><em>GLM-4.7-Flash 轨迹 reward（8×A100，400 次更新）</em></p></div>
     <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_step_time.png" style="width: 100%;"/><p><em>GLM-4.7-Flash 同步与异步步时对比（2×8 A100）</em></p></div>
   </div>

.. dropdown:: 更多训练曲线与异步上限实验图

   .. raw:: html

      <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1_orig_impl_time.png" style="width: 100%;"/><p><em>原版 Search-R1 单步耗时</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_length.png" style="width: 100%;"/><p><em>GLM-4.7-Flash 序列总长度</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_time.png" style="width: 100%;"/><p><em>GLM-4.7-Flash 每 rollout 步耗时</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_turn.png" style="width: 100%;"/><p><em>GLM-4.7-Flash 每条轨迹轮数</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_walltime.png" style="width: 100%;"/><p><em>异步 reward 随墙钟时间变化</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_reward_step.png" style="width: 100%;"/><p><em>异步每步轨迹 reward</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_ratio_abs.png" style="width: 100%;"/><p><em>异步训推 logprob 差（ratio_abs）</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_step_time.png" style="width: 100%;"/><p><em>在途上限 77 对 160：步时</em></p></div>
        <div style="flex: 1 1 30%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_cap_staleness.png" style="width: 100%;"/><p><em>在途上限 77 对 160：staleness</em></p></div>
        <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_step_time.png" style="width: 100%;"/><p><em>Qwen2.5-1.5B 缓冲上限 1×B 对 2×B：步时（11 s 对 3.94 s）</em></p></div>
        <div style="flex: 1 1 45%; text-align: center;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_async_buffer_size.png" style="width: 100%;"/><p><em>Qwen2.5-1.5B 缓冲上限 1×B 对 2×B：缓冲大小</em></p></div>
      </div>
