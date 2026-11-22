GLM-4.7-Flash 的 Search-R1 训练
========================================

使用 ``examples/agent/searchr1/config/train_glm4.7_flash.yaml`` 训练 GLM-4.7-Flash（激活参数约 3B 的 MoE 模型）通过调用搜索工具回答问题。请先按 :doc:`RLinf 安装指南 </rst_source/start/installation>` 与 :doc:`本地 Wiki 检索服务 <retriever>` 完成环境搭建并启动 Qdrant 检索服务。

概述
----------------------------------------

使用本配方在单机 8 卡上以 Megatron-Bridge actor 和 SGLang rollout 训练 GLM-4.7-Flash。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GLM-4.7-Flash（MoE，激活参数约 3B）

   .. grid-item-card:: 算法
      :text-align: center

      带搜索工具调用的多轮强化学习

   .. grid-item-card:: 工具
      :text-align: center

      Qdrant 本地 wiki server

   .. grid-item-card:: 硬件
      :text-align: center

      参考运行使用 8×A100-80G

数据
----------------------------------------

GLM 需要 ``prompt`` 字段为消息列表（``[{"role": "system", ...}, {"role": "user", ...}]``）的样本，由 GLM 自己的 chat template 渲染（``data.apply_chat_template: True``）。HuggingFace 上发布的 ``RLinf/Search-R1-Data`` 是按 Qwen 的 chat template 预先渲染好的字符串，不能直接用于 GLM：GLM 的词表里没有 ``<|im_start|>``，Qwen 的对话标记会被当作普通文本输入模型，而且不会报错。请先把 prompt 转换为消息列表，其余字段（``solutions`` 等）保持不变。

参考运行关闭了思考模式（见下文），因此 user 消息里还去掉了 Search-R1 指令中的 "You must conduct reasoning inside <think> and </think> first every time you get new information." 这一句。

一键上手
----------------------------------------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。训练前先按 :doc:`本地 Wiki 检索服务 <retriever>` 启动 Qdrant 检索服务（检索服务可与训练共用 GPU，请用 ``CUDA_VISIBLE_DEVICES=0`` 启动，约占 1 GB 显存）。

.. tab-set::

   .. tab-item:: 单机（8×A100）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载模型与数据集（按上文将 ``Search-R1-Data`` 的 prompt 转换为消息列表）：

      .. code-block:: bash

         hf download zai-org/GLM-4.7-Flash --local-dir ./GLM-4.7-Flash
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. 启动训练**

      在 ``examples/agent/searchr1/config/train_glm4.7_flash.yaml`` 中填写数据路径、模型路径和检索服务地址：

      .. code-block:: yaml

         data:
           train_data_paths: ["/path/to/data/searchr1_train_glm.jsonl"]
         rollout:
           model:
             model_path: /path/to/model/GLM-4.7-Flash
         tools:
           search:
             server_addr: <server_ip>:8000

      运行以下命令开始训练：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_glm4.7_flash

   .. tab-item:: 多机（2×8 A100）

      **1. 安装环境**

      在两台节点上安装 RLinf，并按 :doc:`多机训练指南 </rst_source/guides/multi_node>` 搭建 Ray 集群（在 ``ray start`` 之前，每台机器设置互不相同的 ``RLINF_NODE_RANK``\ ，head 节点为 0，并激活 RLinf 环境）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      .. code-block:: bash

         hf download zai-org/GLM-4.7-Flash --local-dir ./GLM-4.7-Flash
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. 启动训练**

      在 ``examples/agent/searchr1/config/train_glm4.7_flash_2node_server.yaml`` 中填写相同路径字段（该配置默认 ``cluster.num_nodes: 2``，需要 2 个 8×A100-80G 节点；单机请使用上方“单机”标签页；搜索工具 worker 运行在 0 号节点，``tools.search.server_addr`` 只需从该节点可达），然后在 head 节点启动训练：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_glm4.7_flash_2node_server

配置与并行说明
----------------------------------------

与模型相关的设置在自带 YAML 中已经写好：

.. code-block:: yaml

   data:
     apply_chat_template: True
     apply_chat_template_kwargs:
       enable_thinking: false
       clear_thinking: false
   agentloop:
     agentic_tokenizer: "glm47"
   rollout:
     sglang:
       tool_call_parser: glm47
   actor:
     megatron:
       mbridge: True

- ``agentloop.agentic_tokenizer`` 选择 agent loop 使用的模型相关 tokenization（``default``、``qwen`` 或 ``glm47``，不设置时为 ``default``）。它负责定位推理块的结束位置，使 response 截断在回答中的第一个 ``</search>``，而不是推理过程中写出的 ``</search>``。
- ``rollout.sglang.tool_call_parser`` 必须与所选 tokenizer 类型绑定的 parser 一致，不一致时启动即报错。
- ``enable_thinking`` 传给 GLM 的 chat template，只影响第一轮的生成提示；之后的轮次直接把工具输出拼接到对话中。
- ``actor.megatron.mbridge: True`` 通过 Megatron-Bridge 在进程内加载 HuggingFace 权重，不需要离线把 HuggingFace 转换为 Megatron 格式。

8×A100-80G 上的并行方式（actor 与 rollout 共置）：

- Actor（Megatron）：张量并行 4、专家并行 8、数据并行 2。
- Rollout（SGLang）：一个引擎横跨 8 张卡，注意力数据并行 2、专家并行 8。
- ``rollout.max_running_requests: 512`` 会平分到两个注意力数据并行 rank 上（各 256）。``rollout.cuda_graph_max_bs`` 设为 256，不能小于每个 rank 上的并发数，否则 decode 会退出 CUDA graph，速度反而下降。

跨机 rollout 实例（``train_glm4.7_flash_2node_server.yaml``）有以下要求：

- ``rollout.sglang.backend_type: server``；其他后端在启动时会拒绝跨机布局。
- 每台机器承载相同数量的 rollout 卡，且 tp×pp 能被单机卡数整除（此处为 16 和 8）。
- SGLang 的 ``dp_size`` 大于 1 且跨机时，必须开启 DP attention（``enable_dp_attention: True``）。

两机配置的 rollout 实例开启了 DP attention（``dp_size: 4``，即 attention 张量并行为 4）和 16 路 EP；actor 使用 TP4、EP8、DP4，EP 组限定在单机内，MoE 的 all-to-all 不跨机。其他规模的集群按同样的约束调整 tp / dp / ep 与 ``component_placement``。

如果 NCCL 跨机传输（RoCE/IB）卡住，请确认两台机器上的 GPU 使用同一子网的网卡，例如限定 ``NCCL_IB_HCA``。两台机器都用满 8 卡、本地卡号一致时无需此设置。

如果想让同样的配方以 rollout 与训练重叠的方式运行（actor 与 rollout 分占两台机器），见 :doc:`Search-R1 异步训练 <async>`。

可视化与结果
----------------------------------------

下面是关闭思考模式时 GLM-4.7-Flash 的训练曲线：8×A100-80G，每步 512 个 prompt × 4 个采样，检索服务为 :doc:`本地 Wiki 检索服务 <retriever>` 中的 Qdrant。横轴是参数更新次数：每个 rollout 步更新 4 次（``algorithm.n_minibatches: 4``），因此 100 个 rollout 步对应横轴上约 400 次更新。100 个 rollout 步内，轨迹 reward 从 0.28 上升到 0.54（分别取前 5 步和后 5 步的均值），每条轨迹的平均轮数从 1.7 增加到 2.0，每个 rollout 步约 870 s。

.. raw:: html

   <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png" style="width: 100%;"/>
       <p><em>轨迹 reward</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_length.png" style="width: 100%;"/>
       <p><em>序列总长度</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_time.png" style="width: 100%;"/>
       <p><em>每个 rollout 步耗时（s）</em></p>
     </div>
     <div style="flex: 1 1 45%; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_turn.png" style="width: 100%;"/>
       <p><em>每条轨迹的轮数</em></p>
     </div>
   </div>
   <p style="text-align: center;"><em>GLM-4.7-Flash in RLinf</em></p>
