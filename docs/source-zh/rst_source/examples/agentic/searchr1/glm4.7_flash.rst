GLM-4.7-Flash 的 Search-R1 训练
========================================

使用 `examples/agent/searchr1/config/train_glm4.7_flash.yaml` 训练 GLM-4.7-Flash（激活参数约 3B 的 MoE 模型）通过调用搜索工具回答问题。请先按 :doc:`index` 搭建 RLinf 环境和 Qdrant wiki 检索服务。

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

GLM 需要 `prompt` 字段为消息列表（`[{"role": "system", ...}, {"role": "user", ...}]`）的样本，
由 GLM 自己的 chat template 渲染（`data.apply_chat_template: True`）。
HuggingFace 上发布的 `RLinf/Search-R1-Data` 是按 Qwen 的 chat template 预先渲染好的字符串，
不能直接用于 GLM：GLM 的词表里没有 `<|im_start|>`，Qwen 的对话标记会被当作普通文本输入模型，而且不会报错。
请先把 prompt 转换为消息列表，其余字段（`solutions` 等）保持不变。

参考运行关闭了思考模式（见下文），因此 user 消息里还去掉了 Search-R1 指令中的
"You must conduct reasoning inside <think> and </think> first every time you get new information." 这一句。

配置
----------------------------------------

填写数据路径、模型路径和检索服务地址：

.. code-block:: yaml

   data:
     train_data_paths: ["/path/to/data/searchr1_train_glm.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/GLM-4.7-Flash
   tools:
     search:
       server_addr: <server_ip>:8000

与模型相关的设置在配置中已经写好：

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

- `agentloop.agentic_tokenizer` 选择 agent loop 使用的模型相关 tokenization（`default`、`qwen` 或 `glm47`，
  不设置时为 `default`）。它负责定位推理块的结束位置，使 response 截断在回答中的第一个 `</search>`，
  而不是推理过程中写出的 `</search>`。
- `rollout.sglang.tool_call_parser` 必须与所选 tokenizer 类型绑定的 parser 一致，不一致时启动即报错。
- `enable_thinking` 传给 GLM 的 chat template，只影响第一轮的生成提示；之后的轮次直接把工具输出拼接到对话中。
- `actor.megatron.mbridge: True` 通过 Megatron-Bridge 在进程内加载 HuggingFace 权重，不需要离线把 HuggingFace 转换为 Megatron 格式。

在 8×A100 上训练
----------------------------------------

8×A100-80G 上的并行方式（actor 与 rollout 共置）：

- Actor（Megatron）：张量并行 4、专家并行 8、数据并行 2。
- Rollout（SGLang）：一个引擎横跨 8 张卡，注意力数据并行 2、专家并行 8。
- `rollout.max_running_requests: 512` 会平分到两个注意力数据并行 rank 上（各 256）。
  `rollout.cuda_graph_max_bs` 设为 256，不能小于每个 rank 上的并发数，否则 decode 会退出 CUDA graph，速度反而下降。

检索服务可以与训练共用 GPU。Qdrant 检索服务默认会在每张可见的卡上各启动一个 e5 query 编码器，
因此请用 `CUDA_VISIBLE_DEVICES=0` 启动（约占 1 GB 显存），并在训练之前启动。

运行以下命令开始训练：

.. code-block:: bash

   bash examples/agent/searchr1/run_train.sh train_glm4.7_flash

可视化与结果
----------------------------------------

下面是关闭思考模式时 GLM-4.7-Flash 的训练曲线：8×A100-80G，每步 512 个 prompt × 4 个采样，
检索服务为 :doc:`index` 中的 Qdrant。横轴是参数更新次数：每个 rollout 步更新 4 次（`algorithm.n_minibatches: 4`），
因此 100 个 rollout 步对应横轴上约 400 次更新。100 个 rollout 步内，轨迹 reward 从 0.28 上升到 0.54
（分别取前 5 步和后 5 步的均值），每条轨迹的平均轮数从 1.7 增加到 2.0，每个 rollout 步约 870 s。

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
