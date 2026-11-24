Coding Online RL
================

将 Continue VS Code 扩展接入 RLinf，实时收集代码补全的接受/拒绝反馈并用 PPO 在线更新 Qwen2.5-Coder-1.5B，或使用 LLM 评判完成离线 GRPO 验证。相关实践见 :doc:`智能体落地“最后一公里”初探之Cursor在线强化学习 <../../resources/blog/build_a_coding_online_rl_case>`。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-Coder-1.5B

   .. grid-item-card:: 算法
      :text-align: center

      PPO（在线）· GRPO（离线评判）

   .. grid-item-card:: 任务
      :text-align: center

      IDE 代码补全（Continue / FIM）

   .. grid-item-card:: 硬件
      :text-align: center

      8 卡（在线）· 1–8 卡（离线）

| **你将完成：** 安装依赖与 Continue 插件 → 下载 ``Qwen2.5-Coder-1.5B`` → 启动在线 PPO 或离线 GRPO → 用测试客户端验证链路。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · VS Code 与 Continue VSIX（在线）或 LLM-as-judge API key（离线）。

在线 PPO（``qwen2.5-1.5b-ppo.yaml``）采用单机 8 卡分离式部署（rollout 0–3、inference 4–5、actor 6–7、reward 0–3；端口 8081 提供推理服务、端口 8082 接收反馈）；离线 GRPO（``qwen2.5-1.5b-grpo-llm_judge.yaml``）将各组件共置在单机 1–8 卡上。

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__，通用字段见 :doc:`训练配置 </rst_source/reference/configuration>`。

.. tab-set::

   .. tab-item:: 在线强化学习（PPO + Continue）

      **1. 安装环境**

      安装 RLinf 与 ``httpx``，再从 `RLinf/continue releases <https://github.com/RLinf/continue/releases>`_ 下载修改版 Continue VSIX 安装到 VS Code（运行 ``code --install-extension ./continue-1.3.9.vsix``，或在命令面板选择 ``Extensions: Install from VSIX``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx

      将实际 RLinf 服务地址写入 ``~/.continue/config.yaml``，并在 Continue 的 Models 设置里把 Autocomplete 模型选为 ``my-autocomplete``：

      .. dropdown:: Continue 配置（``~/.continue/config.yaml``）

         .. code-block:: yaml

            models:
              - name: my-autocomplete
                provider: openai
                model: Qwen2.5-Coder-1.5B
                apiBase: http://xxx:8081/v1
                apiKey: xxx
                roles:
                  - autocomplete

            tabAutocompleteOptions:
              enableCompletionTracking: true
              completionTrackingUrl: http://xxx:8082/api/training/submit
              completionTrackingHeaders:
                Authorization: Bearer test-token
                X-Project-ID: test-project
              maxPromptTokens: 1024
              debounceDelay: 350
              multilineCompletions: auto

      **2. 下载模型**

      .. code-block:: bash

         hf download Qwen/Qwen2.5-Coder-1.5B --local-dir ./Qwen2.5-Coder-1.5B

      **3. 启动训练**

      使用 ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml`` 启动在线服务，随后在 VS Code 中打开 Continue 编写代码即可收集反馈并更新模型：

      .. code-block:: bash

         bash examples/agent/coding_online_rl/run_main_coding_online_rl.sh \
           rollout.model.model_path=./Qwen2.5-Coder-1.5B \
           runner.output_dir=./results

   .. tab-item:: 离线验证（GRPO + LLM Judge）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx modelscope

      **2. 下载模型与数据**

      下载基础模型，并从 ModelScope 获取 Python FIM 数据集：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-Coder-1.5B --local-dir ./Qwen2.5-Coder-1.5B
         modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir ./code-fim-v2-python-filtered

      **3. 启动训练**

      导出 LLM-as-judge API 环境变量，使用 ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml`` 启动离线验证：

      .. code-block:: bash

         export LLMASJUDGE_API_URL=your_api_url
         export LLMASJUDGE_API_KEY=your_api_key
         export LLMASJUDGE_MODEL=your_model  # 可选；确保 prompt 适配所选模型
         bash examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh \
           rollout.model.model_path=./Qwen2.5-Coder-1.5B \
           runner.output_dir=./results \
           data.train_data_paths='["./code-fim-v2-python-filtered/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]' \
           data.val_data_paths='["./code-fim-v2-python-filtered/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]'

.. dropdown:: 验证客户端与故障排查

   运行测试客户端，模拟 Continue 发送补全请求并上报反馈：

   .. code-block:: bash

      python examples/agent/coding_online_rl/simple_online_coding_client.py

   - **端口冲突：** 端口 8081 或 8082 被占用时，修改 ``qwen2.5-1.5b-ppo.yaml`` 中的端口配置。
   - **模型加载失败：** 确认 ``rollout.model.model_path`` 指向可访问的本地模型目录。
   - **Continue 连接失败：** 核对 ``~/.continue/config.yaml`` 中的接口地址与网络连通性，或运行 ``simple_online_coding_client.py`` 验证反馈链路。

可视化与结果
------------

查看训练日志与 TensorBoard 曲线；更多指标与 W&B/SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

.. code-block:: bash

   tail -f results/online-ppo-1.5b-pipeline/train.log
   tensorboard --logdir results/Qwen2.5-Coder-1.5B-grpo-llm_judge

checkpoint 保存在 ``results/online-ppo-1.5b-pipeline/checkpoints/``\ （在线 PPO）或 ``results/Qwen2.5-Coder-1.5B-grpo-llm_judge/checkpoints/``\ （离线 GRPO）。
