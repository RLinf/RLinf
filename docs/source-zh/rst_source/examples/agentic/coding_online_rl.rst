代码补全在线强化学习
========================================

使用本配方将 Continue 接入 RLinf，收集代码补全的接受 / 拒绝反馈，并在线更新 Qwen coder 模型。

相关阅读：:doc:`智能体落地“最后一公里”初探之Cursor在线强化学习 <../../resources/blog/build_a_coding_online_rl_case>`。

概述
----------------------------------------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-Coder-1.5B

   .. grid-item-card:: 算法
      :text-align: center

      PPO 在线强化学习或 GRPO 离线验证

   .. grid-item-card:: 反馈
      :text-align: center

      Continue 接受 / 拒绝事件或 LLM-as-judge 标签

   .. grid-item-card:: 硬件
      :text-align: center

      单机 8 卡用于在线 PPO（分离式 ``0-3`` / ``4-5`` / ``6-7``）；单机 1–8 卡用于离线 GRPO（共置）

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 步骤
     - 组件
     - 产出
   * - 实时交互
     - Continue 扩展
     - 向 RLinf 发送代码补全请求
   * - 模型推理
     - RLinf 推理服务
     - 返回代码补全建议
   * - 用户反馈
     - Continue tracking callback
     - 记录接受或拒绝的补全
   * - 在线学习
     - RLinf 训练服务
     - 根据反馈更新 policy

本示例包含在线强化学习（接入 Continue，端口 ``8081`` 提供推理、端口 ``8082`` 接收反馈）和离线验证两条路径，离线验证用大模型模拟人类偏好打分，无需安装 Continue。在线 PPO 配置（``qwen2.5-1.5b-ppo.yaml``）要求单机 8 卡分离式部署（``rollout: 0-3``、``inference: 4-5``、``actor: 6-7``、``reward: 0-3``）；离线 GRPO 配置（``qwen2.5-1.5b-grpo-llm_judge.yaml``）将所有组件共置在单机 1–8 卡（``actor,rollout,reward: all``）。

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。通用路径、runner、rollout 和 cluster 字段见 :doc:`训练配置 <../../reference/configuration>`。

.. tab-set::

   .. tab-item:: 在线强化学习（PPO + Continue）

      **1. 安装环境**

      安装 RLinf 和 HTTP 客户端依赖，并从 `RLinf/continue releases <https://github.com/RLinf/continue/releases>`_ 下载修改版 Continue VSIX 插件安装到 VS Code（命令行执行 ``code --install-extension /path/to/continue-1.3.9.vsix``，或在命令面板运行 ``Extensions: Install from VSIX``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx

      在 ``~/.continue/config.yaml`` 中添加模型与反馈上报配置（将 ``http://xxx:xx/`` 替换为实际的 RLinf 服务地址），然后在 Continue 的“Models”页面将“Autocomplete 模型”设为 ``my-autocomplete``：

      .. dropdown:: Continue 配置（``~/.continue/config.yaml``）

         .. code-block:: yaml

            # 添加一个模型，用于代码补全
            models:
              - name: my-autocomplete
                provider: openai
                model: Qwen2.5-Coder-1.5B
                apiBase: http://xxx:8081/v1
                apiKey: xxx
                roles:
                  - autocomplete

            # 发送用户是否接受代码补全的反馈
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

      在 ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml`` 中设置 ``runner.output_dir`` 和 ``rollout.model.model_path``：

      .. code-block:: yaml

         runner:
           output_dir: /path/to/your/logs

         rollout:
           model:
             model_path: /path/to/your/model

      启动在线强化学习服务（端口 ``8081`` 提供推理 API，端口 ``8082`` 接收用户反馈），然后在 VS Code 中打开 Continue 开始编写代码：

      .. code-block:: bash

         bash examples/agent/coding_online_rl/run_main_coding_online_rl.sh

   .. tab-item:: 离线验证（GRPO + LLM Judge）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx

      **2. 下载模型与数据**

      .. code-block:: bash

         uv pip install modelscope
         hf download Qwen/Qwen2.5-Coder-1.5B --local-dir ./Qwen2.5-Coder-1.5B
         modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir ./code-fim-v2-python-filtered

      **3. 启动训练**

      在 ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml`` 中设置 ``runner.output_dir``、``rollout.model.model_path``、``data.train_data_paths`` 和 ``data.val_data_paths``：

      .. code-block:: yaml

         runner:
           output_dir: /path/to/your/logs

         rollout:
           model:
             model_path: /path/to/your/model

         data:
           train_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]
           val_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]

      导出 LLM-as-judge 环境变量并启动离线验证：

      .. code-block:: bash

         export LLMASJUDGE_API_URL=your_api_url
         export LLMASJUDGE_API_KEY=your_api_key
         export LLMASJUDGE_MODEL=your_model  # 可选；确保 prompt 适配所选模型
         bash examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh

可视化与结果
----------------------------------------

查看日志、TensorBoard 和 checkpoint。通用指标含义见 :doc:`训练指标 <../../reference/metrics>`。

1. **查看日志输出**

   .. code-block:: bash

      tail -f results/online-ppo-1.5b-pipeline/train.log

2. **启动 TensorBoard**

   .. code-block:: bash

      tensorboard --logdir results/Qwen2.5-Coder-1.5B-grpo-llm_judge

3. **检查 checkpoint**

   训练过程中会定期将 checkpoint 保存到 ``results/Qwen2.5-Coder-1.5B-grpo-llm_judge/checkpoints/`` 目录（在线 PPO 保存在 ``results/online-ppo-1.5b-pipeline/checkpoints/``）。

验证客户端
----------------------------------------

运行测试客户端，模拟 Continue 发送代码补全请求并提交反馈数据：

.. code-block:: bash

   python examples/agent/coding_online_rl/simple_online_coding_client.py

故障排查
----------------------------------------

1. **端口冲突**

   如果端口 ``8081`` 或 ``8082`` 被占用，修改 YAML 配置文件中的端口设置。

2. **模型加载失败**

   检查 ``rollout.model.model_path`` 是否指向可访问的本地模型目录。

3. **Continue 连接失败**

   核对 ``~/.continue/config.yaml`` 中的 API 端点地址与网络连通性，或运行 ``simple_online_coding_client.py`` 测试反馈链路。
