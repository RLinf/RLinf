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

   .. grid-item-card:: 服务
      :text-align: center

      ``8081`` 推理服务与 ``8082`` 反馈接收服务

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

本示例同时包含在线强化学习（接入 Continue）和离线验证两条路径。离线验证路径用大模型模拟人类偏好打分，无需安装 Continue。

安装
----------------------------------------

先安装 RLinf，再补充本示例使用的 HTTP 客户端依赖：

.. code-block:: bash

   pip install httpx asyncio

运行离线验证示例时，下载数据集：

.. code-block:: bash

   modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir code-fim-v2-python-filtered

运行
----------------------------------------

配置 Continue 集成
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **安装修改版 Continue 扩展**

   原版 Continue 不支持上传代码补全偏好反馈。请从 `RLinf/continue releases <https://github.com/RLinf/continue/releases>`_ 下载编译好的修改版 VSIX 插件（或自行构建），然后在 VS Code 中安装：

   - 命令行：``code --install-extension /path/to/continue-1.3.9.vsix``
   - 命令面板：按 ``Cmd+Shift+P``，输入 ``Extensions: Install from VSIX``，选择上述文件。

2. **配置 Continue 设置**

   在 ``~/.continue/config.yaml`` 中添加以下配置（将 ``http://xxx:xx/`` 替换为实际的 RLinf 服务地址）：

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

   保存后，从左侧面板打开 Continue 扩展，点击右上角“设置”齿轮按钮，在“Models”页面将“Autocomplete 模型”设为 ``my-autocomplete``。

启动训练服务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通用路径、runner、rollout 和 cluster 字段见 :doc:`训练配置 <../../reference/configuration>`。

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - 模式
     - 配置文件
     - 启动脚本
   * - 在线强化学习（PPO）
     - ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml``
     - ``examples/agent/coding_online_rl/run_main_coding_online_rl.sh``
   * - 离线验证（GRPO）
     - ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml``
     - ``examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh``

1. **修改配置**

   在所选 YAML 中设置 ``runner.output_dir`` 和 ``rollout.model.model_path``：

   .. code-block:: yaml

      runner:
        output_dir: /path/to/your/logs

      rollout:
        model:
          model_path: /path/to/your/model

   离线验证还需在 ``qwen2.5-1.5b-grpo-llm_judge.yaml`` 中配置 ``data.train_data_paths`` 与 ``data.val_data_paths``，并导出 LLM-as-judge 环境变量：

   .. code-block:: yaml

      data:
        train_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]
        val_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]

   .. code-block:: bash

      export LLMASJUDGE_API_URL=your_api_url
      export LLMASJUDGE_API_KEY=your_api_key
      export LLMASJUDGE_MODEL=your_model  # 可选；确保 prompt 适配所选模型

2. **启动训练服务**

   在 RLinf 仓库根目录启动在线强化学习（端口 ``8081`` 提供推理 API，端口 ``8082`` 接收用户反馈）：

   .. code-block:: bash

      bash examples/agent/coding_online_rl/run_main_coding_online_rl.sh

   或启动基于 LLM-as-judge 的离线验证：

   .. code-block:: bash

      bash examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh

使用 Continue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 VS Code 中打开已配置好的 Continue 扩展并开始编写代码。Continue 会向 ``8081`` 端口请求代码补全，并将接受的建议记为正面反馈、拒绝的建议记为负面反馈发送至 ``8082`` 端口，在线更新 policy。

可视化与结果
----------------------------------------

查看日志、TensorBoard 和 checkpoint。通用指标含义见 :doc:`训练指标 <../../reference/metrics>`。

1. **查看日志输出**

   .. code-block:: bash

      tail -f results/ppo-1.5b/train.log

2. **启动 TensorBoard**

   .. code-block:: bash

      tensorboard --logdir results/grpo-1.5b

3. **检查 checkpoint**

   训练过程中会定期将 checkpoint 保存到 ``results/grpo-1.5b/checkpoints/`` 目录。

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
