代码补全在线强化学习
====================

代码补全在线强化学习（Online Coding RL）是 RLinf 框架中的一个重要应用场景。
通过与 Continue 等代码编辑器的集成，获取用户对代码补全的偏好反馈，可以实现近乎实时的代码生成和反馈学习，快速提高代码补全的质量，和对齐用户的偏好。
本示例展示了如何使用 RLinf 框架训练一个能够进行在线代码补全任务的模型。

相关阅读：:doc:`智能体落地“最后一公里”初探之Cursor在线强化学习 <../blog/build_a_coding_online_rl_case>`。

概述
--------

代码补全在线强化学习系统通过以下方式工作：

1. **实时交互**：系统接收来自 Continue 等编辑器的代码补全请求
2. **模型推理**：使用训练好的模型生成代码补全建议
3. **用户反馈**：收集用户对生成代码的接受/拒绝反馈
4. **在线学习**：基于用户反馈实时更新模型参数

这种实时学习机制使得模型能够快速适应用户的编程习惯和偏好。

我们同时提供了针对在线强化学习及离线验证的示例。其中离线验证示例使用大模型模拟人类偏好进行打分，不需要也不支持部署 Continue 在线使用。

运行脚本
--------------

**环境准备**

首先确保您已经安装了 RLinf 框架及其依赖：

.. code-block:: bash

   # 安装额外依赖
   pip install httpx asyncio

如果使用离线验证示例，需要下载数据集：

.. code-block:: bash

   # 安装额外依赖
   modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir code-fim-v2-python-filtered

**配置 Continue 集成**

1. **安装 Continue 扩展**
   
   由于当前 Continue 未支持上传用户对代码补全的偏好反馈，因此我们修改了 Continue 的源码，支持上传用户对代码补全的偏好反馈。
   用户可从 `这里 <https://github.com/RLinf/continue/releases>`_ 获取编译好的修改后的 Continue 插件，或自行构建。

   下载编译好的 Continue 插件后，在 VS Code 中安装。

   方法1: code --install-extension /path/to/continue-1.3.9.vsix"

   方法2: 在 VSCode 中按 Cmd+Shift+P ，输入 'Extensions: Install from VSIX'，选择上述文件

2. **配置 Continue 设置**

   Continue 的配置文件路径为：

   .. code-block:: bash

      ~/.continue/config.yaml

   在 Continue 的配置文件中添加以下设置：

   .. code-block:: yaml

      # 请将 http://xxx:xx/ 替换为实际的 RLinf 在线代码补全服务地址

      # 添加一个模型，用于代码补全
      models:
        - name: my-autocomplete
          provider: openai
          model: Qwen2.5-Coder-1.5B
          apiBase: http://xxx:8081/v1
          apiKey: xxx
          roles:
            - autocomplete

      # 添加发送用户是否接受代码补全的反馈
      tabAutocompleteOptions:
        enableCompletionTracking: true
        completionTrackingUrl: http://xxx:8082/api/training/submit
        completionTrackingHeaders:
          Authorization: Bearer test-token
          X-Project-ID: test-project
        maxPromptTokens: 1024
        debounceDelay: 350
        multilineCompletions: auto

   修改并保存完成后，从左侧面板打开 Continue 扩展，点击右上角的 "设置" 齿轮按钮，在 "Models" 页面确保 "Autocomplete 模型" 选用 my-autocomplete。

**启动训练服务**

1. **准备模型和配置**

   确保您有预训练的模型权重，并修改配置文件，匹配模型路径、需要使用的端口等

   - 对于在线强化学习，修改并使用 examples/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml 文件:
      .. code-block:: yaml

         runner:
           output_dir: /path/to/your/logs

         rollout:
           model_dir: /path/to/your/model


   - 对于离线验证，修改并使用 examples/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml 文件:
      .. code-block:: yaml

         runner:
           output_dir: /path/to/your/logs

         rollout:
           model_dir: /path/to/your/model

         data:
           train_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]
           val_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]

      同时，还需要设置用于模拟反馈的大模型的调用 api_url 及 api_key：

      .. code-block:: bash

         export LLMASJUDGE_API_URL=your_api_url
         export LLMASJUDGE_API_KEY=your_api_key
         export LLMASJUDGE_MODEL=your_model  # not recommended. should fit prompt for your model.

2. **启动 RLinf 训练服务**

   - 对于在线强化学习：
      .. code-block:: bash
      
         # 进入项目目录
         cd /path/to/rlinf_online_rl
         
         # 启动训练服务
         bash examples/coding_online_rl/run_main_coding_online_rl.sh

      这将启动以下服务：

      - **推理服务**：在端口 8081 提供代码补全 API
      - **训练服务**：在端口 8082 接收用户反馈数据

   - 对于离线验证：
      .. code-block:: bash
      
         # 进入项目目录
         cd /path/to/rlinf_online_rl
         
         # 启动训练服务
         bash examples/coding_online_rl/run_main_coding_rl_llm_judge.sh

**与 Continue 联动**

1. **启动 Continue**
   
   在 VS Code 中启动 Continue 扩展，确保它连接到正确的 API 端点。

2. **开始编程**
   
   在 Continue 中开始编写代码，系统将：
   - 自动发送代码补全请求到推理服务
   - 接收模型生成的代码建议
   - 收集您对建议的接受/拒绝反馈

3. **实时学习**
   
   系统会实时处理您的反馈：
   - 接受的建议被标记为正面反馈
   - 拒绝的建议被标记为负面反馈
   - 模型参数根据反馈进行在线更新

**监控训练过程**

您可以通过以下方式监控训练过程：

1. **查看日志输出**
   
   .. code-block:: bash

      # 查看训练日志
      tail -f results/ppo-1.5b/train.log

2. **使用 TensorBoard**
   
   .. code-block:: bash

      # 启动 TensorBoard
      tensorboard --logdir results/grpo-1.5b

3. **检查模型检查点**
   
   训练过程中会定期保存模型检查点到 `results/grpo-1.5b/checkpoints/` 目录。

**测试客户端**

您可以使用提供的测试客户端来验证系统功能：

.. code-block:: bash

   # 运行测试客户端
   python examples/coding_online_rl/simple_test_client.py

测试客户端会模拟 Continue 的行为，发送代码补全请求并提交反馈数据。

**故障排除**

常见问题及解决方案：

1. **端口冲突**
   
   如果端口 8081 或 8082 被占用，请修改配置文件中的端口设置。

2. **模型加载失败**
   
   检查模型路径是否正确，确保模型文件存在且可访问。

3. **Continue 连接失败**
   
   确保 Continue 配置中的 API 端点地址正确，检查网络连接。还可使用 simple_test_client 测试是否能正常收到反馈数据。

通过以上步骤，您就可以成功运行代码补全在线强化学习系统，并实现与 Continue 编辑器的无缝集成。
