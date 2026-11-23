使用 EnvScaler 训练工具调用策略
===============================

这个示例使用 EnvScaler 的 2,550 个强化学习任务，通过 GRPO 训练 Qwen3。
每条轨迹都有独立的业务数据，策略每轮可以调用一个工具查询或修改数据。
任务结束后，环境检查最终状态，以通过的检查项占比作为该轨迹各轮的奖励。

安装
----

准备好 RLinf 的 SGLang agentic 环境和本地 Qwen3-1.7B 权重，在 RLinf 仓库中执行下列命令。
各节点需要通过相同路径读取 EnvScaler 的代码和随仓库提供的 JSON 文件。
环境类和评分函数会执行其中的 Python 代码，因此请使用可信的数据源。

.. code-block:: bash

   git clone https://github.com/RUC-NLPIR/EnvScaler.git ../EnvScaler
   git -C ../EnvScaler checkout 7ab81e557c5992d62d4533e81bfec3d2ca0ae24b
   export ENVSCALER_ROOT="$(cd ../EnvScaler && pwd)"
   export PYTHONPATH="$PWD:$ENVSCALER_ROOT/interact_with_env:${PYTHONPATH:-}"
   python -m pip install openai python-dotenv

运行训练
--------

默认使用非对话模式。策略先读取完整任务，再逐步调用本地工具，最后通过文本回复提交结果。

.. code-block:: bash

   python examples/agent/envscaler/train.py rollout.model.model_path=/path/to/Qwen3-1.7B

对话模式需要一个兼容 OpenAI Chat Completions 的用户模拟服务。
请在各节点启动 Ray 前设置服务地址和密钥。只有模拟用户能看到完整目标，它会发出首条请求、
回答策略的追问，并用 ``###STOP###`` 结束对话。两种模式使用相同的本地评分函数。

.. code-block:: bash

   export USER_OPENAI_BASE_URL=http://user-model-host:8000/v1
   export USER_OPENAI_API_KEY='YOUR_KEY'
   python examples/agent/envscaler/train.py \
       rollout.model.model_path=/path/to/Qwen3-1.7B \
       agentloop.mode=conversational agentloop.user_model=YOUR_SERVED_MODEL
