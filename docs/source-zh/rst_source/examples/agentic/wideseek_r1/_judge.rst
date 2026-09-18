.. This file is a reusable include, not a standalone page.
   Include it from the training and evaluation recipes.

在运行训练或评测之前，请先启动评判模型。WideSeek-R1 使用 LLM 评判器，相比仅依赖精确匹配打分，能够提供更可靠的反馈。

评判模型服务
~~~~~~~~~~~~

默认配置使用 `Qwen3-30B-A3B-Instruct-2507 <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__ 作为评判模型。

使用 SGLang 启动评判服务：

.. code-block:: bash

   python3 -m sglang.launch_server \
      --model-path /PATH/TO/Qwen3-30B-A3B-Instruct-2507 \
      --host 0.0.0.0 \
      --log-level info \
      --context-length 32768 \
      --dp 8

在主实验中，评判模型部署在 8 张 H100 GPU 上。你可以根据可用硬件和吞吐需求减少或增加 ``--dp`` 的值。

然后获取主机 IP 地址，例如：

.. code-block:: bash

   hostname -I

在 YAML 配置中通过以下字段使用该 IP 地址。默认端口为 ``30000``。

.. code-block:: yaml

   agentloop:
     llm_ip: LLM_JUDGE_IP
     llm_port: LLM_JUDGE_PORT

你可以通过以下命令测试：

.. code-block:: bash

   python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip LLM_JUDGE_IP

使用 RLinf 内置的 Rollout Engine 作为评判器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你也可以使用 RLinf 内置的 rollout engine 作为评判器，而不是使用外部服务器。这种方式在 RLinf 框架内部运行评判 LLM，对于本地开发和测试更加方便。

要使用内置的 rollout engine 作为评判器，请在 YAML 配置文件中设置：

.. code-block:: yaml

   agentloop:
     use_local_judge: true  # 在 RLinf 框架内启用本地评判器

然后配置 ``rollout_judge`` 部分，设置你所需的模型和参数：

.. code-block:: yaml

   rollout_judge:
     group_name: "RolloutJudgeGroup"
     gpu_memory_utilization: 0.5
     model:
       model_type: qwen3
       model_path: /PATH/TO/YOUR/JUDGE/MODEL  # 替换为实际路径
       precision: fp16
     rollout_backend: sglang
     tensor_parallel_size: 1
     pipeline_parallel_size: 1
     max_running_requests: 64

使用内置评判器的示例配置文件：

.. list-table::
   :header-rows: 1

   * - 配置
     - 用途
   * - ``examples/agent/wideseek_r1/config/train_qwen3_hybrid_local_judge.yaml``
     - 使用本地评判器训练。
   * - ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch_local_judge.yaml``
     - 使用本地评判器评测 WideSearch。

使用内置评判器时，你不需要启动单独的评判服务器。评判模型将由 RLinf 的 rollout engine 加载和管理。
