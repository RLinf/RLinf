.. This file is a reusable include, not a standalone page.
   Include it from the training and evaluation recipes.

Before running either training or evaluation, start the judge model.
WideSeek-R1 uses an LLM judge to provide more reliable feedback than exact-match
scoring alone.

Judge Model Server
~~~~~~~~~~~~~~~~~~

The default setup uses
`Qwen3-30B-A3B-Instruct-2507 <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__
as the judge model.

Start the judge server with SGLang:

.. code-block:: bash

   python3 -m sglang.launch_server \
      --model-path /PATH/TO/Qwen3-30B-A3B-Instruct-2507 \
      --host 0.0.0.0 \
      --log-level info \
      --context-length 32768 \
      --dp 8

In the main experiments, the judge model was served on 8 H100 GPUs. You can
reduce or increase ``--dp`` based on your available hardware and throughput
requirements.

Then obtain the host IP address, for example:

.. code-block:: bash

   hostname -I

Use that IP address in the YAML configuration through the following fields. The
default port is ``30000``.

.. code-block:: yaml

   agentloop:
     llm_ip: LLM_JUDGE_IP
     llm_port: LLM_JUDGE_PORT

You can test it with:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/utils/sglang_client.py --llm-ip LLM_JUDGE_IP

Using RLinf Built-in Rollout Engine as Judge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can use RLinf's built-in rollout engine as the judge instead
of an external server. This approach runs the judge LLM within the RLinf
framework, which can be more convenient for local development and testing.

To use the built-in rollout engine as judge, set the following configuration in
your YAML file:

.. code-block:: yaml

   agentloop:
     use_local_judge: true  # Enable local judge within RLinf framework

Then configure the ``rollout_judge`` section with your desired model and
settings:

.. code-block:: yaml

   rollout_judge:
     group_name: "RolloutJudgeGroup"
     gpu_memory_utilization: 0.5
     model:
       model_type: qwen3
       model_path: /PATH/TO/YOUR/JUDGE/MODEL  # Replace with actual path
       precision: fp16
     rollout_backend: sglang
     tensor_parallel_size: 1
     pipeline_parallel_size: 1
     max_running_requests: 64

Example configuration files using the built-in judge:

.. list-table::
   :header-rows: 1

   * - Config
     - Purpose
   * - ``examples/agent/wideseek_r1/config/train_qwen3_hybrid_local_judge.yaml``
     - Train with the local judge.
   * - ``examples/agent/wideseek_r1/config/eval_qwen3_widesearch_local_judge.yaml``
     - Evaluate WideSearch with the local judge.

When using the built-in judge, you do not need to start a separate judge server.
The judge model is loaded and managed by RLinf's rollout engine.
