分布式追踪与性能分析
=================================

RLinf 集成了一个轻量级的基于 HTTP 的分布式追踪和分析系统。这使得能够细粒度地追踪所有独立 Ray worker 节点（环境、Actor、Rollout 等）的执行时间线，而不会成为主训练循环的瓶颈。

概览
--------

该追踪系统的工作原理如下：

1. 运行一个轻量级的集中式 **HTTP 追踪服务器** 来接收日志事件。
2. 在每个 Ray worker 进程启动时初始化一个后台 **追踪客户端**。
3. 使用 Cristian 算法自动同步各节点时钟，以确保各个事件的精确时间对齐。
4. 将数据导出为 Chrome Trace JSONL 格式，简单转换后即可在 Perfetto 中可视化。

启动追踪服务器
-------------------------

追踪服务器作为集中采集器。你可以使用位于 ``toolkits/`` 目录下的实用脚本启动它：

.. code-block:: bash

    python3 toolkits/start_trace_server.py \
        --host 127.0.0.1 \
        --port 8888 \
        --file trace_events.jsonl


运行带有追踪功能的训练任务
-----------------------------

要在运行中启用追踪，只需在你的启动脚本中通过 Hydra 原生提供 ``+trace_server_ip`` 以及可选的 ``+trace_server_port``。

.. code-block:: bash

    python3 examples/embodiment/train_embodied_agent.py \
        --config-name libero_spatial_ppo_openpi_pi05 \
        +trace_server_ip=127.0.0.1 \
        +trace_server_port=8888 \
        algorithm.rollout_epoch=2 \
        env.train.total_num_envs=32 \
        actor.micro_batch_size=64 \
        actor.global_batch_size=256 \
        runner.max_steps=3

此次运行生成的所有 Ray worker 将自动与追踪服务器建立连接，并开始异步刷新其执行片段（如 ``generate``、``training_step`` 和 ``env_interact_step``）。

可视化
-------------

在运行完成或执行期间，你可以对追踪数据进行可视化：

1. 找到输出的 JSONL 文件（例如 ``trace_events.jsonl``）。
2. 使用 ``jq`` 将 JSON Lines 格式转换为单一 JSON 数组（Perfetto 需要该格式）：

   .. code-block:: bash

       jq -s . trace_events.jsonl > trace_events.json

3. 打开基于 Chrome 的浏览器并导航至 `Perfetto UI <https://ui.perfetto.dev/>`_ 或 ``chrome://tracing``。
4. 点击 **Open trace file** 并选择你转换后的 ``trace_events.json`` 文件。

你将看到一个交互式甘特图，精确展示各个节点 IP、主机名和进程角色的执行层次、Actor 等待时间和系统瓶颈。
