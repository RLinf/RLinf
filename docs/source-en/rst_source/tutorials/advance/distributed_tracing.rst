Distributed Tracing and Profiling
=================================

RLinf integrates a lightweight, HTTP-based distributed tracing and profiling system. This enables fine-grained tracking of execution timelines across all independent Ray worker nodes (environment, actor, rollout, etc.) without bottlenecking the primary training loop.

Overview
--------

The tracing system works by:

1. Running a lightweight, centralized **HTTP Trace Server** that receives log events.
2. Initializing a background **Trace Client** in every Ray worker process upon boot.
3. Automatically syncing clocks across nodes using Cristian's algorithm to ensure precise chronological alignment of spans.
4. Exporting data into the Chrome Trace JSONL format, visualizable in Perfetto after a simple conversion.

Starting the Trace Server
-------------------------

The trace server acts as the central collector. You can launch it using the utility script located in the ``toolkits/`` directory:

.. code-block:: bash

    python3 toolkits/start_trace_server.py \
        --host 127.0.0.1 \
        --port 8888 \
        --file trace_events.jsonl


Running a Traced Training Job
-----------------------------

To enable tracing for your run, simply provide the ``+trace_server_ip`` and optionally ``+trace_server_port`` Hydra overrides to your entry point script.

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

All Ray workers spawned by this run will automatically establish connections to the trace server and begin flushing asynchronous spans (like ``generate``, ``training_step``, and ``env_interact_step``).

Visualization
-------------

Once your run completes or during execution, you can visualize the trace data:

1. Locate the output JSONL file (e.g., ``trace_events.jsonl``).
2. Use ``jq`` to convert the JSON Lines format into a single JSON array (Perfetto requires this):

   .. code-block:: bash

       jq -s . trace_events.jsonl > trace_events.json

3. Open a Chrome-based browser and navigate to `Perfetto UI <https://ui.perfetto.dev/>`_ or ``chrome://tracing``.
4. Click **Open trace file** and select your converted ``trace_events.json`` file.

You will see an interactive Gantt chart visualizing execution layers, actor wait times, and system bottlenecks labeled precisely by Node IP, Hostname, and Process Role.
