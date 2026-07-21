# Distributed Trace Server

RLinf supports fine-grained distributed instrumentation tracing across driver processes, environment workers, rollout workers, and training actors. The trace server acts as the centralized HTTP collector that all Ray workers flush their traces to asynchronously.

### Starting the Server

Launch the trace server in the background or in a separate terminal:
```bash
python3 toolkits/start_trace_server.py \
    --host 127.0.0.1 \
    --port 8888 \
    --file /workspace/trace_events.jsonl
```
*Note: This command will write all incoming trace events directly to `/workspace/trace_events.jsonl`.*

### Example: E2E Training Run with Tracing Enabled

To capture traces during a training run, pass the trace server configurations natively via Hydra overrides (`+trace_server_ip` and `+trace_server_port`). 

Example using the `train_embodied_agent.py` entrypoint:
```bash
python3 examples/embodiment/train_embodied_agent.py \
    --config-name libero_spatial_ppo_openpi_pi05 \
    +trace_server_ip=127.0.0.1 \
    +trace_server_port=8888 \
    algorithm.rollout_epoch=2 \
    env.train.total_num_envs=32 \
    actor.micro_batch_size=64 \
    actor.global_batch_size=256 \
    runner.max_steps=3
```

### Verification and Visualization

Once training finishes:
1. Verify that `trace_events.jsonl` has been populated (e.g. `wc -l /workspace/trace_events.jsonl`).
2. Verify that metadata for the worker processes (like `driver`, `EnvGroup`, `RolloutGroup`) are present:
   ```bash
   grep '"ph": "M"' /workspace/trace_events.jsonl
   ```
3. Convert the JSON Lines output to a single JSON array using `jq` so it can be parsed by Perfetto:
   ```bash
   jq -s . /workspace/trace_events.jsonl > /workspace/trace_events.json
   ```
4. Open [Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing` in a Chrome-based browser.
5. Drag and drop the `trace_events.json` file to visualize interactive timelines of your distributed execution.
