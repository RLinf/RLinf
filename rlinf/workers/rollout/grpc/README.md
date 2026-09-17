# gRPC Backend for Remote Inference

This module provides gRPC-based remote inference support for RLinf training workflows.

## Architecture

RLinf supports two inference backends during training:

1. **Local inference** (default): Model runs in the rollout worker process
2. **gRPC inference**: Model runs on a remote GPU server, rollout worker connects via gRPC

The gRPC backend is useful when:
- Training on a machine without GPU (e.g., robot control computer)
- The inference model is too large for the control machine
- You want to separate model serving from data collection

## Usage in Training

### 1. Start the gRPC Policy Server

On the GPU server, start the LeRobot gRPC policy server:

```bash
# See examples/embodiment/so101/lerobot_grpc_policy_server.py for SO-101
python examples/embodiment/so101/lerobot_grpc_policy_server.py \
  --checkpoint /path/to/actor-checkpoint \
  --norm-stats /path/to/norm_stats.json \
  --host 0.0.0.0 \
  --port 50051 \
  --device cuda:0
```

Or use the service wrapper:

```bash
# Configure the service
cp examples/embodiment/so101/lerobot_grpc_policy.env.example \
   examples/embodiment/so101/lerobot_grpc_policy.env
# Edit the .env file with your paths

# Start the service
examples/embodiment/so101/lerobot_grpc_policy_service.sh start
```

### 2. Configure Training to Use gRPC Backend

Add the following to your training config (e.g., `realworld_so101_dagger_openpi.yaml`):

```yaml
rollout:
  use_grpc_backend: true
  grpc:
    server_address: "localhost:50051"  # Or remote host:port
    timeout: 30.0  # Request timeout in seconds
```

Then run training as usual:

```bash
python rlinf/train_embodied_agent.py --config-name realworld_so101_dagger_openpi
```

### 3. Example Configurations

See:
- `examples/embodiment/config/realworld_so101_inference_grpc.yaml` - Inference-only with gRPC
- `examples/embodiment/config/realworld_so101_dagger_grpc.yaml` - DAgger training with gRPC

## Implementation Details

### GRPCPolicyAdapter

The `GRPCPolicyAdapter` class implements the `BasePolicy` interface and forwards
`predict_action_batch()` calls to a remote gRPC server using the LeRobot protocol.

Key features:
- Drop-in replacement for local models in `MultiStepRolloutWorker`
- Handles batch observations by processing them sequentially
- Converts between RLinf observation format and LeRobot gRPC protocol
- Maintains compatibility with DAgger expert models (expert still runs locally)

### Integration with MultiStepRolloutWorker

The `init_worker()` method checks `cfg.rollout.use_grpc_backend`:
- If `True`: Creates a `GRPCPolicyAdapter` instead of loading a local model
- If `False`: Standard local model loading (default behavior)

Optimizations like torch.compile and CUDA graphs are automatically skipped for gRPC backend.

## Limitations

- gRPC inference adds network latency (~5-50ms depending on connection)
- Expert models for DAgger always run locally (not over gRPC)
- OPD feature models run locally (not over gRPC)
- Model updates during training require restarting the gRPC server

## Standalone Inference Tools

For standalone inference (not during training), see:
- `examples/embodiment/so101/README.md` - LeRobot-based inference examples
- `toolkits/realworld_check/run_so101_policy.py` - Full inference script with DAgger support

These tools use the LeRobot `RobotClient` and `AsyncInference` protocol, which is more
feature-rich for real-time robot control but separate from the training workflow.
