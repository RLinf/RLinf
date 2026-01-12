# BEHAVIOR Model Evaluation in RLinf

This document explains how to evaluate BEHAVIOR models in RLinf using the worker-based architecture, and contrasts it with the client/server approach used in OpenPI-Comet.

## Overview

RLinf provides two approaches for BEHAVIOR evaluation:
1. **Parallel Worker-Based Evaluation** (Recommended): Fast, scalable, follows RLinf's architecture
2. **Single-Threaded Evaluation**: Provides detailed metrics, useful for debugging

## Architecture Comparison

### OpenPI-Comet Approach (Client/Server)

The OpenPI-Comet evaluation uses a client/server architecture with WebSocket communication:

```
┌─────────────────────┐         WebSocket          ┌──────────────────────┐
│  Policy Server      │◄──────────────────────────►│  Environment Client  │
│  (serve_b1k.py)     │                            │  (eval.py)           │
│                     │                            │                      │
│  - Load Policy      │    Observation             │  - OmniGibson Env    │
│  - Generate Actions │◄───────────────────────────│  - Step Environment  │
│  - B1KPolicyWrapper │                            │  - Collect Metrics   │
│                     │    Actions                 │                      │
│                     │───────────────────────────►│                      │
└─────────────────────┘                            └──────────────────────┘

GPU 1: Run policy model                           GPU 0: Run environment
```

**Limitations:**
- Requires separate processes and WebSocket setup
- Single environment per client
- Network communication overhead
- Manual GPU allocation
- Complex error handling across processes

### RLinf Approach (Worker-Based)

RLinf uses a distributed worker architecture with Ray for communication:

```
┌──────────────────────────────────────────────────────────────┐
│                         Ray Cluster                           │
│                                                                │
│  ┌───────────────────┐                  ┌──────────────────┐ │
│  │  Env Workers      │    Ray Channel   │  Rollout Workers │ │
│  │                   │◄────────────────►│                  │ │
│  │  - OmniGibson     │                  │  - Policy Model  │ │
│  │  - VectorEnv      │   Observations   │  - Action Gen    │ │
│  │  - Metrics        │◄─────────────────│                  │ │
│  │                   │                  │                  │ │
│  │  Parallel Envs:   │   Actions        │  GPU 0-1         │ │
│  │  GPU 0-1          │─────────────────►│                  │ │
│  └───────────────────┘                  └──────────────────┘ │
│                                                                │
│  ┌───────────────────────────────────────────────────────────┤
│  │  EmbodiedEvalRunner                                        │
│  │  - Coordinates workers                                     │
│  │  - Aggregates metrics                                      │
│  │  - Saves results                                           │
│  └────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Automatic GPU placement via Ray
- Parallel environment execution
- Efficient communication via Ray channels
- Scales to multi-node clusters
- Built-in fault tolerance
- Consistent with RLinf training pipeline

## Quick Start

### 1. Parallel Evaluation (Recommended)

Evaluate a trained BEHAVIOR model using RLinf's parallel workers:

```bash
# Set up environment
cd /path/to/RLinf
export EMBODIED_PATH="$(pwd)/examples/embodiment"
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run evaluation
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval \
    runner.eval_policy_path=/path/to/model/checkpoint.pt \
    env.eval.task_idx=0 \
    env.eval.total_num_envs=2 \
    algorithm.eval_rollout_epoch=10
```

### 2. Single-Threaded Evaluation (Detailed Metrics)

For detailed per-instance metrics and debugging:

```bash
# Using the convenience script
bash toolkits/eval_scripts_openpi/eval_behavior.sh \
    --task_name turning_on_radio \
    --model_path /path/to/model/ \
    --policy_type rlinf \
    --action_chunk 32 \
    --max_steps 2000

# Or run directly with Python
python toolkits/eval_scripts_openpi/behavior_eval.py \
    --task_name turning_on_radio \
    --pretrained_path /path/to/model/ \
    --policy_type rlinf \
    --action_chunk 32 \
    --max_steps 2000 \
    --num_episodes_per_instance 1 \
    --log_path ./eval_results
```

## Detailed Workflow

### Worker-Based Evaluation Flow

1. **Initialization Phase**
   ```python
   # Create worker groups
   rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(cluster, ...)
   env_group = EnvWorker.create_group(cfg).launch(cluster, ...)
   
   # Initialize runner
   runner = EmbodiedEvalRunner(cfg, rollout=rollout_group, env=env_group)
   runner.init_workers()
   ```

2. **Evaluation Loop**
   ```python
   # Env workers and rollout workers run concurrently
   env_handle = env.evaluate(input_channel=rollout_channel, output_channel=env_channel)
   rollout_handle = rollout.evaluate(input_channel=env_channel, output_channel=rollout_channel)
   
   # Wait for completion
   env_results = env_handle.wait()
   rollout_handle.wait()
   ```

3. **Communication Flow**
   ```
   EnvWorker:
   1. Reset environments → get observations
   2. Send observations via env_channel
   3. Receive actions via rollout_channel
   4. Step environments with actions
   5. Collect rewards, terminations, metrics
   6. Repeat until episode complete
   
   RolloutWorker:
   1. Receive observations via env_channel
   2. Preprocess observations for model
   3. Generate actions with policy
   4. Send actions via rollout_channel
   5. Repeat until episode complete
   ```

## Configuration

### Environment Configuration

Key configuration parameters in `config/env/behavior_r1pro.yaml`:

```yaml
env_type: behavior
total_num_envs: 2  # Number of parallel environments
max_episode_steps: 2000
max_steps_per_rollout_epoch: 2000
task_idx: 0  # 0-49 for different BEHAVIOR tasks

video_cfg:
  save_video: True
  video_base_dir: ./logs/videos/eval

base_config_name: r1pro_behavior  # R1 Pro robot configuration
```

### Evaluation Configuration

Key parameters in `config/behavior_openvlaoft_eval.yaml`:

```yaml
runner:
  only_eval: True  # Evaluation-only mode
  eval_policy_path: /path/to/checkpoint.pt  # Model checkpoint

algorithm:
  eval_rollout_epoch: 10  # Number of evaluation episodes
  sampling_params:
    temperature_eval: 0.6  # Sampling temperature

env:
  eval:
    total_num_envs: 2  # Parallel environments
    max_steps_per_rollout_epoch: 2000

actor:
  model:
    num_action_chunks: 32  # Must match training config
    num_images_in_input: 3  # Head + 2 wrist cameras
```

## BEHAVIOR Tasks

The BEHAVIOR benchmark includes 50 household manipulation tasks:

| Task Index | Task Name | Description |
|------------|-----------|-------------|
| 0 | turning_on_radio | Turn on a radio |
| 1 | opening_packages | Open packages |
| 2 | packing_grocery_items_into_bags | Pack groceries |
| 3 | cleaning_windows | Clean windows |
| 4 | setting_table | Set dining table |
| ... | ... | ... |
| 49 | ... | ... |

For the complete list, see `rlinf/envs/behavior/behavior_task.jsonl`.

## Metrics

### Parallel Evaluation Metrics

Automatically collected by `EnvWorker`:

- **Success Rate**: Percentage of successful task completions
- **Episode Length**: Average number of steps per episode
- **Episode Return**: Cumulative rewards
- **Task-Specific Metrics**: Provided by OmniGibson

Results are aggregated across all workers and saved to:
- TensorBoard logs: `logs/[experiment]/tensorboard/`
- JSON summaries: `logs/[experiment]/metrics/`

### Single-Threaded Evaluation Metrics

Provides more detailed statistics:

```json
{
  "task_name": "turning_on_radio",
  "total_trials": 10,
  "successful_trials": 7,
  "success_rate": 0.7,
  "avg_episode_length": 342.5,
  "instance_results": [
    {
      "instance_id": 0,
      "success_rate": 1.0,
      "avg_episode_length": 298.0
    },
    ...
  ]
}
```

## Multi-GPU Evaluation

### Single Node, Multiple GPUs

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env: 0  # GPU 0 for environments
    rollout: 1  # GPU 1 for policy
```

### Multi-Node Evaluation

```yaml
cluster:
  num_nodes: 2
  component_placement:
    env: 0-1  # Envs on both nodes
    rollout: 0-1  # Policy on both nodes
```

```bash
# Start Ray cluster
# Node 0 (head):
bash ray_utils/start_ray.sh

# Node 1 (worker):
bash ray_utils/start_ray.sh

# Run evaluation on head node:
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval
```

## Comparison with Training

RLinf's evaluation follows the same architecture as training:

```
Training:                           Evaluation:
┌──────────────┐                   ┌──────────────┐
│ Env Workers  │                   │ Env Workers  │
├──────────────┤                   ├──────────────┤
│ Rollout      │                   │ Rollout      │
│ Workers      │                   │ Workers      │
├──────────────┤                   └──────────────┘
│ Actor        │                   (No actor needed)
│ Workers      │
│ (Training)   │
└──────────────┘
```

Key differences:
- Evaluation doesn't need actor workers (no gradient computation)
- Evaluation uses `only_eval=True` and `eval_rollout_epoch` parameter
- Evaluation loads checkpoint via `eval_policy_path`

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```yaml
   # Reduce number of parallel environments
   env.eval.total_num_envs: 1
   
   # Enable model offloading
   rollout.enable_offload: True
   ```

2. **Ray Connection Issues**
   ```bash
   # Check Ray status
   ray status
   
   # Restart Ray cluster
   ray stop
   bash ray_utils/start_ray.sh
   ```

3. **Environment Hanging**
   ```yaml
   # Reduce max episode steps
   env.eval.max_episode_steps: 1000
   
   # Check OmniGibson logs
   tail -f ~/.local/share/ov/logs/Kit/Isaac-Sim/*/kit.log
   ```

4. **Action Chunk Mismatch**
   ```
   Error: Action chunk size mismatch
   Solution: Ensure actor.model.num_action_chunks matches training config
   ```

## Advanced Usage

### Custom Evaluation Metrics

Extend `BehaviorEnv` to add custom metrics:

```python
# In rlinf/envs/behavior/behavior_env.py

def _record_metrics(self, rewards, infos):
    # Add custom metrics
    for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
        episode_info = {
            "success": info.get("done", {}).get("success", False),
            "episode_length": info.get("episode_length", 0),
            # Add your custom metrics here
            "custom_metric": self.compute_custom_metric(info),
        }
    # ...
```

### Evaluate on Custom Task Instances

```python
# Specify custom instance IDs
cfg.env.eval.instance_ids = [0, 1, 2, 3, 4]  # Only evaluate these instances
```

### Video Recording

```yaml
env:
  eval:
    video_cfg:
      save_video: True
      info_on_video: True  # Overlay metrics on video
      video_base_dir: ./videos/eval
```

## Performance Tips

1. **Optimal Environment Count**: Start with 2-4 environments per GPU
2. **Action Chunks**: Larger chunks (32) are faster than smaller ones (5)
3. **Model Offloading**: Enable for large models to save memory
4. **Pipeline Stages**: Use `pipeline_stage_num: 2` for better throughput

## Summary

**When to use Worker-Based Evaluation:**
- ✅ Evaluating trained models (production)
- ✅ Need fast evaluation across multiple instances
- ✅ Evaluating on multiple GPUs/nodes
- ✅ Consistent with training pipeline

**When to use Single-Threaded Evaluation:**
- ✅ Debugging model behavior
- ✅ Need detailed per-instance statistics
- ✅ Analyzing failure modes
- ✅ Creating evaluation reports

For most use cases, **worker-based evaluation is recommended** as it's faster, scales better, and follows RLinf's architecture.

## References

- [RLinf BEHAVIOR Documentation](../../docs/source-en/rst_source/examples/behavior.rst)
- [OmniGibson Documentation](https://behavior.stanford.edu/omnigibson/)
- [BEHAVIOR Benchmark](https://behavior.stanford.edu/)
- [Example Configs](./config/)

