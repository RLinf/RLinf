# Multi-GPU BEHAVIOR Evaluation Guide

## Overview

This guide explains how to run BEHAVIOR evaluation across multiple GPUs with proper video recording.

## GPU Configuration

### Option 1: 4 GPUs (Recommended)

**Dedicated GPUs for Env and Rollout:**

```yaml
cluster:
  component_placement:
    env: 0-1      # Env workers on GPU 0 and GPU 1
    rollout: 2-3  # Rollout workers on GPU 2 and GPU 3

env:
  eval:
    total_num_envs: 2  # 1 env per GPU (GPU 0 and GPU 1)
```

**Resource allocation:**
- GPU 0: EnvWorker 0 (1 environment)
- GPU 1: EnvWorker 1 (1 environment)
- GPU 2: RolloutWorker 0 (policy inference)
- GPU 3: RolloutWorker 1 (policy inference)

**Advantages:**
- No GPU contention
- Best performance
- Can scale to more envs if needed

---

### Option 2: 2 GPUs (Shared)

**Shared GPUs for Both Env and Rollout:**

```yaml
cluster:
  component_placement:
    env: 0-1      # Env workers on GPU 0 and GPU 1
    rollout: 0-1  # Rollout workers also on GPU 0 and GPU 1 (shared)

env:
  eval:
    total_num_envs: 2  # 1 env per GPU
```

**Resource allocation:**
- GPU 0: EnvWorker 0 + RolloutWorker 0 (shared)
- GPU 1: EnvWorker 1 + RolloutWorker 1 (shared)

**Considerations:**
- GPU memory must fit both env and policy
- Possible GPU memory contention
- Slower than dedicated GPUs

---

### Option 3: Single GPU (Baseline)

**All components on one GPU:**

```yaml
cluster:
  component_placement:
    env: 0        # Env worker on GPU 0
    rollout: 0    # Rollout worker also on GPU 0

env:
  eval:
    total_num_envs: 1  # Single environment
```

---

## Video Recording with Multiple Envs

### Automatic Video Naming

When `total_num_envs > 1`, each environment worker creates a **separate video file** to avoid overwrites:

```
logs/behavior_eval/video/eval/
├── behavior_video_worker0_rank0.mp4  # From EnvWorker 0
└── behavior_video_worker1_rank0.mp4  # From EnvWorker 1
```

The naming format is: `behavior_video_worker{worker_id}_rank{rank}.mp4`

### Implementation

```python
# In behavior_env.py line 85-87
worker_id = f"worker{worker_info.worker_id}_rank{worker_info.rank}"
video_name = f"{video_base_dir}/behavior_video_{worker_id}.mp4"
```

This ensures:
- ✅ No file conflicts between workers
- ✅ Each environment's rollout is recorded separately
- ✅ Easy to identify which worker generated which video

---

## Memory Considerations

### Per-Component Memory Usage (Approximate)

| Component | GPU Memory |
|-----------|-----------|
| **OmniGibson Env** (1 instance) | ~4-6 GB |
| **OpenVLA Policy** (7B params) | ~14-16 GB |
| **Total per GPU** (shared) | ~18-22 GB |

### GPU Requirements by Configuration

| Configuration | Min GPU Memory | Recommended |
|--------------|----------------|-------------|
| **Option 1** (4 GPUs) | 16 GB per GPU | 24 GB per GPU |
| **Option 2** (2 GPUs shared) | 24 GB per GPU | 40 GB per GPU |
| **Option 3** (1 GPU) | 24 GB | 40 GB |

---

## Scaling Guidelines

### Increasing Number of Environments

To run more environments, adjust both `total_num_envs` and GPU placement:

```yaml
# Example: 4 environments on 4 GPUs
cluster:
  component_placement:
    env: 0-3      # 4 env workers
    rollout: 4-7  # 4 rollout workers (dedicated)

env:
  eval:
    total_num_envs: 4  # 1 env per GPU
```

This creates:
- 4 environment workers (GPU 0-3)
- 4 rollout workers (GPU 4-7)
- 4 separate video files

### Balancing Throughput vs Resources

| Setup | Throughput | GPU Usage | Videos |
|-------|-----------|-----------|---------|
| 1 env, 1 GPU | 1x | Minimal | 1 |
| 2 envs, 2 GPUs | ~2x | Medium | 2 |
| 4 envs, 4 GPUs | ~4x | High | 4 |

---

## Example Configurations

### High-Throughput (8 GPUs)

```yaml
cluster:
  component_placement:
    env: 0-3        # 4 env workers
    rollout: 4-7    # 4 rollout workers

env:
  eval:
    total_num_envs: 4
```

### Memory-Constrained (2 GPUs, small model)

```yaml
cluster:
  component_placement:
    env: 0
    rollout: 1

env:
  eval:
    total_num_envs: 1  # Single env to minimize memory
```

---

## Troubleshooting

### Issue: OOM (Out of Memory)

**Symptoms:** `torch.cuda.OutOfMemoryError`

**Solutions:**
1. Reduce `total_num_envs`
2. Use dedicated GPUs for env and rollout
3. Disable high-res cameras: `use_high_res_cameras: False`

### Issue: Videos Overwriting

**Symptoms:** Only one video file exists, or video is corrupted

**Check:**
- Verify worker_info is passed correctly in `behavior_env.py` line 86
- Check that different workers have different worker IDs
- Look for multiple `.mp4` files in video directory

### Issue: Slow Evaluation

**Possible causes:**
1. GPU contention (shared env/rollout on same GPU)
2. Single environment bottleneck

**Solutions:**
1. Use dedicated GPUs (Option 1)
2. Increase `total_num_envs` to parallelize

---

## Best Practices

1. **Start Small**: Test with 1 env on 1 GPU first
2. **Monitor Memory**: Use `nvidia-smi` to check GPU utilization
3. **Scale Gradually**: Add more envs/GPUs incrementally
4. **Save Videos Selectively**: Only enable for final evaluation (uses disk space)
5. **Check Video Files**: Verify all workers created separate videos

---

## Summary

**Recommended Setup for Your Case (2 envs, 2 rollout GPUs):**

```yaml
cluster:
  component_placement:
    env: 0-1      # GPU 0 and GPU 1 for environments
    rollout: 2-3  # GPU 2 and GPU 3 for rollout (or 0-1 if only 2 GPUs)

env:
  eval:
    total_num_envs: 2
    video_cfg:
      save_video: True
      use_high_res_cameras: True
```

**Expected Output:**
- 2 environment workers running in parallel
- 2 rollout workers for policy inference
- 2 video files: `behavior_video_worker0_rank0.mp4`, `behavior_video_worker1_rank0.mp4`
- ~2x evaluation throughput compared to single GPU

