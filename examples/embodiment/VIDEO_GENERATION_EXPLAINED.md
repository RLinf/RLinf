# Video Generation in BEHAVIOR Evaluation - Updated Behavior

## New Behavior (After Fix)

**Each environment generates its own video file.**

### Key Changes

Previously:
- ❌ 1 worker with N environments → **1 video** (only recorded first environment)

Now:
- ✅ 1 worker with N environments → **N videos** (one per environment)

## How It Works Now

### Video File Naming

Each video file is named with:
- Worker rank
- GPU ID  
- Environment index within that worker

**Format:** `behavior_video_rank{rank}_gpu{gpu}_env{env_idx}.mp4`

### Example 1: 2 Workers, 4 Environments Total

```yaml
cluster:
  component_placement:
    env: 0-1      # 2 workers
    rollout: 2-3  # 2 workers

env:
  eval:
    total_num_envs: 4  # 4 environments total
```

**Distribution:**
- Worker 0 (GPU 0): Manages 2 environments (env 0, env 1)
- Worker 1 (GPU 1): Manages 2 environments (env 0, env 1)

**Videos Generated:**
```
behavior_video_rank0_gpu0_env0.mp4  # Worker 0, Environment 0
behavior_video_rank0_gpu0_env1.mp4  # Worker 0, Environment 1
behavior_video_rank1_gpu1_env0.mp4  # Worker 1, Environment 0
behavior_video_rank1_gpu1_env1.mp4  # Worker 1, Environment 1
```

**Total: 4 videos** ✅

### Example 2: 4 Workers, 4 Environments Total

```yaml
cluster:
  component_placement:
    env: 0-3      # 4 workers
    rollout: 0-3  # 4 workers

env:
  eval:
    total_num_envs: 4  # 4 environments total
```

**Distribution:**
- Worker 0 (GPU 0): Manages 1 environment (env 0)
- Worker 1 (GPU 1): Manages 1 environment (env 0)
- Worker 2 (GPU 2): Manages 1 environment (env 0)
- Worker 3 (GPU 3): Manages 1 environment (env 0)

**Videos Generated:**
```
behavior_video_rank0_gpu0_env0.mp4  # Worker 0, Environment 0
behavior_video_rank1_gpu1_env0.mp4  # Worker 1, Environment 0
behavior_video_rank2_gpu2_env0.mp4  # Worker 2, Environment 0
behavior_video_rank3_gpu3_env0.mp4  # Worker 3, Environment 0
```

**Total: 4 videos** ✅

### Example 3: 1 Worker, 4 Environments Total

```yaml
cluster:
  component_placement:
    env: 0        # 1 worker
    rollout: 0    # 1 worker

env:
  eval:
    total_num_envs: 4  # 4 environments total
```

**Distribution:**
- Worker 0 (GPU 0): Manages 4 environments (env 0, 1, 2, 3)

**Videos Generated:**
```
behavior_video_rank0_gpu0_env0.mp4  # Worker 0, Environment 0
behavior_video_rank0_gpu0_env1.mp4  # Worker 0, Environment 1
behavior_video_rank0_gpu0_env2.mp4  # Worker 0, Environment 2
behavior_video_rank0_gpu0_env3.mp4  # Worker 0, Environment 3
```

**Total: 4 videos** ✅

## Formula

**Number of videos = `total_num_envs`**

Regardless of how many workers you use, you will get one video per environment.

## Code Changes Summary

### 1. Multiple Video Writers

**Before:**
```python
self._video_writer = None  # Single video writer
```

**After:**
```python
self._video_writers = []  # List of video writers, one per environment
```

### 2. Initialization

**Before:**
```python
# Create one video writer
self.video_writer = create_video_writer(...)
```

**After:**
```python
# Create one video writer per environment
for env_idx in range(self.cfg.total_num_envs):
    video_id = f"rank{rank}_gpu{gpu}_env{env_idx}"
    video_writer = create_video_writer(...)
    self._video_writers.append(video_writer)
```

### 3. Writing Videos

**Before:**
```python
# Only write first environment's observations
for sensor_data in raw_obs[0].values():
    ...
write_video(..., video_writer=self.video_writer)
```

**After:**
```python
# Write each environment's observations to its own video
for env_idx, env_obs in enumerate(raw_obs):
    video_writer = self.video_writers[env_idx]
    for sensor_data in env_obs.values():
        ...
    write_video(..., video_writer=video_writer)
```

## Benefits

1. **✅ True Parallel Evaluation**: Each environment's rollout is recorded separately
2. **✅ Independent Analysis**: Can analyze each environment's trajectory independently
3. **✅ No Data Loss**: All environments are recorded, not just the first one
4. **✅ Flexible Deployment**: Works regardless of worker count

## Configuration Guidelines

### For Maximum Throughput

Use more workers to distribute environments:

```yaml
cluster:
  component_placement:
    env: 0-7      # 8 workers
    rollout: 0-7  # 8 workers

env:
  eval:
    total_num_envs: 8  # 1 env per worker
```

**Result:** 8 videos, fast parallel execution

### For Memory-Constrained Setup

Use fewer workers, each managing multiple environments:

```yaml
cluster:
  component_placement:
    env: 0-1      # 2 workers
    rollout: 2-3  # 2 workers

env:
  eval:
    total_num_envs: 8  # 4 envs per worker
```

**Result:** 8 videos, lower memory usage per GPU

### For Testing

Use single worker:

```yaml
cluster:
  component_placement:
    env: 0        # 1 worker
    rollout: 0    # 1 worker

env:
  eval:
    total_num_envs: 2  # 2 envs on one worker
```

**Result:** 2 videos, minimal resource usage

## Verification

After running evaluation, check the video count:

```bash
ls -lh logs/behavior_eval/video/eval/ | grep behavior_video | wc -l
```

This should equal `total_num_envs`.

## Performance Considerations

### Video Writing Overhead

Each video writer adds a small overhead. For large-scale evaluation:

1. **Disable videos during development:**
   ```yaml
   env.eval.video_cfg.save_video: False
   ```

2. **Enable only for final runs:**
   ```yaml
   env.eval.video_cfg.save_video: True
   ```

### Disk Space

Each video file is approximately:
- Low-res (128x128): ~10-50 MB per 1000 steps
- High-res (480x480/960x960): ~100-500 MB per 1000 steps

For `total_num_envs: 8` with high-res:
- Disk usage: ~800 MB - 4 GB per evaluation run

## Summary

**Key Takeaway:** With the updated code, `total_num_envs` directly determines the number of videos generated, making it intuitive and predictable.

```
total_num_envs = 4  →  4 videos
total_num_envs = 8  →  8 videos
total_num_envs = 1  →  1 video
```

Worker configuration affects performance and memory usage, but not the number of videos! 🎥


