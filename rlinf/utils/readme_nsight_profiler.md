# Nsight Systems Profiler for RLinf

RLinf supports NVIDIA Nsight Systems profiling that co-exists with Ray. This enables GPU kernel-level visibility into all training components (actor, rollout, env, channel workers) with multi-layer NVTX annotations — from high-level worker methods down to individual GLOO/NCCL transfers.

## Prerequisites

- `nsys` CLI (Nsight Systems) installed and **in `PATH` at `ray start` time**
- `nvtx` Python package
- Ray >= 2.8 (for `runtime_env.nsight` support)

### Installation

```bash
# 1. Install Nsight Systems CLI (requires root)
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxkbcommon-x11-0 libxcb-xinput0 libxcb-cursor0 \
    libnss3 libxcomposite1 libxdamage1 libxtst6
cd /tmp
wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb
dpkg -i nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb
rm -f nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb

# 2. Install nvtx Python package
pip install nvtx

# 3. Verify
nsys --version
python -c "import nvtx; print('nvtx OK')"
```

**Important**: Ray workers inherit environment variables from `ray start` time.
Ensure `nsys` is in `PATH` *before* starting Ray:

```bash
export PATH="/usr/local/bin:$PATH"
which nsys  # must succeed
ray stop && ray start --head
```

## Configuration

Add the `nsight_profiler` section to your YAML config:

```yaml
nsight_profiler:
  steps: [2, 3]           # Which training steps to profile. null = disabled (zero overhead).
  discrete: false          # false = one nsys-rep segment per step (recommended)

  nsight_options:          # Passed to Ray runtime_env["nsight"]
    t: "cuda,cudnn,cublas,nvtx,osrt"
    cuda-memory-usage: "true"
    capture-range: "cudaProfilerApi"
    capture-range-end: "repeat-shutdown:10"

  actor:
    enable: true
    all_ranks: true
    ranks: []
  rollout:
    enable: true
    all_ranks: true
    ranks: []
  env:
    enable: true
    all_ranks: true
    ranks: []
```

### Key options

| Option | Description |
|--------|-------------|
| `steps` | List of training step indices to profile. `null` = disabled. Warmup steps (e.g. `[2, 3]`) recommended to skip initialization noise. |
| `discrete` | `false` (recommended): one continuous `.nsys-rep` per step. `true`: one per `@annotate` function. |
| `nsight_options` | Passed directly to Ray `runtime_env["nsight"]`. `capture-range: "cudaProfilerApi"` required. Add `osrt` for OS-level socket tracing (GLOO). |
| `capture-range-end` | `"repeat-shutdown:10"` allows multiple start/stop cycles without killing the worker. |
| Per-role config | `enable`, `all_ranks`, `ranks` control which workers are launched under nsys. |

## Architecture overview

The profiling feature has 6 layers, each independently zero-overhead when disabled:

```
┌─────────────────────────────────────────────────────────────────┐
│ F: YAML Config → Train Entrypoint                               │
│    nsight_profiler config → _get_role_nsight_options(role)       │
│    → WorkerGroup.launch(nsight_options=...)                      │
│    → Channel.create(nsight_options=...)                          │
├─────────────────────────────────────────────────────────────────┤
│ A: Profiler Infrastructure                                       │
│    cluster.py: runtime_env["nsight"] = nsight_options            │
│    worker_group.py: per-rank output naming                       │
│    worker.py: start_profile/stop_profile → cudaProfilerApi       │
│    nsight_profiler.py: NsightProfiler, @annotate, nvtx_range    │
├─────────────────────────────────────────────────────────────────┤
│ E: Runner Control (per-step profiling window)                    │
│    EmbodiedRunner / AsyncEmbodiedRunner / AsyncPPORunner         │
│    _should_profile(step) → _start_profiling → _stop_profiling   │
├─────────────────────────────────────────────────────────────────┤
│ D: Worker Method Annotations (@NsightProfiler.annotate)          │
│    env/step, rollout/predict, actor/run_training, etc.           │
├─────────────────────────────────────────────────────────────────┤
│ B: Channel NVTX Telemetry                                        │
│    channel.py: put/get payload size, queue depth, sub-phases    │
│    channel_worker.py: recv_from_producer, enqueue, dequeue,     │
│                        send_to_consumer (in ChannelWorker proc) │
├─────────────────────────────────────────────────────────────────┤
│ C: Collective & Process Group NVTX                               │
│    collective_group.py: send/recv type, serialize/deserialize,  │
│                          metadata vs cpu/accel payload           │
│    multi_channel_pg.py: pg/send|recv backend, bytes, D2H/H2D   │
│    ↓ dist.broadcast() → nsys auto-captures NCCL kernel / GLOO  │
└─────────────────────────────────────────────────────────────────┘
```

## How it works

```
Runner (controller)
│
├─ Checks if current step is in nsight_profiler.steps
├─ Calls start_profile(step) on actor, rollout, env worker groups
│   └─ Workers call torch.cuda.profiler.start()
│      (nsys with capture-range=cudaProfilerApi begins recording)
│      set_channel_nvtx_enabled(True)
│
├─ Training step executes normally
│   └─ Multi-layer NVTX ranges recorded in the timeline
│
└─ Calls stop_profile() on all worker groups
    └─ Workers call torch.cuda.profiler.stop()
       → nsys writes .nsys-rep segment

ChannelWorkers run separately with capture-range=none (always recording)
and set_channel_nvtx_enabled(True) at init time.
```

## NVTX annotation table

### Layer D: Worker methods

| Worker | Method | NVTX Label | What it captures |
|--------|--------|------------|-----------------|
| EnvWorker | `recv_chunk_actions()` | `env/recv_actions` | Channel.get wait time |
| EnvWorker | `env_interact_step()` | `env/step` | Env physics + rendering |
| EnvWorker | `send_env_batch()` | `env/send_obs` | Channel.put to rollout |
| EnvWorker | `interact()` | `env/interact` | Full epoch envelope |
| RolloutWorker | `recv_env_output()` | `rollout/recv_obs` | Channel.get wait time |
| RolloutWorker | `predict()` | `rollout/predict` | Model inference |
| RolloutWorker | `send_chunk_actions()` | `rollout/send_actions` | Channel.put to env |
| RolloutWorker | `send_rollout_result()` | `rollout/send_traj` | Channel.put to env (rollout result) |
| RolloutWorker | `generate_one_epoch()` | `rollout/generate_epoch` | One epoch loop |
| RolloutWorker | `generate()` | `rollout/generate` | Full generate envelope |
| ActorWorker | `recv_rollout_trajectories()` | `actor/recv_traj` | Channel.get from env |
| ActorWorker | `compute_advantages_and_returns()` | `actor/compute_adv` | GAE computation |
| ActorWorker | `run_training()` | `actor/run_training` | Training update (fwd+bwd+optim) |
| ActorWorker | `sync_model_to_rollout()` | `actor/sync_weights_to_rollout` | Direct P2P weight send |

### Layer B: Channel telemetry

| Location | NVTX Label | Info embedded |
|----------|------------|---------------|
| Channel.put | `channel/{name}/put bytes=... cpu=... gpu=... pickle=...` | Payload size breakdown |
| Channel.put | `channel/{name}/put/send` | Worker.send phase |
| Channel.put | `channel/{name}/put/enqueue` | AsyncChannelWork wait |
| Channel.get | `channel/{name}/get key=... src=...` | Outer envelope |
| Channel.get | `channel/{name}/get/wait_recv` | Worker.recv blocking wait |
| ChannelWorker | `cw/{name}/recv_from_producer src=...` | Recv from producer worker |
| ChannelWorker | `cw/{name}/enqueue key=... qsize_before=N` | Queue insert + depth |
| ChannelWorker | `cw/{name}/dequeue key=... qsize_before=N` | Queue wait + depth |
| ChannelWorker | `cw/{name}/send_to_consumer dst=...` | Send to consumer worker |

### Layer C: Collective & process group

| Location | NVTX Label | Info embedded |
|----------|------------|---------------|
| CollectiveGroup | `collective/send type=TENSOR_LIST transport=NCCL peer=0` | Object type + transport |
| CollectiveGroup | `collective/recv type=OBJECT peer=0` | Object type |
| CollectiveGroup | `collective/serialize` / `collective/deserialize` | Pickle/unpickle timing |
| CollectiveGroup | `collective/send_tensor_list/metadata n_tensors=5` | Metadata phase |
| CollectiveGroup | `collective/send_tensor_list/cpu_payload n=3 bytes=76800` | CPU tensor transfer |
| CollectiveGroup | `collective/send_tensor_list/accel_payload n=2 bytes=76800 mode=NCCL` | GPU tensor transfer + IPC/NCCL mode |
| MultiChannelPG | `pg/send backend=NCCL bytes=38400 dtype=float32` | Final backend + byte count |
| MultiChannelPG | `pg/recv backend=GLOO bytes=342 dtype=uint8` | Final backend + byte count |
| MultiChannelPG | `pg/D2H_copy bytes=... dtype=...` | GPU→CPU memcpy (PCIe) |
| MultiChannelPG | `pg/H2D_copy bytes=... dtype=...` | CPU→GPU memcpy (PCIe) |

## Output files

Profiling results are saved to `/tmp/ray/session_latest/logs/nsight/` on each node.

### Business workers (capture-range: cudaProfilerApi)

One `.nsys-rep` per profiled step:
```
ActorGroup_rank0_pid{PID}.1.nsys-rep     ← steps[0] (e.g. step 2)
ActorGroup_rank0_pid{PID}.2.nsys-rep     ← steps[1] (e.g. step 3)
RolloutGroup_rank0_pid{PID}.1.nsys-rep
EnvGroup_rank0_pid{PID}.1.nsys-rep
```

The `.N` suffix is nsys's internal capture segment counter (1-indexed), **not** the training step number. Segment N corresponds to `steps[N-1]` in your config.

### ChannelWorkers (capture-range: none, always recording)

One file covering the entire training lifetime:
```
Channel_Env_rank0_pid{PID}.nsys-rep
Channel_Rollout_rank0_pid{PID}.nsys-rep
Channel_Actor_rank0_pid{PID}.nsys-rep
Channel_ReplayBuffer_rank0_pid{PID}.nsys-rep
```

### Viewing results

1. Copy `.nsys-rep` files to a machine with Nsight Systems GUI
2. Open **all files** for a multi-process timeline view (time-aligned)
3. NVTX ranges appear as labeled colored blocks; expand threads to see background collective work

## Physical data paths (2-node async example)

```
Path 1: Env ↔ Rollout (obs & actions) — same node = loopback
  EnvWorker → GLOO(loopback) → Channel_Env → GLOO(loopback) → RolloutWorker

Path 2: Env → Actor (trajectory) — cross-node = IB
  EnvWorker → GLOO → Channel_ReplayBuffer → GLOO(IB) → ActorWorker

Path 3: Actor → Rollout (weight sync) — direct P2P, no Channel
  ActorWorker → Worker.send(NCCL/IB) → RolloutWorker
```

Key insight: Channel data always passes through a **ChannelWorker** process (2 hops).
Weight sync uses **direct P2P** between actor and rollout (1 hop, no ChannelWorker).

## Reading hardware-level metrics from nsys

| What you want | Where to look in nsys GUI |
|---------------|--------------------------|
| NCCL/RDMA transfer time | `pg/send backend=NCCL bytes=...` NVTX range → align with CUDA HW → Kernels row (`ncclKernel_Broadcast`) |
| GLOO/TCP transfer time | `pg/send backend=GLOO bytes=...` NVTX range → align with OS Runtime row (`sendmsg` / `write` syscall) |
| PCIe D2H/H2D time | `pg/D2H_copy bytes=...` or `pg/H2D_copy bytes=...` → align with CUDA HW → Memory row (`cudaMemcpyAsync`) |
| Pickle serialization | `collective/serialize` / `collective/deserialize` NVTX range duration |
| Queue backpressure | `cw/{name}/enqueue qsize_before=N` (ChannelWorker side) — growing N means consumer is slower than producer |
| Effective bandwidth | `bytes / duration` from any `pg/send` or `pg/recv` range |

## Troubleshooting

### `nsight is not installed` (Ray runtime env error)

Ray's nsight plugin runs `shutil.which("nsys")` in the worker process. If `nsys` is not in `PATH` when `ray start` was called, workers won't find it. Fix:

```bash
export PATH="/usr/local/bin:$PATH"
ray stop && ray start --head   # must restart Ray
```

### nsys kills worker processes after first `cudaProfilerStop`

Missing `capture-range-end`. nsys defaults to terminating after the first stop. Fix:

```yaml
nsight_options:
  capture-range: "cudaProfilerApi"
  capture-range-end: "repeat-shutdown:10"
```

**Do not** add `stop-on-exit` — it is not a valid `nsys profile` argument and
will cause Ray runtime env creation to hang silently.

### `WARNING: CPU IP/backtrace sampling not supported`

Caused by `kernel.perf_event_paranoid` being too restrictive. **Does not affect
GPU profiling, NVTX, or CUDA kernel tracing.** Fix (requires root on host):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
```

### `WARNING: GPU initialization took too long`

Harmless. Fix by running the NVIDIA persistence daemon:

```bash
sudo nvidia-persistenced --persistence-mode
```

### Async runner: bubble between profiled steps

In async mode, env/rollout are long-running tasks. `start_profile`/`stop_profile`
are Ray remote calls that must wait for async actors to yield. This creates gaps
between profiled steps. This is expected — the data within each profiled step is
still accurate.

## Disabling profiling

Set `steps: null` in the config (or omit the `nsight_profiler` section entirely). When disabled:
- No `nsight` key in Ray `runtime_env` (workers not launched under nsys)
- All `@NsightProfiler.annotate` decorators are no-ops (zero overhead)
- Channel/collective NVTX is disabled
- ChannelWorkers are not profiled
