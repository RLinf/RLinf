# Nsight Systems Profiler for RLinf

RLinf supports NVIDIA Nsight Systems profiling with multi-layer NVTX annotations for the embodied training pipeline. When enabled, workers are launched under `nsys` via Ray `runtime_env["nsight"]`, and per-step profiling is controlled by the runner.

For full documentation, see the Sphinx tutorials:
- **Code design**: `docs/source-en/rst_source/tutorials/advance/nsight_code_design.rst`
- **Timeline analysis guide**: `docs/source-en/rst_source/tutorials/advance/nsight_profiler_guide.rst`

## Quick start

### Prerequisites

```bash
# 1. Install Nsight Systems CLI (if not already in your container)
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxkbcommon-x11-0 libxcb-xinput0 libxcb-cursor0 \
    libnss3 libxcomposite1 libxdamage1 libxtst6
cd /tmp && wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb \
    && dpkg -i nsight-systems-*.deb && rm -f nsight-systems-*.deb

# 2. Install nvtx Python package
pip install nvtx

# 3. Verify and restart Ray (workers inherit PATH from ray start)
nsys --version && python -c "import nvtx; print('nvtx OK')"
export PATH="/usr/local/bin:$PATH"
ray stop && ray start --head
```

### YAML configuration

```yaml
nsight_profiler:
  steps: [2, 3]           # training steps to profile (null = disabled)
  discrete: false
  nsight_options:
    t: "cuda,cudnn,cublas,nvtx,osrt"
    cuda-memory-usage: "true"
    capture-range: "cudaProfilerApi"
    capture-range-end: "repeat-shutdown:10"
  actor:
    enable: true
    all_ranks: true
  rollout:
    enable: true
    all_ranks: true
  env:
    enable: true
    all_ranks: true
```

### Output files

Results are saved to `/tmp/ray/session_latest/logs/nsight/` on each node:

```
ActorGroup_rank0_pid{PID}.1.nsys-rep      # step 2 (segment 1)
ActorGroup_rank0_pid{PID}.2.nsys-rep      # step 3 (segment 2)
RolloutGroup_rank0_pid{PID}.1.nsys-rep
EnvGroup_rank0_pid{PID}.1.nsys-rep
Channel_Env_rank0_pid{PID}.nsys-rep       # always-on recording
Channel_Rollout_rank0_pid{PID}.nsys-rep
```

Open all files in Nsight Systems GUI for a time-aligned multi-process timeline.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `nsight is not installed` (Ray error) | Ensure `nsys` is in PATH before `ray start` |
| nsys kills worker after first stop | Add `capture-range-end: "repeat-shutdown:10"` |
| `CPU IP/backtrace sampling not supported` | Harmless; fix with `sudo sysctl -w kernel.perf_event_paranoid=1` |
| Bubble between profiled steps (async) | Expected — async actors must yield before start/stop takes effect |

## Disabling

Set `steps: null` or omit `nsight_profiler` entirely. All instrumentation becomes zero-overhead.
