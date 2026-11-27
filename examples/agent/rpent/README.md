# RPent Multi-Turn Embodied GRPO Training

This example trains a 4B Vision-Language-Action reasoning model (Qwen3.5-4B) with synchronous multi-turn GRPO and oversampling across multiple manipulation simulators (LIBERO and RoboTwin, with extensible registry support for RoboCasa, RoboDojo and MetaWorld) served by [RPent](https://github.com/RLinf/RPent).

## Architecture

```
                     ┌──────────────────────────────────────────────────────────┐
                     │            8-GPU Single-Node Layout                      │
                     ├────────────────────────────┬─────────────────────────────┤
                     │ GPUs 0–3 (Pure-RL Image)   │ GPUs 4–7 (Isolated Venvs)   │
                     │                            │                             │
                     │ • RPentGRPORunner          │ • SAM3 Pool (envs/sam3)     │
                     │ • MegatronActor (DP/TP)    │     1 server/GPU + router   │
                     │ • SGLang Rollout (DP=4)    │ • LIBERO Pool (envs/libero) │
                     │   + External Qwen-VL       │     16 warm EGL workers     │
                     │     Capture Processor      │ • RoboTwin Pool             │
                     │ • RPentAgentLoopWorker     │     (envs/robotwin)         │
                     │   (async tool loop)        │     16 warm SAPIEN workers  │
                     └────────────────────────────┴─────────────────────────────┘
```

### Core Modules

| File | Responsibility |
| :--- | :--- |
| `rlinf/runners/rpent_runner.py` | Dedicated `RPentGRPORunner` implementing Sync GRPO + oversampling, dynamic cohort scheduling, and startup `validate_sim_pool_capacity()` checks. |
| `rlinf/agents/rpent/agent_loop.py` | Async multi-turn tool-calling agent loop (`RPentAgentLoopWorker`), binary `/execute` HTTP client, validation retries, and infra retry handling. |
| `rlinf/agents/rpent/history.py` | `EpisodeConversation` + `AppendOnlyHistory` (Pydantic-AI–compatible compaction, token-ID prefix caching, prefix-chain merging for Megatron training) using zero-Base64 raw `bytes` references. |
| `rlinf/agents/rpent/simulators.py` | Multi-simulator registry, task JSONL parsing, and balanced per-domain sampling quotas. |
| `rlinf/hybrid_engines/sglang/multimodal/qwen_vl.py` | External SGLang multimodal processor (`SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`) with LRU image-tensor caching — **zero SGLang source patches**. |
| `rlinf/hybrid_engines/sglang/multimodal/replay.py` | Local tmpfs image staging (`stage_local_images`) + safetensors replay payload capture for exact Megatron vision-input reconstruction. |
| `rlinf/hybrid_engines/megatron/qwen_vl_inputs.py` | Qwen3.5 VL multimodal input assembly for Megatron forward/backward without modifying Megatron-LM sources. |
| `examples/agent/rpent/config/train_qwen3.5_4b_rpent_grpo.yaml` | 8-GPU Sync GRPO + oversampling config (`seq_length=262144`, `max_turns=50`, `max_new_tokens=16384`, `group_size=8`). |
| `examples/agent/rpent/config/train_qwen3.5_4b_rpent_grpo_48gpu.yaml` | 48-GPU cross-cluster Sync GRPO config (`32 × 8` oversampled to `384`, early per-simulator straggler cancellation, collocated `/pool/{sleep,wake}`, `max_fraction=0.25`). |

## Key Optimizations

1. **Zero engine patches & isolated environments**:
   - No source modifications to Megatron-LM or SGLang; SGLang loads `rlinf.hybrid_engines.sglang.multimodal` via `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`, and Qwen3.5 GDN `out_norm` weights are converted cleanly in `Qwen35Convertor` (with fp32 `+1.0` promotion before bf16 cast).
   - The RL training container remains pure-RL (only `fla-core` for Megatron Qwen3.5 `GatedDeltaNet`), while `sam3`, `libero`, and `robotwin` each run in their own isolated virtual environment.
2. **Full oversampling concurrency with startup validation**:
   - `tools.rpent.episodes_per_worker: null` removes client-side semaphore throttling when simulator pools are sized to `sampling_batch_size × group_size` (e.g., 16 LIBERO + 16 RoboTwin = 32 concurrent episodes).
   - `RPentGRPORunner` queries every simulator `/health` endpoint before step 0 and fails fast if resident workers are below the oversampling quota.
3. **Deferred simulator rendering**:
   - Training runs with `tools.rpent.record_artifacts: false`, disabling per-physics-step offscreen RGB-D rendering in LIBERO and RoboTwin while preserving `record_artifacts: true` for evaluation video capture.
4. **End-to-end zero-Base64 image pipeline**:
   - Binary `/execute` HTTP framing streams raw PNG bytes from RPent to `RPentAgentLoopWorker`.
   - `EpisodeConversation` and `AppendOnlyHistory` keep lightweight `rpent-image://<fid>` placeholders and reuse raw `bytes` by reference without deep copies.
   - `stage_local_images()` stages each unique PNG once by content hash to local tmpfs (`/tmp/rlinf-mm-imgs-<pid>/`) so SGLang receives file paths instead of repeated multi-megabyte Base64 JSON payloads.

## Quick Start

### 1. Prepare Task JSONL Files

```bash
python examples/agent/rpent/prepare_tasks.py \
  --libero-out /path/to/data/libero/train.jsonl \
  --robotwin-out /path/to/data/robotwin/train.jsonl
```

In the MetaWorld Python 3.11 environment, with both repositories on `PYTHONPATH`, prepare the MT50, ML45 train and ML45 held-out pools for native index 49 under `${RPENT_ROOT}/data/metaworld`:

```bash
python examples/agent/rpent/prepare_tasks.py \
  --metaworld \
  --seeds 49 \
  --output-dir "${RPENT_ROOT:-/path/to/rpent_ppo}/data/metaworld"
```

To mix MetaWorld into the 8-GPU GRPO config (`train_qwen3.5_4b_rpent_grpo.yaml`, which sets `rollout_batch_size: 2`, `sampling_batch_size: 4`, `group_size: 8` → `16` rollout / `32` sampling episodes total), **replace** the default `env.rpent` block (`libero` + `robotwin`) with the `libero` + `metaworld` block below (do not merge it on top of `robotwin` without adjusting `rollout_episodes` / `sampling_episodes` so they still sum to `16` / `32`), and launch the RPent MetaWorld pool with at least `--workers 16` (matching `sampling_episodes: 16`) so startup capacity validation passes:

```yaml
env:
  rpent:
    libero:
      endpoint: http://127.0.0.1:28561
      train_data_paths: ${oc.env:RPENT_ROOT,/path/to/rpent_ppo}/data/libero/train.jsonl
      rollout_episodes: 8
      sampling_episodes: 16
    metaworld:
      endpoint: http://127.0.0.1:28565
      train_data_paths: ${oc.env:RPENT_ROOT,/path/to/rpent_ppo}/data/metaworld/ml45-train.jsonl
      rollout_episodes: 8
      sampling_episodes: 16
```

> **MetaWorld runtime notes**: RPent clamps `max_episode_steps` to MetaWorld's native **500-step horizon** regardless of larger requested values. MetaWorld's fixed-wrist Sawyer enforces a strict **4-DoF `move_to` schema (`xyz`-only)** — passing `orientation_rpy` triggers a validation retry with MetaWorld's 4-DoF hint, while observations still report the calibrated live `pose.orientation_rpy`.

### 2. Start Simulator & SAM3 Services (GPUs 4–7)

Using the RPent service supervisor (see `RPent/robots/rl/README.md` and `RPent/tools/rl/sim_services.example.yaml`):

```bash
PYTHONPATH=/path/to/RPent:/path/to/RLinf \
  python /path/to/RPent/tools/rl/serve.py --config /path/to/sim_services.yaml
```

### 3. Launch 8-GPU Sync GRPO Training (GPUs 0–3)

```bash
bash examples/agent/rpent/run_train.sh \
  runner.experiment_name=rpent-qwen3.5-4b-grpo \
  runner.max_steps=100 \
  data.seed=42
```

For a 1-step smoke test verifying weight sync, full 32-episode concurrency, and Megatron GRPO update:

```bash
bash examples/agent/rpent/run_train.sh \
  runner.experiment_name=rpent-qwen3.5-4b-grpo-smoke \
  runner.max_steps=1 \
  data.seed=42
```

> **Note on CUDA IPC weight sync (driver 535.x)**: `actor.env_vars` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` and `PYTORCH_ALLOC_CONF=expandable_segments:False` so `storage._share_cuda_()` succeeds across Actor→Rollout IPC transfers.

## 48-GPU Cross-Cluster Training (`train_qwen3.5_4b_rpent_grpo_48gpu.yaml`)

For large-scale synchronous GRPO (`rollout_batch_size=32`, `sampling_batch_size=48`, `group_size=8` → **384 concurrent oversampled episodes**: 24 LIBERO + 24 RoboTwin groups, taking the fastest 16 complete groups per simulator), RLinf coordinates a 4-node / 32-GPU Ray cluster with multi-node RPent `SimPoolRouter` frontends spanning heterogeneous H100 and A100 clusters.

### Reference 48-GPU Topology (5× H100 + 1× A100 via Jump Host)

| Node | Role | Per-GPU Allocation | Node Total |
| :--- | :--- | :--- | :--- |
| `h100` (8× H100) | Dedicated Sim + SAM3 (`sleepable: false`) | `4 LIBERO + 6 RoboTwin + 2 SAM3` | `32 LIBERO + 48 RoboTwin + 16 SAM3` |
| `h100_1`–`h100_4` (32× H100) | Ray RL Cluster + Collocated Sim (`sleepable: true`) | `5 LIBERO + 2 RoboTwin` + SGLang (`DP=32`) / Megatron (`TP=2, DP=16`) | `160 LIBERO + 64 RoboTwin + 32 RL` |
| `a100` (8× A100) | Remote Dedicated Sim + SAM3 (`sleepable: false`, bridged via jump-host SSH tunnel) | `10 RoboTwin + 1 SAM3` | `80 RoboTwin + 8 SAM3` |
| **Total (48 GPUs)** | | | **192 LIBERO + 192 RoboTwin + 24 SAM3 (queue=2048) + 32 RL GPUs** |

### Multi-Node Features & Launch

1. **Early per-simulator straggler cancellation**: As soon as either simulator meets its `rollout_episodes / group_size` quota (16 complete groups), `RPentGRPORunner` immediately calls `RPentAgentLoopWorker.cancel_candidate_ids()` to abort unselected stragglers in that domain, freeing SGLang KV/Mamba cache and GPU compute for the other domain.
2. **Collocated pool sleep/wake lifecycle**: With `tools.rpent.sleep_sim_pools_during_training: true`, `RPentGRPORunner` invokes `POST /pool/sleep` on every router before Megatron `recompute_logprobs` + backward (`~8.8s`, freeing collocated H100 VRAM to `~5.5 GiB` while dedicated `h100`/`a100` nodes stay warm) and `POST /pool/wake` (`~20.6s`) after weight sync.
3. **Long-tail context compression (`max_fraction: 0.25`)**: Compacts conversation history at 25% of the 262k window (`~65k` tokens) to bound late-turn prefill/decode tail latency across 384 concurrent rollouts.

```bash
# On each of the 4 RL nodes (set NODE_RANK=0..3 and ulimit -n 1048576):
# Start Ray with --temp-dir=/tmp/rpent_ppo/ray (backed by /dev/shm/rpent_ppo)
# Then launch from head node (h100_1):
ulimit -n 1048576
export RPENT_ROOT=/path/to/rpent_ppo
export RPENT_LOCAL_CACHE=/tmp/rpent_ppo        # symlink -> /dev/shm/rpent_ppo
export RPENT_ROUTER_HOST=<H100_HEAD_IP>
export RAY_ADDRESS=<H100_HEAD_IP>:29710
export NO_PROXY=127.0.0.1,localhost,<VPC_CIDR>

python examples/agent/rpent/train.py \
  --config-name train_qwen3.5_4b_rpent_grpo_48gpu
```

### Cross-Cluster Production Issues & Fixes

The following issues were hit and resolved during 48-GPU cross-cluster bring-up and verified end-to-end across full Step 1 (`32/32` weight sync, `384/384` zero-queue concurrency, `17.97%` success rate, `retry=0.0`, `loss=0.0125`, `peak VRAM=79.2 GiB`):

| Issue / Failure | Root Cause | Fix & Operational Guidance |
| :--- | :--- | :--- |
| **`AttributeError` in `rlinf/utils/patcher.py` on H100 (Hopper)** | Hopper's `deep_gemm` inserts non-module sentinel objects lacking `__dict__` into `sys.modules`. | Guarded `_Patcher` in `rlinf/utils/patcher.py` with `if not hasattr(value, "__dict__"): continue`. |
| **`GenerateInput(rank=0)` `SIGABRT: Too many open files`** | 32-GPU Ray/Gloo `ChannelWorker` opens `>1,400` TCP FDs (64 Gloo groups + Ray sockets), exceeding container default `ulimit -n 1024`. | Set `ulimit -n 1048576` in Ray node startup, `run_train.sh`, and SSH tunnel scripts (and `resource.setrlimit` in RPent `server.py`/`serve.py`). |
| **`httpx.RemoteProtocolError: Server disconnected without sending a response`** | Client `httpx` default `keepalive_expiry=5.0` raced with `uvicorn`'s default `timeout_keep_alive=5` across the cross-cluster SSH tunnel, and bare open errors hid HTTP response details. | Set `AgentLoopWorker._client()` `keepalive_expiry=30.0` (strictly below RPent server `timeout_keep_alive=300`), included `resp.text[:300]` in `HTTPStatusError`, mapped router `TransportError` → HTTP `503`, and set `max_infra_retries: 3`. |
| **Shared BeeGFS short-reads & container `/` overlay eviction** | Concurrent reads of Python packages, `@triton.jit` sources (`fla-core` `OSError: could not get source code`), and simulator assets from BeeGFS caused intermittent 0-byte reads; copying ~45 GB into container `/tmp` risked 100 GB overlay eviction. | Stage `RLinf`, `RPent`, `envs/rl-overlay`, `checkpoint-216`, simulator venvs, and assets into node-local RAM tmpfs (`/dev/shm/rpent_ppo`, symlinked at `/tmp/rpent_ppo`), and point Ray `--temp-dir` to `/tmp/rpent_ppo/ray`. Keeps `/` overlay at `<1.2 GiB / 100 GiB (1%–2%)` and enables `~42s` RAM-to-RAM node recovery (`1.07 GB/s`). |
| **Megatron backward OOM on collocated RL+Sim GPUs** | Peak 32-GPU Megatron (`TP=2, DP=16`) backward reaches `79.2 GiB / 81.5 GiB` on H100; resident collocated sim workers leave insufficient VRAM. | Enable `tools.rpent.sleep_sim_pools_during_training: true` and `sleepable: true` on collocated RPent pools (`h100_1`–`h100_4`) while keeping dedicated `h100`/`a100` pools warm (`sleepable: false`). |
| **`json.decoder.JSONDecodeError` in SGLang `l2norm_fwd_kernel` after pod crash / disk full** | Interrupted writes leave 0-byte `*.json` files inside `/root/.triton/cache`, causing all SGLang workers on that node to crash during prefill. | Audit `/root/.triton/cache` for 0-byte or invalid JSON files (`find /root/.triton/cache -type f -size 0`) before launch, and sync a verified cache from a healthy peer node. |
| **Transient SGLang `TCPStore` `EADDRINUSE` during 8-way startup** | Concurrent `tp=1` SGLang workers on the same host can occasionally race on ephemeral ports at startup. | Cleanly retry launch once Ray reports `0.0/32.0 GPU` used and routers report `active_sessions: 0`. |

## Unit Tests

```bash
PYTHONPATH=/path/to/RLinf:/path/to/RPent:/path/to/Megatron-LM \
  pytest tests/unit_tests/test_rpent_grpo.py -v
```
