# RPent Multi-Turn Embodied GRPO Training

This example trains a 4B Vision-Language-Action reasoning model (Qwen3.5-4B) with synchronous multi-turn GRPO and oversampling across multiple manipulation simulators (LIBERO and RoboTwin, with extensible registry support for RoboCasa and RoboDojo) served by [RPent](https://github.com/RLinf/RPent).

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

## Unit Tests

```bash
PYTHONPATH=/path/to/RLinf:/path/to/RPent:/path/to/Megatron-LM \
  pytest tests/unit_tests/test_rpent_grpo.py -v
```
