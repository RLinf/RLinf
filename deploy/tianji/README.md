# Tianji HIL Deployment Launchers (`deploy/tianji`)

This directory contains the versioned, machine-agnostic launchers and preflight helpers for running Tianji Marvin Pro real-world workflows directly from the `RLinf` repository (`tianji_hil` branch).

> [!NOTE]
> Existing desktop deployment snapshots on `tianji1` (such as `~/Stage2-30HZ-1118v1` and `~/Stage2-30HZ-sub6-1118v1`) remain untouched until an operator explicitly validates and switches to this versioned directory.

---

## 1. Machine-Local Configuration (`site.env`)

Copy [site.env.example](site.env.example) to an **untracked** `site.env` (searched in order: `$TIANJI_SITE_ENV`, `deploy/tianji/site.env`, `$RLINF_ROOT/site.env`, `~/.config/tianji/site.env`):

```bash
cp deploy/tianji/site.env.example deploy/tianji/site.env
```

Key environment variables configured in `site.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RLINF_ROOT` | Auto-detected repo root | Root of the `RLinf` checkout |
| `LOOPINFRA_ROOT` | `$HOME/LoopInfra` | Unified `LoopInfra` checkout used for SDK kinematics & `.loopinfra-python` |
| `RLINF_PYTHON` | `$HOME/miniconda3/envs/train/bin/python` | GPU/training Python interpreter (also defaults `TIANJI_CPU_PYTHON` / `TIANJI_GPU_PYTHON`) |
| `LOOPINFRA_PYTHON` | `$LOOPINFRA_ROOT/.loopinfra-python` | Isolated LoopInfra collector Python interpreter |
| `TIANJI_MODEL_DIR` | `$RLINF_ROOT/models/stage1_v1` | Stage1 VLA checkpoint directory (`TIANJI_RLT_STAGE1_MODEL_PATH`) |
| `TIANJI_RUNS_DIR` | `$RLINF_ROOT/runs` | Root directory for run logs, HIL archives, and checkpoints |
| `TIANJI_RUNTIME_DIR` | `$RLINF_ROOT/.runtime/tianji` | Runtime sockets, collector generated configs, and preflight state |
| `TIANJI_CAR_NUM_SUBTASKS` | `15` | Car subtask count (`15` for `1118v1`, `6` for `sub6`) |
| `TIANJI_CAR_BANNER` | `Stage2-30HZ-1118v1` (or `Stage2-30HZ-sub6-1118v1`) | HIL panel banner & desktop launcher name |

Never commit filled `site.env` files, tokens, or hardware IDs.

---

## 2. Workflows & Entrypoints

### LoopInfra Collector & Preflight

```bash
# Offline preflight check (no hardware required)
bash deploy/tianji/check_tianji_local.sh --mode offline

# Start or reuse the validated LoopInfra collector
bash deploy/tianji/run_tianji_wuji_collector.sh
```

### 1. Model Inference (Stage1 VLA & Stage2 Residual)

```bash
# Stage1 VLA inference (auto-selects joint vs eef via TIANJI_ACTION_SPACE)
bash deploy/tianji/run_tianji_stage1_local.sh

# Stage2 Joint Residual inference
bash deploy/tianji/run_tianji_stage2_infer_local.sh /path/to/actor/model_state_dict/full_weights.pt
```

### 2. DAgger Data Collection (`hil_mode=dagger`)

Uses [tianji_dagger_collect.yaml](../../examples/embodiment/config/tianji_dagger_collect.yaml) with `hil_mode=dagger`, `episode_archive=true`, `tactile_raw_archive=true`, `discard_episodes=true`, `runner.deploy_actor_updates=false`, and RL optimizer updates disabled (`rlt_schedule.max_updates_per_train_step=0`, `replay_buffer.auto_save=false`):

```bash
bash deploy/tianji/run_tianji_dagger_local.sh
```

### 3. Online RL (Async 2-GPU Residual Actor-Critic)

```bash
# Fresh async Stage2 run (starts/waits for collector & opens :8879 HIL panel)
bash deploy/tianji/start_tianji_stage2_async.sh

# Resume from latest or explicit checkpoint
bash deploy/tianji/resume_tianji_stage2_async.sh /path/to/checkpoints/global_step_N
```

### 4. Generalized Car Deployment (`1118v1` / `sub6`)

Supports both `15`-subtask (`TIANJI_CAR_NUM_SUBTASKS=15`) and `6`-subtask (`TIANJI_CAR_NUM_SUBTASKS=6`) configurations, plus `--dagger` mode:

```bash
# Stage & verify assets from NAS (pass --source or set TIANJI_CAR_NAS_SOURCE)
bash deploy/tianji/stage2_30hz_1118v1.sh --cache --source /path/to/nas/car_checkpoint_root
bash deploy/tianji/stage2_30hz_1118v1.sh --verify

# Optional: install desktop icon after collector verification
bash deploy/tianji/stage2_30hz_1118v1.sh --install

# Run inference (--run) or DAgger collection (--dagger)
bash deploy/tianji/stage2_30hz_1118v1.sh --run
bash deploy/tianji/stage2_30hz_1118v1.sh --dagger
```

### 5. Plug-and-Play Sim2Real Toy-Car Inference (`tianji_stage1_joint_h60`, SFT & RLT)

Use [run_tianji_sim2real_car.sh](run_tianji_sim2real_car.sh) and [check_sim2real_car_model.py](check_sim2real_car_model.py) for 30 Hz, 16D joint-space Sim2Real toy-car checkpoints (`horizon=60`, `tianji_stage1_joint_h60`). Both pure-SFT checkpoints (`model.safetensors`, `use_rlt=False`) and RLT Stage1 checkpoints (`full_weights.pt`, `use_rlt=True`) are auto-detected, and both single global prompts and `prompts.json` (1, 4, 12, or 24 subtasks) are supported.

Default control-loop alignment (`docs/inference.md`): `--no-rtc-guidance` + control-side 10-step deviation-decay Smoothstep chunk blend (`--blend-steps 10`), per-step velocity clamp (`--max-vel-rad-s 0.7854`, 45 deg/s), elapsed-time stale step tracking, and online `ChunkExecutionMonitor` writing `<recorder_dir>/chunk_diagnostics.jsonl`. Pass `--rtc-guidance` to enable RLinf RTC guidance or `--fast 1|2` for `TIANJI_FAST_INFER`.

```bash
# 1. Offline GPU smoke test & 20-iter latency benchmark (no hardware or LoopInfra needed)
bash deploy/tianji/run_tianji_sim2real_car.sh --model /path/to/ckpt --check --iters 20 --fast 0

# 2. Live LoopInfra cameras/joints + GPU inference, read-only (NO robot motion)
bash deploy/tianji/run_tianji_sim2real_car.sh --model /path/to/ckpt --read-only

# 3. Real 30 Hz 16D joint execution on Tianji (HIL panel on :8879)
#    Uses <model>/prompts.json if present, or defaults to the single global toy-car prompt.
bash deploy/tianji/run_tianji_sim2real_car.sh --model /path/to/ckpt --run

# Override prompt(s) or tune blending / RTC explicitly:
bash deploy/tianji/run_tianji_sim2real_car.sh --model /path/to/ckpt --prompt "Install four wheels and screws on a wheel-less toy vehicle using the toy drill." --run
bash deploy/tianji/run_tianji_sim2real_car.sh --model /path/to/ckpt --prompts /path/to/prompts.json --blend-steps 10 --run
bash deploy/tianji/run_tianji_sim2real_car.sh --model /path/to/ckpt --rtc-guidance --run

# 4. Real 30 Hz execution + DAgger episode archive (hil_mode=dagger)
bash deploy/tianji/run_tianji_sim2real_car.sh --model /path/to/ckpt --dagger
```

### Episode Archive & Checkpoint Utilities

```bash
# Validate archived DAgger/RL episodes for LeRobot/offline-RL completeness
python3 deploy/tianji/check_tianji_episode_archive.py "$TIANJI_RUNS_DIR/<run_id>"

# Find newest complete Stage2 checkpoint
python3 deploy/tianji/latest_tianji_stage2_checkpoint.py "$TIANJI_RUNS_DIR"
```

---

## 3. Related Documentation

- Master Tianji docs index: [docs/tianji/README.md](../../docs/tianji/README.md)
- Detailed runbooks & HIL panel guide: [docs/tianji/workflows.md](../../docs/tianji/workflows.md)
- `RLinf` ↔ `LoopInfra` contract: [docs/tianji/interface_contract.md](../../docs/tianji/interface_contract.md)
- Repo & branch governance: [docs/tianji/repo_management.md](../../docs/tianji/repo_management.md)
