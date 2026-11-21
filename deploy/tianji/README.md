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
