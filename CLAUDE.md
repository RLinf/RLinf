# CLAUDE.md

Quick-start reference for Claude Code working on RLinf. For full contribution rules see [CONTRIBUTING.md](CONTRIBUTING.md); for AI agent coding guide see [AGENTS.md](AGENTS.md).

## Build & Install

Python **3.10-3.11** (default 3.11.14). Uses `uv` for venv and dependency management.

```bash
# Embodied (most common)
bash requirements/install.sh embodied --model <model> --env <env>
# Models: openvla, openvla-oft, openpi, gr00t, dexbotic
# Envs:   maniskill_libero, behavior, metaworld, calvin, isaaclab, robocasa, franka, frankasim, robotwin, habitat, opensora, wan

# Agentic (reasoning/LLM with Megatron)
bash requirements/install.sh agentic

# Docs only
bash requirements/install.sh docs
```

Extras in `pyproject.toml`: `embodied`, `agentic-sglang`, `agentic-vllm`, `franka`. The venv is `.venv` by default; activate with `source .venv/bin/activate`.

## Running Tests

```bash
# Activate the openpi environment before running tests
source switch_env openpi

# Unit tests (from repo root)
pytest tests/unit_tests/ -v
# Single test
pytest tests/unit_tests/test_worker.py -v
pytest tests/unit_tests/test_worker.py::TestWorkerClass::test_method -v

# E2e tests (require GPU, Ray, and installed model/env)
bash tests/e2e_tests/embodied/run.sh <config_name>
bash tests/e2e_tests/reasoning/run.sh <config_name>
# E2e configs live under tests/e2e_tests/{embodied,reasoning,agent,sft}/*.yaml
```

## Linting & Formatting

Pre-commit hooks: **ruff** (lint + format) and **commit-check** (message format, branch name, sign-off).

```bash
pip install pre-commit
pre-commit install --hook-type commit-msg
pre-commit run --all-files          # check everything
ruff check --preview --fix .        # lint only
ruff format .                       # format only
```

Ruff config: `line-length = 88`, `target-version = "py311"`, Google-style docstrings, isort with `known-first-party = ["rlinf"]`. Docstring rules currently enforced only for `rlinf/scheduler/`.

## Architecture Overview

- **Ray** for distributed process management; **Hydra** for hierarchical YAML config.
- **Cluster** (`rlinf/scheduler/cluster/`) manages nodes; **Placement** strategies (`rlinf/scheduler/placement/`) map components (actor, rollout, env, reward) to nodes/GPUs.
- **Workers** (`rlinf/workers/`) are Ray remote actors — Actor (FSDP or Megatron backend), Rollout (HF, SGLang, or vLLM engine), Env (sync/async), Reward, Replay Buffer. Workers communicate via **Channels** (`rlinf/scheduler/channel/`).
- **Runners** (`rlinf/runners/`) drive the training loop: rollout -> reward -> advantage -> actor update. Each task type (embodied sync/async, reasoning, agent, SFT, eval) has its own runner.
- **Algorithms** (`rlinf/algorithms/`) — advantage functions (GAE, GRPO, Reinforce++), losses (PPO, GRPO, SAC), rewards (math, code, VQA, etc.) — all registered via decorators.
- **Models** (`rlinf/models/`) — embodiment policies (OpenVLA, OpenPI, GR00T, MLP/CNN/Flow) and reasoning wiring.
- **Environments** (`rlinf/envs/`) — ManiSkill, LIBERO, IsaacLab, CALVIN, MetaWorld, Behavior, RoboCasa, etc.
- **Config** (`rlinf/config.py`) — `build_config`, `validate_cfg`, `SupportedModel`, `SupportedEnvType` enums.
- Entry scripts live in `examples/` (e.g., `examples/embodiment/train_embodied_agent.py`).

## Key Development Patterns

**Worker subclassing**: Inherit from `Worker`, implement `initialize()` and your API. Use `self.log_info` / `self.log_warning` / `self.log_error` — never `print`.

**Registering new algorithms**: Use `@register_advantage("name")`, `@register_policy_loss("name")` from `rlinf/algorithms/registry`. Set `algorithm.adv_type` / `algorithm.loss_type` in YAML.

**Registering models/envs**: Add to `SupportedModel` / `SupportedEnvType` enums in `rlinf/config.py`. For envs, also add to `get_env_cls()` in `rlinf/envs/__init__.py` with lazy imports.

**Logging**: Outside workers use `from rlinf.utils.logging import get_logger; logger = get_logger()`.

**Config YAML rules**:
- All values must be static — no computed fields in YAML
- Never overwrite user-facing config fields in code (treat as read-only)
- Copy existing configs as templates rather than writing from scratch

## Commit Conventions

[Conventional Commits](https://www.conventionalcommits.org/) with **mandatory `Signed-off-by`** (enforced by pre-commit hook).

```bash
git commit -s -m "feat(embodiment): add new env wrapper"
# Types: feat, fix, docs, style, refactor, test, chore
```

The pre-commit `check-commit-signoff` hook will reject commits without `Signed-off-by`. The `check-message` hook enforces the `<type>(<scope>): <description>` format.

## Progress Tracking

**IMPORTANT**: After completing each phase or significant milestone, you **MUST** update `progress.md` (repo root) with the current status, completed steps, and any issues encountered. This ensures continuity across context compressions.

---

## Current Task: Pi06 Pipeline E2E Testing for Open-Source Release

**Goal**: 对 pi06 功能代码进行 E2E pipeline 测试，确保代码可发布。

**详细计划**: 见 `pi06_test_plan.md`

**测试 Skill**: 见 `pi06_test_skill/`（check 通过后移至 `.cursor/skills/`）

**Reference repo**: `/mnt/project_rlinf/liuzhihao/RLinf_pi06_release_worktree_2`

### 当前进度

按以下顺序执行，每步通过后再进入下一步：

#### Part 0: vla_lib 外部依赖内部化（待执行）

7个文件 import 自外部 `vla_lib` 包，需要内部化：
- `rlinf/models/embodiment/vla_lib_value_model/value_policy_config.py`
- `rlinf/models/embodiment/vla_lib_value_model/modeling_critic.py`
- `rlinf/models/embodiment/vla_lib_value_model/value_policy.py`
- `rlinf/datasets/vla_lib/advantage_mixture_dataset.py`
- `rlinf/datasets/vla_lib/value_mixture_dataset.py`
- `rlinf/datasets/vla_lib/io_processing/value_transforms.py`
- `rlinf/datasets/vla_lib/io_processing/__init__.py`

#### Part 1: 测试 Skill 和 Config（待 check）

已生成文件：
- `pi06_test_skill/SKILL.md` — 主 skill
- `pi06_test_skill/task_example/step1-5_*.md` — 每步的可变参数 + 命令 + 验证
- `examples/*/config/*_test.yaml` — 5个测试配置
- `tests/e2e_tests/pi06_pipeline/run_pipeline_test.sh` — 自动化脚本

#### Part 2: 执行测试（Part 1 check 后执行）

```
Phase A: vla_lib 内部化
Phase B: 深度逻辑核查 (对比 reference repo)
Phase C: 顺序运行 Step 1-5
Phase D: 修复问题
```

### 测试 Pipeline

```
[Step 1] compute_returns   → stats.json 有 return/reward
[Step 2] vla_lib_sft       → checkpoint 生成 (5步, freeze_vlm)
[Step 3] compute_advantages → advantages_test.parquet (True/False 标签)
[Step 4] cfg_sft            → 5步训练完成
[Step 5] debug_one_iter     → 5步离线训练完成
```

### 关键路径

| 数据 | 路径 |
|------|------|
| 数据集 | `/mnt/project_rlinf_hs/liuzhihao/data_to_use/libero_10_3shot` |
| 基础模型 | `/mnt/project_rlinf/liuzhihao/vla_lib_fork_gaofeng/pretrained_models/pi05_base_pytorch` |
| 原始 repo | `/mnt/project_rlinf/liuzhihao/RLinf_pi06_release_worktree_2` |
| 外部 vla_lib | `/mnt/project_rlinf/liuzhihao/vla_lib_fork_gaofeng/` |

### 新增模块总览

**Models**:
- `rlinf/models/embodiment/openpi_cfg/` — OpenPI CFG action model
- `rlinf/models/embodiment/vla_lib_value_model/` — VLA lib value model

**Workers**:
- `rlinf/workers/cfg/fsdp_cfg_worker.py` — FSDP CFG worker
- `rlinf/workers/actor/debug_fsdp_actor_worker_cfg.py` — Debug CFG actor
- `rlinf/workers/vla_lib_sft/fsdp_value_sft_worker.py` — Value SFT worker

**Runners**:
- `rlinf/runners/debug_pi06_runner.py` — Debug pi06 runner
- `rlinf/runners/vla_lib_sft_runner.py` — VLA lib SFT runner

**Datasets** (`rlinf/datasets/` — 全新模块):
- `rlinf/datasets/vla_lib/` — value/advantage/mixture datasets, config, io_processing
- `rlinf/datasets/dataloaders/` — DataLoader 实现
- `rlinf/datasets/factory.py` — Dataset factory
- `rlinf/datasets/transforms/` — Tokenize transforms

**Offline processing** (`examples/process/`):
- `compute_returns.py`, `compute_advantages.py`, `recompute_advantages_from_value_reward.py`

**Entry points**:
- `examples/cfg/train_cfg_sft.py` — CFG SFT
- `examples/vla_lib_sft/train_vla_lib_sft.py` — Value model SFT
- `examples/embodiment/train_debug_one_iter.py` — Debug one-iter
