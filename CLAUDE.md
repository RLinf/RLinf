# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> For a fuller orientation to architecture and extension guides, also read `AGENTS.md`.

## Commands

### Installation
```bash
# Python 3.10–3.11 required (see pyproject.toml requires-python)

# Recommended: install with uv
uv venv && source .venv/bin/activate
UV_TORCH_BACKEND=auto uv sync

# Or install from requirements (picks up model/env deps)
bash requirements/install.sh embodied --model <model> --env <env>
```

### Linting and Formatting
```bash
pip install pre-commit
pre-commit install --hook-type commit-msg
pre-commit run --all-files   # runs Ruff (lint + format) and commit checks
```

### Testing
```bash
# Unit tests
pytest tests/unit_tests

# Single unit test
pytest tests/unit_tests/test_channel.py
pytest tests/unit_tests/test_channel.py::TestChannel::test_send_recv -v

# Scheduler doctests
pytest --doctest-modules rlinf/scheduler

# All e2e tests (GPU-heavy; CI only runs these when run-ci label is added)
pytest tests/e2e_tests/
```

### Running Training
```bash
# Single-node embodied RL
ray start --head
python examples/embodiment/train_embodied_agent.py --config-name <config_name>
# Or: bash examples/embodiment/run_embodiment.sh <config_name>

# Standalone evaluation
bash examples/embodiment/eval_embodiment.sh <config_name>
```

### Documentation
```bash
cd docs && make html
```

## Architecture

RLinf is a distributed RL infrastructure for embodied and agentic AI, built on **Ray** (process management) and **Hydra** (configuration).

### Execution Flow

1. Entry script (e.g. `examples/embodiment/train_embodied_agent.py`) reads Hydra YAML config.
2. Builds a **Cluster** (Ray must be running); determines **component placement** (actor, rollout, env, reward, agent).
3. Creates **WorkerGroup**s via Ray; workers communicate via channels.
4. A **Runner** drives the loop: rollout → reward → advantage → actor update.

### Key Packages (`rlinf/`)

| Module | Role |
|---|---|
| `config.py` | `build_config` / `validate_cfg`; `SupportedModel`, `SupportedEnvType` enums |
| `scheduler/` | Cluster, Worker, WorkerGroup, channel, manager, placement, dynamic_scheduler |
| `workers/` | Actor (FSDP/Megatron), rollout (HF/SGLang/vLLM), env (sync/async), reward, replay buffer |
| `runners/` | Training loop drivers: embodied, reasoning, coding_online_rl, agent, SFT, eval |
| `algorithms/` | Advantage functions, policy losses, reward registry (PPO, GRPO, SAC, etc.) |
| `models/` | Embodiment models (OpenVLA, π₀, GR00T, MLP/CNN/Flow) and reasoning wiring |
| `envs/` | Env integrations; `get_env_cls()` in `envs/__init__.py` is the dispatch point |
| `hybrid_engines/` | SGLang/vLLM rollout integration |
| `utils/` | Logging, placement, distributed training, checkpoint, resharding |

### Configuration

- YAML configs live under `examples/*/config/`. Copy existing configs as templates.
- **No dynamic values or calculations in YAML.** Compute derived values in `config.py`.
- **Config fields are read-only in code.** Never overwrite user-set fields programmatically.
- Key top-level config sections: `cluster`, `task`, `model`, `runner`, `algorithm`, `data`.

### Backends

- **Training:** FSDP (default/flexible) or Megatron-LM (large-scale).
- **Rollout:** SGLang, vLLM, or HuggingFace.
- **Algorithms:** PPO, GRPO, Reinforce++, DAPO, SAC, CrossQ, RLPD, SAC-Flow, DSRL.

## Extension Patterns

### New Algorithm (advantage / loss / reward)
- Implement function → decorate with `@register_advantage("name")` / `@register_policy_loss("name")` from `rlinf/algorithms/registry`.
- Set `algorithm.adv_type` / `algorithm.loss_type` in YAML.
- Rewards: add class under `rlinf/algorithms/rewards/`, call `register_reward("name", cls)` in `rewards/__init__.py`.

### New Embodied Model
1. Add `MY_MODEL = ("my_model", "embodied")` to `SupportedModel` in `rlinf/config.py`.
2. Create `rlinf/models/embodiment/my_model/`; inherit `BasePolicy`, implement `default_forward` and `predict_action_batch`.
3. Add branches in actor/rollout workers to instantiate the model.
4. Add install logic to `requirements/install.sh`; add Dockerfile stage, CI job, and e2e config under `tests/e2e_tests/embodied/`.

### New Environment
1. Add `MY_ENV = "my_env"` to `SupportedEnvType` in `rlinf/envs/__init__.py`.
2. Add lazy-import branch in `get_env_cls()`.
3. Create `rlinf/envs/my_env/` with a gym-style env (`reset`, `step`, `observation_space`, `action_space`).
4. Add action-format branch in `rlinf/envs/action_utils.py` if needed.
5. Add env-specific validation in `rlinf/config.py`; add Dockerfile stage, CI job, and e2e config.

## Code Style

- **Google Python Style Guide.** Ruff enforces lint/format at `line-length: 88`.
- **Docstrings and type hints** required on all public APIs in `rlinf/scheduler/` (Ruff docstring rules `D` are only enforced there via per-file-ignores in `pyproject.toml`); expected elsewhere.
- **Logging:** in `Worker` use `self.log_info` / `self.log_warning` / `self.log_error`; elsewhere use `from rlinf.utils.logging import get_logger`.
- **No `print` statements.**
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) format `<type>(<scope>): <description>`, signed-off (`git commit -s`).
- **PRs** require the same title format; add `run-ci` label only when a full CI run is needed.
- **All user-facing changes** must include tests and documentation validated by a reviewer.
- If something is unclear, add `TODO(agent)` and note the limitation rather than guessing.

## Multi-Node Setup

```bash
# On every node, BEFORE ray start:
export RLINF_NODE_RANK=<0..N-1>   # unique per node
export RLINF_COMM_NET_DEVICES=...  # optional: specify network interface

# Head node:
ray start --head --port=6379 --node-ip-address=<head_ip>

# Worker nodes:
ray start --address=<head_ip>:6379

# Launch training ONLY on head node; set cluster.num_nodes in YAML.
```
