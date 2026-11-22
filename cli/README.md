# rlinf-cli

Lightweight, zero-dependency CLI launcher for [RLinf](https://github.com/RLinf/RLinf) — a distributed reinforcement learning infrastructure for embodied and agentic AI.

## Install

```bash
# Recommended: install globally as a standalone tool
uv tool install rlinf-cli

# Or with pip / pipx
pip install rlinf-cli
```

Or run one-off without installing:

```bash
uvx --from rlinf-cli rlinf install embodied --model openvla --env maniskill_libero
```

## Quick Start

Inside an [RLinf](https://github.com/RLinf/RLinf) checkout (or with `RLINF_REPO_PATH` / `REPO_PATH` set):

```bash
# 1. Inspect supported targets, models, environments, engines, and platforms
rlinf install --list

# 2. Install an environment (backed by requirements/install.sh)
rlinf install embodied --model openvla --env maniskill_libero

# 3. Verify Python, accelerators, rendering libraries, Ray, and Hugging Face
rlinf doctor

# 4. Discover, inspect, validate, and launch recipes
rlinf list
rlinf show maniskill_ppo_mlp
rlinf train maniskill_ppo_mlp --check
rlinf train maniskill_ppo_mlp
```

`rlinf-cli` delegates directly to the active checkout's `rlinf/cli/` and automatically resolves `--venv`, `$VIRTUAL_ENV`, or the repository's `.venv`, so it stays in sync across branches and works before any virtual environment exists.

## Links

- Repository: https://github.com/RLinf/RLinf
- Documentation: https://rlinf.readthedocs.io/
- Issues: https://github.com/RLinf/RLinf/issues
