# rlinf-cli

Lightweight, zero-dependency CLI launcher for [RLinf](https://github.com/RLinf/RLinf) — a distributed reinforcement learning infrastructure for embodied and agentic AI.

## Install

```bash
# Recommended: install globally as a standalone tool
uv tool install rlinf-cli

# Or with pip / pipx
pip install rlinf-cli

# Enable automatic venv activation on `rlinf install` and `rlinf use`
# (supports bash, zsh, fish, and pwsh)
rlinf shell-init --install
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

# 2. Install an environment (auto-activates in the current shell when the shell hook is loaded)
rlinf install embodied --model openvla --env maniskill_libero

# 3. Switch between installed virtual environments
rlinf use --list
rlinf use .venv-openpi

# 4. Verify Python, accelerators, rendering libraries, Ray, and Hugging Face
rlinf doctor

# 5. Discover, inspect, validate, and launch recipes
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
