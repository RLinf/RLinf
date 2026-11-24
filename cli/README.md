# rlinf-cli

Lightweight, zero-dependency CLI launcher for [RLinf](https://github.com/RLinf/RLinf) — a distributed reinforcement learning infrastructure for embodied and agentic AI.

## Quick Start

```bash
# 1. Clone RLinf
git clone https://github.com/RLinf/RLinf.git && cd RLinf

# 2. Install the lightweight CLI once
pip install rlinf-cli          # or: uv tool install rlinf-cli

# 3. Install an environment (auto-installs shell hook for future shells)
rlinf install embodied --model openvla --env maniskill_libero

# 4. Run immediately — no manual `source .venv/bin/activate` needed
rlinf doctor
rlinf train maniskill_ppo_mlp
```

`rlinf` automatically detects `.venv`, loads its `bin/activate` environment exports, and delegates to the active checkout's `rlinf/cli/`.

## Multiple Environments & Shell Switching

```bash
# Install another environment into a separate venv
rlinf install embodied --model openpi --env libero --venv .venv-openpi

# Run directly with --venv
rlinf train libero_spatial_grpo_openpi --venv .venv-openpi

# Or switch your current shell (bash, zsh, fish, pwsh)
rlinf shell-init               # one-time setup (also added automatically by `rlinf install`)
rlinf use --list
rlinf use .venv-openpi
```

You can also run one-off without installing globally:

```bash
uvx --from rlinf-cli rlinf install embodied --model openvla --env maniskill_libero
```

## Links

- Repository: https://github.com/RLinf/RLinf
- Documentation: https://rlinf.readthedocs.io/
- Issues: https://github.com/RLinf/RLinf/issues
