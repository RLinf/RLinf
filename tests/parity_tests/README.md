# Automated Testing

The RLinf automated testing toolkit batch-runs embodied training experiments, detects run status, and performs automatic baseline comparison analysis. It supports automatic sequential execution, automatic Python environment switching, and log-based completion/crash detection.

## Environment and Configuration

### Task List

Define the experiments to run in the `TASKS` array in [`run_all.sh`](run_all.sh). Each entry has the format `ENV_NAME MODEL_NAME VENV_NAME YAML_ARG T_NODES T_STEPS T_SAVE`:

- `ENV_NAME` — environment name (e.g. `maniskill_libero`, `behavior`, `isaaclab`, `metaworld`, `calvin`)
- `MODEL_NAME` — model name (e.g. `openvla`, `openvla-oft`, `openpi`, `gr00t`, `mlp`)
- `VENV_NAME` — virtual environment name under `VENV_BASE_DIR` (typically matches `MODEL_NAME`; override per-task when needed)
- `YAML_ARG` — corresponding YAML configuration file name
- `T_NODES` — number of nodes required
- `T_STEPS` — total training steps
- `T_SAVE` — save interval (`-1` means no checkpoint saving)

Example:

```bash
TASKS=(
    "maniskill_libero openvla-oft openvla-oft maniskill_ppo_openvlaoft 1 100 -1"
    "maniskill_libero openpi openpi libero_goal_ppo_openpi 1 100 -1"
    "maniskill_libero openpi openpi maniskill_ppo_mlp 1 100 -1"
)
```

### Python Environments and Model Assets

Install the corresponding Python virtual environments and update `VENV_BASE_DIR` in `run_all.sh`:

```bash
# Install a single model+environment combination
# --venv = <parent_dir>/<venv_name>; the parent dir is the install location
# and the trailing segment is the venv name (should match MODEL_NAME).
bash requirements/install.sh embodied --model openvla --env maniskill_libero --venv /workspace/venv/openvla

# Point VENV_BASE_DIR at the directory holding your venvs
export VENV_BASE_DIR="/path/to/venvs"
```

`VENV_BASE_DIR` should follow the layout used by the official RLinf images — one subdirectory per `VENV_NAME`, each a standard Python virtual environment:

```text
/path/to/venvs/
├── openvla/bin/activate
├── openvla-oft/bin/activate
├── openpi/bin/activate
└── ...
```

Download model weights and asset files, then update paths in the YAML configs:

```bash
# Model weights
hf download gen-robot/openvla-7b-rlvla-warmup --local-dir openvla-7b-rlvla-warmup

# ManiSkill assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
```

> Tip: instead of downloading model weights and LoRA checkpoints by hand, you can ask a coding agent (e.g. Claude Code) to scan every YAML config referenced by `run_all.sh`, collect all `model.*` paths and `lora_path` entries, and download/place them automatically under `/workspace/models/`.

## Running the Tests

Start automated testing:

```bash
bash ./tests/parity_tests/run_all.sh
```

`run_all.sh` accepts `--similarity-method` to choose the baseline-comparison metric (default `pearson`). Available options: `spearman`, `mse`, `mae`, `cosine`, `dtw`, `all`:

```bash
bash ./tests/parity_tests/run_all.sh --similarity-method pearson
```

Execution flow: the node runs each task sequentially, checking logs to decide whether the threshold has been reached or the run has crashed. If the threshold is reached, the task is skipped; if all runs crashed, it is marked failed; otherwise the matching virtual environment is activated and training begins. After all tasks complete, a final summary is printed and automatic log analysis, curve plotting, and baseline comparison are performed.

## Output

After all tasks complete, a final summary is printed:

```text
=========================================================
                    FINAL SUMMARY
=========================================================
Total tasks:        6
Success:            4
Skipped (crashed):  2
=========================================================
```

A detailed running summary is also generated, showing each task's status. Comparison analysis runs automatically, producing a `success_once` curve plot under `logs/` along with similarity metrics against the baseline.
