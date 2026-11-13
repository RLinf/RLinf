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

### Baseline Logs Directory

Baseline comparison is driven by `BASELINE_DIR` in [`run_all.sh`](run_all.sh), which defaults to `$REPO_PATH/logs_baseline`. Before running the tests, create this directory and populate it with a reference (baseline) run log for **every** experiment listed in `TASKS`; missing baselines are skipped with a warning and no similarity metric is produced for that task.

Layout — one subdirectory per experiment, whose name **ends with** `-<YAML_ARG>`, containing a `run_embodiment.log` file:

```text
$REPO_PATH/logs_baseline/
├── 2025-01-15_10-00-00-maniskill_ppo_openvla/
│   └── run_embodiment.log
├── 2025-01-15_11-00-00-libero_goal_ppo_openpi/
│   └── run_embodiment.log
├── 2025-01-15_12-00-00-maniskill_ppo_mlp/
│   └── run_embodiment.log
└── ...
```

The `<YAML_ARG>` suffix must match the fourth field of the corresponding `TASKS` entry exactly (that's how `_find_baseline_log` in [`compare_baseline.py`](compare_baseline.py) locates the reference run). Any prefix (timestamp, run id, machine name, …) is fine — only the trailing `-<YAML_ARG>` matters.

To generate baselines, run `run_all.sh` once on a known-good commit, then copy the produced experiment directories from `$REPO_PATH/logs/` into `$REPO_PATH/logs_baseline/`. To point at a different location, edit `BASELINE_DIR` in [`run_all.sh`](run_all.sh).

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

## QAM numerical reference check

QAM training uses RLinf's PyTorch implementation. The optional numerical check imports the official JAX implementation only in its test process. It compares losses, gradients, adjoint targets, and an optimizer update for action horizons 1 and 2 with matched weights and noise.

Use a clean checkout of [ColinQiyangLi/qam](https://github.com/ColinQiyangLi/qam) at commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`. Install the reference's JAX, Flax, Optax, and other Python dependencies in the test environment as well as RLinf's dependencies. The test skips when `QAM_REFERENCE_ROOT`, JAX, or Flax is absent, and fails when a configured checkout has the wrong revision or tracked modifications. It is a manual optional check; the OGBench e2e CI job does not install JAX or download the reference repository.

```bash
git clone https://github.com/ColinQiyangLi/qam.git /path/to/qam-reference
git -C /path/to/qam-reference checkout 2726d767c9a0a7a46d49693f0391f73dc2cf58ac
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  QAM_REFERENCE_ROOT=/path/to/qam-reference \
  python -m pytest tests/parity_tests/test_qam_reference.py -v -rs
```

The CPU unit cases live with the components they exercise. Run them in separate processes, as unit-test CI does:

```bash
python -m pytest tests/unit_tests/test_data.py -k ogbench
python -m pytest tests/unit_tests/test_models.py -k qam
python -m pytest tests/unit_tests/test_qam.py
```

After installing OGBench, run the GPU integration test through the standard offline launcher. It uses a small model, trains to step 8, resumes from step 4, compares model/optimizer/RNG/sampler state and later training metrics exactly, and evaluates a checkpoint without further training. The test also exercises the OGBench environment's public reset, step, chunk, terminal, and close contracts. Set `OGBENCH_DATASET_DIR` to reuse an existing dataset cache; otherwise the first run downloads about 255 MiB.

```bash
python -m pytest tests/e2e_tests/embodied/test_qam.py -v --basetemp=/path/to/new-test-output
```

Choose a new output directory: pytest clears `--basetemp` before the run. The subdirectories retain console logs, TensorBoard events, resolved configs, and checkpoints. These short tests check integration and numerical continuity, not benchmark performance. GitHub runs the embodied e2e job only when a maintainer applies the `run-ci` label to a non-draft PR.
