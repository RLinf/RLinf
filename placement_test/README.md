# placement_test

This folder contains YAML configs and scripts for **placement throughput tests** (env/rollout/actor placements + pipeline stage settings) and for running them in batch.

## Generate YAMLs in batch

Use `placement_test/generate_yaml_batch.py` (or the `.sh` wrapper) to generate a set of YAMLs under `placement_test/` from a given **base YAML**.

### What the generator changes (and what it preserves)

The generator **only modifies** these fields:

- `env.train.total_num_envs`
- `rollout.pipeline_stage_num`
- `cluster.component_placement`

Everything else is copied from your base YAML unchanged (env/task/model/algorithm/optimizer/etc.).

### What YAMLs will be generated

For each `total_num_envs` value, the generator creates:

- **collocated**: `pipeline_stage_num = 1`, placement tag `env07-rollout07` (all on `0-7`)
- **hybrid**: `pipeline_stage_num = 2`, placement tag `env03-rollout47` (env on `0-3`, rollout on `4-7`)
- **extra hybrid (optional)**: if `envnum % 3 == 0`, also create `env01-rollout27` with `pipeline_stage_num = 2`

These tags match the existing naming conventions and placement patterns already used in `placement_test/*.yaml`.

### Naming (`--prefix auto` and manual `--prefix`)

By default, `--prefix auto` infers a short `env_model` prefix from the base YAML Hydra defaults:

- env tag: from `defaults: - env/<...>@env.train` (family = first token before `_`, e.g. `maniskill`, `robotwin`)
- model tag: from `defaults: - model/<...>@actor.model` (full name between `model/` and `@`, e.g. `openvla`, `openvla_oft`)

Example output prefix: `maniskill_openvla`, `robotwin_openvla_oft`.

If you want a longer prefix including task/algo (e.g. `robotwin_beat_block_hammer_grpo_openvlaoft`), pass it explicitly via `--prefix`.

### Dry-run (print filenames only)

```bash
python3 placement_test/generate_yaml_batch.py \
  --base-yaml examples/embodiment/config/aaa_maniskill_ppo_openvla.yaml \
  --envnum-start 96 --envnum-end 128 --envnum-step 16 \
  --include-env01-rollout27-when-divisible-by-3 \
  --dry-run
```

### Write files (recommended wrapper)

```bash
bash placement_test/generate_yaml_batch.sh \
  --base-yaml examples/embodiment/config/aaa_maniskill_ppo_openvla.yaml \
  --envnum-start 96 --envnum-end 512 --envnum-step 16 \
  --include-env01-rollout27-when-divisible-by-3
```

### Example: RobotWin + OpenVLA-OFT, custom prefix

```bash
bash placement_test/generate_yaml_batch.sh \
  --base-yaml examples/embodiment/config/robotwin_beat_block_hammer_grpo_openvlaoft.yaml \
  --prefix robotwin_beat_block_hammer_grpo_openvlaoft \
  --envnum-start 96 --envnum-end 512 --envnum-step 16 \
  --include-env01-rollout27-when-divisible-by-3
```

## Run generated YAMLs continuously

Batch execution is driven by `placement_test/run_test_batch.sh`.

### 1) Add YAML names into the TASKS list

Open `placement_test/run_test_batch.sh` and edit the `TASKS=( ... )` array.

**Important**: each entry should be the YAML basename **without** `.yaml`, for example:

- `maniskill_openvla-envnum96-env03-rollout47-actor07-pipelinestage2`
- `robotwin_beat_block_hammer_grpo_openvlaoft-envnum384-env07-rollout07-actor07-pipelinestage1`

### 2) Run on single node (rank 0 only)

```bash
bash placement_test/run_test_batch.sh
```

The script will:

- stop any existing Ray processes
- start Ray head via `ray_utils/start_ray.sh`
- run each task via `placement_test/run_placement_test.sh <yaml_basename>`
- cleanup between tasks

### 3) Run on multi-node (rank 0 + workers)

The same script supports worker nodes via the `RANK` env var:

- on head node:

```bash
export RANK=0
bash placement_test/run_test_batch.sh
```

- on each worker node:

```bash
export RANK=1
bash placement_test/run_test_batch.sh
```

`run_test_batch.sh` uses a shared `task_sync.txt` flag file (under the repo root) to broadcast which YAML is currently running, so worker nodes can join the Ray cluster at the right time.

### Notes

- The batch script is intentionally aggressive about cleanup (`ray stop --force`, `pkill python`, `/dev/shm/ray/*`) to avoid interference between runs.
- If you change GPU counts or component placements beyond the presets, ensure the generated `cluster.component_placement` matches your hardware and placement assumptions.

