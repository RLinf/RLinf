---
name: test-install
description: Test that requirements/install.sh works for an embodied model/env by building its venv and running the matching CI e2e test. Use when asked to test or verify an install, check a new model/env installs cleanly, run the e2e test for a model/env, confirm a venv works, or check that the model/checkpoint paths referenced by an e2e config actually exist on disk.
---

Verify that `requirements/install.sh` actually works for an embodied model/env:
build its venv, confirm the e2e config's model paths exist, then run the matching
CI e2e test against that venv. The harness is
[.cursor/skills/test-install/driver.py](.cursor/skills/test-install/driver.py) —
it reads the install command, env vars, and test config **straight out of**
[.github/workflows/embodied-e2e-tests.yml](.github/workflows/embodied-e2e-tests.yml),
so it never drifts from CI. Drive everything through that script.

All paths below are relative to the repo root (the dir with `requirements/install.sh`).

## Prerequisites

This runs on the **embodied CI runner** (or an equivalent box): NVIDIA GPUs, the
shared `/workspace/dataset/` tree (models, LIBERO, etc.), `uv`, and a `python3`
that has PyYAML. No `apt-get` needed — the driver only orchestrates. Quick check:

```bash
nvidia-smi -L | head -1
ls -d /workspace/dataset >/dev/null && python3 -c "import yaml" && echo "env OK"
```

If `python3` lacks PyYAML, the driver auto-reexecs under `uv run --with pyyaml`,
so it works regardless.

## Run (agent path)

The driver has seven subcommands. **Start with the read-only ones** (`list`,
`resolve`, `check-paths`) — they're instant and tell you exactly what CI does
before you spend an hour on an install.

```bash
# What model/env combos does CI cover, and which configs do they run?
python3 .cursor/skills/test-install/driver.py list

# Show the exact install command + env vars + test configs for one combo:
python3 .cursor/skills/test-install/driver.py resolve gr00t_n1d6 maniskill_libero

# Do the model/checkpoint paths an e2e config needs actually exist on disk?
python3 .cursor/skills/test-install/driver.py check-paths libero_spatial_ppo_gr00t_n1d6

# Same check across every e2e config at once (great pre-flight / PR check):
python3 .cursor/skills/test-install/driver.py check-all
```

`check-paths` classifies every absolute path in the config: `[model/input]`
(must exist — a missing one fails the run before training and returns exit 1),
`[output dir]` (created by the run, may be missing), `[path]` (informational).

### The full pipeline

`run` does install → check-paths (per config) → test, stopping a test whose
required model paths are missing. **Always preview with `--dry-run` first** — it
prints the exact shell (install line + CI test step) without executing:

```bash
python3 .cursor/skills/test-install/driver.py run gr00t_n1d6 maniskill_libero \
    --venv /workspace/test-venvs/gr00t_n1d6 --dry-run
```

Drop `--dry-run` to actually build and test. Or drive the two halves separately
(faster iteration — install once, test many):

```bash
# Preview the install (env vars + install.sh line, with --venv and --use-mirror
# injected). Drop --dry-run to build the venv; add --no-mirror to skip mirrors:
python3 .cursor/skills/test-install/driver.py install gr00t_n1d6 maniskill_libero \
    --venv /workspace/test-venvs/gr00t_n1d6 --dry-run

# Run the matching e2e test against any venv (verbatim CI test step, venv path
# swapped in). This one is cheap — dummy SAC, 2 epochs — so run it for real:
python3 .cursor/skills/test-install/driver.py test realworld_dummy_sac_cnn \
    --venv /opt/venv/openvla
```

A full model install (e.g. `gr00t_n1d6`) is heavy — it clones repos and builds
flash-attn — so preview with `--dry-run`, then run it where you can afford the
time. The dummy-SAC `test` above completes in a few minutes against the prebuilt
`/opt/venv/openvla`.

A real test run spins up Ray, the env/rollout/policy workers, and trains for the
config's (deliberately tiny) epoch count. The dummy-SAC smoke above finishes in a
few minutes and writes TensorBoard output under the config's `log_path`
(`/workspace/results/<config>/tensorboard/`) — that directory appearing with
fresh files is your "it worked" signal.

### Clean up the venv when you're done

A test venv is heavy (e.g. `gr00t_n1d6` is ~600M) and is throwaway — it exists
only to prove the install + e2e work. **After the test finishes, `rm -rf` the
`--venv` path you built** unless the user asked to keep it for reuse:

```bash
rm -rf /workspace/test-venvs/<model>
```

Don't touch the shared caches (`/workspace/dataset/.uv`, `.uv_cache`) or the
prebuilt `/opt/venv/*` — only the per-test venv you created. Cleaning up keeps
`/workspace/test-venvs/` from accumulating stale multi-hundred-MB trees across
runs. (Cleanup is for the venv only; the `/workspace/results/<config>/` outputs
are your proof the test ran — leave them or mention them.)

### When there's no matching test

If a model/env has an install but **no** e2e job in the workflow, `run` installs
and then tells you there's nothing to run — **ask the user what to run** rather
than guessing. If you have a config name that isn't wired into CI, `test
<config> --runner run|run_async|run_offline` runs it directly (pick the runner;
`run` is the default).

## Verifying a brand-new model/env (the common case)

When someone adds `install_<model>_model()` + an e2e job (see the
`add-install-docker-ci-e2e` and `install-check` skills), confirm it end to end:

```bash
# 1. CI parsed it correctly and the install command looks right:
python3 .cursor/skills/test-install/driver.py resolve <model> <env>
# 2. The SFT checkpoint the e2e config points at is actually on this box:
python3 .cursor/skills/test-install/driver.py check-paths <its_config>
# 3. Full build + test (preview with --dry-run, then drop it to run for real):
python3 .cursor/skills/test-install/driver.py run <model> <env> \
    --venv /workspace/test-venvs/<model> --dry-run
```

If step 2 reports `MISS [model/input]`, the install can be perfect and the e2e
will still fail — the dataset/checkpoint just isn't staged on this runner. Surface
that to the user; it's not an install bug.

## Gotchas

- **`install`/`run` need `--venv <path>`** — it's injected as `install.sh --venv`.
  Use an absolute path (e.g. `/workspace/test-venvs/<model>`); a bare name lands
  relative to the repo root. The `test` step then sources `<path>/bin/activate`.
- **`--venv` reuses an existing venv** if one is already at that path (install.sh
  validates the Python version and reuses it). For a truly clean install, point at
  a fresh path or `rm -rf` it first.
- **The driver runs CI's shell verbatim**, including its `export UV_PATH=/workspace/dataset/.uv`
  etc. That's intentional — it reproduces CI exactly. It also means the install
  writes into the shared uv cache, same as CI.
- **`--use-mirror` is added to every install by default** (faster downloads),
  even for CI jobs that don't list it. Pass `--no-mirror` to `install`/`run` to
  turn it off. It's never duplicated if the CI job already has it.
- **Some jobs use `--platform amd`/`ascend`** (ROCm/Ascend runners) — `resolve`
  shows the platform; those won't install on an NVIDIA box.
- **Some jobs do extra setup inside the test step** (e.g. `cp .../maniskill_assets/assets`
  into the repo, or `export ROBOT_PLATFORM=ALOHA`). The driver replays the whole
  CI test step, so those are included automatically — but they assume the asset
  dirs exist under `/workspace/dataset/`.
- **A duplicated config in `list`** (e.g. `d4rl_iql_mujoco,d4rl_iql_mujoco`) just
  means the job runs that config twice with different flags (FSDP on/off). Normal.

## Troubleshooting

- **`No CI job for model=… env=…`** — that combo isn't in the workflow. Run
  `list` to see valid pairs; the model/env strings must match `--model`/`--env`
  in `install.sh` exactly (`maniskill_libero`, not `libero`).
- **`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`** in
  a test run — benign protobuf/TF-on-import noise from the workers, not a failure.
  Look for the Ray `Placement(...)` lines and the rollout progress bar to confirm
  real progress.
- **`error: run this from inside the RLinf repo`** — `cd` to the repo root (the
  dir containing `requirements/install.sh`) before invoking the driver.
- **Test exits 0 immediately but nothing trained** — you backgrounded it with `&`;
  the launcher returns 0 while training detaches. Run it in the foreground, or
  wait on the real `train_embodied_agent.py` pid.
- **`install` hangs at `uv sync` with ~0 CPU and no `.uv_cache` writes** — almost
  always a dead `http(s)_proxy` env var on the box (e.g. a local
  `127.0.0.1:10809` that isn't forwarding), leaving idle ESTABLISHED `:443`
  connections. `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy
  ALL_PROXY` before running the install. Not an install.sh bug.
- **`setup_mirror` fails: `cannot overwrite multiple values ... insteadOf`** —
  prior interrupted runs left duplicate `url.<mirror>.insteadOf` entries in the
  global git config. Clear them with
  `git config --global --unset-all url."https://ghfast.top/github.com/".insteadOf`
  then retry. Also environmental, not an install.sh bug.
