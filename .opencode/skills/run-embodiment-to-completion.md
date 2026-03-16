---
name: run-embodiment-to-completion
description: Run an embodiment job to real completion using a persistent log file. Kill stale runs before every retry, diagnose the first root-cause error from the log, and keep iterating until the run exits cleanly or a documented blocker remains within the allowed edit scope.
---

# Run Embodiment Script to Completion

Use this skill when the user wants an embodiment job such as `examples/embodiment/run_embodiment.sh <config_name>` to actually finish, not merely initialize.

## Non-negotiables

- Do not treat startup, model loading, or partial rollout progress as success.
- For this NaVid/Habitat debugging workflow, directly run `bash examples/embodiment/run_embodiment.sh habitat_r2r_ppo_navid` and do not substitute a different config name unless the user explicitly tells you to.
- Do not pipe the training command through `tail`. Redirect stdout/stderr to a file and inspect that file.
- Do not use `wc -l` as your primary log-inspection method. Start from a tail window, and if the error is incomplete, expand the inspected line range backward until you reach a clearly normal milestone, then read forward top-to-bottom.
- Before every retry, kill the previous run and confirm it is gone.
- Diagnose the first real exception, not the later cascade (`ActorDiedError`, `ray.kill`, etc.).
- Respect user-imposed edit boundaries. For this `habitat_r2r_ppo_navid` workflow, prefer fixes in NaVid/Habitat-local config, model, and env code before touching RLinf framework-wide code.
- If the first credible fix appears to live in shared RLinf framework code, do **not** implement it as a “temporary” or “obvious” fix. Treat that as an out-of-scope blocker until you have proved no local NaVid/Habitat fix is viable.
- Before editing any file under `rlinf/workers/`, `rlinf/runners/`, `rlinf/algorithms/`, `rlinf/scheduler/`, or `rlinf/utils/`, stop and explicitly re-check whether the same outcome can be achieved in config, `rlinf/models/embodiment/navid/` integration code, or `rlinf/envs/habitat/` instead.
- When checking GPU utilization or occupancy, first read `examples/embodiment/config/habitat_r2r_ppo_navid.yaml` and determine the relevant GPU indices from `cluster.component_placement`. Monitor only those GPUs for actor/rollout/env placement instead of assuming the whole server is dedicated to this run.
- Keep monitoring sleeps at 120 seconds or less so the debug loop stays responsive.

## Standard Loop

## Edit Scope for `habitat_r2r_ppo_navid`

In this workflow, **shared-code** means RLinf framework code that is reused across multiple models, envs, or training flows. If you edit it, you are changing behavior beyond NaVid-on-Habitat.

### Preferred local edit surface

Try to keep fixes inside these paths first:

- `examples/embodiment/config/habitat_r2r_ppo_navid.yaml`
- `examples/embodiment/config/model/navid.yaml`
- `examples/embodiment/config/env/habitat_r2r.yaml`
- `rlinf/models/embodiment/navid/` except for the original NaVid implementation paths listed below
- `rlinf/envs/habitat/`
- `rlinf/envs/habitat/extensions/config/vlnce_r2r_navid.yaml`
- `VLN-CE/config/navid.yaml`
- NaVid/Habitat-targeted tests such as `tests/unit_tests/test_navid_value_shapes.py`

### Original NaVid implementation: do not modify

Treat these as vendored or upstream NaVid implementation and do not edit them during this debug workflow:

- `rlinf/models/embodiment/navid/model/`
- `rlinf/models/embodiment/navid/processor/`
- `rlinf/models/embodiment/navid/constants.py`
- `rlinf/models/embodiment/navid/conversation.py`
- `rlinf/models/embodiment/navid/mm_utils.py`

If a bug appears to originate there, adapt the surrounding RLinf-local integration, config, or Habitat-side code instead of patching the original NaVid sources directly.

### Treat as shared RLinf framework code

Avoid editing these unless the root cause cannot be fixed in the local surface above:

- `rlinf/config.py`
- `rlinf/envs/__init__.py`
- `rlinf/models/__init__.py`
- `rlinf/envs/action_utils.py`
- `rlinf/workers/`
- `rlinf/runners/`
- `rlinf/algorithms/`
- `rlinf/scheduler/`
- `rlinf/utils/`
- generic model code under `rlinf/models/embodiment/` outside `navid/`
- original NaVid upstream implementation under `rlinf/models/embodiment/navid/model/` and `rlinf/models/embodiment/navid/processor/`

Example: `rlinf/workers/actor/fsdp_actor_worker.py` is shared-code. Even if a NaVid/Habitat failure finally raises there, do not patch it first in this workflow.

### Decision rule before editing

Ask: **Is this file specific to NaVid or Habitat, or is it general RLinf infrastructure?**

- If it is specific to NaVid/Habitat and is not part of the original NaVid implementation, it is in scope.
- If it is general infrastructure used by multiple stacks, treat it as shared-code and avoid editing it unless no local fix is viable.
- If it lives under `rlinf/models/embodiment/navid/model/` or `rlinf/models/embodiment/navid/processor/`, treat it as protected upstream code and do not edit it in this workflow.

If the first real root cause appears to require a shared-code change, stop the loop and hand off with the exact blocker instead of widening scope implicitly.

### Hard stop before shared-code edits

If you are about to edit a shared RLinf file because it looks like the fastest or cleanest fix, stop.

Run this checklist first:

1. Name the exact local files already inspected under config, `rlinf/models/embodiment/navid/`, and `rlinf/envs/habitat/`.
2. State why each local surface cannot safely fix the root cause.
3. Confirm that the proposed shared-code edit would affect non-NaVid or non-Habitat training flows.
4. If any answer is missing, do **not** edit shared code.

For this workflow, a plausible shared-code fix is still **not allowed** unless you first document why the local edit surface is exhausted. “The bug manifests in `rlinf/workers/`” is not enough by itself; the decision must be based on edit scope, not only stack trace location.

If you already made a broader edit while debugging, revert it immediately and continue from the local edit surface or produce a blocked-state handoff.

### 1. Clean up the previous attempt

Before rerunning, kill the prior launcher and trainer:

```bash
pkill -f "run_embodiment.sh" || true
pkill -f "train_embodied_agent.py" || true
sleep 2
pgrep -af "run_embodiment.sh|train_embodied_agent.py" || true
```

If GPUs still look busy and the failure smells like OOM, first determine which GPUs belong to this run from `examples/embodiment/config/habitat_r2r_ppo_navid.yaml` under `cluster.component_placement`, then inspect only those GPUs and their remaining processes before changing code or config. Do not treat activity on unrelated GPUs as evidence that the NaVid/Habitat run is still alive or out of memory.

### 2. Start the run with a real log file

Use a persistent log file from the start:

```bash
rm -f output.log
bash examples/embodiment/run_embodiment.sh habitat_r2r_ppo_navid > output.log 2>&1 &
echo $! > output.pid
```

A shell timeout does not mean the run failed. It may still be running in the background.

### 3. Monitor the log file, not the live process

Use file-based inspection only:

```bash
tail -n 100 output.log
sed -n 'START,ENDp' output.log
```

Start with `tail -n 100 output.log`. If that tail does not contain the full error, expand the inspected range backward with `sed -n 'START,ENDp' output.log` until you reach the last clearly normal milestone, then read forward top-to-bottom to locate the first root-cause failure.

Use short monitoring intervals only. Never sleep more than 120 seconds between log checks.

When GPU monitoring is needed during the loop, scope it to the GPUs implied by the config. For the current `habitat_r2r_ppo_navid` config, `cluster.component_placement` shows `actor: 4`, `rollout: 5`, and `env: 5`, so GPUs 4 and 5 are the first ones to inspect. Other GPUs on the server may belong to unrelated workloads.

### 4. Separate noise from failure

Common non-fatal noise in Habitat/NaVid runs includes:
- Hydra `_self_` warning
- Gym deprecation warning
- Habitat lighting/OpenGL capability dumps
- Long checkpoint loading
- Rollout progress such as `Generating Rollout Epochs: ...`

Do not stop on those. Keep reading until you find the first exception or the run exits cleanly.

### 5. Fix the root cause in the allowed scope

- Prefer the smallest correct fix.
- Do not silence errors or add catch-all fallbacks that hide the problem.
- If the user restricts edits to model/env/config files, do not patch shared RLinf utilities or framework dispatch paths. Revert any broader change and relocate the fix into NaVid/Habitat-local files when possible.
- If the root cause is visible inside a shared framework stack trace but the allowed fix surface is local, treat the shared stack frame as a symptom until you rule out local dtype/shape/config/integration causes.
- Do not convert a blocked local-scope investigation into a framework patch. Stop, capture the blocker, and hand off instead.
- After each fix, rerun from step 1.

## What "Finished" Means

The job is finished only when one of these is true:
- the training or eval run exits cleanly;
- the requested run reaches its configured terminal condition;
- the user explicitly accepts the current stopping point.

A timed-out monitoring command is not completion.

## NaVid / Habitat Lessons from `habitat_r2r_ppo_navid`

Use these as concrete checks when the config or stack is similar:

- OOM can happen during actor init or optimizer warmup after checkpoints load. Check for stale processes and real GPU occupancy before assuming the config alone is wrong.
- For OOM triage on multi-tenant servers, map GPU checks to `cluster.component_placement` first. In the current config, actor runs on GPU 4 while rollout and env run on GPU 5, so unrelated activity on other GPUs should not drive the diagnosis.
- `env.train.max_steps_per_rollout_epoch` must be divisible by `actor.model.num_action_chunks`. For NaVid configs inheriting `model/navid.yaml`, `num_action_chunks` is `3`, so values like `4` are invalid.
- `Warning: NaVid could not parse action ... using 'no_op'` is not necessarily the crash. Keep reading for the next real exception.
- A rollout can reach 75% or 100% and still fail later in trajectory collation or advantage computation. Continue past rollout generation and into actor-side processing.
- When you see a late Ray failure (`ActorDiedError`, collective manager death, `ray.kill`), look earlier in the log for the first worker exception. In the session, the real root cause appeared earlier as a rollout or trajectory error, not the later actor death.
- If a model-specific payload shape or type breaks shared trajectory handling, prefer adapting the model or env-local data shape instead of modifying a shared base utility unless the user explicitly approves framework-wide changes.
- Even when a local root cause propagates into `rlinf/workers/actor/` or `rlinf/algorithms/`, do not patch those shared files first. For example, a NaVid/Habitat dtype mismatch that finally asserts inside PPO loss still requires checking NaVid/Habitat-local outputs, config, and integration boundaries before changing the framework.

## Blocked-State Handoff

If you cannot continue without violating the allowed edit scope, stop only after capturing:
- the exact command you ran;
- the current log file path;
- the last normal milestone in the log;
- the first root-cause stack trace;
- the specific reason the next fix would require prohibited shared-code changes.

State the blocker explicitly in the handoff, for example: "The remaining candidate fix is in `rlinf/workers/actor/fsdp_actor_worker.py`, which is shared RLinf framework code and therefore out of scope for this NaVid/Habitat-local workflow."
