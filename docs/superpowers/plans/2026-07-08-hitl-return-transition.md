# HITL Return Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic opt-in transition window that smooths successful HITL return targets immediately before the first teleop frame.

**Architecture:** Keep the feature in `compute_returns.py`, where the returns sidecar is produced, so value SFT and advantage computation consume one shared target definition. Add a small config resolver plus a focused smoothing helper, then thread the resolved step count through the existing parquet and dataset processing functions.

**Tech Stack:** Python, NumPy, PyArrow, OmegaConf, pytest, Hydra YAML configs.

---

## File Structure

- Modify `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
  - Add `_resolve_hitl_transition_steps()` for config resolution.
  - Add `_smooth_pre_teleop_returns()` for the linear pre-teleop ramp.
  - Add `hitl_transition_steps` to `compute_hitl_aware_returns_for_episode()`.
  - Thread the resolved step count through `_process_single_parquet()`, `process_dataset()`, and `compute_returns()`.
- Modify `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`
  - Set `action_horizon: 10`.
  - Set `hitl_transition_chunks: 2`.
- Modify `tests/unit_tests/test_aloha_recap_returns.py`
  - Add direct unit tests for the return transition formula and edge cases.
  - Add config-threading tests for parquet and multi-dataset processing.
  - Add a config smoke test for the ALOHA sandwich return config.

---

### Task 1: Add Direct Return Transition Behavior

**Files:**
- Modify: `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
- Modify: `tests/unit_tests/test_aloha_recap_returns.py`

- [ ] **Step 1: Write failing direct unit tests**

In `tests/unit_tests/test_aloha_recap_returns.py`, update the import block:

```python
from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
    _process_single_parquet,
    _resolve_hitl_transition_steps,
    compute_hitl_aware_returns_for_episode,
    compute_returns_for_episode,
)
```

Add these tests immediately after `test_successful_episode_split_at_first_teleop_frame`:

```python
def test_successful_hitl_transition_steps_ramp_pre_teleop_returns_only() -> None:
    returns, rewards = compute_hitl_aware_returns_for_episode(
        episode_length=6,
        is_success=True,
        teleop_mask=np.asarray([0, 0, 0, 1, 1, 0], dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
        hitl_transition_steps=2,
    )

    np.testing.assert_allclose(
        returns,
        np.asarray([-302.0, -151.5, -2.0, -2.0, -1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        rewards,
        np.asarray([-1.0, -1.0, -300.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )


def test_successful_hitl_transition_clamps_to_short_prefix() -> None:
    returns, rewards = compute_hitl_aware_returns_for_episode(
        episode_length=4,
        is_success=True,
        teleop_mask=np.asarray([0, 1, 1, 0], dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
        hitl_transition_steps=5,
    )

    np.testing.assert_allclose(
        returns,
        np.asarray([-2.0, -2.0, -1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        rewards,
        np.asarray([-300.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )


def test_resolve_hitl_transition_steps_prefers_explicit_steps() -> None:
    assert (
        _resolve_hitl_transition_steps(
            hitl_transition_steps=3,
            hitl_transition_chunks=2,
            action_horizon=10,
        )
        == 3
    )


def test_resolve_hitl_transition_steps_uses_chunks_and_action_horizon() -> None:
    assert (
        _resolve_hitl_transition_steps(
            hitl_transition_steps=None,
            hitl_transition_chunks=2,
            action_horizon=10,
        )
        == 20
    )


def test_resolve_hitl_transition_steps_requires_action_horizon_for_chunks() -> None:
    with pytest.raises(ValueError, match="hitl_transition_chunks requires"):
        _resolve_hitl_transition_steps(
            hitl_transition_steps=None,
            hitl_transition_chunks=2,
            action_horizon=None,
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit_tests/test_aloha_recap_returns.py::test_successful_hitl_transition_steps_ramp_pre_teleop_returns_only \
  tests/unit_tests/test_aloha_recap_returns.py::test_successful_hitl_transition_clamps_to_short_prefix \
  tests/unit_tests/test_aloha_recap_returns.py::test_resolve_hitl_transition_steps_prefers_explicit_steps \
  tests/unit_tests/test_aloha_recap_returns.py::test_resolve_hitl_transition_steps_uses_chunks_and_action_horizon \
  tests/unit_tests/test_aloha_recap_returns.py::test_resolve_hitl_transition_steps_requires_action_horizon_for_chunks \
  -q
```

Expected: FAIL during collection with `ImportError: cannot import name '_resolve_hitl_transition_steps'`.

- [ ] **Step 3: Implement minimal direct behavior**

In `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`, add these helpers after `compute_returns_for_episode()`:

```python
def _resolve_hitl_transition_steps(
    hitl_transition_steps: int | None = None,
    hitl_transition_chunks: int | None = None,
    action_horizon: int | None = None,
) -> int:
    """Resolve HITL transition config to a non-negative frame count."""
    if hitl_transition_steps is not None:
        steps = int(hitl_transition_steps)
        if steps < 0:
            raise ValueError("data.hitl_transition_steps must be >= 0")
        return steps

    chunks = 0 if hitl_transition_chunks is None else int(hitl_transition_chunks)
    if chunks < 0:
        raise ValueError("data.hitl_transition_chunks must be >= 0")
    if chunks == 0:
        return 0
    if action_horizon is None:
        raise ValueError(
            "data.hitl_transition_chunks requires data.action_horizon when "
            "data.hitl_transition_steps is not set"
        )

    horizon = int(action_horizon)
    if horizon <= 0:
        raise ValueError(
            "data.action_horizon must be > 0 when HITL transition chunks are used"
        )
    return chunks * horizon


def _smooth_pre_teleop_returns(
    prefix_returns: np.ndarray,
    suffix_returns: np.ndarray,
    hitl_transition_steps: int,
) -> np.ndarray:
    """Apply a linear return ramp over the final pre-teleop prefix frames."""
    if (
        hitl_transition_steps <= 0
        or len(prefix_returns) == 0
        or len(suffix_returns) == 0
    ):
        return prefix_returns

    start = max(0, len(prefix_returns) - hitl_transition_steps)
    window = len(prefix_returns) - start
    if window <= 0:
        return prefix_returns

    smoothed = prefix_returns.astype(np.float32, copy=True)
    alpha = np.linspace(1.0 / window, 1.0, num=window, dtype=np.float32)
    target_end = float(suffix_returns[0])
    source = smoothed[start:]
    smoothed[start:] = ((1.0 - alpha) * source + alpha * target_end).astype(np.float32)
    return smoothed
```

Update the `compute_hitl_aware_returns_for_episode()` signature:

```python
def compute_hitl_aware_returns_for_episode(
    episode_length: int,
    is_success: bool,
    teleop_mask: np.ndarray,
    gamma: float,
    failure_reward: float,
    hitl_transition_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
```

Inside `compute_hitl_aware_returns_for_episode()`, replace the final return block:

```python
    prefix_returns = _smooth_pre_teleop_returns(
        prefix_returns=prefix_returns,
        suffix_returns=suffix_returns,
        hitl_transition_steps=hitl_transition_steps,
    )
    return (
        np.concatenate([prefix_returns, suffix_returns]).astype(np.float32),
        np.concatenate([prefix_rewards, suffix_rewards]).astype(np.float32),
    )
```

- [ ] **Step 4: Run direct tests to verify pass**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit_tests/test_aloha_recap_returns.py::test_successful_episode_split_at_first_teleop_frame \
  tests/unit_tests/test_aloha_recap_returns.py::test_successful_hitl_transition_steps_ramp_pre_teleop_returns_only \
  tests/unit_tests/test_aloha_recap_returns.py::test_successful_hitl_transition_clamps_to_short_prefix \
  tests/unit_tests/test_aloha_recap_returns.py::test_resolve_hitl_transition_steps_prefers_explicit_steps \
  tests/unit_tests/test_aloha_recap_returns.py::test_resolve_hitl_transition_steps_uses_chunks_and_action_horizon \
  tests/unit_tests/test_aloha_recap_returns.py::test_resolve_hitl_transition_steps_requires_action_horizon_for_chunks \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
  tests/unit_tests/test_aloha_recap_returns.py
git commit -s -m "feat: add HITL return transition helper"
```

---

### Task 2: Thread Transition Steps Through Return Processing

**Files:**
- Modify: `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
- Modify: `tests/unit_tests/test_aloha_recap_returns.py`

- [ ] **Step 1: Write failing threading tests**

In `tests/unit_tests/test_aloha_recap_returns.py`, add this test after `test_process_single_parquet_accepts_lerobot_column_teleop_mask`:

```python
def test_process_single_parquet_applies_hitl_transition_steps(tmp_path) -> None:
    pq_file = tmp_path / "episode.parquet"
    table = pa.table(
        {
            "episode_index": pa.array([0, 0, 0, 0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2, 3, 4, 5], type=pa.int64()),
            "is_success": pa.array([True, True, True, True, True, True]),
            "teleop_mask": pa.array([[0], [0], [0], [1], [1], [0]]),
            "task_index": pa.array([0, 0, 0, 0, 0, 0], type=pa.int64()),
            "task": pa.array(["pick"] * 6),
        }
    )
    pq.write_table(table, pq_file)

    result = _process_single_parquet(
        str(pq_file),
        dataset_type="rollout",
        gamma=1.0,
        failure_reward=-300.0,
        tasks={},
        hitl_aware_returns=True,
        hitl_transition_steps=2,
    )

    np.testing.assert_allclose(
        result.column("return").to_numpy(),
        np.asarray([-302.0, -151.5, -2.0, -2.0, -1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.column("reward").to_numpy(),
        np.asarray([-1.0, -1.0, -300.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )
    assert result.column("done").to_pylist() == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
```

Add this test after `test_compute_returns_uses_per_entry_hitl_aware_override`:

```python
def test_compute_returns_resolves_per_entry_hitl_transition_settings(
    tmp_path, monkeypatch
) -> None:
    first_dataset = tmp_path / "first"
    second_dataset = tmp_path / "second"
    first_dataset.mkdir()
    second_dataset.mkdir()
    captured_transition_steps = []

    def fake_process_dataset(**kwargs):
        captured_transition_steps.append(kwargs["hitl_transition_steps"])
        return {"return": {"min": 0.0, "max": 0.0}, "reward": {}}

    monkeypatch.setattr(returns_module, "process_dataset", fake_process_dataset)
    cfg = OmegaConf.create(
        {
            "data": {
                "data_root": None,
                "train_data_paths": [
                    {
                        "dataset_path": str(first_dataset),
                        "type": "rollout",
                        "hitl_transition_chunks": 2,
                        "action_horizon": 10,
                    },
                    {
                        "dataset_path": str(second_dataset),
                        "type": "rollout",
                        "hitl_transition_steps": 3,
                        "hitl_transition_chunks": 2,
                        "action_horizon": 10,
                    },
                ],
                "dataset_type": "rollout",
                "gamma": 1.0,
                "failure_reward": -300.0,
                "hitl_aware_returns": True,
                "hitl_transition_chunks": 0,
                "action_horizon": None,
                "num_workers": 1,
                "tag": None,
            }
        }
    )

    returns_module.compute_returns(cfg)

    assert captured_transition_steps == [20, 3]
```

- [ ] **Step 2: Run threading tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit_tests/test_aloha_recap_returns.py::test_process_single_parquet_applies_hitl_transition_steps \
  tests/unit_tests/test_aloha_recap_returns.py::test_compute_returns_resolves_per_entry_hitl_transition_settings \
  -q
```

Expected: FAIL with `TypeError` for an unexpected `hitl_transition_steps` keyword or missing `hitl_transition_steps` in captured kwargs.

- [ ] **Step 3: Thread `hitl_transition_steps` through function signatures**

In `compute_returns.py`, update `_process_single_parquet()`:

```python
def _process_single_parquet(
    pq_file: str,
    dataset_type: str,
    gamma: float,
    failure_reward: float,
    tasks: dict[int, str],
    hitl_aware_returns: bool = False,
    hitl_transition_steps: int = 0,
) -> pa.Table | None:
```

Update its call to `compute_hitl_aware_returns_for_episode()`:

```python
            ep_returns, ep_rewards = compute_hitl_aware_returns_for_episode(
                episode_length=ep_length,
                is_success=is_success,
                teleop_mask=teleop_col[ep_start:ep_end],
                gamma=gamma,
                failure_reward=failure_reward,
                hitl_transition_steps=hitl_transition_steps,
            )
```

Update `process_dataset()`:

```python
def process_dataset(
    dataset_path: Path,
    output_path: Path | None,
    dataset_type: str,
    gamma: float,
    failure_reward: float,
    num_workers: int = 8,
    tag: str | None = None,
    hitl_aware_returns: bool = False,
    hitl_transition_steps: int = 0,
) -> dict:
```

Update the process log line:

```python
    logger.info(
        f"  Type: {dataset_type}, Gamma: {gamma}, "
        f"Failure reward: {failure_reward}, HITL-aware returns: {hitl_aware_returns}, "
        f"HITL transition steps: {hitl_transition_steps}"
    )
```

Update both `_process_single_parquet()` call sites inside `process_dataset()` to pass `hitl_transition_steps` after `hitl_aware_returns`:

```python
                hitl_aware_returns,
                hitl_transition_steps,
```

- [ ] **Step 4: Resolve transition config in `compute_returns()`**

In `compute_returns()`, after `hitl_aware_returns = cfg.data.get("hitl_aware_returns", False)`, add:

```python
    default_hitl_transition_steps = cfg.data.get("hitl_transition_steps", None)
    default_hitl_transition_chunks = cfg.data.get("hitl_transition_chunks", 0)
    default_action_horizon = cfg.data.get("action_horizon", None)
```

Inside the `for entry in datasets_list:` loop, before appending to `datasets_to_process`, add:

```python
            entry_hitl_transition_steps = _resolve_hitl_transition_steps(
                hitl_transition_steps=entry.get(
                    "hitl_transition_steps", default_hitl_transition_steps
                ),
                hitl_transition_chunks=entry.get(
                    "hitl_transition_chunks", default_hitl_transition_chunks
                ),
                action_horizon=entry.get("action_horizon", default_action_horizon),
            )
```

Add this key to that `datasets_to_process.append({...})` dict:

```python
                    "hitl_transition_steps": entry_hitl_transition_steps,
```

In the single-dataset branch, before `datasets_to_process.append({...})`, add:

```python
        hitl_transition_steps = _resolve_hitl_transition_steps(
            hitl_transition_steps=default_hitl_transition_steps,
            hitl_transition_chunks=default_hitl_transition_chunks,
            action_horizon=default_action_horizon,
        )
```

Add this key to the single-dataset `datasets_to_process.append({...})` dict:

```python
                "hitl_transition_steps": hitl_transition_steps,
```

Update the `process_dataset()` call near the end of `compute_returns()`:

```python
            hitl_transition_steps=ds_config["hitl_transition_steps"],
```

- [ ] **Step 5: Run threading tests to verify pass**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit_tests/test_aloha_recap_returns.py::test_process_single_parquet_applies_hitl_transition_steps \
  tests/unit_tests/test_aloha_recap_returns.py::test_compute_returns_resolves_per_entry_hitl_transition_settings \
  -q
```

Expected: PASS.

- [ ] **Step 6: Run existing HITL return tests to verify no regression**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit_tests/test_aloha_recap_returns.py::test_successful_episode_split_at_first_teleop_frame \
  tests/unit_tests/test_aloha_recap_returns.py::test_process_single_parquet_accepts_lerobot_column_teleop_mask \
  tests/unit_tests/test_aloha_recap_returns.py::test_compute_returns_uses_per_entry_hitl_aware_override \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
  tests/unit_tests/test_aloha_recap_returns.py
git commit -s -m "feat: thread HITL return transition config"
```

---

### Task 3: Enable ALOHA Sandwich Transition Config

**Files:**
- Modify: `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`
- Modify: `tests/unit_tests/test_aloha_recap_returns.py`

- [ ] **Step 1: Write failing config smoke test**

In `tests/unit_tests/test_aloha_recap_returns.py`, add this import near the top:

```python
from pathlib import Path
```

Add this test at the end of the file:

```python
def test_aloha_sandwich_compute_returns_config_enables_two_transition_chunks() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        repo_root
        / "examples"
        / "offline_rl"
        / "config"
        / "aloha_sandwich_recap_compute_returns.yaml"
    )

    assert cfg.data.hitl_aware_returns is True
    assert cfg.data.action_horizon == 10
    assert cfg.data.hitl_transition_chunks == 2
    assert (
        _resolve_hitl_transition_steps(
            hitl_transition_steps=cfg.data.get("hitl_transition_steps", None),
            hitl_transition_chunks=cfg.data.hitl_transition_chunks,
            action_horizon=cfg.data.action_horizon,
        )
        == 20
    )
```

- [ ] **Step 2: Run config smoke test to verify failure**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit_tests/test_aloha_recap_returns.py::test_aloha_sandwich_compute_returns_config_enables_two_transition_chunks \
  -q
```

Expected: FAIL with `omegaconf.errors.ConfigAttributeError` for missing `action_horizon` or `hitl_transition_chunks`.

- [ ] **Step 3: Update ALOHA sandwich return config**

In `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`, update the `data` block to include:

```yaml
  gamma: 1.0
  action_horizon: 10
  failure_reward: -300.0
  hitl_aware_returns: true
  hitl_transition_chunks: 2
  tag: sandwich_fail300
```

The resulting top-level `data` block should keep the existing dataset path, `dataset_type`, `tag`, and `num_workers` values.

- [ ] **Step 4: Run config smoke test to verify pass**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit_tests/test_aloha_recap_returns.py::test_aloha_sandwich_compute_returns_config_enables_two_transition_chunks \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml \
  tests/unit_tests/test_aloha_recap_returns.py
git commit -s -m "config: enable ALOHA HITL return transition"
```

---

### Task 4: Full Verification

**Files:**
- Verify: `examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
- Verify: `examples/offline_rl/config/aloha_sandwich_recap_compute_returns.yaml`
- Verify: `tests/unit_tests/test_aloha_recap_returns.py`

- [ ] **Step 1: Run all ALOHA return tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit_tests/test_aloha_recap_returns.py -q
```

Expected: PASS for every test in `test_aloha_recap_returns.py`.

- [ ] **Step 2: Run adjacent ALOHA RECAP transform tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit_tests/test_aloha_recap_transforms.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Ruff check on modified files**

Run:

```bash
.venv/bin/python -m ruff check \
  examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
  tests/unit_tests/test_aloha_recap_returns.py
```

Expected: PASS with no lint errors.

- [ ] **Step 4: Run Ruff format check on modified files**

Run:

```bash
.venv/bin/python -m ruff format --check \
  examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
  tests/unit_tests/test_aloha_recap_returns.py
```

Expected: PASS with no formatting changes required.

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: clean working tree after the three task commits, or only unrelated pre-existing files if the implementation was done in a dirty workspace. No generated data files should be staged.

---

## Optional Data-Level Validation After Pipeline Rerun

This validation requires recomputing ALOHA sandwich returns with the new config. It is not part of the unit-test implementation commit because it rewrites dataset sidecar artifacts outside the repo.

After running return computation, inspect episode 025 with:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21/meta/returns_sandwich_fail300.parquet")
df = pd.read_parquet(path)
ep = df[df.episode_index == 25].sort_values("frame_index")
for frame in range(1738, 1759):
    row = ep[ep.frame_index == frame].iloc[0]
    print(frame, float(row["return"]), float(row["return"]) / 3614.0)
PY
```

Expected after rerun: frames `1738` through `1757` form a smooth downward ramp toward frame `1758`, instead of frame `1757` staying near normalized `-0.083` and frame `1758` jumping directly to normalized `-0.199`.
