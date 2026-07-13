#!/usr/bin/env bash
set -euo pipefail

: "${UNIFORM_DATA_ROOT:?Set the root containing step0, step20, ..., step200}"
UNIFORM_STEPS=${UNIFORM_STEPS:-"0 20 40 60 80 100 120 140 160 180 200"}
: "${DUALVIEW_SFT_DATA_ROOT:?Set the processed dataset output root}"

raw_args=()
for step in ${UNIFORM_STEPS}; do
  raw_args+=(--raw-data-path "${UNIFORM_DATA_ROOT}/step${step}")
done

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}

"${PYTHON_BIN}" examples/reward/preprocess_qwentrend_terminal_success_dataset.py \
  "${raw_args[@]}" \
  --output-dir "${DUALVIEW_SFT_DATA_ROOT}" \
  --window-size 5 \
  --val-split 0.1 \
  --max-positive 8000 \
  --negative-positive-ratio 3 \
  --hard-negatives-per-episode 3 \
  --success-exclusion-steps 8 \
  --near-terminal-positives-per-episode 1 \
  --success-positive-lead-steps 4 \
  --online-interval "${ONLINE_INTERVAL:-5}" \
  --workers 32 \
  --seed 42

"${PYTHON_BIN}" - "${DUALVIEW_SFT_DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])


def episode_paths(split):
    paths = set()
    rows = 0
    with (root / split / "segments.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            paths.add(row["source_episode_path"])
            rows += 1
    return paths, rows


train_episodes, train_rows = episode_paths("train")
eval_episodes, eval_rows = episode_paths("eval")
overlap = train_episodes & eval_episodes
if overlap:
    raise SystemExit(f"episode split leakage: {len(overlap)} overlapping episodes")
print(
    f"episode split OK: train={len(train_episodes)} episodes/{train_rows} windows, "
    f"eval={len(eval_episodes)} episodes/{eval_rows} windows, overlap=0"
)
