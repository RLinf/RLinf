#!/usr/bin/env bash
set -euo pipefail

export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_PATH"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"
export RAY_ADDRESS=local HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard

run_dir=$(mktemp -d "${TMPDIR:-/tmp}/g17.XXXXXX")
trap 'rm -rf -- "$run_dir"' EXIT

train() {
    RAY_TMPDIR="$run_dir/ray-$1" python examples/sft/train_vla_sft.py \
        --config-path "$REPO_PATH/tests/e2e_tests/sft" --config-name "$config" \
        "runner.max_steps=$1" "runner.logger.log_path=$output" "${@:2}"
}

config=$1
output="$run_dir/train"
experiment="$output/gr00t_n1d7_spatial_sft"
checkpoints="$experiment/checkpoints"

train 1 actor.fsdp_config.save_full_model_weights=false
train 2 "runner.resume_dir=$checkpoints/global_step_1"

python - "$experiment" <<'PY'
import math
import sys
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint import FileSystemReader

experiment = Path(sys.argv[1])
losses = {}
for event_file in (experiment.parent / "tensorboard").glob("events.out.tfevents.*"):
    events = EventAccumulator(str(event_file)).Reload()
    losses.update((event.step, event.value) for event in events.Scalars("train/loss"))
assert set(losses) == {0, 1} and all(map(math.isfinite, losses.values())), losses

previous = None
for step in (1, 2):
    actor = experiment / "checkpoints" / f"global_step_{step}" / "actor"
    assert FileSystemReader(actor / "dcp_checkpoint").read_metadata().state_dict_metadata
    rng_states = torch.load(actor / "rng.pt", map_location="cpu", weights_only=False)
    assert len(rng_states) == 2 and all(rng_states), (actor, "missing rank RNG state")

    rank_states = torch.load(actor / "data.pt", map_location="cpu", weights_only=False)
    assert len(rank_states) == 2, (actor, "missing rank data state")
    cursors = []
    # The smoke config uses one iterator per rank (num_workers=0).
    for rank, state in enumerate(rank_states):
        cursor = state["fetcher_state"]["dataset_iter_state"]
        assert cursor["identity"][0] == rank, (actor, cursor)
        assert state["_num_yielded"] == step, (actor, rank, state["_num_yielded"])
        cursors.append((cursor["epoch"], cursor["shard_index"], cursor["sample_offset"]))
    if previous is not None:
        assert all(old < new for old, new in zip(previous, cursors)), (previous, cursors)
    previous = cursors
PY

python toolkits/gr00t/export_sft.py \
    --weights "$checkpoints/global_step_2/actor/model_state_dict/full_weights.pt" \
    --base-model "$GR00T_SFT_MODEL_PATH" \
    --processor "$experiment/gr00t/rank_0/processor" \
    --output "$run_dir/export"

RAY_TMPDIR="$run_dir/ray-eval" bash evaluations/run_eval.sh libero libero_spatial_gr00t_n1d7_eval \
    "runner.logger.log_path=$run_dir/eval" \
    "rollout.model.model_path=$run_dir/export" \
    "rollout.model.backbone_model_path=$GR00T_SFT_BACKBONE_PATH" \
    env.eval.total_num_envs=2 env.eval.rollout_epoch=1 \
    env.eval.max_episode_steps=8 env.eval.max_steps_per_rollout_epoch=8

python - "$run_dir/eval/tensorboard" <<'PY'
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

events = EventAccumulator(sys.argv[1]).Reload()
assert events.Scalars("eval/num_trajectories")[-1].value == 2
assert 0 <= events.Scalars("eval/success_once")[-1].value <= 1
PY
