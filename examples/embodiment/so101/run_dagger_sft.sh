#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${RLINF_PYTHON:-${root_dir}/.venv/bin/python}"
: "${SO101_DAGGER_DATASET_PATH:?Set SO101_DAGGER_DATASET_PATH to a LeRobot DAgger dataset}"
: "${SO101_SFT_CHECKPOINT:?Set SO101_SFT_CHECKPOINT to the starting SO-101 checkpoint}"
: "${PI05_BASE_CHECKPOINT:?Set PI05_BASE_CHECKPOINT to the PI05 base checkpoint}"
: "${SO101_NORM_STATS_PATH:?Set SO101_NORM_STATS_PATH to norm_stats.json}"
export EMBODIED_PATH="${root_dir}/examples/sft"
export PYTHONPATH="${root_dir}:${PYTHONPATH:-}"
exec "${python_bin}" "${root_dir}/examples/sft/train_vla_sft.py" \
  --config-name so101_dagger_openpi_pi05 "$@"
