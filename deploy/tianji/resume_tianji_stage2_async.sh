#!/usr/bin/env bash
# Resume a full Stage2 checkpoint through the same collector + async entry.
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# == 0 )) || [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: resume_tianji_stage2_async.sh CHECKPOINT_DIR [--subtask-prompts FILE] [Hydra overrides]'
  echo 'CHECKPOINT_DIR is checkpoints/global_step_N (contains actor/), not a model .pt file.'
  if (( $# == 0 )); then exit 2; fi
  exit 0
fi
checkpoint="$(realpath -e -- "$1")" || exit 2
shift
if [[ ! -d "$checkpoint/actor" ]]; then
  echo "Not a Stage2 checkpoint directory (actor/ missing): $checkpoint" >&2
  exit 2
fi
for arg in "$@"; do
  case "$arg" in
    runner.resume_dir=*|+runner.resume_dir=*|++runner.resume_dir=*|~runner.resume_dir*)
      echo 'Pass the resume checkpoint only as the first argument.' >&2; exit 2 ;;
  esac
done
checkpoint="${checkpoint//\\/\\\\}"
checkpoint="${checkpoint//\"/\\\"}"
exec bash "$launcher/start_tianji_stage2_async.sh" "runner.resume_dir=\"$checkpoint\"" \
  "algorithm.replay_buffer.load_path=null" "$@"
