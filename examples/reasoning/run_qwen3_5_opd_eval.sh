#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /mnt/public2/suheng/venvs/rlinf-agentic/bin/activate
export TMPDIR=/mnt/public2/suheng/tmp TMP=/mnt/public2/suheng/tmp TEMP=/mnt/public2/suheng/tmp
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

base_model=/mnt/public2/suheng/models/Qwen3.5-2B
verl_repo=/mnt/public2/suheng/repos/verl
shopt -s nullglob
rlinf_steps=("$PWD"/logs/*/checkpoints/global_step_160)
verl_steps=("$verl_repo"/checkpoints/qwen3_5_dapo_opd/*/global_step_160)
if (( ${#rlinf_steps[@]} == 0 || ${#verl_steps[@]} == 0 )); then
    echo "Expected step 160 checkpoints for both RLinf and verl; found ${#rlinf_steps[@]} and ${#verl_steps[@]}." >&2
    exit 1
fi

output_root=${EVAL_OUTPUT_DIR:-"$PWD/logs/math-eval"}
mkdir -p "$output_root"
run_dir=$(mktemp -d "$(realpath "$output_root")/run-XXXXXXXX")
echo "Writing results to $run_dir"

evaluate_model() {
    local model_path=$1 config
    shift
    for config in eval_qwen3.5-2b eval_qwen3.5-2b-aime2026 eval_qwen3.5-2b-math500; do
        bash examples/reasoning/run_eval_math.sh "$config" "$@" \
            "rollout.model.model_path=$model_path" \
            "runner.output_dir=$run_dir"
    done
}

evaluate_model "$base_model" "$@"

for step in "${rlinf_steps[@]}"; do
    name="rlinf-$(basename "$(dirname "$(dirname "$step")")")-step160"
    model_path="$run_dir/hf/$name"
    python examples/reasoning/export_qwen3_5_to_hf.py \
        --base-model "$base_model" --weights "$step/actor/model_state_dict/full_weights.pt" \
        --output-dir "$model_path"
    evaluate_model "$model_path" "$@"
done

for step in "${verl_steps[@]}"; do
    name="verl-$(basename "$(dirname "$step")")-step160"
    model_path="$run_dir/hf/$name"
    (cd "$verl_repo" && python -m verl.model_merger merge --backend fsdp \
        --local_dir "$step/actor" --target_dir "$model_path" --use_cpu_initialization)
    evaluate_model "$model_path" "$@"
done

cat "$run_dir/summary.csv"
