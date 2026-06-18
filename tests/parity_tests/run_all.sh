#!/bin/bash

# Parse command line arguments
SIMILARITY_METHOD="pearson"
while [[ $# -gt 0 ]]; do
    case $1 in
        --similarity-method)
            SIMILARITY_METHOD="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 1. Core path configuration
# Auto-detect script directory and compute REPO_PATH (two levels up from tests/parity_tests)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="${REPO_PATH:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
export LOG_DIR="$REPO_PATH/logs"
export WORKDIR="$REPO_PATH/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${REPO_PATH}/tests/parity_tests:${PYTHONPATH:-}"
export SYNC_FLAG_FILE="$REPO_PATH/ray_utils/task_sync.txt"
export FORCE_REBUILD=1

# Virtual environment configuration
# Override via environment: export VENV_BASE_DIR=/your/venvs/path
export VENV_BASE_DIR="${VENV_BASE_DIR:-/path/to/venvs}"

# Pre-flight check: ensure VENV_BASE_DIR has been configured
if [ "$VENV_BASE_DIR" = "/path/to/venvs" ]; then
    echo "ERROR: VENV_BASE_DIR is not configured."
    echo "Please set it to your virtual environments directory before running:"
    echo "  export VENV_BASE_DIR=/path/to/your/venvs"
    echo "  bash tests/parity_tests/run_all.sh"
    exit 1
fi
# 2. Task list
#    Format: ENV_NAME MODEL_NAME VENV_NAME YAML_ARG T_NODES T_STEPS T_SAVE
#    ENV_NAME: Environment name (maniskill_libero, behavior, isaaclab, metaworld, calvin, etc.)
#    MODEL_NAME: Model name (openvla, openvla-oft, openpi, gr00t, mlp, etc.)
#    VENV_NAME: Virtual environment name (defaults to MODEL_NAME; override per-task when needed)
#    YAML_ARG: Configuration file name
TASKS=(
    "maniskill_libero openvla-oft openvla-oft maniskill_ppo_openvlaoft 1 100 -1"
    "maniskill_libero openpi openpi libero_goal_ppo_openpi 1 100 -1"
    "maniskill_libero openpi openpi libero_goal_ppo_openpi_pi05 1 100 -1"
    "maniskill_libero openpi openpi maniskill_ppo_mlp 1 100 -1"
    "maniskill_libero openpi openpi maniskill_ppo_openpi 1 100 -1"
    "maniskill_libero openpi openpi maniskill_ppo_openpi_pi05 1 100 -1"

    # "maniskill_libero openvla openvla maniskill_ppo_openvla 1 120 -1"
    # "maniskill_libero gr00t gr00t libero_10_ppo_gr00t 1 120 -1"

    # "maniskill_libero openpi openpi gsenv_ppo_openpi_pi05 1 120 -1"
    # "maniskill_libero openpi openpi maniskill_async_ppo_openpi 1 120 -1"
    # "maniskill_libero openpi openpi maniskill_async_ppo_openpi_pi05 1 120 -1"
    # "maniskill_libero openvla openvla maniskill_async_ppo_openvla 1 120 -1"
    # "maniskill_libero openvla-oft openvla-oft maniskill_async_ppo_openvlaoft 1 120 -1"
    # "maniskill_libero openpi openpi libero_spatial_async_ppo_openpi 1 120 -1"
    # "maniskill_libero openpi openpi libero_object_async_ppo_openpi_pi05 1 120 -1"
    # "maniskill_libero openvla openvla maniskill_grpo_openvla 1 120 -1"
    # "maniskill_libero openvla-oft openvla-oft maniskill_grpo_openvlaoft 1 120 -1"
    # "maniskill_libero openpi openpi libero_10_grpo_openpi 1 120 -1"
    # "maniskill_libero openpi openpi libero_spatial_grpo_openpi_pi05 1 120 -1"
    # "maniskill_libero openpi openpi libero_spatial_0_grpo_mlp 1 1000 -1"
    # "maniskill_libero openpi openpi maniskill_sac_mlp 1 1000 -1"

    # "behavior openpi openpi behavior_ppo_openpi 1 120 -1"
    # "calvin openpi openpi calvin_abc_d_ppo_openpi 1 120 -1"
    # "calvin openpi openpi calvin_abcd_d_ppo_openpi_pi05 1 120 -1"
    # "robotwin openvla-oft openvla-oft robotwin_place_empty_cup_ppo_openvlaoft 1 120 -1"
    # "isaaclab gr00t gr00t isaaclab_franka_stack_cube_ppo_gr00t 1 120 -1"
    # "frankasim mlp mlp frankasim_ppo_mlp 1 1000 -1"

    # "robotwin openvla-oft openvla-oft robotwin_beat_block_hammer_grpo_openvlaoft 1 120 -1"
    # "wan openvla-oft openvla-oft wan_libero_goal_grpo_openvlaoft 1 120 -1"

    # "frankasim mlp mlp frankasim_sac_cnn_async 1 120 -1"

    # Tasks supplemented from workflow
    # "maniskill_libero openvla openvla maniskill_sac_mlp 1 120 -1"
    # "maniskill_libero openvla openvla maniskill_sac_mlp_async 1 120 -1"
    # "maniskill_libero openvla openvla maniskill_sac_flow_state 1 120 -1"
    # "maniskill_libero openvla openvla realworld_dummy_sac_cnn 1 120 -1"
    # "frankasim openvla openvla frankasim_ppo_mlp 1 120 -1"
    # "frankasim openvla openvla frankasim_sac_cnn_async 1 120 -1"
    # "maniskill_libero openvla-oft openvla-oft libero_goal_grpo_openvlaoft 1 120 -1"
    # "behavior openvla-oft openvla-oft behavior_ppo_openvlaoft 1 120 -1"
    # "robotwin openvla-oft openvla-oft robotwin_grpo_openvlaoft 1 120 -1"
    # "maniskill_libero gr00t gr00t libero_spatial_ppo_gr00t 1 120 -1"
    # "isaaclab gr00t gr00t isaaclab_ppo_gr00t 1 120 -1"
    # "maniskill_libero openpi openpi maniskill_ppo_openpi05 1 120 -1"
    # "maniskill_libero openpi openpi libero_spatial_ppo_openpi 1 120 -1"
    # "maniskill_libero openpi openpi libero_spatial_ppo_openpi05 1 120 -1"
    # "maniskill_libero openpi openpi libero_spatial_dsrl_openpi 1 120 -1"
    # "maniskill_libero openpi openpi maniskill_ppo_co_training_openpi_pi05 1 120 -1"
    # "metaworld openpi openpi metaworld_50_ppo_openpi 1 120 -1"
    # "calvin openpi openpi calvin_ppo_openpi 1 120 -1"
    # "maniskill_libero openpi openpi robocasa_grpo_openpi 1 120 -1"
    # "maniskill_libero openvla-oft openvla-oft opensora_libero_spatial_grpo_openvlaoft 1 120 -1"
    # "maniskill_libero openvla-oft openvla-oft wan_libero_spatial_grpo_openvlaoft 1 120 -1"
)

export RANK=${RANK:-0}
export NUM_GPUS_PER_NODE=8

# Define switch_to_env function for local virtual environment activation
function switch_to_env() {
    local venv_name="$1"
    local venv_path="${VENV_BASE_DIR}/${venv_name}"
    if [ ! -d "$venv_path" ]; then
        echo "Environment $venv_name does not exist in $VENV_BASE_DIR."
        exit 1
    fi
    source "$venv_path/bin/activate"
}

# Define unified cleanup function
function cleanup() {
    echo "[$(date +%T)] Performing aggressive cleanup..."
    # Try to stop ray, skip if command not found
    command -v ray >/dev/null 2>&1 && ray stop --force || echo "Ray command not found, skipping ray stop"
    pkill -9 -u $(whoami) python >/dev/null 2>&1
    pkill -9 -u $(whoami) ray >/dev/null 2>&1
    rm -rf /dev/shm/ray/* 2>/dev/null
    sleep 3
}

# ---------------- RANK branch logic ----------------

if [ "$RANK" -eq 0 ]; then
    # ================= HEAD NODE logic =================
    
    # Clean up all residual signals before starting
    rm -f "$SYNC_FLAG_FILE"
    cleanup
    
    # Task statistics counters
    TOTAL_TASKS=${#TASKS[@]}
    CURRENT_TASK_INDEX=0
    SKIPPED_THRESHOLD=0
    SKIPPED_CRASHED=0
    SUCCESS_COUNT=0
    FAILED_COUNT=0

    for TASK_STR in "${TASKS[@]}"; do
        read -r ENV_NAME MODEL_NAME VENV_NAME YAML_ARG T_NODES T_STEPS T_SAVE <<< "$TASK_STR"
        
        CURRENT_TASK_INDEX=$((CURRENT_TASK_INDEX + 1))
        
        echo "========================================================="
        echo "TASK [$CURRENT_TASK_INDEX/$TOTAL_TASKS]: $YAML_ARG | ENV: $ENV_NAME | MODEL: $MODEL_NAME"
        echo "========================================================="
        
        # Check if any log for this experiment exists and determine status
        # Call check.py with log directory and experiment name
        # It will check all matching logs and return aggregated status
        echo "Checking training status for experiment: $YAML_ARG"
        CHECK_RESULT=$(python3 "$REPO_PATH/tests/parity_tests/check.py" "$LOG_DIR" --experiment "$YAML_ARG" --threshold=100 --format=simple 2>&1)
        
        # Parse result: reached,crashed
        REACHED=$(echo "$CHECK_RESULT" | cut -d',' -f1)
        CRASHED=$(echo "$CHECK_RESULT" | cut -d',' -f2)
        
        echo "Check result: reached=$REACHED,  crashed=$CRASHED"
        
        # If threshold reached in any log, skip task
        if [ "$REACHED" = "True" ]; then
            echo ">>> SKIP: Task already reached threshold in previous run"
            SKIPPED_THRESHOLD=$((SKIPPED_THRESHOLD + 1))
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            continue
        fi
        
        # If all runs crashed (none reached), skip task
        if [ "$CRASHED" = "True" ]; then
            echo ">>> SKIP: Task crashed in all previous runs"
            SKIPPED_CRASHED=$((SKIPPED_CRASHED + 1))     
            FAILED_COUNT=$((FAILED_COUNT + 1))
            continue
        fi
        
        # No matching logs found or logs exist but neither reached nor crashed
        echo ">>> START: $YAML_ARG Task not completed yet, Starting ..."

        # 1. Ensure Worker sees signal has disappeared (cleanup phase)
        rm -f "$SYNC_FLAG_FILE"
        sleep 5 # Give time for shared filesystem sync

        # 2. Determine virtual environment path (VENV_NAME comes from task list)
        cd "$REPO_PATH" || exit
        VENV_PATH="${VENV_BASE_DIR}/${VENV_NAME}"
        
        echo "Building environment: model=$MODEL_NAME, env=$ENV_NAME"
        echo "Virtual environment path: $VENV_PATH"
        
        # Ensure venv base directory exists
        mkdir -p "$VENV_BASE_DIR"
        
        # Set environment variables (refer to workflow settings)
        unset UV_DEFAULT_INDEX
        export UV_PATH=${UV_PATH:-/mnt/public/dataset/.uv}
        export UV_LINK_MODE=${UV_LINK_MODE:-symlink}
        export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/public/dataset/.uv_cache}
        export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/mnt/public/dataset/.uv_python}
        
        # Set specific paths based on environment
        case "$ENV_NAME" in
            maniskill_libero)
                export LIBERO_PATH=${LIBERO_PATH:-/mnt/public/dataset/LIBERO}
                ;;
            behavior)
                export BEHAVIOR_PATH=${BEHAVIOR_PATH:-/mnt/public/dataset/BEHAVIOR-1K}
                export ISAAC_SIM_WHEEL_PATH=${ISAAC_SIM_WHEEL_PATH:-/mnt/public/dataset/isaac_sim_wheels}
                ;;
            isaaclab)
                export ISAAC_LAB_PATH=${ISAAC_LAB_PATH:-/mnt/public/dataset/IsaacLab}
                export GR00T_PATH=${GR00T_PATH:-/mnt/public/dataset/Isaac-GR00T/}
                ;;
            calvin)
                export CALVIN_PATH=${CALVIN_PATH:-/mnt/public/dataset/calvin}
                ;;
            frankasim)
                export SERL_PATH=${SERL_PATH:-/mnt/public/dataset/serl}
                ;;
            robotwin)
                export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/mnt/public/dataset/RoboTwin}
                ;;
        esac
        
        # Set specific paths based on model
        case "$MODEL_NAME" in
            gr00t)
                export GR00T_PATH=${GR00T_PATH:-/mnt/public/dataset/Isaac-GR00T/}
                ;;
            openvla-oft)
                case "$ENV_NAME" in
                    opensora)
                        export OPENSORA_PATH=${OPENSORA_PATH:-/mnt/public/dataset/opensora}
                        ;;
                    wan)
                        export WAN_PATH=${WAN_PATH:-/mnt/public/dataset/wan}
                        ;;
                esac
                ;;
        esac
        
        # 4. Activate environment
        switch_to_env "$VENV_NAME"
        echo "Activated virtual environment: $VENV_NAME"

        # 5. Write new signal and start Ray Head
        echo "$VENV_NAME" > "$SYNC_FLAG_FILE"
        echo "Head: Signal sent. Starting Ray Head..."
        
        export NODES=$T_NODES
        export STEPS=$T_STEPS
        export SAVE_INTER=$T_SAVE
        export TOKENIZERS_PARALLELISM=false

        # Start Ray and wait for cluster ready
        bash ray_utils/start_ray.sh

        # 6. Execute task (refer to run_embodiment.sh logic)
        cd "$WORKDIR" || exit
        echo "Executing training..."
        
        # Set environment variables required by run_embodiment.sh
        export MUJOCO_GL="egl"
        export PYOPENGL_PLATFORM="egl"
        export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
        
        # Set special environment variables based on task (refer to workflow)
        case "$YAML_ARG" in
            robotwin_*)
                export ROBOT_PLATFORM=${ROBOT_PLATFORM:-ALOHA}
                export ROBOTWIN_PATH=${ROBOTWIN_PATH:-/mnt/public/dataset/RoboTwin}
                export PYTHONPATH=${ROBOTWIN_PATH}:$PYTHONPATH
                ;;
            behavior_*)
                export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-/mnt/public/dataset/behavior-datasets}
                export ISAAC_PATH=${ISAAC_PATH:-/mnt/public/dataset/isaac-sim}
                ;;
            isaaclab_*)
                # Isaac Lab environment variables already set during build
                ;;
        esac
        
        # Execute training script
        bash "${WORKDIR}/run_embodiment.sh" "$YAML_ARG" 2>&1 | tee "${YAML_ARG}_run.log"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -ne 0 ]; then
            echo "！！！CRITICAL ERROR: $YAML_ARG failed with Code $EXIT_CODE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            rm -f "$SYNC_FLAG_FILE"
            cleanup
            # No longer exit directly, continue to next task
            sleep 10
            continue
        fi

        # 7. Task successful, clear signal, prepare for next round
        echo "Task $YAML_ARG completed successfully."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        rm -f "$SYNC_FLAG_FILE"
        cleanup
        sleep 10
    done

    # Print final statistics
    echo ""
    echo "========================================================="
    echo "                    FINAL SUMMARY                        "
    echo "========================================================="
    echo "Total tasks:        $TOTAL_TASKS"
    echo "Success:            $SUCCESS_COUNT"
    echo "Skipped (crashed):  $FAILED_COUNT"
    echo "========================================================="
    
    # Run comprehensive log analysis using analyze_logs.py
    echo ""
    echo "========================================================="
    echo "         COMPREHENSIVE LOG ANALYSIS                      "
    echo "========================================================="
    
    # Define baseline directory for comparison
    BASELINE_DIR="$REPO_PATH/logs_baseline"
    
    # Create output directory for analysis results
    ANALYSIS_OUTPUT="$LOG_DIR/analysis_results"
    mkdir -p "$ANALYSIS_OUTPUT"
    
    if [ -d "$LOG_DIR" ]; then
        echo "Parsing Logs ..."
        echo "Conducting Baseline Comparison ..."
        echo ""
        # Run analyze_logs.py with baseline comparison
        python3 "$REPO_PATH/tests/parity_tests/analyze_logs.py" "$LOG_DIR" \
            --output-dir "$ANALYSIS_OUTPUT" \
            --baseline-dir "$BASELINE_DIR" \
            --step 100 \
            --similarity-method "$SIMILARITY_METHOD" \
            --similarity-threshold 0.2 \
            --skip-comparison-plot \
            2>/dev/null || echo "Failed to run log analysis"
        
        echo ""
        echo "========================================================="
        echo "Analysis results saved to: $ANALYSIS_OUTPUT"
        echo "========================================================="
    else
        echo "Log directory not found: $LOG_DIR"
    fi
    
    echo ""
    echo "ALL TASKS COMPLETED!"

else
    # ================= WORKER NODE logic =================
    LAST_PROCESSED_ENV=""

    while true; do
        if [ ! -f "$SYNC_FLAG_FILE" ]; then
            echo "[$(date +%T)] Worker: Waiting for signal..."
            LAST_PROCESSED_ENV="" # Signal disappeared, reset record
            sleep 5
            continue
        fi

        CURRENT_ENV=$(cat "$SYNC_FLAG_FILE" | tr -d '[:space:]')
        
        # If signal file is empty or environment hasn't changed, continue waiting
        if [ -z "$CURRENT_ENV" ] || [ "$CURRENT_ENV" == "$LAST_PROCESSED_ENV" ]; then
            sleep 2
            continue
        fi

        echo "[$(date +%T)] Worker: New Signal [$CURRENT_ENV]. Initializing..."
        
        # 1. Switch environment and sync cleanup
        cd "$REPO_PATH" || exit
        # CURRENT_ENV is the VENV_NAME written by the head node
        switch_to_env "$CURRENT_ENV"
        cleanup
        
        # 2. Start Ray and join cluster
        echo "Worker: Joining Ray cluster with env $CURRENT_ENV..."
        bash ray_utils/start_ray.sh
        
        LAST_PROCESSED_ENV="$CURRENT_ENV"

        # 3. Block and wait for task to finish (signal file deleted by Head)
        echo "Worker: Training in progress..."
        while [ -f "$SYNC_FLAG_FILE" ]; do
            # Check if received signal changed mid-task (though probability is low)
            TMP_ENV=$(cat "$SYNC_FLAG_FILE" 2>/dev/null | tr -d '[:space:]')
            if [ "$TMP_ENV" != "$CURRENT_ENV" ] && [ -n "$TMP_ENV" ]; then
                echo "Worker: Signal changed mid-task! Re-initializing..."
                break
            fi
            sleep 10
        done
        
        echo "Worker: Task finished signal detected. Cleaning up..."
        cleanup
    done
fi

