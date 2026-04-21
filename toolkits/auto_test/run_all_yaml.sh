#!/bin/bash

# 1. Core path configuration
# Auto-detect script directory as REPO_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="${REPO_PATH:-$SCRIPT_DIR}"
export LOG_DIR="$REPO_PATH/logs"
export WORKDIR="$REPO_PATH/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export SYNC_FLAG_FILE="$REPO_PATH/ray_utils/task_sync.txt"
export FORCE_REBUILD=1

# Virtual environment configuration
export VENV_BASE_DIR="/mnt/public/xttx/xusi/rlinf_venv"
# 2. Task list
#    Format: ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE
#    ENV_NAME: Environment name (maniskill_libero, behavior, isaaclab, metaworld, calvin, etc.)
#    MODEL_NAME: Model name (openvla, openvla-oft, openpi, gr00t, mlp, etc.)
#    YAML_ARG: Configuration file name
TASKS=(
    # Original tasks
    "maniskill_libero openpi maniskill_ppo_mlp 1 1000 -1"
    "maniskill_libero openpi libero_goal_ppo_openpi 1 120 -1"
    "maniskill_libero openpi libero_goal_ppo_openpi_pi05 1 120 -1"
    # "maniskill_libero openpi maniskill_ppo_openpi 1 10 -1"   
    # "maniskill_libero openpi maniskill_ppo_openpi_pi05 1 120 -1"   
    # "maniskill_libero openvla maniskill_ppo_openvla 1 120 -1"
    # "maniskill_libero openvla-oft maniskill_ppo_openvlaoft 1 120 -1"
    # "maniskill_libero gr00t libero_10_ppo_gr00t 1 120 -1"

    # "maniskill_libero openpi gsenv_ppo_openpi_pi05 1 120 -1"   
    # "maniskill_libero openpi maniskill_async_ppo_openpi 1 120 -1"   
    # "maniskill_libero openpi maniskill_async_ppo_openpi_pi05 1 120 -1"   
    # "maniskill_libero openvla maniskill_async_ppo_openvla 1 120 -1"   
    # "maniskill_libero openvla-oft maniskill_async_ppo_openvlaoft 1 120 -1"   
    # "maniskill_libero openpi libero_spatial_async_ppo_openpi 1 120 -1"   
    # "maniskill_libero openpi libero_object_async_ppo_openpi_pi05 1 120 -1"   
    # "maniskill_libero openvla maniskill_grpo_openvla 1 120 -1"
    # "maniskill_libero openvla-oft maniskill_grpo_openvlaoft 1 120 -1"
    # "maniskill_libero openpi libero_10_grpo_openpi 1 120 -1"
    # "maniskill_libero openpi libero_spatial_grpo_openpi_pi05 1 120 -1"
    # "maniskill_libero openvla-oft libero_10_grpo_openvlaoft 1 120 -1"
    # "maniskill_libero openpi libero_spatial_0_grpo_mlp 1 1000 -1"
    # "maniskill_libero openpi maniskill_sac_mlp 1 1000 -1"   
    # maniskill tests completed

    # "behavior openpi behavior_ppo_openpi 1 120 -1"   
    # "calvin openpi calvin_abc_d_ppo_openpi 1 120 -1"   
    # "calvin openpi calvin_abcd_d_ppo_openpi_pi05 1 120 -1"   
    # "robotwin openvla-oft robotwin_place_empty_cup_ppo_openvlaoft 1 120 -1"   
    # "isaaclab gr00t isaaclab_franka_stack_cube_ppo_gr00t 1 120 -1"   
    # "frankasim mlp frankasim_ppo_mlp 1 1000 -1"   

    # "robotwin openvla-oft robotwin_beat_block_hammer_grpo_openvlaoft 1 120 -1"   
    # "wan openvla-oft wan_libero_goal_grpo_openvlaoft 1 120 -1"   

    # "frankasim mlp frankasim_sac_cnn_async 1 120 -1"   
    
    # Tasks supplemented from workflow
    # "maniskill_libero openvla maniskill_sac_mlp 1 120 -1"
    # "maniskill_libero openvla maniskill_sac_mlp_async 1 120 -1"
    # "maniskill_libero openvla maniskill_sac_flow_state 1 120 -1"
    # "maniskill_libero openvla realworld_dummy_sac_cnn 1 120 -1"
    # "frankasim openvla frankasim_ppo_mlp 1 120 -1"
    # "frankasim openvla frankasim_sac_cnn_async 1 120 -1"
    # "maniskill_libero openvla-oft libero_goal_grpo_openvlaoft 1 120 -1"
    # "behavior openvla-oft behavior_ppo_openvlaoft 1 120 -1"
    # "robotwin openvla-oft robotwin_grpo_openvlaoft 1 120 -1"
    # "maniskill_libero gr00t libero_spatial_ppo_gr00t 1 120 -1"
    # "isaaclab gr00t isaaclab_ppo_gr00t 1 120 -1"
    # "maniskill_libero openpi maniskill_ppo_openpi05 1 120 -1"
    # "maniskill_libero openpi libero_spatial_ppo_openpi 1 120 -1"
    # "maniskill_libero openpi libero_spatial_ppo_openpi05 1 120 -1"
    # "maniskill_libero openpi libero_spatial_dsrl_openpi 1 120 -1"
    # "maniskill_libero openpi maniskill_ppo_co_training_openpi_pi05 1 120 -1"
    # "metaworld openpi metaworld_50_ppo_openpi 1 120 -1"
    # "calvin openpi calvin_ppo_openpi 1 120 -1"
    # "maniskill_libero openpi robocasa_grpo_openpi 1 120 -1"
    # "maniskill_libero openvla-oft opensora_libero_spatial_grpo_openvlaoft 1 120 -1"
    # "maniskill_libero openvla-oft wan_libero_spatial_grpo_openvlaoft 1 120 -1"
)

export RANK=${RANK:-0}
export NUM_GPUS_PER_NODE=8

# Define unified cleanup function
function super_cleanup() {
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
    super_cleanup
    
    # Task statistics counters
    TOTAL_TASKS=${#TASKS[@]}
    CURRENT_TASK_INDEX=0
    SKIPPED_THRESHOLD=0
    SKIPPED_CRASHED=0
    SUCCESS_COUNT=0
    FAILED_COUNT=0

    for TASK_STR in "${TASKS[@]}"; do
        read -r ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE <<< "$TASK_STR"
        
        CURRENT_TASK_INDEX=$((CURRENT_TASK_INDEX + 1))
        
        echo "========================================================="
        echo "TASK [$CURRENT_TASK_INDEX/$TOTAL_TASKS]: $YAML_ARG | ENV: $ENV_NAME | MODEL: $MODEL_NAME"
        echo "========================================================="
        
        # Check if log file exists, determine if task should be skipped
        # Log directory format: timestamp-YAML_ARG (e.g., 20260414-11:04:12-libero_90_grpo_openvlaoft)
        # Fuzzy match and select the latest log directory
        LATEST_LOG_DIR=""
        if [ -d "$LOG_DIR" ]; then
            # Find all directories matching *-YAML_ARG format, sort by name (timestamp format ensures lexicographic order = chronological order)
            LATEST_LOG_DIR=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*-${YAML_ARG}" 2>/dev/null | sort -r | head -n 1)
        fi
        
        if [ -n "$LATEST_LOG_DIR" ] && [ -d "$LATEST_LOG_DIR" ]; then
            LOG_FILE="${LATEST_LOG_DIR}/run_embodiment.log"
            echo "Found existing log file: $LOG_FILE"
            echo "Checking training status..."
            # Call Python script to check log status (using simple format output: reached,crashed)
            CHECK_RESULT=$(python3 "$REPO_PATH/check.py" "$LOG_FILE" --threshold=100 --format=simple 2>&1)
            
            # Parse result: reached,crashed
            REACHED=$(echo "$CHECK_RESULT" | cut -d',' -f1)
            CRASHED=$(echo "$CHECK_RESULT" | cut -d',' -f2)
            
            echo "Check result: reached=$REACHED,\t crashed=$CRASHED"
            
            # If threshold reached, skip task
            if [ "$REACHED" = "True" ]; then
                echo ">>> SKIP: Task already reached threshold (10% of total steps)"
                SKIPPED_THRESHOLD=$((SKIPPED_THRESHOLD + 1))
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                continue
            fi
            
            # If previously ran but crashed, skip task
            if [ "$CRASHED" = "True" ]; then
                echo ">>> SKIP: Task crashed before reaching threshold"
                SKIPPED_CRASHED=$((SKIPPED_CRASHED + 1))     
                FAILED_COUNT=$((FAILED_COUNT + 1))
                continue
            fi
        else
            echo ">>> START: $YAML_ARG Task not started yet, Starting ..."        
        fi

        # 1. Ensure Worker sees signal has disappeared (cleanup phase)
        rm -f "$SYNC_FLAG_FILE"
        sleep 5 # Give time for shared filesystem sync

        # 2. Determine virtual environment path (format: ENV_NAME_MODEL_NAME)
        cd "$REPO_PATH" || exit
        VENV_NAME="${ENV_NAME}_${MODEL_NAME}"
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
        source switch_env $MODEL_NAME
        echo "Activated virtual environment: $MODEL_NAME"

        # 5. Write new signal and start Ray Head
        echo "$ENV_NAME" > "$SYNC_FLAG_FILE"
        echo "Head: Signal sent. Starting Ray Head..."
        
        export NODES=$T_NODES
        export STEPS=$T_STEPS
        export SAVE_INTER=$T_SAVE
        export TOKENIZERS_PARALLELISM=false

        # Start Ray and wait for cluster ready
        bash ray_utils/start_ray.sh
        # TOTAL_GPUS=$(($T_NODES * $NUM_GPUS_PER_NODE))
        # bash ray_utils/check_ray.sh "$TOTAL_GPUS"

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
        
        # TODO
        # Execute training script
        bash "${WORKDIR}/run_embodiment.sh" "$YAML_ARG" 2>&1 | tee "${YAML_ARG}_run.log"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -ne 0 ]; then
            echo "！！！CRITICAL ERROR: $YAML_ARG failed with Code $EXIT_CODE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            rm -f "$SYNC_FLAG_FILE"
            super_cleanup
            # No longer exit directly, continue to next task
            sleep 10
            continue
        fi

        # 7. Task successful, clear signal, prepare for next round
        echo "Task $YAML_ARG completed successfully."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        rm -f "$SYNC_FLAG_FILE"
        super_cleanup
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
    
    # Print success_once results for each task
    echo ""
    echo "========================================================="
    echo "              SUCCESS_ONCE @ STEP 100                    "
    echo "========================================================="
    if [ -d "$LOG_DIR" ]; then
        python3 "$REPO_PATH/parse_success_once.py" "$LOG_DIR" --step 100 2>/dev/null || echo "Failed to parse success_once results"
    else
        echo "Log directory not found: $LOG_DIR"
    fi
    echo "========================================================="
    
    # Print success_once curves for each task
    echo ""
    echo "========================================================="
    echo "         SUCCESS_ONCE CURVES FOR ALL TASKS               "
    echo "========================================================="
    
    # Create directory for saving curves
    CURVES_DIR="$LOG_DIR/success_once_curves"
    mkdir -p "$CURVES_DIR"
    
    for TASK_STR in "${TASKS[@]}"; do
        read -r ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE <<< "$TASK_STR"
        
        # Find the latest log directory for this task
        LATEST_LOG_DIR=""
        if [ -d "$LOG_DIR" ]; then
            LATEST_LOG_DIR=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*-${YAML_ARG}" 2>/dev/null | sort -r | head -n 1)
        fi
        
        if [ -n "$LATEST_LOG_DIR" ] && [ -d "$LATEST_LOG_DIR" ]; then
            LOG_FILE="${LATEST_LOG_DIR}/run_embodiment.log"
            if [ -f "$LOG_FILE" ]; then
                echo ""
                echo "--- Processing: $YAML_ARG ---"
                CURVE_OUTPUT="${CURVES_DIR}/${YAML_ARG}_success_once_curve.png"
                DATA_OUTPUT="${CURVES_DIR}/${YAML_ARG}_success_once_data.csv"
                python3 "$REPO_PATH/parse_success_once.py" "$LOG_FILE" --plot "$CURVE_OUTPUT" --plot-data "$DATA_OUTPUT" 2>/dev/null || echo "Failed to process $YAML_ARG"
            fi
        fi
    done
        
    echo ""
    echo "========================================================="
    echo "Curves saved to: $CURVES_DIR"
    echo "========================================================="
    
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
        source switch_env "$CURRENT_ENV"
        super_cleanup
        
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
        super_cleanup
    done
fi

