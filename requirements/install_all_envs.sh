#!/bin/bash
# 批量安装所有任务所需的环境
# 根据 TASKS 列表提取的唯一环境和模型组合

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 从任务列表提取的唯一环境和模型组合
# 格式: "ENV_NAME MODEL_NAME VENV_NAME"
# 注意: mlp 模型不在 install.sh 支持的模型列表中，需要特殊处理

ENV_MODEL_COMBINATIONS=(
    # maniskill_libero 环境
    "maniskill_libero openpi maniskill_libero_openpi"
    "maniskill_libero openvla maniskill_libero_openvla"
    "maniskill_libero openvla-oft maniskill_libero_openvla_oft"
    "maniskill_libero gr00t maniskill_libero_gr00t"
    # mlp 模型需要特殊处理 - 使用基础环境安装
    "maniskill_libero mlp maniskill_libero_mlp"
    
    # robotwin 环境
    "robotwin openvla-oft robotwin_openvla_oft"
    
    # wan 环境 (world model)
    "wan openvla-oft wan_openvla_oft"
    
    # frankasim 环境
    # mlp 模型需要特殊处理
    "frankasim mlp frankasim_mlp"
)

# 安装选项
USE_MIRROR="--use-mirror"
NO_ROOT=""
INSTALL_RLINF=""

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-mirror)
                USE_MIRROR=""
                shift
                ;;
            --no-root)
                NO_ROOT="--no-root"
                shift
                ;;
            --install-rlinf)
                INSTALL_RLINF="--install-rlinf"
                shift
                ;;
            -h|--help)
                echo "Usage: bash install_all_envs.sh [options]"
                echo ""
                echo "Options:"
                echo "  --no-mirror      Do not use mirrors for downloads"
                echo "  --no-root        Skip system dependency installation (for non-root users)"
                echo "  --install-rlinf  Install RLinf itself into the virtual environment"
                echo "  -h, --help       Show this help message"
                echo ""
                echo "This script will install the following environment-model combinations:"
                printf "  - %-20s %-15s %s\n" "ENV_NAME" "MODEL_NAME" "VENV_NAME"
                printf "  - %-20s %-15s %s\n" "------" "----------" "---------"
                for combo in "${ENV_MODEL_COMBINATIONS[@]}"; do
                    read -r env model venv <<< "$combo"
                    printf "  - %-20s %-15s %s\n" "$env" "$model" "$venv"
                done
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# 安装单个环境
install_env() {
    local env_name="$1"
    local model_name="$2"
    local venv_name="$3"
    
    log_info "============================================"
    log_info "Installing: env=$env_name, model=$model_name, venv=$venv_name"
    log_info "============================================"
    
    cd "$REPO_ROOT"
    
    if [[ "$model_name" == "mlp" ]]; then
        # mlp 模型不在支持的模型列表中，只安装环境依赖
        # 使用 openvla 作为基础模型安装环境，因为 mlp 不需要特殊的模型依赖
        log_warn "mlp model is not in supported models. Installing environment only with openvla as base."
        
        if [[ "$env_name" == "maniskill_libero" ]]; then
            bash "$SCRIPT_DIR/install.sh" embodied \
                --model openvla \
                --env maniskill_libero \
                --venv "$venv_name" \
                $USE_MIRROR \
                $NO_ROOT \
                $INSTALL_RLINF
        elif [[ "$env_name" == "frankasim" ]]; then
            # frankasim + mlp: 安装 frankasim 环境
            bash "$SCRIPT_DIR/install.sh" embodied \
                --model openvla \
                --env frankasim \
                --venv "$venv_name" \
                $USE_MIRROR \
                $NO_ROOT \
                $INSTALL_RLINF
        else
            log_error "Unsupported env for mlp model: $env_name"
            return 1
        fi
    else
        # 标准安装
        bash "$SCRIPT_DIR/install.sh" embodied \
            --model "$model_name" \
            --env "$env_name" \
            --venv "$venv_name" \
            $USE_MIRROR \
            $NO_ROOT \
            $INSTALL_RLINF
    fi
    
    if [[ $? -eq 0 ]]; then
        log_info "Successfully installed: $venv_name"
    else
        log_error "Failed to install: $venv_name"
        return 1
    fi
}

main() {
    parse_args "$@"
    
    log_info "Starting batch installation of all environments..."
    log_info "Options: USE_MIRROR=$USE_MIRROR, NO_ROOT=$NO_ROOT, INSTALL_RLINF=$INSTALL_RLINF"
    
    local total=${#ENV_MODEL_COMBINATIONS[@]}
    local current=0
    local failed=0
    local failed_list=()
    
    for combo in "${ENV_MODEL_COMBINATIONS[@]}"; do
        current=$((current + 1))
        read -r env_name model_name venv_name <<< "$combo"
        
        log_info "Progress: [$current/$total] Processing $venv_name..."
        
        if install_env "$env_name" "$model_name" "$venv_name"; then
            log_info "[$current/$total] Completed: $venv_name"
        else
            failed=$((failed + 1))
            failed_list+=("$venv_name")
            log_error "[$current/$total] Failed: $venv_name"
        fi
    done
    
    log_info "============================================"
    log_info "Batch installation completed!"
    log_info "Total: $total, Success: $((total - failed)), Failed: $failed"
    
    if [[ $failed -gt 0 ]]; then
        log_error "Failed installations:"
        for name in "${failed_list[@]}"; do
            log_error "  - $name"
        done
        exit 1
    fi
    
    log_info "All environments installed successfully!"
}

main "$@"