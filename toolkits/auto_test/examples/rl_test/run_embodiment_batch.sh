#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_embodiment.sh"

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "run script not found: $RUN_SCRIPT" >&2
    exit 1
fi

CONFIG_SPECS=()

if [ "$#" -gt 0 ]; then
    CONFIG_SPECS=("$@")
else
    RAW_CONFIG_SPECS="${RLINF_CONFIG_SPECS:-}"
    if [ -n "$RAW_CONFIG_SPECS" ]; then
        while IFS= read -r config_spec; do
            config_spec="$(trim_whitespace "$config_spec")"
            if [ -n "$config_spec" ]; then
                CONFIG_SPECS+=("$config_spec")
            fi
        done <<< "$RAW_CONFIG_SPECS"
    fi
fi

if [ "${#CONFIG_SPECS[@]}" -eq 0 ]; then
    RAW_CONFIG_NAMES="${RLINF_CONFIG_NAMES:-}"
    if [ -n "$RAW_CONFIG_NAMES" ]; then
        NORMALIZED_CONFIG_NAMES="$(printf '%s' "$RAW_CONFIG_NAMES" | tr ', ' '\n\n')"
        while IFS= read -r config_name; do
            config_name="$(trim_whitespace "$config_name")"
            if [ -n "$config_name" ]; then
                CONFIG_SPECS+=("$config_name")
            fi
        done <<< "$NORMALIZED_CONFIG_NAMES"
    fi
fi

if [ "${#CONFIG_SPECS[@]}" -eq 0 ] && [ -n "${RLINF_CONFIG_NAME:-}" ]; then
    CONFIG_SPECS=("$RLINF_CONFIG_NAME")
fi

if [ "${#CONFIG_SPECS[@]}" -eq 0 ]; then
    echo "no config specs provided. Set RLINF_CONFIG_SPECS / RLINF_CONFIG_NAMES / RLINF_CONFIG_NAME or pass args." >&2
    exit 1
fi

echo "Batch config specs:"
for config_spec in "${CONFIG_SPECS[@]}"; do
    echo "  - ${config_spec}"
done

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
FAILURES=()
TOTAL_CONFIGS="${#CONFIG_SPECS[@]}"
CURRENT_INDEX=0

for config_spec in "${CONFIG_SPECS[@]}"; do
    CURRENT_INDEX=$((CURRENT_INDEX + 1))
    config_name="$config_spec"
    env_name="${RLINF_DEFAULT_ENV:-}"

    if [[ "$config_spec" == *,* ]]; then
        config_name="$(trim_whitespace "${config_spec%%,*}")"
        env_name="$(trim_whitespace "${config_spec#*,}")"
    fi

    if [ -z "$config_name" ]; then
        echo "[${CURRENT_INDEX}/${TOTAL_CONFIGS}] Invalid config spec: ${config_spec}" >&2
        exit 1
    fi

    echo
    echo "[${CURRENT_INDEX}/${TOTAL_CONFIGS}] Starting config: ${config_name}"

    if [ -n "$env_name" ]; then
        echo "[${CURRENT_INDEX}/${TOTAL_CONFIGS}] Switching env: ${env_name}"
        source switch_env "$env_name"
        export RLINF_ACTIVE_ENV="$env_name"
        echo "[${CURRENT_INDEX}/${TOTAL_CONFIGS}] Python after switch: $(which python)"
    fi

    if bash "$RUN_SCRIPT" "$config_name"; then
        echo "[${CURRENT_INDEX}/${TOTAL_CONFIGS}] Finished config: ${config_name}"
        continue
    fi

    exit_code=$?
    echo "[${CURRENT_INDEX}/${TOTAL_CONFIGS}] Failed config: ${config_name} (exit=${exit_code})" >&2

    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
        FAILURES+=("${config_name}:${exit_code}")
        continue
    fi

    exit "$exit_code"
done

if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "Batch finished with failures: ${FAILURES[*]}" >&2
    exit 1
fi

echo "Batch finished successfully."
