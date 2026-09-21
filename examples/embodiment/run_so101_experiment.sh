#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_dir}/.venv/bin/python}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

command="${1:-mock}"
shift || true
case "${command}" in
    mock)
        exec "${python_bin}" "${repo_dir}/toolkits/realworld_check/run_so101_mock.py" "$@"
        ;;
    grpc-check)
        exec "${python_bin}" "${repo_dir}/toolkits/realworld_check/check_so101_grpc.py" "$@"
        ;;
    control)
        control_file="${RLINF_KEYBOARD_CONTROL_FILE:-/tmp/rlinf-so101.keys}"
        key="${1:-}"
        case "${key}" in
            s|start) command_name="start" ;;
            " "|space|intervene) command_name="space" ;;
            c|success) command_name="success" ;;
            a|abort|failure) command_name="failure" ;;
            r|release) command_name="release" ;;
            q|quit) command_name="quit" ;;
            *) echo "usage: $0 control {s|space|c|a|r|q}" >&2; exit 2 ;;
        esac
        printf '%s\n' "${command_name}" >> "${control_file}"
        echo "sent ${key} -> ${command_name} (${control_file})"
        ;;
    eval)
        control_file="${RLINF_KEYBOARD_CONTROL_FILE:-/tmp/rlinf-so101.keys}"
        : > "${control_file}"
        export RLINF_KEYBOARD_CONTROL_FILE="${control_file}"
        server_address="${SO101_SERVER_ADDRESS:-127.0.0.1:50052}"
        policy_id="${SO101_POLICY_ID:-so101-pi05-openpi-rlinf-step10000}"
        follower_port="${SO101_FOLLOWER_PORT:-/dev/ttyACM1}"
        follower_id="${SO101_FOLLOWER_ID:-my_awesome_follower_arm}"
        camera="${SO101_CAMERA:-/dev/video0}"
        exec "${python_bin}" "${repo_dir}/evaluations/eval_embodied_agent.py" \
            --config-path "${repo_dir}/examples/embodiment/config" \
            --config-name realworld_so101_eval_openpi_grpc \
            "rollout.grpc.server_address=${server_address}" \
            "rollout.grpc.policy_id=${policy_id}" \
            "cluster.node_groups.1.hardware.configs.0.serial_port=${follower_port}" \
            "cluster.node_groups.1.hardware.configs.0.calibration_id=${follower_id}" \
            "cluster.node_groups.1.hardware.configs.0.camera_serials=[${camera}]" \
            "$@"
        ;;
    *)
        echo "usage: $0 {mock|grpc-check|control|eval} [options]" >&2
        exit 2
        ;;
esac
