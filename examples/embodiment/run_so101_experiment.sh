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
        intervention_file="${RLINF_KEYBOARD_INTERVENTION_CONTROL_FILE:-${control_file}.intervention}"
        key="${1:-}"
        case "${key}" in
            s|start) command_name="start"; target_file="${control_file}" ;;
            " "|space|intervene) command_name="space"; target_file="${intervention_file}" ;;
            c|success) command_name="success"; target_file="${control_file}" ;;
            a|abort|failure) command_name="failure"; target_file="${control_file}" ;;
            r|release) command_name="release"; target_file="${intervention_file}" ;;
            q|quit) command_name="quit"; target_file="${control_file}" ;;
            *) echo "usage: $0 control {s|space|c|a|r|q}" >&2; exit 2 ;;
        esac
        printf '%s\n' "${command_name}" >> "${target_file}"
        echo "sent ${key} -> ${command_name} (${target_file})"
        ;;
    eval)
        control_file="${RLINF_KEYBOARD_CONTROL_FILE:-/tmp/rlinf-so101.keys}"
        intervention_file="${RLINF_KEYBOARD_INTERVENTION_CONTROL_FILE:-${control_file}.intervention}"
        : > "${control_file}"
        : > "${intervention_file}"
        export RLINF_KEYBOARD_CONTROL_FILE="${control_file}"
        export RLINF_KEYBOARD_INTERVENTION_CONTROL_FILE="${intervention_file}"
        if [[ -z "${RLINF_COMM_NET_DEVICES:-}" ]] && command -v ip >/dev/null 2>&1; then
            # WSL interface names vary; pass the default route's interface to
            # Gloo before Ray starts so c10d does not guess from the hostname.
            RLINF_COMM_NET_DEVICES="$(ip route show default 2>/dev/null | awk 'NR == 1 {print $5}')"
            if [[ -n "${RLINF_COMM_NET_DEVICES}" ]]; then
                export RLINF_COMM_NET_DEVICES
            fi
        fi
        device_env="${SO101_DEVICE_ENV:-${HOME}/.config/so101/devices.env}"
        if [[ -r "${device_env}" ]]; then
            # shellcheck disable=SC1090
            source "${device_env}"
        fi
        server_address="${SO101_SERVER_ADDRESS:-127.0.0.1:50052}"
        policy_id="${SO101_POLICY_ID:-so101-pi05}"
        config_name="${SO101_EVAL_CONFIG:-realworld_so101_eval_openpi_grpc}"
        follower_port="${SO101_FOLLOWER_PORT:-/dev/ttyACM1}"
        follower_id="${SO101_FOLLOWER_ID:-FOLLOWER_CALIB}"
        leader_port="${SO101_LEADER_PORT:-/dev/ttyACM0}"
        leader_id="${SO101_LEADER_ID:-LEADER_CALIB}"
        camera="${SO101_CAMERA:-/dev/video0}"
        leader_overrides=()
        if [[ "${config_name}" == "realworld_so101_grpc_dagger_offline" ]]; then
            leader_overrides=(
                "env.eval.teleop.0.so101_leader.port=${leader_port}"
                "env.eval.teleop.0.so101_leader.calibration_id=${leader_id}"
            )
        fi
        exec "${python_bin}" "${repo_dir}/evaluations/eval_embodied_agent.py" \
            --config-path "${repo_dir}/examples/embodiment/config" \
            --config-name "${config_name}" \
            "rollout.grpc.server_address=${server_address}" \
            "rollout.grpc.policy_id=${policy_id}" \
            "cluster.node_groups.1.hardware.configs.0.serial_port=${follower_port}" \
            "cluster.node_groups.1.hardware.configs.0.calibration_id=${follower_id}" \
            "cluster.node_groups.1.hardware.configs.0.camera_serials=[${camera}]" \
            "${leader_overrides[@]}" \
            "$@"
        ;;
    *)
        echo "usage: $0 {mock|grpc-check|control|eval} [options]" >&2
        exit 2
        ;;
esac
