#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stages an Ascend CI job's working tree, virtualenv, Python runtime, and
# editable dependency checkouts on the local root disk (/var/tmp) while keeping
# large wheel caches and model assets on /workspace/dataset (NFS).
#
# Root disk is bounded (~100 GB), so directory names are keyed by RUNNER_NAME
# (at most one active directory per runner), stale state from prior cancelled
# runs is wiped before staging, and staging falls back to GITHUB_WORKSPACE if
# /var/tmp has less than 15 GiB free.

ascend_stage_dep_repo() {
    local var_name="$1"
    local src_dir="${!var_name:-}"
    if [ "${ASCEND_LOCAL_STAGE:-0}" != "1" ] || [ -z "$src_dir" ] || [ ! -d "$src_dir" ]; then
        return 0
    fi
    local dst_dir="${ASCEND_LOCAL_ROOT}/deps/${var_name,,}"
    mkdir -p "$dst_dir"
    tar -C "$src_dir" --exclude=.git --exclude=build -cf - . | tar -C "$dst_dir" -xf -
    export "${var_name}=${dst_dir}"
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "${var_name}=${dst_dir}" >> "$GITHUB_ENV"
    fi
}

case "${1:-}" in
    setup)
        RUNNER_SLOT="${RUNNER_NAME:-ascend}"
        ASCEND_LOCAL_ROOT="/var/tmp/rlinf-ci/${RUNNER_SLOT}"
        # Keep socket base path short (<20 bytes) for Unix domain socket limits.
        RAY_TMPDIR="/tmp/ray-${RUNNER_SLOT##*-}"

        ray stop --force >/dev/null 2>&1 || true
        rm -rf "$ASCEND_LOCAL_ROOT" "$RAY_TMPDIR" \
               "${GITHUB_WORKSPACE:?}/.venv" "${GITHUB_WORKSPACE}/../results"
        mkdir -p "$RAY_TMPDIR"

        unset UV_DEFAULT_INDEX
        export UV_PATH=/workspace/dataset/.uv
        export UV_CACHE_DIR=/workspace/dataset/.uv_cache

        # Require >= 15 GiB free on /var/tmp before staging onto root disk.
        avail_kb=$(df -Pk /var/tmp | awk 'NR==2 {print $4}')
        if [ "${avail_kb:-0}" -ge 15728640 ]; then
            ASCEND_LOCAL_STAGE=1
            mkdir -p /var/tmp/uv_python "$ASCEND_LOCAL_ROOT"
            py_arch="cpython-3.11.14-linux-aarch64-gnu"
            if [ -d "/workspace/dataset/.uv_python/${py_arch}" ] && [ ! -d "/var/tmp/uv_python/${py_arch}" ]; then
                tmp_py="/var/tmp/uv_python/.tmp.${py_arch}.$$"
                rm -rf "$tmp_py"
                cp -a "/workspace/dataset/.uv_python/${py_arch}" "$tmp_py"
                mv -T "$tmp_py" "/var/tmp/uv_python/${py_arch}" 2>/dev/null || rm -rf "$tmp_py"
            fi
            if [ -d "/var/tmp/uv_python/${py_arch}" ]; then
                ln -sfn "/var/tmp/uv_python/${py_arch}" /var/tmp/uv_python/cpython-3.11-linux-aarch64-gnu
                export UV_PYTHON_INSTALL_DIR=/var/tmp/uv_python
            else
                export UV_PYTHON_INSTALL_DIR=/workspace/dataset/.uv_python
            fi

            if [ "$(stat -c %d "$GITHUB_WORKSPACE")" = "$(stat -c %d /var/tmp)" ]; then
                ASCEND_WORK_DIR="$GITHUB_WORKSPACE"
            else
                ASCEND_WORK_DIR="${ASCEND_LOCAL_ROOT}/repo"
                mkdir -p "$ASCEND_WORK_DIR"
                git -C "$GITHUB_WORKSPACE" archive HEAD | tar -C "$ASCEND_WORK_DIR" -xf -
            fi
            echo "[ascend-ci] Staged workspace on local disk: ${ASCEND_WORK_DIR} (free=$((avail_kb / 1048576)) GiB)"
        else
            ASCEND_LOCAL_STAGE=0
            ASCEND_WORK_DIR="$GITHUB_WORKSPACE"
            export UV_LINK_MODE=symlink
            export UV_PYTHON_INSTALL_DIR=/workspace/dataset/.uv_python
            echo "[ascend-ci] Low disk on /var/tmp ($((avail_kb / 1048576)) GiB free); using symlinked venv at ${ASCEND_WORK_DIR}"
        fi

        export ASCEND_LOCAL_STAGE ASCEND_LOCAL_ROOT ASCEND_WORK_DIR RAY_TMPDIR
        for dep_var in DECORD_PATH GR00T_PATH STARVLA_PATH DREAMZERO_PATH; do
            ascend_stage_dep_repo "$dep_var"
        done

        if [ -n "${GITHUB_ENV:-}" ]; then
            {
                echo "ASCEND_WORK_DIR=${ASCEND_WORK_DIR}"
                echo "ASCEND_LOCAL_ROOT=${ASCEND_LOCAL_ROOT}"
                echo "RAY_TMPDIR=${RAY_TMPDIR}"
                echo "UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}"
            } >> "$GITHUB_ENV"
        fi
        cd "$ASCEND_WORK_DIR"
        ;;

    cleanup)
        RUNNER_SLOT="${RUNNER_NAME:-ascend}"
        ASCEND_LOCAL_ROOT="/var/tmp/rlinf-ci/${RUNNER_SLOT}"
        RAY_TMPDIR="/tmp/ray-${RUNNER_SLOT##*-}"
        ray stop --force >/dev/null 2>&1 || true
        rm -rf "$ASCEND_LOCAL_ROOT" "$RAY_TMPDIR" \
               "${GITHUB_WORKSPACE:?}/.venv" "${GITHUB_WORKSPACE}/../results"
        find /workspace/results -maxdepth 3 -type d \( -name checkpoints -o -name converted_ckpts \) \
             -prune -exec rm -rf {} + 2>/dev/null || true
        ;;

    *)
        echo "Usage: source $0 setup | bash $0 cleanup" >&2
        exit 1
        ;;
esac
