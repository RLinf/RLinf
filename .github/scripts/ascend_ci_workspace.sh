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

# Stages an Ascend CI job's working tree, virtualenv, local uv wheel cache,
# Python runtime, and editable dependency checkouts on the local root disk
# (/var/tmp) while keeping large model assets on /workspace/dataset (NFS).
#
# Keeping UV_CACHE_DIR (/var/tmp/uv_cache) and .venv on the same /var/tmp
# filesystem lets uv use instant hardlinks (zero extra data blocks per .venv
# and 100% local-disk Python imports). Root disk is ~100 GB, so:
#   - Runner scratch paths are keyed by RUNNER_NAME (at most one per runner)
#     and wiped both before setup and in `if: always()` cleanup.
#   - If free space on /var/tmp drops below 18 GiB, `uv cache prune --ci`
#     reclaims unzipped archive trees first; if still below 10 GiB, the job
#     falls back to /workspace/dataset/.uv_cache with UV_LINK_MODE=symlink.

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

        avail_kb=$(df -Pk /var/tmp | awk 'NR==2 {print $4}')
        if [ "${avail_kb:-0}" -lt 18874368 ] && [ -d /var/tmp/uv_cache ]; then
            echo "[ascend-ci] /var/tmp below 18 GiB free; pruning unzipped entries in /var/tmp/uv_cache"
            uv cache prune --ci --cache-dir /var/tmp/uv_cache >/dev/null 2>&1 || true
            avail_kb=$(df -Pk /var/tmp | awk 'NR==2 {print $4}')
        fi

        if [ "${avail_kb:-0}" -ge 10485760 ]; then
            ASCEND_LOCAL_STAGE=1
            mkdir -p /var/tmp/uv_cache /var/tmp/uv_python "$ASCEND_LOCAL_ROOT"
            export UV_CACHE_DIR=/var/tmp/uv_cache
            export UV_LINK_MODE=hardlink

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
            echo "[ascend-ci] Using local /var/tmp workspace + hardlinked uv_cache: ${ASCEND_WORK_DIR} (free=$((avail_kb / 1048576)) GiB)"
        else
            ASCEND_LOCAL_STAGE=0
            ASCEND_WORK_DIR="$GITHUB_WORKSPACE"
            export UV_CACHE_DIR=/workspace/dataset/.uv_cache
            export UV_LINK_MODE=symlink
            export UV_PYTHON_INSTALL_DIR=/workspace/dataset/.uv_python
            echo "[ascend-ci] Low disk on /var/tmp ($((avail_kb / 1048576)) GiB free); falling back to NFS uv_cache with symlinked venv"
        fi

        export ASCEND_LOCAL_STAGE ASCEND_LOCAL_ROOT ASCEND_WORK_DIR RAY_TMPDIR
        git_cfg_idx="${GIT_CONFIG_COUNT:-0}"
        export "GIT_CONFIG_KEY_${git_cfg_idx}=safe.directory"
        export "GIT_CONFIG_VALUE_${git_cfg_idx}=*"
        export GIT_CONFIG_COUNT=$((git_cfg_idx + 1))

        for dep_var in DECORD_PATH GR00T_PATH STARVLA_PATH DREAMZERO_PATH; do
            ascend_stage_dep_repo "$dep_var"
        done

        if [ -n "${GITHUB_ENV:-}" ]; then
            {
                echo "ASCEND_WORK_DIR=${ASCEND_WORK_DIR}"
                echo "ASCEND_LOCAL_ROOT=${ASCEND_LOCAL_ROOT}"
                echo "RAY_TMPDIR=${RAY_TMPDIR}"
                echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
                echo "UV_LINK_MODE=${UV_LINK_MODE}"
                echo "UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}"
                echo "GIT_CONFIG_KEY_${git_cfg_idx}=safe.directory"
                echo "GIT_CONFIG_VALUE_${git_cfg_idx}=*"
                echo "GIT_CONFIG_COUNT=${GIT_CONFIG_COUNT}"
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
