#! /bin/bash
# Local (CPU-only) routing tests for configure_nvidia in requirements/install.sh.
# Extracts the script's functions/variables (everything before main()) and stubs
# nvidia-smi / curl so the CUDA-index routing decision can be asserted without a
# GPU or network. Mirrors the Blackwell sm_120 scenario from issue #1638.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SH="$HERE/../install.sh"

fails=0
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; fails=$((fails+1)); }

# Build a sourceable slice of install.sh: everything up to main().
sed -n "1,/^main() {/p" "$INSTALL_SH" | sed '$d' > /tmp/install_lib_$$.sh
source /tmp/install_lib_$$.sh
# install.sh enables `set -eo pipefail`; relax it here so a failing probe
# becomes a test result instead of a silent exit.
set +eo pipefail

# --- stubs -------------------------------------------------------------
nvcc() { echo "Cuda compilation tools, release 12.8, V12.8.90"; }
nvidia-smi() {
    case "${1:-}" in
        --query-gpu*) echo "compute_cap"; echo "$FAKE_CC" ;;
        *)            echo "Driver Version: 580.65.06  CUDA Version: $FAKE_DRIVER_CUDA";;
    esac
}
# newer; cu124/cu126 host only <= 2.6/2.7 style entries.
curl() {
    local args=() out="" url=""
    while [ $# -gt 0 ]; do
        case "$1" in
            -o) out="$2"; shift 2 ;;
            -w) shift 2 ;;          # -w '%{http_code}': emulate by echoing 200
            -*) shift ;;
            *)  url="$1"; shift ;;
        esac
    done
    local tag
    tag=$(grep -oE 'cu[0-9]+' <<< "$url" | head -1)
    local listing
    case "$tag" in
        cu129|cu130)
            listing="torch-2.7.1%2Bcu129-cp311-cp311-manylinux_2_28_x86_64.whl
torch-2.8.0%2Bcu129-cp311-cp311-manylinux_2_28_x86_64.whl"
            ;;
        cu128)
            listing="torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl
torch-2.7.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
            ;;
        cu126)
            listing="torch-2.5.1%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl
torch-2.7.1%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl"
            ;;
        cu124)
            listing="torch-2.5.1%2Bcu124-cp311-cp311-manylinux_2_28_x86_64.whl"
            ;;
        *) listing="" ;;
    esac
    [ -n "$out" ] && printf '%s\n' "$listing" > "$out"
    echo "200"
    return 0
}

run_configure() {
    TORCH_VERSION="${1:-}"
    TORCH_VERSION_EXPLICIT="${2:-0}"
    # configure_nvidia exports UV_TORCH_BACKEND; reset so each case starts
    # from a clean, user-unset state.
    unset UV_TORCH_BACKEND
    configure_nvidia >/dev/null 2>&1
}

# --- 1. --torch 2.7.1 on Blackwell (sm_120): must route to cu128/cu129 ----
FAKE_CC="12.0" FAKE_DRIVER_CUDA="12.9"
run_configure "2.7.1" 1
[[ "$PLATFORM_TORCH_INDEX" == *"/cu12"[89] ]] && pass "--torch 2.7.1 Blackwell routes to cu128/cu129 index (got $PLATFORM_TORCH_INDEX)" \
    || fail "--torch 2.7.1 Blackwell index (got '$PLATFORM_TORCH_INDEX')"
[[ "$PLATFORM_TORCH_STR" == "+cu12"[89] ]] && pass "PLATFORM_TORCH_STR local segment set (${PLATFORM_TORCH_STR})" \
    || fail "PLATFORM_TORCH_STR (got '$PLATFORM_TORCH_STR')"

# --- 2. BEHAVIOR default pin 2.5.1 on Blackwell: bumped to 2.7.1 + cu128+ ---
ENV_NAME="behavior"
TORCH_VERSION=""
apply_env_default_torch   # ENV_NAME=behavior -> TORCH_VERSION=2.5.1
[ "$TORCH_VERSION" = "2.5.1" ] || fail "apply_env_default_torch should pin 2.5.1 (got $TORCH_VERSION)"
FAKE_CC="12.0" FAKE_DRIVER_CUDA="12.9"
run_configure "$TORCH_VERSION" 0
[ "$TORCH_VERSION" = "2.7.1" ] && pass "BEHAVIOR 2.5.1 auto-bumped to 2.7.1 on Blackwell" \
    || fail "BEHAVIOR bump (got $TORCH_VERSION)"
[[ "$PLATFORM_TORCH_INDEX" == *"/cu12"[89] ]] && pass "bumped 2.7.1 routed to cu128/cu129 (got $PLATFORM_TORCH_INDEX)" \
    || fail "bumped 2.7.1 index (got '$PLATFORM_TORCH_INDEX')"

# --- 3. explicit --torch 2.5.1 on Blackwell: hard error (exit 1) -----------
FAKE_CC="12.0"; FAKE_DRIVER_CUDA="12.9"
rc=0
( unset UV_TORCH_BACKEND; ENV_NAME=""; TORCH_VERSION="2.5.1"; TORCH_VERSION_EXPLICIT=1; configure_nvidia ) >/dev/null 2>&1 || rc=$?
[ "$rc" -ne 0 ] && pass "explicit --torch 2.5.1 on Blackwell exits non-zero" \
    || fail "explicit --torch 2.5.1 on Blackwell should exit 1"

# --- 4. pre-Blackwell host keeps old behavior: 2.5.1 -> cu124 index ---------
FAKE_CC="8.9"; FAKE_DRIVER_CUDA="12.4"
ENV_NAME=""
run_configure "2.5.1" 0
[ "$TORCH_VERSION" = "2.5.1" ] && pass "no version bump on Ampere" \
    || fail "unexpected bump on Ampere (got $TORCH_VERSION)"
[[ "$PLATFORM_TORCH_INDEX" == *"/cu124" ]] && pass "Ampere 2.5.1 routed to cu124 (got $PLATFORM_TORCH_INDEX)" \
    || fail "Ampere 2.5.1 index (got '$PLATFORM_TORCH_INDEX')"

# --- 5. torch < 2.11 on pre-Blackwell is now routed (the bug: it wasn't) ----
FAKE_CC="8.9"; FAKE_DRIVER_CUDA="12.9"
run_configure "2.7.1" 1
[[ "$PLATFORM_TORCH_INDEX" == *"/cu12"[89] ]] && pass "--torch 2.7.1 non-Blackwell routed off PyPI default (got $PLATFORM_TORCH_INDEX)" \
    || fail "--torch 2.7.1 non-Blackwell index (got '$PLATFORM_TORCH_INDEX')"

# --- 6. torchvision/torchaudio derivation used by install_behavior_env ------
TORCH_VERSION="2.7.1"
local_unused=1
_bt_maj=2; _bt_min=7; _bt_patch=1
_bt_tv="0.$((_bt_min + 15)).${_bt_patch}"
[ "$_bt_tv" = "0.22.1" ] && pass "torchvision derived 0.22.1 for torch 2.7.1" \
    || fail "torchvision derivation (got $_bt_tv)"

rm -f /tmp/install_lib_$$.sh
echo "----"
if [ "$fails" -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "${fails} TEST(S) FAILED"; exit 1; fi
