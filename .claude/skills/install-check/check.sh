#!/usr/bin/env bash
# check.sh — lint requirements/install.sh (and its docker/Dockerfile coverage)
# against RLinf install conventions.
#
# This is the harness for the `install-check` skill. It statically scans
# install.sh and flags lines that likely violate the conventions in SKILL.md,
# then checks that every model/env in SUPPORTED_MODELS / SUPPORTED_ENVS has a
# matching install invocation in docker/Dockerfile. Findings are *candidates for
# review*, not hard errors — some are legitimate (see SKILL.md Gotchas).
#
# Usage:   bash .claude/skills/install-check/check.sh [install.sh] [Dockerfile]
# Default: requirements/install.sh  and  <repo>/docker/Dockerfile (derived).
# Exit:    0 if no candidates found, 1 otherwise.

set -uo pipefail

F="${1:-requirements/install.sh}"
if [ ! -f "$F" ]; then
    echo "check.sh: cannot find install script at '$F'." >&2
    echo "Pass the path explicitly: bash check.sh requirements/install.sh" >&2
    exit 2
fi
# Derive the Dockerfile from the install script's location unless given.
REPO_ROOT="$(cd "$(dirname "$F")/.." && pwd)"
D="${2:-$REPO_ROOT/docker/Dockerfile}"

HITS=0
section() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
hit()     { HITS=$((HITS + 1)); printf '  %s\n' "$1"; }

# ---------------------------------------------------------------------------
# Req 3: system packages must go through requirements/embodied/sys_deps.sh,
# never apt/dnf/yum/pacman inside install.sh (so all distros are covered).
# ---------------------------------------------------------------------------
section "Req 3 — system package managers inside install.sh (use sys_deps.sh instead)"
m=$(grep -nE '^[[:space:]]*(sudo[[:space:]]+)?(apt-get|apt|dnf|yum|pacman)[[:space:]]' "$F" \
    | grep -vE 'command -v|detect_pkg_manager' || true)
if [ -n "$m" ]; then
    while IFS= read -r line; do hit "$line"; done <<< "$m"
    echo "  FIX: move the package(s) into requirements/embodied/sys_deps.sh"
    echo "       (add to every install_deps_<pkgmgr> function), then drop the call here."
else
    echo "  ok — no direct package-manager calls"
fi

# ---------------------------------------------------------------------------
# Req 6: don't override core deps (ray) and minimise torch-family overrides.
# ---------------------------------------------------------------------------
section "Req 6 — core dependency overrides (ray / explicit torch pins)"
m=$(grep -nE 'uv pip install[^#]*\bray==' "$F" || true)
if [ -n "$m" ]; then
    while IFS= read -r line; do hit "$line  <-- ray is a core dep; do NOT pin it per-env"; done <<< "$m"
else
    echo "  ok — no ray override"
fi
m=$(grep -nE 'uv pip install[^#]*\b(torch|torchvision|torchaudio)==' "$F" || true)
if [ -n "$m" ]; then
    echo "  review — explicit torch-family pins (prefer --torch flag / apply_torch_override,"
    echo "           or 'pushd ~' before installing so pyproject overrides don't apply):"
    while IFS= read -r line; do hit "$line"; done <<< "$m"
fi

# ---------------------------------------------------------------------------
# Req 5: don't edit RLinf's own pyproject.toml from install.sh except via the
# apply_torch_override utility. Editing a *cloned dependency's* pyproject after
# cd-ing into it is fine.
# ---------------------------------------------------------------------------
section "Req 5 — pyproject.toml edits outside apply_torch_override"
lo=$(grep -n '^apply_torch_override()' "$F" | head -1 | cut -d: -f1)
hi=$(grep -n '^install_uv()' "$F" | head -1 | cut -d: -f1)
m=$(grep -nE 'pyproject\.toml' "$F" | grep -E 'sed -i|>> |>>"|cp |tee ' || true)
if [ -n "$m" ]; then
    found=0
    while IFS= read -r line; do
        ln=${line%%:*}
        if [ -n "$lo" ] && [ -n "$hi" ] && [ "$ln" -ge "$lo" ] && [ "$ln" -lt "$hi" ]; then
            continue   # inside the sanctioned utility
        fi
        hit "$line"; found=1
    done <<< "$m"
    if [ "$found" -eq 1 ]; then
        echo "  NOTE: editing a *cloned repo's* pyproject after 'cd <repo>' is OK (req 5)."
        echo "        Editing RLinf's own pyproject is NOT — route it through apply_torch_override."
    else
        echo "  ok — only apply_torch_override edits pyproject.toml"
    fi
else
    echo "  ok — no out-of-band pyproject.toml edits"
fi

# ---------------------------------------------------------------------------
# Req 4: every git dependency must be forked into github.com/RLinf OR pinned to
# a revision / branch / tag.
# ---------------------------------------------------------------------------
section "Req 4 — git deps that are neither RLinf-forked nor pinned"
awk '
  /github\.com\// {
    if ($0 ~ /^[[:space:]]*#/) next
    if ($0 ~ /git config/) next                     # mirror insteadOf setup, not a dep
    if ($0 ~ /clone_or_reuse_repo\(\)/) next        # the helper definition itself
    if ($0 ~ /releases\/download/) next             # prebuilt wheel URLs, not source
    org=$0; sub(/.*github\.com\//, "", org); sub(/[\/".].*/, "", org)
    pinned = ($0 ~ /@[0-9a-fA-F]{7,}/ || $0 ~ /@v?[0-9]/ || $0 ~ /@[A-Za-z][A-Za-z0-9._\/-]+/ || $0 ~ /-b[[:space:]]/)
    forked = (org == "RLinf")
    if (!forked && !pinned) printf "  %d: %s\n", NR, $0
  }
' "$F" | while IFS= read -r l; do hit "$l"; done
unpinned=$(awk '
  /github\.com\// {
    if ($0 ~ /^[[:space:]]*#/) next
    if ($0 ~ /git config/) next
    if ($0 ~ /clone_or_reuse_repo\(\)/) next
    if ($0 ~ /releases\/download/) next
    org=$0; sub(/.*github\.com\//,"",org); sub(/[\/".].*/,"",org)
    pinned=($0 ~ /@[0-9a-fA-F]{7,}/ || $0 ~ /@v?[0-9]/ || $0 ~ /@[A-Za-z][A-Za-z0-9._\/-]+/ || $0 ~ /-b[[:space:]]/)
    if (org!="RLinf" && !pinned) c++
  } END{print c+0}' "$F")
if [ "$unpinned" -gt 0 ]; then
    HITS=$((HITS + unpinned))
    echo "  FIX: pin with @<commit|tag|branch> or -b <branch>, OR fork into github.com/RLinf"
    echo "  NOTE: a clone that is checked out to a fixed hash later in its function"
    echo "        (e.g. NVIDIA/Isaac-GR00T -> git checkout 7d5a455) is effectively pinned."
else
    echo "  ok — all github source deps are RLinf-forked or pinned"
fi

# ---------------------------------------------------------------------------
# Req 1: reuse common utilities instead of re-implementing them. Raw 'git clone'
# outside the clone_or_reuse_repo helper usually means the env should call it.
# ---------------------------------------------------------------------------
section "Req 1 — raw 'git clone' that should use clone_or_reuse_repo"
m=$(grep -nE '(^|[^_])git clone' "$F" | grep -vE 'clone_or_reuse_repo|git clone "\$@"' || true)
if [ -n "$m" ]; then
    while IFS= read -r line; do hit "$line"; done <<< "$m"
    echo "  FIX: prefer clone_or_reuse_repo ENV_VAR DEFAULT_DIR URL [git args...]"
    echo "       unless this is a genuinely multi-repo build (e.g. franka catkin ws)."
else
    echo "  ok — all clones go through clone_or_reuse_repo"
fi

# ---------------------------------------------------------------------------
# Docker: every model/env in SUPPORTED_MODELS / SUPPORTED_ENVS must be wired
# into docker/Dockerfile (a `--model <m>` / `--env <e>` install invocation).
# A new model/env install MUST be accompanied by the matching Dockerfile change.
# ---------------------------------------------------------------------------
section "Docker — models/envs in install.sh missing from docker/Dockerfile"
if [ ! -f "$D" ]; then
    hit "Dockerfile not found at '$D' — pass it as the 2nd argument."
else
    # boundary = next char is not part of a name (handles '--env libero;' and
    # avoids '--env libero' matching inside '--env liberopro').
    bnd='([^a-zA-Z0-9_-]|$)'
    models=$(grep -E '^SUPPORTED_MODELS=' "$F" | sed -E 's/^[^(]*\(//; s/\).*//; s/"//g')
    envs=$(grep -E '^SUPPORTED_ENVS='   "$F" | sed -E 's/^[^(]*\(//; s/\).*//; s/"//g')
    miss_m=""; miss_e=""
    for x in $models; do grep -Eq -- "--model ${x}${bnd}" "$D" || miss_m="$miss_m $x"; done
    for x in $envs;   do grep -Eq -- "--env ${x}${bnd}"   "$D" || miss_e="$miss_e $x"; done
    if [ -n "$miss_m" ]; then
        for x in $miss_m; do hit "model '$x' has no '--model $x' install in docker/Dockerfile"; done
    fi
    if [ -n "$miss_e" ]; then
        for x in $miss_e; do hit "env   '$x' has no '--env $x' install in docker/Dockerfile"; done
    fi
    if [ -z "$miss_m$miss_e" ]; then
        echo "  ok — every model and env is referenced in docker/Dockerfile"
    else
        echo "  FIX: add a build stage + install invocation in docker/Dockerfile for each"
        echo "       (FROM embodied-common-image AS embodied-<env>-image; base-image line;"
        echo "       RUN bash requirements/install.sh ... --model <m> --env <e>). See"
        echo "       the add-install-docker-ci-e2e skill for the full stage pattern."
        echo "  NOTE: a few utility/real-robot targets intentionally have no image"
        echo "        (e.g. dummy, d4rl, gim_arm, dosw1) — keep those out consciously."
    fi
fi

printf '\n\033[1m== Summary ==\033[0m\n'
if [ "$HITS" -eq 0 ]; then
    echo "No convention candidates found in $F."
    exit 0
fi
echo "$HITS candidate(s) flagged — review each against SKILL.md."
echo "Heuristics over-report; a flagged line may be a legitimate exception."
exit 1
