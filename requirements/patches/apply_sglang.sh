#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_VERSION=$(python -c 'from importlib.metadata import version; print(version("sglang"))')
case "$SGLANG_VERSION" in
    0.5.12.post1) PATCH_FILE="$SCRIPT_DIR/sglang_${SGLANG_VERSION}.patch" ;;
    *) exit 0 ;;
esac
SGLANG_PACKAGE=$(python -c 'from importlib.util import find_spec; print(next(iter(find_spec("sglang").submodule_search_locations)))')

# Run outside the checkout so Git does not filter paths under a local .venv.
apply=(git -C / apply --unsafe-paths --directory="$SGLANG_PACKAGE")
# An existing installation may already have the Student top-k changes.
exclude=()
while IFS=$'\t' read -r _ _ path; do
    if "${apply[@]}" --include="$SGLANG_PACKAGE/$path" --check --reverse "$PATCH_FILE" >/dev/null 2>&1; then
        exclude+=("--exclude=$SGLANG_PACKAGE/$path")
    fi
done < <(git -C / apply --numstat "$PATCH_FILE")
"${apply[@]}" "${exclude[@]}" "$PATCH_FILE"
