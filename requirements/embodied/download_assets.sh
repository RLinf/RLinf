#!/bin/bash

set -euo pipefail

DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME}
SUPPORT_LIST=("maniskill" "openpi")
GITHUB_PREFIX=${GITHUB_PREFIX:-""}
USE_MIRRORS=${USE_MIRRORS:-0}
ASSETS=()

print_help() {
	cat <<EOF
Usage: bash download_assets.sh [--dir DIR] [--assets NAMES] [--use-mirror]

Options:
  --dir DIR         Root directory to store all downloaded assets.
					Default: \$DOWNLOAD_DIR or \$HOME.

  --assets NAMES    Comma-separated list of assets to download.

  --use-mirror      Use mirrors (HuggingFace / GitHub) for faster downloads.
					Mirrors are also picked up automatically when HF_ENDPOINT /
					GITHUB_PREFIX are already exported (e.g. by install.sh).

Examples:
  bash requirements/embodied/download_assets.sh --assets maniskill
  bash requirements/embodied/download_assets.sh --dir /opt/.assets --assets maniskill,openpi
  bash requirements/embodied/download_assets.sh --use-mirror --assets maniskill,openpi
EOF
}

# Configure HuggingFace / GitHub mirrors when requested. This is needed when the
# script is run on its own (e.g. a standalone Docker RUN) and does not inherit the
# mirror env vars that install.sh's setup_mirror exports. Values mirror install.sh.
setup_mirror() {
	if [ "$USE_MIRRORS" -eq 1 ]; then
		export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
		export GITHUB_PREFIX=${GITHUB_PREFIX:-https://ghfast.top/}
	fi
}

download_maniskill_assets() {
	local root_dir=$1

	# ManiSkill assets
	export MS_ASSET_DIR="${root_dir}/.maniskill"
	if [ -d "$MS_ASSET_DIR" ]; then
		echo "[download_assets] ManiSkill assets already exist at $MS_ASSET_DIR, skipping download."
	else
		mkdir -p "$MS_ASSET_DIR"
        # Ensure mani_skill is installed
        if ! python -c "import mani_skill" &> /dev/null; then
            echo "mani_skill is not installed. Please install it first." >&2
            exit 1
        fi
		if [ "$USE_MIRRORS" -eq 1 ]; then
			# mani_skill.utils.download_asset hardcodes huggingface.co / github.com
			# URLs in DATA_SOURCES and fetches them with urllib, which ignores
			# HF_ENDPOINT and git's insteadOf. Rewrite the in-memory URLs to the
			# mirrors before downloading instead of calling the module directly.
			for uid in bridge_v2_real2sim widowx250s; do
				python - "$uid" <<'PYEOF'
import os, sys
from mani_skill.utils.download_asset import main, parse_args
from mani_skill.utils.assets import data as ds

hf = os.environ.get("HF_ENDPOINT", "").rstrip("/")
gh = os.environ.get("GITHUB_PREFIX", "")
for src in ds.DATA_SOURCES.values():
    url = getattr(src, "url", None)
    if not url:
        continue
    if hf and url.startswith("https://huggingface.co"):
        src.url = hf + url[len("https://huggingface.co"):]
    elif gh and url.startswith("https://github.com"):
        src.url = gh + url
main(parse_args([sys.argv[1], "-y"]))
PYEOF
			done
		else
			python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
			python -m mani_skill.utils.download_asset widowx250s -y
		fi
	fi

	# SAPIEN assets (PhysX)
	export PHYSX_VERSION=105.1-physx-5.3.1.patch0
	export PHYSX_DIR="${root_dir}/.sapien/physx/${PHYSX_VERSION}"
	if [ -f "$PHYSX_DIR/linux-so.zip" ] || [ -d "$PHYSX_DIR" ] && compgen -G "$PHYSX_DIR/*" > /dev/null; then
		echo "[download_assets] SAPIEN PhysX assets already exist at $PHYSX_DIR, skipping download."
	else
		mkdir -p "$PHYSX_DIR"
		wget -O "$PHYSX_DIR/linux-so.zip" "${GITHUB_PREFIX}https://github.com/sapien-sim/physx-precompiled/releases/download/${PHYSX_VERSION}/linux-so.zip"
		unzip "$PHYSX_DIR/linux-so.zip" -d "$PHYSX_DIR" && rm "$PHYSX_DIR/linux-so.zip"
	fi
}

download_openpi_assets() {
	local root_dir=$1

	export TOKENIZER_DIR="${root_dir}/.cache/openpi/"

	if [ -f "$TOKENIZER_DIR/paligemma_tokenizer.model" ]; then
		echo "[download_assets] OpenPI tokenizer already exists at $TOKENIZER_DIR, skipping download."
	else
		mkdir -p "$TOKENIZER_DIR"
		hf download RLinf/openpi_tokenizer --local-dir "$TOKENIZER_DIR"
	fi
}

parse_args() {
	while [ "$#" -gt 0 ]; do
		case "$1" in
			-h|--help)
				print_help
				exit 0
				;;
			--dir)
				if [ -z "${2:-}" ]; then
					echo "--dir requires a directory argument." >&2
					exit 1
				fi
				DOWNLOAD_DIR="$2"
				shift 2
				;;
			--assets)
				if [ -z "${2:-}" ]; then
					echo "--assets requires a comma-separated list of asset names." >&2
					exit 1
				fi
				IFS=',' read -r -a ASSETS <<<"$2"
				shift 2
				;;
			--use-mirror)
				USE_MIRRORS=1
				shift
				;;
			--*)
				echo "Unknown option: $1" >&2
				echo "Use --help to see available options." >&2
				exit 1
				;;
			*)
				echo "Unexpected positional argument: $1" >&2
				echo "Use --help to see usage." >&2
				exit 1
				;;
		esac
	done
}

main() {
	parse_args "$@"

	if [ ${#ASSETS[@]} -eq 0 ]; then
		echo "No assets specified. See --help for usage." >&2
		exit 1
	fi

	setup_mirror

	mkdir -p "$DOWNLOAD_DIR"

	for asset in "${ASSETS[@]}"; do
		case "$asset" in
			maniskill)
				download_maniskill_assets "$DOWNLOAD_DIR"
				;;
			openpi)
				download_openpi_assets "$DOWNLOAD_DIR"
				;;
			*)
				echo "Unknown asset group: $asset. Supported: ${SUPPORT_LIST[*]}" >&2
				exit 1
				;;
		esac
	done
}

main "$@"
