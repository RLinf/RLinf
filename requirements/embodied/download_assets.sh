#!/bin/bash

set -euo pipefail

DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME}
SUPPORT_LIST=("maniskill" "openpi")
GITHUB_PREFIX=${GITHUB_PREFIX:-""}
ASSETS=()

print_help() {
	cat <<EOF
Usage: bash download_assets.sh [--dir DIR] [--assets NAMES]

Options:
  --dir DIR         Root directory to store all downloaded assets.
					Default: \$DOWNLOAD_DIR or \$HOME.

  --assets NAMES    Comma-separated list of assets to download.

Examples:
  bash requirements/embodied/download_assets.sh --assets maniskill
  bash requirements/embodied/download_assets.sh --dir /opt/.assets --assets maniskill,openpi
EOF
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
		python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
		python -m mani_skill.utils.download_asset widowx250s -y
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
