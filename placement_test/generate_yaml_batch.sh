#!/usr/bin/env bash
set -euo pipefail

# Generate a batch of placement_test YAMLs from a base YAML.
#
# Example:
#   bash placement_test/generate_yaml_batch.sh \
#     --base-yaml examples/embodiment/config/aaa_maniskill_ppo_openvla.yaml \
#     --envnum-start 96 --envnum-end 512 --envnum-step 16 \
#     --include-env01-rollout27-when-divisible-by-3
#
# Prefix:
#   By default, filenames use --prefix auto which infers something like "maniskill_openvla"
#   from the base YAML's Hydra defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${REPO_ROOT}/placement_test/generate_yaml_batch.py" "$@"

