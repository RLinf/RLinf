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

"""Unified OpenPI Pi0 and Pi0.5 checkpoint convertor.

Dispatches to four modes over a shared core:

    jax_to_openpi_rlinf              JAX Pi0/Pi05 checkpoint -> OpenPI_RLinf layout
    openpi_pytorch_to_openpi_rlinf   OpenPI PyTorch layout -> OpenPI_RLinf layout
    sft_to_openpi_rlinf              RLinf SFT full_weights.pt -> OpenPI_RLinf
                                   layout selected by ``--config-name`` and ``--dtype``
    openpi_rlinf_to_openpi_pytorch   OpenPI_RLinf layout -> OpenPI PyTorch layout

Usage::

    python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi_rlinf \\
        --input-model       /path/to/jax_checkpoint \\
        --input-norm-stats  /path/to/norm_stats.json \\
        --output-model      /path/to/out_openpi_rlinf \\
        --output-norm-stats /path/to/out_openpi_rlinf/physical-intelligence/behavior/norm_stats.json

Run ``--mode <mode> --help`` for the per-mode arguments.
"""

from __future__ import annotations

import argparse

from rlinf.utils.ckpt_convertor.openpi import (
    jax_to_openpi_rlinf,
    openpi_pytorch_to_openpi_rlinf,
    openpi_rlinf_to_openpi_pytorch,
    pt_to_safetensors,
)

# Public mode names describe the layouts explicitly. Internally, the conversion
# kernels retain the original terminology: old = OpenPI PyTorch, new =
# OpenPI_RLinf.
_MODES = {
    "jax_to_openpi_rlinf": jax_to_openpi_rlinf,
    "openpi_pytorch_to_openpi_rlinf": openpi_pytorch_to_openpi_rlinf,
    "sft_to_openpi_rlinf": pt_to_safetensors,
    "openpi_rlinf_to_openpi_pytorch": openpi_rlinf_to_openpi_pytorch,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with one subparser per mode."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name, module in _MODES.items():
        sub = subparsers.add_parser(name, help=module.__doc__.splitlines()[0])
        module.add_arguments(sub)
        sub.set_defaults(_run=module.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``--mode`` and arguments, then run the selected convertor."""
    # Accept the ``--mode <name>`` spelling the package advertises by normalizing
    # it to the subcommand form argparse subparsers expect.
    raw = list(argv) if argv is not None else None
    if raw is None:
        import sys

        raw = sys.argv[1:]
    normalized: list[str] = []
    i = 0
    while i < len(raw):
        token = raw[i]
        if token == "--mode" and i + 1 < len(raw):
            normalized.append(raw[i + 1])
            i += 2
            continue
        if token.startswith("--mode="):
            normalized.append(token.split("=", 1)[1])
            i += 1
            continue
        normalized.append(token)
        i += 1

    parser = build_parser()
    args = parser.parse_args(normalized)
    args._run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
