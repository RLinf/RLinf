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

"""Command-line argument parsing and subcommand dispatch for ``rlinf``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

from .catalog import (
    RecipeFamily,
    RecipeMode,
    RecipeSpec,
    discover_recipes,
    filter_recipes,
    find_repo_root,
    resolve_recipe,
)
from .doctor import render_doctor
from .launcher import build_launch_plan, execute_launch_plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlinf",
        description=(
            "Thin unified CLI for discovering, checking, and launching RLinf "
            "training, evaluation, and data collection runs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # rlinf list
    list_parser = subparsers.add_parser(
        "list",
        help="List curated or filtered RLinf recipes.",
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Show every discovered recipe instead of the curated default list.",
    )
    list_parser.add_argument(
        "--family",
        choices=[f.value for f in RecipeFamily],
        help="Filter recipes by task family.",
    )
    list_parser.add_argument(
        "--mode",
        choices=[m.value for m in RecipeMode],
        help="Filter recipes by execution mode.",
    )
    list_parser.add_argument(
        "--env",
        help="Filter recipes whose name or subgroup matches an environment token.",
    )
    list_parser.add_argument(
        "--model",
        help="Filter recipes whose name matches a model token.",
    )
    list_parser.add_argument(
        "--search",
        help="Filter recipes by substring across family, subgroup, and name.",
    )

    # rlinf train / eval / collect
    def _add_launch_subparser(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(
            name,
            help=help_text,
            description=(
                f"{help_text} Any extra arguments (e.g. runner.max_epochs=10 "
                "or +key=value) are forwarded verbatim to Hydra."
            ),
        )
        sub.add_argument(
            "target",
            help="Recipe name (from 'rlinf list') or path to a local YAML file.",
        )
        sub.add_argument(
            "--family",
            choices=[f.value for f in RecipeFamily],
            help="Disambiguate a recipe name or set the family for an external YAML.",
        )
        sub.add_argument(
            "--subgroup",
            help="Optional subgroup (e.g. searchr1, wideseek_r1, rstar2, coding_online_rl, calc_x, libero).",
        )
        sub.add_argument(
            "--log-dir",
            type=Path,
            help="Explicit log directory passed as runner.logger.log_path.",
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the resolved command and environment without launching.",
        )
        sub.set_defaults(benchmark=None, use_async=False)
        return sub

    train_parser = _add_launch_subparser(
        "train",
        "Launch an RLinf training recipe or custom YAML.",
    )
    train_parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Force the async embodied training script (train_async.py).",
    )

    eval_parser = _add_launch_subparser(
        "eval",
        "Launch an RLinf evaluation recipe or custom YAML.",
    )
    eval_parser.add_argument(
        "--benchmark",
        help="Evaluation benchmark subgroup (e.g. libero, robotwin, realworld).",
    )

    _add_launch_subparser(
        "collect",
        "Launch an RLinf data collection recipe or custom YAML.",
    )

    # rlinf doctor
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect Python, accelerators, rendering libraries, Ray, and Hugging Face.",
    )
    doctor_parser.add_argument(
        "--network",
        action="store_true",
        help="Probe HF_ENDPOINT reachability with a 5-second HEAD request.",
    )

    return parser


def _format_recipe_table(
    recipes: list[RecipeSpec],
    *,
    featured: bool,
    total_count: int,
) -> str:
    if not recipes:
        return (
            "No matching recipes found. Run 'rlinf list --all' to view all recipes.\n"
        )

    family_w = max(len("FAMILY"), *(len(r.family.value) for r in recipes))
    mode_w = max(len("MODE"), *(len(r.mode.value) for r in recipes))
    name_w = max(len("RECIPE"), *(len(r.qualified_name) for r in recipes))

    lines = [
        f"{'FAMILY':<{family_w}}  {'MODE':<{mode_w}}  {'RECIPE':<{name_w}}",
        f"{'-' * family_w}  {'-' * mode_w}  {'-' * name_w}",
    ]
    for r in recipes:
        lines.append(
            f"{r.family.value:<{family_w}}  {r.mode.value:<{mode_w}}  {r.qualified_name:<{name_w}}"
        )

    if featured:
        lines.append(
            f"\nShowing {len(recipes)} featured recipes out of {total_count} total. "
            "Pass --all or filter with --family / --env / --model / --search."
        )
    else:
        lines.append(f"\n{len(recipes)} recipe(s) matched.")
    return "\n".join(lines) + "\n"


def _handle_list(
    args: argparse.Namespace,
    *,
    repo_root: Path | None,
    stream: TextIO,
) -> int:
    root = find_repo_root(repo_root)
    all_recipes = discover_recipes(root)
    has_filters = any(
        [args.all, args.family, args.mode, args.env, args.model, args.search]
    )
    featured_only = not has_filters
    selected = filter_recipes(
        all_recipes,
        featured_only=featured_only,
        family=RecipeFamily(args.family) if args.family else None,
        mode=RecipeMode(args.mode) if args.mode else None,
        env=args.env,
        model=args.model,
        search=args.search,
    )
    stream.write(
        _format_recipe_table(
            selected,
            featured=featured_only,
            total_count=len(all_recipes),
        )
    )
    stream.flush()
    return 0


def _handle_launch(
    args: argparse.Namespace,
    overrides: list[str],
    mode: RecipeMode,
    *,
    repo_root: Path | None,
    stream: TextIO,
) -> int:
    root = find_repo_root(repo_root)
    spec = resolve_recipe(
        args.target,
        mode,
        root,
        family=RecipeFamily(args.family) if args.family else None,
        subgroup=args.subgroup,
        benchmark=args.benchmark,
        use_async=args.use_async,
    )
    plan = build_launch_plan(
        spec,
        root,
        overrides,
        log_dir=args.log_dir,
    )
    return execute_launch_plan(plan, dry_run=args.dry_run, stream=stream)


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    stream: TextIO | None = None,
    err_stream: TextIO | None = None,
) -> int:
    """Entry point for the ``rlinf`` command-line interface.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        repo_root: Optional explicit repository root (primarily for testing).
        stream: Output stream (defaults to ``sys.stdout``).
        err_stream: Error stream (defaults to ``sys.stderr``).

    Returns:
        Process exit code (``0`` on success, non-zero on error).
    """
    out = sys.stdout if stream is None else stream
    err = sys.stderr if err_stream is None else err_stream
    parser = _build_parser()

    raw_args = list(sys.argv[1:] if argv is None else argv)
    # Launch subcommands forward unknown arguments verbatim to Hydra, while
    # ``list`` and ``doctor`` validate all flags strictly.
    if raw_args and raw_args[0] in {"train", "eval", "collect"}:
        args, overrides = parser.parse_known_args(raw_args)
    else:
        args = parser.parse_args(raw_args)
        overrides = []

    try:
        if args.command == "list":
            return _handle_list(args, repo_root=repo_root, stream=out)
        if args.command == "doctor":
            return render_doctor(
                repo_root=repo_root,
                check_network=args.network,
                stream=out,
            )
        mode_map = {
            "train": RecipeMode.TRAIN,
            "eval": RecipeMode.EVAL,
            "collect": RecipeMode.COLLECT,
        }
        return _handle_launch(
            args,
            overrides,
            mode_map[args.command],
            repo_root=repo_root,
            stream=out,
        )
    except (FileNotFoundError, ValueError) as exc:
        err.write(f"error: {exc}\n")
        err.flush()
        return 2
