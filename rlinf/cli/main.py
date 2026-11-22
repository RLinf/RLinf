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
import difflib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from .catalog import (
    RecipeFamily,
    RecipeMode,
    RecipeSpec,
    discover_recipes,
    filter_recipes,
    find_repo_root,
    inspect_recipe,
    resolve_recipe,
    resolve_recipe_any_mode,
)
from .doctor import render_doctor
from .launcher import build_launch_plan, execute_launch_plan

_KNOWN_LAUNCH_FLAGS: tuple[str, ...] = (
    "--family",
    "--subgroup",
    "--benchmark",
    "--log-dir",
    "--dry-run",
    "--check",
    "--async",
    "--help",
)

# Native Hydra CLI flags that may legitimately be passed alongside overrides.
_HYDRA_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "-c",
        "--cfg",
        "-m",
        "--multirun",
        "--info",
        "--resolve",
        "-p",
        "--package",
        "--hydra-help",
        "--shell-completion",
        "-cd",
        "--config-dir",
        "-cp",
        "--config-path",
        "-cn",
        "--config-name",
    }
)


def _get_version() -> str:
    try:
        return importlib.metadata.version("rlinf")
    except importlib.metadata.PackageNotFoundError:
        return "0.4.0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlinf",
        description=(
            "Thin unified CLI for discovering, inspecting, checking, and "
            "launching RLinf training, evaluation, and data collection runs."
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
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
        help="Filter recipes by space-separated tokens across family, mode, subgroup, and name.",
    )
    list_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show entry script and config file paths alongside each recipe.",
    )
    list_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )

    # rlinf show
    show_parser = subparsers.add_parser(
        "show",
        help="Inspect a recipe's config path, placement, env requirements, and launch command.",
    )
    show_parser.add_argument(
        "target",
        help="Recipe name (from 'rlinf list') or path to a local YAML file.",
    )
    show_parser.add_argument(
        "--mode",
        choices=[m.value for m in RecipeMode],
        help="Disambiguate recipes that exist under multiple modes.",
    )
    show_parser.add_argument(
        "--family",
        choices=[f.value for f in RecipeFamily],
        help="Disambiguate a recipe name or set the family for an external YAML.",
    )
    show_parser.add_argument(
        "--subgroup",
        help="Optional subgroup (e.g. searchr1, wideseek_r1, libero).",
    )
    show_parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw YAML configuration file.",
    )
    show_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON metadata.",
    )

    # rlinf train / eval / collect
    def _add_launch_subparser(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(
            name,
            help=help_text,
            description=(
                f"{help_text} Any extra Hydra overrides (e.g. runner.max_epochs=10, "
                "+key=value, ~key) are forwarded verbatim. Use '--' to forward "
                "arbitrary raw flags."
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
        sub.add_argument(
            "--check",
            action="store_true",
            help="Validate Hydra config composition (--cfg job) without starting Ray.",
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
    doctor_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )

    return parser


def _validate_forwarded_overrides(overrides: list[str]) -> None:
    """Catch mistyped CLI flags (like ``--dr-run``) before forwarding to Hydra."""
    for token in overrides:
        if not token.startswith("-") or "=" in token:
            continue
        flag_name = token.split("=", 1)[0]
        if flag_name in _HYDRA_CLI_FLAGS:
            continue
        close = difflib.get_close_matches(
            flag_name, _KNOWN_LAUNCH_FLAGS, n=2, cutoff=0.5
        )
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown option '{token}'.{hint} Pass '--' before raw flags to "
            "forward them verbatim."
        )


def _format_recipe_table(
    recipes: list[RecipeSpec],
    repo_root: Path,
    *,
    featured: bool,
    total_count: int,
    verbose: bool = False,
) -> str:
    if not recipes:
        return (
            "No matching recipes found. Run 'rlinf list --all' to view all recipes.\n"
        )

    family_w = max(len("FAMILY"), *(len(r.family.value) for r in recipes))
    mode_w = max(len("MODE"), *(len(r.mode.value) for r in recipes))
    name_w = max(len("RECIPE"), *(len(r.qualified_name) for r in recipes))

    if verbose:
        scripts = [str(r.entry_script.relative_to(repo_root)) for r in recipes]
        configs = [str(r.config_file.relative_to(repo_root)) for r in recipes]
        script_w = max(len("SCRIPT"), *(len(s) for s in scripts))
        config_w = max(len("CONFIG"), *(len(c) for c in configs))
        lines = [
            f"{'FAMILY':<{family_w}}  {'MODE':<{mode_w}}  {'RECIPE':<{name_w}}  "
            f"{'SCRIPT':<{script_w}}  {'CONFIG':<{config_w}}",
            f"{'-' * family_w}  {'-' * mode_w}  {'-' * name_w}  "
            f"{'-' * script_w}  {'-' * config_w}",
        ]
        for r, s, c in zip(recipes, scripts, configs):
            lines.append(
                f"{r.family.value:<{family_w}}  {r.mode.value:<{mode_w}}  "
                f"{r.qualified_name:<{name_w}}  {s:<{script_w}}  {c:<{config_w}}"
            )
    else:
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


def _format_show_summary(info: dict[str, Any]) -> str:
    lines = [
        f"Recipe:    {info['qualified_name']} ({info['family']}/{info['mode']})",
        f"Config:    {info['config_file']}",
        f"Entry:     {info['entry_script']}",
    ]
    if info["task_type"] is not None:
        lines.append(f"Task type: {info['task_type']}")
    if info["num_nodes"] is not None:
        lines.append(f"Nodes:     {info['num_nodes']}")
    if info["component_placement"] is not None:
        lines.append(f"Placement: {info['component_placement']}")
    if info["max_epochs"] is not None or info["max_steps"] is not None:
        lines.append(
            f"Schedule:  max_epochs={info['max_epochs']}, max_steps={info['max_steps']}"
        )
    if info["model_path"]:
        lines.append(f"Model:     {info['model_path']}")
    if info["defaults"]:
        lines.append(f"Defaults:  {', '.join(info['defaults'])}")
    if info["required_env_vars"]:
        status_items = [
            f"{var} [{'set' if os.environ.get(var) else 'UNSET'}]"
            for var in info["required_env_vars"]
        ]
        lines.append(f"Env vars:  {', '.join(status_items)}")
    lines.append(f"Run:       {info['launch_command']}")
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
    if args.as_json:
        payload = [
            {
                "family": r.family.value,
                "mode": r.mode.value,
                "name": r.name,
                "qualified_name": r.qualified_name,
                "config_file": str(r.config_file.relative_to(root)),
                "entry_script": str(r.entry_script.relative_to(root)),
            }
            for r in selected
        ]
        stream.write(json.dumps(payload, indent=2) + "\n")
    else:
        stream.write(
            _format_recipe_table(
                selected,
                root,
                featured=featured_only,
                total_count=len(all_recipes),
                verbose=args.verbose,
            )
        )
    stream.flush()
    return 0


def _handle_show(
    args: argparse.Namespace,
    *,
    repo_root: Path | None,
    stream: TextIO,
) -> int:
    root = find_repo_root(repo_root)
    spec = resolve_recipe_any_mode(
        args.target,
        root,
        mode=RecipeMode(args.mode) if args.mode else None,
        family=RecipeFamily(args.family) if args.family else None,
        subgroup=args.subgroup,
    )
    if args.raw:
        stream.write(spec.config_file.read_text(encoding="utf-8"))
        stream.flush()
        return 0

    info = inspect_recipe(spec, root)
    if args.as_json:
        stream.write(json.dumps(info, indent=2) + "\n")
    else:
        stream.write(_format_show_summary(info))
    stream.flush()
    return 0


def _handle_launch(
    args: argparse.Namespace,
    overrides: list[str],
    mode: RecipeMode,
    *,
    repo_root: Path | None,
    stream: TextIO,
    err_stream: TextIO,
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
    return execute_launch_plan(
        plan,
        dry_run=args.dry_run,
        check_only=args.check,
        stream=stream,
        err_stream=err_stream,
    )


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
    if not raw_args:
        parser.print_help(out)
        return 0

    # Launch subcommands forward Hydra overrides verbatim while still catching
    # mistyped CLI flags. Everything after '--' is forwarded unconditionally.
    if raw_args[0] in {"train", "eval", "collect"}:
        if "--" in raw_args:
            sep = raw_args.index("--")
            cli_part = raw_args[:sep]
            passthrough = raw_args[sep + 1 :]
        else:
            cli_part = raw_args
            passthrough = []
        args, unknown = parser.parse_known_args(cli_part)
        try:
            _validate_forwarded_overrides(unknown)
        except ValueError as exc:
            err.write(f"error: {exc}\n")
            err.flush()
            return 2
        overrides = [*unknown, *passthrough]
    else:
        args = parser.parse_args(raw_args)
        overrides = []

    try:
        if args.command == "list":
            return _handle_list(args, repo_root=repo_root, stream=out)
        if args.command == "show":
            return _handle_show(args, repo_root=repo_root, stream=out)
        if args.command == "doctor":
            return render_doctor(
                repo_root=repo_root,
                check_network=args.network,
                as_json=args.as_json,
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
            err_stream=err,
        )
    except (FileNotFoundError, ValueError) as exc:
        err.write(f"error: {exc}\n")
        err.flush()
        return 2
