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

"""Recipe discovery and target resolution for the ``rlinf`` CLI."""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import yaml

_OC_ENV_RE = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(,[^}]*)?\}")

# Repo-derived path env vars automatically populated by the CLI launcher.
_AUTO_MANAGED_ENV_VARS: frozenset[str] = frozenset(
    {
        "REPO_PATH",
        "EMBODIED_PATH",
        "EVALUATIONS_PATH",
        "DIFFUSION_PATH",
        "REWARD_PATH",
        "VLM_PATH",
    }
)


class RecipeFamily(str, Enum):
    """Top-level task families shipped under ``examples/`` and ``evaluations/``."""

    EMBODIED = "embodied"
    SFT = "sft"
    OFFLINE_RL = "offline_rl"
    DIFFUSION = "diffusion"
    REASONING = "reasoning"
    AGENT = "agent"
    REWARD = "reward"


class RecipeMode(str, Enum):
    """Execution mode selected by the CLI subcommand."""

    TRAIN = "train"
    EVAL = "eval"
    COLLECT = "collect"


@dataclass(frozen=True)
class RecipeSpec:
    """Resolved launch target for a single Hydra configuration file."""

    name: str
    family: RecipeFamily
    mode: RecipeMode
    config_file: Path
    config_dir: Path
    entry_script: Path
    subgroup: str = ""
    env_profile: str = "embodied_sim"
    inject_log_path: bool = True
    searchpath: tuple[Path, ...] = ()

    @property
    def qualified_name(self) -> str:
        """Return ``subgroup/name`` when scoped to a subgroup, otherwise ``name``."""
        return f"{self.subgroup}/{self.name}" if self.subgroup else self.name


# Canonical starting recipes surfaced by default in ``rlinf list``.
FEATURED_RECIPES: tuple[tuple[RecipeFamily, RecipeMode, str], ...] = (
    (RecipeFamily.EMBODIED, RecipeMode.TRAIN, "maniskill_ppo_mlp"),
    (RecipeFamily.EMBODIED, RecipeMode.TRAIN, "maniskill_ppo_openvla_quickstart"),
    (RecipeFamily.EMBODIED, RecipeMode.TRAIN, "maniskill_ppo_openvlaoft_quickstart"),
    (RecipeFamily.EMBODIED, RecipeMode.TRAIN, "libero_spatial_ppo_openpi_pi05"),
    (RecipeFamily.EMBODIED, RecipeMode.TRAIN, "robotwin_adjust_bottle_ppo_openpi_pi05"),
    (RecipeFamily.EMBODIED, RecipeMode.TRAIN, "realworld_charger_sac_cnn_async"),
    (RecipeFamily.EMBODIED, RecipeMode.EVAL, "libero_spatial_openpi_pi05_eval"),
    (RecipeFamily.EMBODIED, RecipeMode.COLLECT, "realworld_collect_data"),
    (RecipeFamily.SFT, RecipeMode.TRAIN, "libero_sft_openpi"),
    (RecipeFamily.OFFLINE_RL, RecipeMode.TRAIN, "d4rl_iql_mujoco"),
    (RecipeFamily.DIFFUSION, RecipeMode.TRAIN, "sd3_nft_ocr"),
    (RecipeFamily.REASONING, RecipeMode.TRAIN, "qwen2.5-1.5b-grpo-megatron"),
    (RecipeFamily.AGENT, RecipeMode.TRAIN, "train_qwen2.5"),
    (RecipeFamily.AGENT, RecipeMode.TRAIN, "train_qwen3_hybrid"),
    (RecipeFamily.REWARD, RecipeMode.TRAIN, "reward_training"),
)

_FAMILY_CONFIG_ROOTS: dict[RecipeFamily, str] = {
    RecipeFamily.EMBODIED: "examples/embodiment/config",
    RecipeFamily.SFT: "examples/sft/config",
    RecipeFamily.OFFLINE_RL: "examples/offline_rl/config",
    RecipeFamily.DIFFUSION: "examples/diffusion/config",
    RecipeFamily.REASONING: "examples/reasoning/config/math",
    RecipeFamily.AGENT: "examples/agent/searchr1/config",
    RecipeFamily.REWARD: "examples/reward/config",
}

_AGENT_SUBGROUP_ALIASES: dict[str, str] = {
    "searchr1": "searchr1",
    "wideseek": "wideseek_r1",
    "wideseek_r1": "wideseek_r1",
    "rstar2": "rstar2",
    "coding_online_rl": "coding_online_rl",
    "agentlightning": "calc_x",
    "calc_x": "calc_x",
}


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the RLinf checkout root containing ``examples/`` and ``rlinf/``.

    Args:
        start: Optional starting directory or explicit repo root.

    Returns:
        Resolved path to the repository root.

    Raises:
        FileNotFoundError: If no RLinf checkout can be found from ``start``,
            ``RLINF_REPO_PATH``, the current working directory, or the installed
            package path.
    """
    candidates: list[Path] = []
    if start is not None:
        resolved = start.resolve()
        candidates.append(resolved)
        candidates.extend(resolved.parents)
    else:
        env_root = os.environ.get("RLINF_REPO_PATH")
        if env_root:
            candidates.append(Path(env_root).expanduser().resolve())
        cwd = Path.cwd().resolve()
        candidates.append(cwd)
        candidates.extend(cwd.parents)
        pkg_root = Path(__file__).resolve().parents[2]
        candidates.append(pkg_root)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "examples").is_dir() and (candidate / "rlinf").is_dir():
            return candidate

    searched = ", ".join(str(p) for p in candidates[:4])
    raise FileNotFoundError(
        "Could not locate an RLinf repository root containing 'examples/' and "
        f"'rlinf/'. Searched: {searched}. Set RLINF_REPO_PATH or run inside an "
        "RLinf checkout."
    )


def _classify_embodied_config(repo_root: Path, cfg_file: Path) -> RecipeSpec:
    stem = cfg_file.stem
    lower = stem.lower()
    is_realworld = "realworld" in lower
    env_profile = "embodied_real" if is_realworld else "embodied_sim"

    if "collect" in lower:
        if "collect_dataset" in lower:
            entry = repo_root / "examples/reward/realworld_collect_process_dataset.py"
        elif is_realworld:
            entry = repo_root / "examples/embodiment/collect_real_data.py"
        else:
            entry = repo_root / "examples/embodiment/train_embodied_agent.py"
        mode = RecipeMode.COLLECT
    elif "async" in lower:
        entry = repo_root / "examples/embodiment/train_async.py"
        mode = RecipeMode.TRAIN
    elif "d4rl" in lower or "offline" in lower:
        return RecipeSpec(
            name=stem,
            family=RecipeFamily.OFFLINE_RL,
            mode=RecipeMode.TRAIN,
            config_file=cfg_file,
            config_dir=cfg_file.parent,
            entry_script=repo_root / "examples/embodiment/train_offline_rl.py",
            env_profile=env_profile,
            inject_log_path=True,
        )
    else:
        entry = repo_root / "examples/embodiment/train_embodied_agent.py"
        mode = RecipeMode.TRAIN

    return RecipeSpec(
        name=stem,
        family=RecipeFamily.EMBODIED,
        mode=mode,
        config_file=cfg_file,
        config_dir=cfg_file.parent,
        entry_script=entry,
        env_profile=env_profile,
        inject_log_path=True,
    )


def _iter_yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix in {".yaml", ".yml"}
        and not p.stem.startswith("base_")
        and p.stem != "tp_comm_overlap_cfg"
    )


def _add_agent_specs(
    specs: list[RecipeSpec],
    repo_root: Path,
    subgroup: str,
    config_rel: str,
    classify: Callable[[str], tuple[RecipeMode, str] | None],
) -> None:
    cfg_dir = repo_root / config_rel
    for cfg_file in _iter_yaml_files(cfg_dir):
        result = classify(cfg_file.stem)
        if result is None:
            continue
        mode, entry_rel = result
        specs.append(
            RecipeSpec(
                name=cfg_file.stem,
                family=RecipeFamily.AGENT,
                mode=mode,
                config_file=cfg_file,
                config_dir=cfg_dir,
                entry_script=repo_root / entry_rel,
                subgroup=subgroup,
                env_profile="agent_reason",
                inject_log_path=False,
            )
        )


def discover_recipes(repo_root: Path) -> list[RecipeSpec]:
    """Discover all runnable recipe YAML configs across an RLinf checkout.

    Args:
        repo_root: Repository root containing ``examples/`` and ``evaluations/``.

    Returns:
        Sorted list of discovered :class:`RecipeSpec` entries.
    """
    specs: list[RecipeSpec] = []

    # Embodied training and collection configs.
    embodied_dir = repo_root / "examples/embodiment/config"
    for cfg_file in _iter_yaml_files(embodied_dir):
        specs.append(_classify_embodied_config(repo_root, cfg_file))

    # Embodied evaluation configs under evaluations/<benchmark>/.
    eval_root = repo_root / "evaluations"
    if eval_root.is_dir():
        for bench_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
            benchmark = bench_dir.name
            env_profile = (
                "embodied_real" if benchmark == "realworld" else "embodied_sim"
            )
            for cfg_file in _iter_yaml_files(bench_dir):
                specs.append(
                    RecipeSpec(
                        name=cfg_file.stem,
                        family=RecipeFamily.EMBODIED,
                        mode=RecipeMode.EVAL,
                        config_file=cfg_file,
                        config_dir=bench_dir,
                        entry_script=repo_root / "evaluations/eval_embodied_agent.py",
                        subgroup=benchmark,
                        env_profile=env_profile,
                        inject_log_path=True,
                    )
                )

    # SFT configs.
    sft_dir = repo_root / "examples/sft/config"
    for cfg_file in _iter_yaml_files(sft_dir):
        is_vlm = "vlm" in cfg_file.stem.lower()
        specs.append(
            RecipeSpec(
                name=cfg_file.stem,
                family=RecipeFamily.SFT,
                mode=RecipeMode.TRAIN,
                config_file=cfg_file,
                config_dir=sft_dir,
                entry_script=repo_root
                / (
                    "examples/sft/train_vlm_sft.py"
                    if is_vlm
                    else "examples/sft/train_vla_sft.py"
                ),
                env_profile="sft_vlm" if is_vlm else "sft_vla",
                inject_log_path=True,
            )
        )

    # Offline RL training configs (CFG-RL, RECAP value SFT, STEAM value SFT).
    offline_dir = repo_root / "examples/offline_rl/config"
    for cfg_file in _iter_yaml_files(offline_dir):
        lower = cfg_file.stem.lower()
        if lower.startswith("cfg_rl"):
            offline_entry = (
                repo_root
                / "examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py"
            )
        elif "recap_value" in lower:
            offline_entry = (
                repo_root
                / "examples/offline_rl/advantage_labeling/recap/train_value.py"
            )
        elif "steam_value" in lower:
            offline_entry = (
                repo_root
                / "examples/offline_rl/advantage_labeling/steam/train_steam.py"
            )
        else:
            continue
        specs.append(
            RecipeSpec(
                name=cfg_file.stem,
                family=RecipeFamily.OFFLINE_RL,
                mode=RecipeMode.TRAIN,
                config_file=cfg_file,
                config_dir=offline_dir,
                entry_script=offline_entry,
                env_profile="embodied_sim",
                inject_log_path=True,
            )
        )

    # Diffusion configs.
    diffusion_dir = repo_root / "examples/diffusion/config"
    for cfg_file in _iter_yaml_files(diffusion_dir):
        specs.append(
            RecipeSpec(
                name=cfg_file.stem,
                family=RecipeFamily.DIFFUSION,
                mode=RecipeMode.TRAIN,
                config_file=cfg_file,
                config_dir=diffusion_dir,
                entry_script=repo_root / "examples/embodiment/train_embodied_agent.py",
                env_profile="diffusion",
                inject_log_path=True,
            )
        )

    # Reasoning configs (math and vqa).
    for subgroup in ("math", "vqa"):
        reason_dir = repo_root / "examples/reasoning/config" / subgroup
        for cfg_file in _iter_yaml_files(reason_dir):
            specs.append(
                RecipeSpec(
                    name=cfg_file.stem,
                    family=RecipeFamily.REASONING,
                    mode=RecipeMode.TRAIN,
                    config_file=cfg_file,
                    config_dir=reason_dir,
                    entry_script=repo_root / "examples/reasoning/main_grpo.py",
                    subgroup=subgroup,
                    env_profile="agent_reason",
                    inject_log_path=False,
                )
            )

    # Agent configs.
    _add_agent_specs(
        specs,
        repo_root,
        subgroup="searchr1",
        config_rel="examples/agent/searchr1/config",
        classify=lambda stem: (
            (RecipeMode.EVAL, "examples/agent/searchr1/eval.py")
            if "eval" in stem.lower()
            else (RecipeMode.TRAIN, "examples/agent/searchr1/train.py")
        ),
    )
    _add_agent_specs(
        specs,
        repo_root,
        subgroup="wideseek_r1",
        config_rel="examples/agent/wideseek_r1/config",
        classify=lambda stem: (
            (RecipeMode.EVAL, "examples/agent/wideseek_r1/eval.py")
            if "eval" in stem.lower()
            else (RecipeMode.TRAIN, "examples/agent/wideseek_r1/train.py")
        ),
    )
    _add_agent_specs(
        specs,
        repo_root,
        subgroup="rstar2",
        config_rel="examples/agent/rstar2/config",
        classify=lambda _stem: (
            RecipeMode.TRAIN,
            "examples/agent/rstar2/main_rstar2.py",
        ),
    )
    _add_agent_specs(
        specs,
        repo_root,
        subgroup="coding_online_rl",
        config_rel="examples/agent/coding_online_rl/config",
        classify=lambda stem: (
            (
                RecipeMode.TRAIN,
                "examples/agent/coding_online_rl/main_coding_rl_llm_judge.py",
            )
            if "llm_judge" in stem.lower()
            else (
                RecipeMode.TRAIN,
                "examples/agent/coding_online_rl/main_coding_online_rl.py",
            )
        ),
    )
    _add_agent_specs(
        specs,
        repo_root,
        subgroup="calc_x",
        config_rel="examples/agent/agentlightning/calc_x/config",
        classify=lambda stem: (
            RecipeMode.EVAL if "eval" in stem.lower() else RecipeMode.TRAIN,
            "examples/agent/agentlightning/calc_x/main.py",
        ),
    )

    # Reward configs (model training, real-world teleop eval, dataset collection).
    reward_dir = repo_root / "examples/reward/config"
    for cfg_file in _iter_yaml_files(reward_dir):
        lower = cfg_file.stem.lower()
        if "teleop" in lower:
            reward_mode = RecipeMode.EVAL
            reward_entry = repo_root / "examples/reward/eval_realworld_teleop.py"
        elif "collect" in lower:
            reward_mode = RecipeMode.COLLECT
            reward_entry = (
                repo_root / "examples/reward/realworld_collect_process_dataset.py"
            )
        else:
            reward_mode = RecipeMode.TRAIN
            reward_entry = repo_root / "examples/reward/train_reward_model.py"
        specs.append(
            RecipeSpec(
                name=cfg_file.stem,
                family=RecipeFamily.REWARD,
                mode=reward_mode,
                config_file=cfg_file,
                config_dir=reward_dir,
                entry_script=reward_entry,
                env_profile="reward",
                inject_log_path=True,
            )
        )

    return sorted(
        specs,
        key=lambda s: (s.family.value, s.mode.value, s.subgroup, s.name),
    )


def filter_recipes(
    recipes: list[RecipeSpec],
    *,
    featured_only: bool = False,
    family: RecipeFamily | None = None,
    mode: RecipeMode | None = None,
    env: str | None = None,
    model: str | None = None,
    search: str | None = None,
) -> list[RecipeSpec]:
    """Filter discovered recipes by curated set, family, mode, or substring.

    Args:
        recipes: Full list of discovered recipes.
        featured_only: When ``True``, keep only canonical featured recipes.
        family: Optional task family filter.
        mode: Optional execution mode filter.
        env: Optional case-insensitive environment token filter (matches name or subgroup).
        model: Optional case-insensitive model token filter (matches recipe name).
        search: Optional case-insensitive substring matched against name, subgroup, and family.

    Returns:
        Filtered list of :class:`RecipeSpec` entries.
    """
    filtered = recipes
    if family is not None:
        filtered = [r for r in filtered if r.family == family]
    if mode is not None:
        filtered = [r for r in filtered if r.mode == mode]
    if env:
        token = env.lower()
        filtered = [
            r
            for r in filtered
            if token in r.name.lower() or token in r.subgroup.lower()
        ]
    if model:
        token = model.lower()
        filtered = [r for r in filtered if token in r.name.lower()]
    if search:
        tokens = [t for t in search.lower().split() if t]
        filtered = [
            r
            for r in filtered
            if all(
                token in r.name.lower()
                or token in r.subgroup.lower()
                or token in r.family.value
                or token in r.mode.value
                for token in tokens
            )
        ]
    if featured_only:
        order = {key: idx for idx, key in enumerate(FEATURED_RECIPES)}
        featured = [r for r in filtered if (r.family, r.mode, r.name) in order]
        featured.sort(key=lambda r: order[(r.family, r.mode, r.name)])
        return featured
    return filtered


def _collect_default_yaml_files(yaml_path: Path, search_dirs: list[Path]) -> list[Path]:
    """Resolve one level of ``defaults`` entries to existing YAML files."""
    parsed = _read_yaml_dict(yaml_path)
    raw_defaults = parsed.get("defaults")
    if not isinstance(raw_defaults, list):
        return []

    found: list[Path] = []
    seen: set[Path] = set()
    for item in raw_defaults:
        rel_candidates: list[str] = []
        if isinstance(item, str):
            if item.startswith("override ") or item == "_self_":
                continue
            rel_candidates.append(item.split("@", 1)[0].strip())
        elif isinstance(item, dict):
            for key, val in item.items():
                key_str = str(key).split("@", 1)[0].strip()
                if key_str.startswith("override ") or not isinstance(val, str):
                    continue
                rel_candidates.append(f"{key_str}/{val}")

        for rel in rel_candidates:
            rel_path = rel if rel.endswith((".yaml", ".yml")) else f"{rel}.yaml"
            for base_dir in search_dirs:
                candidate = (base_dir / rel_path).resolve()
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    found.append(candidate)
                    break
    return found


def extract_required_env_vars(
    yaml_path: Path,
    search_dirs: list[Path] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(searchpath_vars, body_vars)`` referenced via ``${oc.env:VAR}`` without a default.

    Inspects ``yaml_path`` and any direct ``defaults:`` YAML files found under
    ``search_dirs`` so inherited env requirements (e.g. ``POLARIS_DATA_PATH``)
    are surfaced before launch.

    Args:
        yaml_path: Path to a Hydra YAML configuration file.
        search_dirs: Optional config roots used to resolve ``defaults:`` entries.

    Returns:
        Two sorted lists of environment variable names: those required inside
        ``hydra.searchpath`` (evaluated before CLI overrides) and those required
        elsewhere in the YAML or its direct defaults.
    """
    try:
        raw = yaml_path.read_text(encoding="utf-8")
    except OSError:
        return [], []

    parsed = _read_yaml_dict(yaml_path)
    searchpath_vars: set[str] = set()
    hydra_cfg = parsed.get("hydra")
    if isinstance(hydra_cfg, dict):
        sp = hydra_cfg.get("searchpath")
        if isinstance(sp, list):
            for entry in sp:
                if isinstance(entry, str):
                    for match in _OC_ENV_RE.finditer(entry):
                        if match.group(2) is None:
                            searchpath_vars.add(match.group(1))

    body_vars: set[str] = set()
    files_to_scan = [yaml_path]
    if search_dirs:
        files_to_scan.extend(_collect_default_yaml_files(yaml_path, search_dirs))

    for cfg_file in files_to_scan:
        try:
            content = (
                raw if cfg_file == yaml_path else cfg_file.read_text(encoding="utf-8")
            )
        except OSError:
            continue
        for line in content.splitlines():
            code_part = line.split("#", 1)[0]
            for match in _OC_ENV_RE.finditer(code_part):
                if match.group(2) is None and match.group(1) not in searchpath_vars:
                    body_vars.add(match.group(1))

    return sorted(searchpath_vars), sorted(body_vars)


def inspect_recipe(spec: RecipeSpec, repo_root: Path) -> dict[str, Any]:
    """Extract a concise, human- and JSON-friendly metadata summary for ``spec``.

    Args:
        spec: Resolved :class:`RecipeSpec`.
        repo_root: Repository root used to compute relative paths when inside repo.

    Returns:
        Dictionary of summary fields for ``rlinf show``.
    """
    parsed = _read_yaml_dict(spec.config_file)
    search_dirs = [
        spec.config_dir,
        *spec.searchpath,
        repo_root / _FAMILY_CONFIG_ROOTS[spec.family],
    ]
    searchpath_vars, body_vars = extract_required_env_vars(
        spec.config_file, search_dirs=search_dirs
    )
    user_env_vars = sorted(
        (set(searchpath_vars) | set(body_vars)) - _AUTO_MANAGED_ENV_VARS
    )

    def _rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            return str(path)

    defaults_list: list[str] = []
    raw_defaults = parsed.get("defaults")
    if isinstance(raw_defaults, list):
        for item in raw_defaults:
            if (
                isinstance(item, str)
                and not item.startswith("override ")
                and item != "_self_"
            ):
                defaults_list.append(item)
            elif isinstance(item, dict):
                for k, v in item.items():
                    if not str(k).startswith("override "):
                        defaults_list.append(f"{k}: {v}")

    cluster = parsed.get("cluster") if isinstance(parsed.get("cluster"), dict) else {}
    runner = parsed.get("runner") if isinstance(parsed.get("runner"), dict) else {}
    actor = parsed.get("actor") if isinstance(parsed.get("actor"), dict) else {}
    actor_model = actor.get("model") if isinstance(actor.get("model"), dict) else {}
    rollout = parsed.get("rollout") if isinstance(parsed.get("rollout"), dict) else {}
    rollout_model = (
        rollout.get("model") if isinstance(rollout.get("model"), dict) else {}
    )

    model_path = actor_model.get("model_path") or rollout_model.get("model_path")
    return {
        "name": spec.name,
        "qualified_name": spec.qualified_name,
        "family": spec.family.value,
        "mode": spec.mode.value,
        "config_file": _rel(spec.config_file),
        "entry_script": _rel(spec.entry_script),
        "defaults": defaults_list,
        "num_nodes": cluster.get("num_nodes"),
        "component_placement": cluster.get("component_placement"),
        "task_type": runner.get("task_type"),
        "max_epochs": runner.get("max_epochs"),
        "max_steps": runner.get("max_steps"),
        "model_path": model_path,
        "required_env_vars": user_env_vars,
        "launch_command": f"rlinf {spec.mode.value} {spec.qualified_name}",
    }


def resolve_recipe_any_mode(
    target: str,
    repo_root: Path,
    *,
    mode: RecipeMode | None = None,
    family: RecipeFamily | None = None,
    subgroup: str | None = None,
) -> RecipeSpec:
    """Resolve ``target`` across modes (used by ``rlinf show``).

    Args:
        target: Recipe name, ``subgroup/name``, or path to a local YAML file.
        repo_root: RLinf repository root.
        mode: Optional explicit mode filter.
        family: Optional family filter.
        subgroup: Optional subgroup filter.

    Returns:
        Resolved :class:`RecipeSpec`.

    Raises:
        FileNotFoundError: If ``target`` looks like a missing YAML file path.
        ValueError: If no recipe matches or multiple recipes match ambiguously.
    """
    if mode is not None:
        return resolve_recipe(target, mode, repo_root, family=family, subgroup=subgroup)

    candidate_path = Path(target).expanduser()
    if candidate_path.is_file():
        inferred_mode = (
            RecipeMode.EVAL
            if "eval" in candidate_path.stem.lower()
            else (
                RecipeMode.COLLECT
                if "collect" in candidate_path.stem.lower()
                else RecipeMode.TRAIN
            )
        )
        return resolve_recipe(
            target, inferred_mode, repo_root, family=family, subgroup=subgroup
        )
    if target.endswith((".yaml", ".yml")) and (
        os.sep in target or target.startswith(".")
    ):
        raise FileNotFoundError(
            f"Config file not found: '{target}'. Expected an existing YAML file "
            "or a recipe name from 'rlinf list'."
        )

    all_recipes = discover_recipes(repo_root)
    normalized = target.removesuffix(".yaml").removesuffix(".yml")
    lookup_subgroup = subgroup
    lookup_name = normalized
    if "/" in normalized:
        lookup_subgroup, lookup_name = normalized.split("/", 1)
    if lookup_subgroup is not None:
        lookup_subgroup = _AGENT_SUBGROUP_ALIASES.get(
            lookup_subgroup.lower(), lookup_subgroup
        )

    pool = all_recipes
    if family is not None:
        pool = [r for r in pool if r.family == family]
    if lookup_subgroup is not None:
        pool = [r for r in pool if r.subgroup == lookup_subgroup]

    matches = [r for r in pool if r.name == lookup_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ", ".join(
            f"--mode {m.mode.value} --family {m.family.value} {m.qualified_name}"
            for m in matches
        )
        raise ValueError(f"Ambiguous recipe '{target}'. Disambiguate with: {options}.")

    all_names = sorted({r.name for r in all_recipes})
    close = difflib.get_close_matches(lookup_name, all_names, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(
        f"Unknown recipe '{target}'.{hint} Run 'rlinf list --all' to view available recipes."
    )


def _read_yaml_dict(yaml_path: Path) -> dict:
    try:
        parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _yaml_declares_searchpath(parsed: dict) -> bool:
    hydra_cfg = parsed.get("hydra")
    return isinstance(hydra_cfg, dict) and bool(hydra_cfg.get("searchpath"))


def _infer_external_family_and_subgroup(
    yaml_path: Path, parsed: dict
) -> tuple[RecipeFamily, str]:
    """Infer task family and optional subgroup for an external/e2e YAML file."""
    parent_parts = [p.lower() for p in yaml_path.parent.parts]
    for part in reversed(parent_parts):
        if part in _AGENT_SUBGROUP_ALIASES:
            return RecipeFamily.AGENT, _AGENT_SUBGROUP_ALIASES[part]
    if "offline_rl" in parent_parts:
        return RecipeFamily.OFFLINE_RL, ""
    if "sft" in parent_parts:
        if "reward" in yaml_path.stem.lower():
            return RecipeFamily.REWARD, ""
        return RecipeFamily.SFT, ""
    if "reasoning" in parent_parts or "dynamic_scheduler" in parent_parts:
        return RecipeFamily.REASONING, ""
    if "diffusion" in parent_parts:
        return RecipeFamily.DIFFUSION, ""
    if "reward" in parent_parts:
        return RecipeFamily.REWARD, ""

    runner = parsed.get("runner")
    task_type = runner.get("task_type") if isinstance(runner, dict) else None
    if isinstance(task_type, str):
        lower_task = task_type.lower()
        if "sft" in lower_task:
            return RecipeFamily.SFT, ""
        if "reasoning" in lower_task:
            return RecipeFamily.REASONING, ""
        if "agent" in lower_task:
            return RecipeFamily.AGENT, ""
        if "offline" in lower_task:
            return RecipeFamily.OFFLINE_RL, ""

    lower_name = yaml_path.stem.lower()
    if "sft" in lower_name:
        return RecipeFamily.SFT, ""
    if "d4rl" in lower_name or "offline" in lower_name:
        return RecipeFamily.OFFLINE_RL, ""
    return RecipeFamily.EMBODIED, ""


def _resolve_agent_entry(
    repo_root: Path, subgroup: str, mode: RecipeMode, lower: str
) -> Path:
    canonical = _AGENT_SUBGROUP_ALIASES.get(subgroup.lower(), subgroup.lower())
    if canonical == "searchr1":
        return repo_root / (
            "examples/agent/searchr1/eval.py"
            if mode == RecipeMode.EVAL
            else "examples/agent/searchr1/train.py"
        )
    if canonical == "wideseek_r1":
        return repo_root / (
            "examples/agent/wideseek_r1/eval.py"
            if mode == RecipeMode.EVAL
            else "examples/agent/wideseek_r1/train.py"
        )
    if canonical == "rstar2":
        return repo_root / "examples/agent/rstar2/main_rstar2.py"
    if canonical == "coding_online_rl":
        return repo_root / (
            "examples/agent/coding_online_rl/main_coding_rl_llm_judge.py"
            if "llm_judge" in lower
            else "examples/agent/coding_online_rl/main_coding_online_rl.py"
        )
    if canonical == "calc_x":
        return repo_root / "examples/agent/agentlightning/calc_x/main.py"

    supported = ", ".join(sorted(set(_AGENT_SUBGROUP_ALIASES.values())))
    raise ValueError(
        f"Unknown agent subgroup '{subgroup}'. Expected one of: {supported}."
    )


def _build_external_spec(
    repo_root: Path,
    yaml_path: Path,
    mode: RecipeMode,
    family: RecipeFamily | None,
    subgroup: str | None,
    use_async: bool,
) -> RecipeSpec:
    parsed = _read_yaml_dict(yaml_path)
    inferred_family, inferred_subgroup = _infer_external_family_and_subgroup(
        yaml_path, parsed
    )
    resolved_family = family or inferred_family
    resolved_subgroup = (
        _AGENT_SUBGROUP_ALIASES.get(subgroup.lower(), subgroup)
        if subgroup
        else inferred_subgroup
    )
    has_searchpath = _yaml_declares_searchpath(parsed)
    config_root = repo_root / _FAMILY_CONFIG_ROOTS[resolved_family]
    lower = yaml_path.stem.lower()

    if resolved_family == RecipeFamily.AGENT:
        if not resolved_subgroup:
            supported = ", ".join(sorted(set(_AGENT_SUBGROUP_ALIASES.values())))
            raise ValueError(
                f"Cannot determine agent entry script for '{yaml_path}'. "
                f"Pass '--subgroup' with one of: {supported}."
            )
        entry_script = _resolve_agent_entry(repo_root, resolved_subgroup, mode, lower)
        env_profile = "agent_reason"
        searchpath: tuple[Path, ...] = ()
    elif mode == RecipeMode.EVAL:
        entry_script = repo_root / "evaluations/eval_embodied_agent.py"
        env_profile = "embodied_real" if "realworld" in lower else "embodied_sim"
        searchpath = (
            () if has_searchpath else (repo_root / "examples/embodiment/config",)
        )
    elif mode == RecipeMode.COLLECT:
        if "collect_dataset" in lower:
            entry_script = (
                repo_root / "examples/reward/realworld_collect_process_dataset.py"
            )
        elif "collect_data" in lower or "realworld" in lower:
            entry_script = repo_root / "examples/embodiment/collect_real_data.py"
        else:
            entry_script = repo_root / "examples/embodiment/train_embodied_agent.py"
        env_profile = "embodied_real" if "realworld" in lower else "embodied_sim"
        searchpath = (
            () if has_searchpath else (repo_root / "examples/embodiment/config",)
        )
    elif resolved_family == RecipeFamily.SFT:
        is_vlm = "vlm" in lower
        entry_script = repo_root / (
            "examples/sft/train_vlm_sft.py"
            if is_vlm
            else "examples/sft/train_vla_sft.py"
        )
        env_profile = "sft_vlm" if is_vlm else "sft_vla"
        searchpath = () if has_searchpath else (config_root,)
    elif resolved_family == RecipeFamily.OFFLINE_RL:
        if "cfg" in lower:
            entry_script = (
                repo_root
                / "examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py"
            )
        elif "recap" in lower:
            entry_script = (
                repo_root
                / "examples/offline_rl/advantage_labeling/recap/train_value.py"
            )
        elif "steam" in lower:
            entry_script = (
                repo_root
                / "examples/offline_rl/advantage_labeling/steam/train_steam.py"
            )
        else:
            entry_script = repo_root / "examples/embodiment/train_offline_rl.py"
        env_profile = "embodied_sim"
        searchpath = (
            ()
            if has_searchpath
            else (config_root, repo_root / "examples/embodiment/config")
        )
    elif resolved_family == RecipeFamily.DIFFUSION:
        entry_script = repo_root / "examples/embodiment/train_embodied_agent.py"
        env_profile = "diffusion"
        searchpath = () if has_searchpath else (config_root,)
    elif resolved_family == RecipeFamily.REASONING:
        entry_script = repo_root / "examples/reasoning/main_grpo.py"
        env_profile = "agent_reason"
        searchpath = ()
    elif resolved_family == RecipeFamily.REWARD:
        if "teleop" in lower:
            entry_script = repo_root / "examples/reward/eval_realworld_teleop.py"
        elif "collect" in lower:
            entry_script = (
                repo_root / "examples/reward/realworld_collect_process_dataset.py"
            )
        else:
            entry_script = repo_root / "examples/reward/train_reward_model.py"
        env_profile = "reward"
        searchpath = () if has_searchpath else (config_root,)
    else:
        entry_script = (
            repo_root / "examples/embodiment/train_async.py"
            if (use_async or "async" in lower)
            else repo_root / "examples/embodiment/train_embodied_agent.py"
        )
        env_profile = "embodied_real" if "realworld" in lower else "embodied_sim"
        searchpath = (
            () if has_searchpath else (repo_root / "examples/embodiment/config",)
        )

    return RecipeSpec(
        name=yaml_path.stem,
        family=resolved_family,
        mode=mode,
        config_file=yaml_path,
        config_dir=yaml_path.parent,
        entry_script=entry_script,
        subgroup=resolved_subgroup,
        env_profile=env_profile,
        # External and e2e YAMLs own their runner.logger.log_path unless --log-dir is passed.
        inject_log_path=False,
        searchpath=searchpath,
    )


def resolve_recipe(
    target: str,
    mode: RecipeMode,
    repo_root: Path,
    *,
    family: RecipeFamily | None = None,
    subgroup: str | None = None,
    benchmark: str | None = None,
    use_async: bool = False,
) -> RecipeSpec:
    """Resolve a recipe name or local YAML path to a concrete :class:`RecipeSpec`.

    Args:
        target: Recipe name (with or without ``.yaml``), ``subgroup/name``, or
            path to a local YAML file (in-repo or out-of-tree).
        mode: Target execution mode (``train``, ``eval``, or ``collect``).
        repo_root: RLinf repository root.
        family: Optional family filter to disambiguate names or steer external YAMLs.
        subgroup: Optional subgroup name (e.g. ``searchr1``, ``math``, ``libero``).
        benchmark: Optional evaluation benchmark subgroup (alias for ``subgroup``).
        use_async: Force the async embodied training entrypoint.

    Returns:
        Resolved :class:`RecipeSpec`.

    Raises:
        FileNotFoundError: If ``target`` looks like a file path that does not exist.
        ValueError: If no matching recipe exists or multiple recipes match ambiguously.
    """
    all_recipes = discover_recipes(repo_root)
    effective_subgroup = benchmark or subgroup

    # 1. Existing local YAML path (either inside the repo or out-of-tree).
    candidate_path = Path(target).expanduser()
    if candidate_path.is_file():
        resolved_path = candidate_path.resolve()
        for spec in all_recipes:
            if spec.config_file.resolve() == resolved_path:
                return _apply_mode_overrides(spec, repo_root, mode, use_async)
        return _build_external_spec(
            repo_root, resolved_path, mode, family, effective_subgroup, use_async
        )

    if target.endswith((".yaml", ".yml")) and (
        os.sep in target or target.startswith(".")
    ):
        raise FileNotFoundError(
            f"Config file not found: '{target}'. Expected an existing YAML file "
            "or a recipe name from 'rlinf list'."
        )

    # 2. Named recipe lookup (optional ``subgroup/name`` and optional ``.yaml`` suffix).
    normalized = target.removesuffix(".yaml").removesuffix(".yml")
    lookup_subgroup = effective_subgroup
    lookup_name = normalized
    if "/" in normalized:
        lookup_subgroup, lookup_name = normalized.split("/", 1)
    if lookup_subgroup is not None:
        lookup_subgroup = _AGENT_SUBGROUP_ALIASES.get(
            lookup_subgroup.lower(), lookup_subgroup
        )

    pool = [r for r in all_recipes if r.mode == mode]
    if family is not None:
        pool = [r for r in pool if r.family == family]
    if lookup_subgroup is not None:
        pool = [r for r in pool if r.subgroup == lookup_subgroup]

    matches = [r for r in pool if r.name == lookup_name]

    # Mirror evaluations/run_eval.sh: allow evaluating an embodied training config
    # directly when no dedicated config exists under evaluations/.
    if (
        not matches
        and mode == RecipeMode.EVAL
        and family in (None, RecipeFamily.EMBODIED)
    ):
        embodied_pool = [
            r
            for r in all_recipes
            if r.family == RecipeFamily.EMBODIED
            and r.config_dir == repo_root / "examples/embodiment/config"
            and r.name == lookup_name
        ]
        if embodied_pool:
            base = embodied_pool[0]
            return RecipeSpec(
                name=base.name,
                family=RecipeFamily.EMBODIED,
                mode=RecipeMode.EVAL,
                config_file=base.config_file,
                config_dir=base.config_dir,
                entry_script=repo_root / "evaluations/eval_embodied_agent.py",
                subgroup=lookup_subgroup or "",
                env_profile=base.env_profile,
                inject_log_path=True,
            )

    if len(matches) == 1:
        return _apply_mode_overrides(matches[0], repo_root, mode, use_async)

    if len(matches) > 1:
        commands = ", ".join(
            f"'rlinf {mode.value} --family {m.family.value} {m.qualified_name}'"
            for m in matches
        )
        raise ValueError(
            f"Ambiguous recipe '{target}' for mode '{mode.value}'. "
            f"Disambiguate with one of: {commands}."
        )

    # Helpful error message with close matches and wrong-mode command suggestions.
    other_matches = [r for r in all_recipes if r.name == lookup_name]
    if other_matches:
        suggestions = ", ".join(
            f"'rlinf {m.mode.value} {m.qualified_name}'" for m in other_matches
        )
        raise ValueError(
            f"Recipe '{target}' is not a {mode.value} recipe. "
            f"Did you mean: {suggestions}?"
        )

    mode_names = sorted({r.name for r in all_recipes if r.mode == mode})
    close = difflib.get_close_matches(lookup_name, mode_names, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(
        f"Unknown {mode.value} recipe '{target}'.{hint} "
        "Run 'rlinf list --all' to inspect available recipes."
    )


def _apply_mode_overrides(
    spec: RecipeSpec,
    repo_root: Path,
    mode: RecipeMode,
    use_async: bool,
) -> RecipeSpec:
    entry = spec.entry_script
    resolved_mode = spec.mode
    if mode == RecipeMode.EVAL and spec.family == RecipeFamily.EMBODIED:
        entry = repo_root / "evaluations/eval_embodied_agent.py"
        resolved_mode = RecipeMode.EVAL
    elif (
        use_async and spec.family == RecipeFamily.EMBODIED and mode == RecipeMode.TRAIN
    ):
        entry = repo_root / "examples/embodiment/train_async.py"

    if entry == spec.entry_script and resolved_mode == spec.mode:
        return spec
    return RecipeSpec(
        name=spec.name,
        family=spec.family,
        mode=resolved_mode,
        config_file=spec.config_file,
        config_dir=spec.config_dir,
        entry_script=entry,
        subgroup=spec.subgroup,
        env_profile=spec.env_profile,
        inject_log_path=spec.inject_log_path,
        searchpath=spec.searchpath,
    )
