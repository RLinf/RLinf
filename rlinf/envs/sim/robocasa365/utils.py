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

"""Utility helpers for RoboCasa365."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

import numpy as np
from omegaconf import OmegaConf

_LEGACY_TASK_DESC_MAP = {
    "OpenSingleDoor": "open cabinet or microwave door",
    "CloseSingleDoor": "close cabinet or microwave door",
    "OpenDoubleDoor": "open double cabinet doors",
    "CloseDoubleDoor": "close double cabinet doors",
    "OpenDrawer": "open drawer",
    "CloseDrawer": "close drawer",
    "PnPCounterToCab": "pick and place from counter to cabinet",
    "PnPCabToCounter": "pick and place from cabinet to counter",
    "PnPCounterToSink": "pick and place from counter to sink",
    "PnPSinkToCounter": "pick and place from sink to counter",
    "PnPCounterToStove": "pick and place from counter to stove",
    "PnPStoveToCounter": "pick and place from stove to counter",
    "PnPCounterToMicrowave": "pick and place from counter to microwave",
    "PnPMicrowaveToCounter": "pick and place from microwave to counter",
    "TurnOnMicrowave": "turn on microwave",
    "TurnOffMicrowave": "turn off microwave",
    "TurnOnSinkFaucet": "turn on sink faucet",
    "TurnOffSinkFaucet": "turn off sink faucet",
    "TurnSinkSpout": "turn sink spout",
    "TurnOnStove": "turn on stove",
    "TurnOffStove": "turn off stove",
    "CoffeeSetupMug": "setup mug for coffee",
    "CoffeeServeMug": "serve coffee into mug",
    "CoffeePressButton": "press coffee machine button",
}

_OFFICIAL_PROMPT_INFO_KEY = "ep_meta"
_OFFICIAL_PROMPT_LANG_KEY = "lang"


def _cfg_to_python(value: Any) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _split_camel_case(name: str) -> str:
    name = name.split("/", 1)[-1]
    name = name.removeprefix("Kitchen")
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)
    return " ".join(token.lower() for token in tokens) if tokens else name.lower()


def _official_prompt_from_info(info_single: dict[str, Any]) -> str:
    value = info_single[_OFFICIAL_PROMPT_INFO_KEY][_OFFICIAL_PROMPT_LANG_KEY]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.ndarray) and value.shape == ():
        scalar_value = value.item()
        if isinstance(scalar_value, str):
            return scalar_value.strip()
    raise TypeError(
        "RoboCasa365 official prompt info "
        f"'{_OFFICIAL_PROMPT_INFO_KEY}.{_OFFICIAL_PROMPT_LANG_KEY}' must contain "
        f"a string, got {type(value).__name__}."
    )


def _normalize_task_filter(task_filter: Any) -> dict[str, list[str]]:
    if task_filter is None:
        return {"include": [], "exclude": []}
    if isinstance(task_filter, dict):
        include = _ensure_list(task_filter.get("include"))
        exclude = _ensure_list(task_filter.get("exclude"))
    else:
        include = _ensure_list(task_filter)
        exclude = []
    return {
        "include": [str(pattern) for pattern in include if pattern],
        "exclude": [str(pattern) for pattern in exclude if pattern],
    }


def _pattern_matches(pattern: str, text: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], text, flags=re.IGNORECASE) is not None
    return pattern.lower() in text.lower()


def _task_matches_filter(
    task_spec: dict[str, Any], task_filter: dict[str, list[str]]
) -> bool:
    haystack = " | ".join(
        str(
            task_spec.get(key, "")
            if key != "metadata"
            else task_spec.get("metadata_view", {})
        )
        for key in ("task_name", "env_name", "task_description", "benchmark_selection")
    )

    include = task_filter["include"]
    if include and not any(_pattern_matches(pattern, haystack) for pattern in include):
        return False

    exclude = task_filter["exclude"]
    if exclude and any(_pattern_matches(pattern, haystack) for pattern in exclude):
        return False

    return True


def _guess_task_mode(
    task_name: str, task_soup: Optional[str], metadata: dict[str, Any]
) -> Optional[str]:
    mode = metadata.get("task_mode") or metadata.get("mode")
    if isinstance(mode, str) and mode:
        return mode

    if task_soup:
        lowered = task_soup.lower()
        if "atomic" in lowered:
            return "atomic"
        if "composite" in lowered:
            return "composite"

    lowered_name = task_name.lower()
    if "atomic" in lowered_name:
        return "atomic"
    if "composite" in lowered_name:
        return "composite"
    return None


def _build_benchmark_selection(
    task_source: str,
    split: Optional[str],
    task_soups: list[str],
    dataset_source: Optional[str],
) -> str:
    if task_source == "legacy":
        return "legacy"

    parts = [part for part in [dataset_source, split] if part]
    if task_soups:
        parts.append("+".join(task_soups))
    return "/".join(parts) if parts else "registry"


def _get_task_horizon(task_name: str, fallback_horizon: int) -> int:
    try:
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        return int(get_task_horizon(task_name))
    except Exception:
        return fallback_horizon


def load_robocasa365_task_specs(cfg: Any) -> list[dict[str, Any]]:
    """Load and filter RoboCasa365 task specs from an environment config."""

    task_source = str(cfg.get("task_source", "dataset_registry"))
    if task_source != "dataset_registry":
        raise ValueError(
            "RoboCasa365 task loading currently requires "
            f"task_source='dataset_registry', got {task_source!r}."
        )

    dataset_source = cfg.get("dataset_source", None)
    if dataset_source is not None:
        dataset_source = str(dataset_source)
    split = cfg.get("split", None)
    if split is not None:
        split = str(split)
    task_soups = [
        str(soup) for soup in _ensure_list(_cfg_to_python(cfg.get("task_soup", None)))
    ]
    task_mode = cfg.get("task_mode", None)
    if task_mode is not None:
        task_mode = str(task_mode)
    task_filter = _normalize_task_filter(_cfg_to_python(cfg.get("task_filter", None)))
    benchmark_selection = cfg.get(
        "benchmark_selection",
        _build_benchmark_selection(
            task_source=task_source,
            split=split,
            task_soups=task_soups,
            dataset_source=dataset_source,
        ),
    )

    try:
        from robocasa.utils.dataset_registry_utils import get_ds_meta, get_ds_soup
    except ImportError as exc:
        raise ImportError(
            "RoboCasa365 benchmark selection requires "
            "robocasa.utils.dataset_registry_utils. Install a RoboCasa version "
            "that includes the benchmark dataset registry."
        ) from exc

    task_names: list[str] = []
    if task_soups:
        if split is None:
            raise ValueError(
                "split must be provided when task_soup is set for RoboCasa365."
            )
        for task_soup in task_soups:
            entries = get_ds_soup(
                task_set=task_soup,
                split=split,
                source=dataset_source or "human",
            )
            task_names.extend(str(task_entry["task"]) for task_entry in entries)
    else:
        task_names = [
            str(task)
            for task in _ensure_list(_cfg_to_python(cfg.get("task_names", None)))
        ]

    fallback_horizon = int(cfg.get("max_episode_steps", 300))
    task_specs = []
    for task_name in _dedupe_preserve_order(task_names):
        try:
            metadata = (
                get_ds_meta(
                    task_name,
                    split=split,
                    source=dataset_source or "human",
                )
                or {}
            )
        except Exception:
            metadata = {}

        task_label = task_name.split("/", 1)[-1]
        task_soup = metadata.get("task_soup")
        if not task_soup and task_soups:
            task_soup = task_soups[0] if len(task_soups) == 1 else "+".join(task_soups)
        selected_task_mode = _guess_task_mode(task_label, task_soup, metadata)
        metadata_view = {
            "task_name": task_label,
            "env_name": f"robocasa/{task_label}",
            "task_source": "dataset_registry",
            "dataset_source": dataset_source or "human",
            "split": split,
            "task_soup": task_soup,
            "benchmark_selection": benchmark_selection,
            "task_mode": selected_task_mode,
        }
        task_spec = {
            "task_name": task_label,
            "env_name": f"robocasa/{task_label}",
            "task_description": _split_camel_case(task_label),
            "horizon": _get_task_horizon(task_label, fallback_horizon),
            "metadata_view": {
                key: value for key, value in metadata_view.items() if value is not None
            },
            "task_mode": selected_task_mode,
            "benchmark_selection": benchmark_selection,
        }
        if task_mode and selected_task_mode and selected_task_mode != task_mode:
            continue
        if not _task_matches_filter(task_spec, task_filter):
            continue
        task_specs.append(task_spec)

    if not task_specs:
        raise ValueError(
            "No RoboCasa365 tasks were selected. "
            "Check split/task_soup/task_filter/task_mode."
        )
    return task_specs


def resolve_robocasa365_episode_horizons(
    *,
    task_horizons: Iterable[int],
    max_episode_steps: int,
    episode_horizon_source: str,
) -> tuple[int, ...]:
    """Resolve per-environment horizons from the configured source."""

    registry_horizons = tuple(int(horizon) for horizon in task_horizons)
    if not registry_horizons:
        raise ValueError("RoboCasa365 requires at least one episode horizon.")
    if any(horizon <= 0 for horizon in registry_horizons):
        raise ValueError(
            "RoboCasa365 task horizons must be positive, got "
            f"{list(registry_horizons)}."
        )

    source = str(episode_horizon_source)
    if source == "task_horizon":
        return registry_horizons
    if source == "max_episode_steps":
        if max_episode_steps <= 0:
            raise ValueError(
                "env max_episode_steps must be positive when "
                "episode_horizon_source='max_episode_steps', got "
                f"{max_episode_steps}."
            )
        return (int(max_episode_steps),) * len(registry_horizons)
    raise ValueError(
        "RoboCasa365 episode_horizon_source must be one of "
        "{'task_horizon', 'max_episode_steps'}, got "
        f"{source!r}."
    )


def resolve_robocasa365_rollout_budget(
    *,
    episode_horizons: Iterable[int],
    num_action_chunks: int,
) -> int:
    """Resolve the shared rollout budget for the selected episode horizons."""

    horizons = tuple(int(horizon) for horizon in episode_horizons)
    if not horizons:
        raise ValueError("RoboCasa365 requires at least one episode horizon.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            f"RoboCasa365 episode horizons must be positive, got {list(horizons)}."
        )
    if num_action_chunks <= 0:
        raise ValueError(
            f"actor.model.num_action_chunks must be positive, got {num_action_chunks}."
        )

    max_horizon = max(horizons)
    return (
        (max_horizon + num_action_chunks - 1) // num_action_chunks
    ) * num_action_chunks


def validate_robocasa365_eval_horizons(
    *,
    episode_horizons: Iterable[int],
    max_steps_per_rollout_epoch: int,
) -> int:
    """Validate that one eval rollout can finish every selected episode.

    Each environment truncates at its resolved horizon. The rollout worker,
    however, advances all evaluation environments for one shared number of
    action chunks, so that shared budget must cover the largest resolved
    horizon.

    Returns:
        The maximum resolved episode horizon.
    """

    horizons = [int(horizon) for horizon in episode_horizons]
    if not horizons:
        raise ValueError("RoboCasa365 eval requires at least one episode horizon.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            f"RoboCasa365 episode horizons must be positive, got {horizons}."
        )
    if max_steps_per_rollout_epoch <= 0:
        raise ValueError(
            "env.eval.max_steps_per_rollout_epoch must be positive, got "
            f"{max_steps_per_rollout_epoch}."
        )
    max_episode_horizon = max(horizons)
    if max_steps_per_rollout_epoch < max_episode_horizon:
        raise ValueError(
            "RoboCasa365 eval rollout is shorter than at least one resolved "
            "episode horizon. The shared rollout budget must cover the largest "
            "resolved horizon so failed episodes are counted. "
            f"max_episode_horizon={max_episode_horizon}, "
            f"max_steps_per_rollout_epoch={max_steps_per_rollout_epoch}. Increase "
            "env.eval.max_steps_per_rollout_epoch to at least the maximum resolved "
            "episode horizon."
        )
    return max_episode_horizon
