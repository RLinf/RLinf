#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PlacementSpec:
    env_tag: str
    rollout_tag: str
    actor_tag: str
    component_placement: dict[str, str]


PLACEMENTS: dict[str, PlacementSpec] = {
    # Matches existing files in `placement_test/`
    # openvla-envnum*-env07-rollout07-actor07-pipelinestage1.yaml
    "collocated": PlacementSpec(
        env_tag="env07",
        rollout_tag="rollout07",
        actor_tag="actor07",
        component_placement={"actor": "0-7", "env": "0-7", "rollout": "0-7"},
    ),
    # openvla-envnum*-env03-rollout47-actor07-pipelinestage2.yaml
    "hybrid": PlacementSpec(
        env_tag="env03",
        rollout_tag="rollout47",
        actor_tag="actor07",
        component_placement={"actor": "0-7", "env": "0-3", "rollout": "4-7"},
    ),
    # openvla-envnum*-env01-rollout27-actor07-pipelinestage2.yaml
    "hybrid_env01_rollout27": PlacementSpec(
        env_tag="env01",
        rollout_tag="rollout27",
        actor_tag="actor07",
        component_placement={"actor": "0-7", "env": "0-1", "rollout": "2-7"},
    ),
}


def _set_nested(cfg: dict[str, Any], path: str, value: Any) -> None:
    cur: dict[str, Any] = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _ensure_mapping(cfg: dict[str, Any], path: str) -> dict[str, Any]:
    cur: dict[str, Any] = cfg
    parts = path.split(".")
    for p in parts:
        nxt = cur.get(p)
        if nxt is None or not isinstance(nxt, dict):
            cur[p] = {}
        cur = cur[p]
    return cur


def _render_name(prefix: str, envnum: int, spec: PlacementSpec, pipeline_stage_num: int) -> str:
    return (
        f"{prefix}-envnum{envnum}-{spec.env_tag}-{spec.rollout_tag}-{spec.actor_tag}-"
        f"pipelinestage{pipeline_stage_num}"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Base YAML must be a mapping, got {type(data)} from {path}")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            width=120,
            allow_unicode=True,
        )


def _infer_defaults_name(base_cfg: dict[str, Any], *, kind: str) -> str | None:
    """Infer env/model *name* from Hydra `defaults` entries.

    We extract the full `<name>` portion between `{kind}/` and `@...`.

    Example:
      - "model/openvla_oft@actor.model" -> "openvla_oft" (do NOT truncate on underscore)
      - "env/maniskill_put_on_plate_in_scene_25_main@env.train" -> "maniskill_put_on_plate_in_scene_25_main"
    """
    defaults = base_cfg.get("defaults")
    if not isinstance(defaults, list):
        return None

    prefix = f"{kind}/"
    for item in defaults:
        if isinstance(item, str):
            # e.g. "env/maniskill_put_on_plate_in_scene_25_main@env.train"
            if item.startswith(prefix):
                after = item[len(prefix) :]
                name = after.split("@", 1)[0]
                if not name:
                    continue
                return name
        elif isinstance(item, dict):
            # e.g. {"env/foo@env.train": None} in some Hydra styles
            for k in item.keys():
                if isinstance(k, str) and k.startswith(prefix):
                    after = k[len(prefix) :]
                    name = after.split("@", 1)[0]
                    if not name:
                        continue
                    return name
    return None


def _infer_prefix(base_cfg: dict[str, Any], fallback: str) -> str:
    env_name = _infer_defaults_name(base_cfg, kind="env")
    model_name = _infer_defaults_name(base_cfg, kind="model")

    # For env, keep a short "family" (first token) since env configs often include task names.
    env_family = env_name.split("_", 1)[0] if env_name else None
    # For model, keep the full name (e.g., openvla_oft is distinct from openvla).
    model_tag = model_name if model_name else None
    if env_family and model_tag:
        return f"{env_family}_{model_tag}"
    if model_tag:
        return model_tag
    return fallback


def _generate_one(
    base_cfg: dict[str, Any],
    *,
    envnum: int,
    pipeline_stage_num: int,
    placement_spec: PlacementSpec,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    # Only mutate the requested key fields.
    _set_nested(cfg, "env.train.total_num_envs", int(envnum))
    _set_nested(cfg, "rollout.pipeline_stage_num", int(pipeline_stage_num))

    cluster_cp = _ensure_mapping(cfg, "cluster.component_placement")
    cluster_cp.clear()
    cluster_cp.update(placement_spec.component_placement)

    return cfg


def _iter_envnums(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError(f"{step=} must be > 0")
    if end < start:
        raise ValueError(f"{end=} must be >= {start=}")
    return list(range(start, end + 1, step))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate placement_test YAMLs from a base YAML. "
            "Only modifies env.train.total_num_envs, rollout.pipeline_stage_num, cluster.component_placement."
        )
    )
    parser.add_argument(
        "--base-yaml",
        required=True,
        type=Path,
        help="Base YAML path to copy from.",
    )
    parser.add_argument(
        "--outdir",
        default=Path("placement_test"),
        type=Path,
        help="Output directory for generated YAMLs (default: placement_test).",
    )
    parser.add_argument(
        "--prefix",
        default="auto",
        help=(
            "Filename prefix before envnum... Use 'auto' to infer from base YAML defaults "
            "(e.g., maniskill_openvla)."
        ),
    )
    parser.add_argument("--envnum-start", type=int, required=True)
    parser.add_argument("--envnum-end", type=int, required=True)
    parser.add_argument("--envnum-step", type=int, default=16)
    parser.add_argument(
        "--include-env01-rollout27-when-divisible-by-3",
        action="store_true",
        help="If set, generate extra hybrid env01/rollout27 when envnum is divisible by 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print filenames, do not write files.",
    )
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_yaml)
    outdir: Path = args.outdir
    prefix = (
        _infer_prefix(base_cfg, fallback="openvla")
        if str(args.prefix).lower() == "auto"
        else str(args.prefix)
    )

    envnums = _iter_envnums(args.envnum_start, args.envnum_end, args.envnum_step)
    if not envnums:
        return 0

    written: list[Path] = []
    for envnum in envnums:
        # Collocated: pipeline_stage_num=1
        spec = PLACEMENTS["collocated"]
        name = _render_name(prefix, envnum, spec, pipeline_stage_num=1)
        out_path = outdir / f"{name}.yaml"
        cfg = _generate_one(
            base_cfg,
            envnum=envnum,
            pipeline_stage_num=1,
            placement_spec=spec,
        )
        if args.dry_run:
            print(out_path.as_posix())
        else:
            _dump_yaml(out_path, cfg)
            written.append(out_path)

        # Hybrid: pipeline_stage_num=2
        spec = PLACEMENTS["hybrid"]
        name = _render_name(prefix, envnum, spec, pipeline_stage_num=2)
        out_path = outdir / f"{name}.yaml"
        cfg = _generate_one(
            base_cfg,
            envnum=envnum,
            pipeline_stage_num=2,
            placement_spec=spec,
        )
        if args.dry_run:
            print(out_path.as_posix())
        else:
            _dump_yaml(out_path, cfg)
            written.append(out_path)

        # Extra: env01 rollout27 when divisible by 3.
        if args.include_env01_rollout27_when_divisible_by_3 and envnum % 3 == 0:
            spec = PLACEMENTS["hybrid_env01_rollout27"]
            name = _render_name(prefix, envnum, spec, pipeline_stage_num=2)
            out_path = outdir / f"{name}.yaml"
            cfg = _generate_one(
                base_cfg,
                envnum=envnum,
                pipeline_stage_num=2,
                placement_spec=spec,
            )
            if args.dry_run:
                print(out_path.as_posix())
            else:
                _dump_yaml(out_path, cfg)
                written.append(out_path)

    if not args.dry_run:
        rel = [p.as_posix() for p in written]
        print(f"Wrote {len(rel)} YAMLs under {outdir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

