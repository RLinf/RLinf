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

"""The YAML files a Hydra config composes, and the values they set.

Composition is approximated: every file the defaults list can pull in is
included, and the job's values are the union of all scalars in those files
plus command-line overrides. A union can only add values, so pruning with it
keeps every branch the real merged config could take.
"""

import posixpath
import re

import yaml
from pyscan import KEY_MARK
from snapshot import Snapshot

_yaml_cache: dict[str, object] = {}


def load_yaml(snap: Snapshot, path: str):
    key = snap.blob_id(path) or path
    if key not in _yaml_cache:
        try:
            _yaml_cache[key] = yaml.safe_load(snap.read(path) or "")
        except yaml.YAMLError:
            _yaml_cache[key] = None
    return _yaml_cache[key]


def scalars(data) -> set[str]:
    out: set[str] = set()
    stack = [data]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str):
            out.add(x.lower())
        elif x is not None:
            out.add(str(x).lower())
    return out


def config_keys(data) -> set[str]:
    """Every mapping key, marked so it cannot collide with a value."""
    out: set[str] = set()
    stack = [data]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            out.update(KEY_MARK + str(k).lower() for k in x)
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return out


def _norm_path(p: str) -> str:
    p = re.sub(r"^file://", "", p)
    p = re.sub(r"\$\{oc\.env:REPO_PATH[^}]*\}/?", "", p)
    p = p.replace("${REPO_PATH}/", "").replace("$REPO_PATH/", "")
    return posixpath.normpath(p).lstrip("/") if p else ""


def _defaults_entries(data) -> list[str]:
    """``group/name`` paths from a defaults list (packages and overrides stripped)."""
    out = []
    if not isinstance(data, dict):
        return out
    for item in data.get("defaults") or []:
        if isinstance(item, str):
            entry = item
        elif isinstance(item, dict) and len(item) == 1:
            ((group, name),) = item.items()
            group = str(group).replace("override ", "").replace("optional ", "").strip()
            names = name if isinstance(name, list) else [name]
            for n in names:
                if n is not None:
                    out.append(f"{group.split('@')[0]}/{n}")
            continue
        else:
            continue
        entry = entry.replace("override ", "").replace("optional ", "").strip()
        if entry == "_self_" or entry.startswith("hydra/"):
            continue
        out.append(entry.split("@")[0])
    return [e for e in out if not e.startswith("hydra/")]


def compose(
    snap: Snapshot,
    config_dir: str,
    config_name: str,
    overrides: list[str] = (),
) -> tuple[set[str], set[str]]:
    """Files and values of ``config_dir/config_name`` with its defaults."""
    config_dir = _norm_path(config_dir)
    primary = (
        f"{config_dir}/{config_name}.yaml" if config_dir else f"{config_name}.yaml"
    )
    roots = [config_dir]
    data = load_yaml(snap, primary)
    if isinstance(data, dict):
        hydra_cfg = data.get("hydra") or {}
        for sp in (
            (hydra_cfg.get("searchpath") or []) if isinstance(hydra_cfg, dict) else []
        ):
            if isinstance(sp, str) and not sp.startswith("pkg://"):
                roots.append(_norm_path(sp))
    files: set[str] = set()
    values: set[str] = set()
    queue = [primary]
    while queue:
        path = queue.pop()
        if path in files or not snap.exists(path):
            continue
        files.add(path)
        data = load_yaml(snap, path)
        values |= scalars(data) | config_keys(data)
        here = posixpath.dirname(path)
        for entry in _defaults_entries(data):
            entry = entry.lstrip("/")
            for base in [here, *roots]:
                cand = (
                    posixpath.normpath(f"{base}/{entry}.yaml")
                    if base
                    else f"{entry}.yaml"
                )
                if snap.exists(cand):
                    queue.append(cand)
        # Values that name another repo file (a dataset spec, a nested config).
        for v in scalars(data):
            if "/" in v and v.endswith((".yaml", ".yml", ".json")):
                cand = _norm_path(v)
                if snap.exists(cand):
                    files.add(cand)
    for ov in overrides:
        key, _, value = ov.lstrip("+~").partition("=")
        if value:
            values.add(value.strip("'\"").lower())
            values.add(KEY_MARK + key.rsplit(".", 1)[-1].lower())
            if "/" in key or key in ("model", "env"):
                # A group override such as env/train=libero_spatial.
                for base in roots:
                    cand = f"{base}/{key.split('@')[0]}/{value}.yaml"
                    if snap.exists(cand):
                        files.add(cand)
                        data = load_yaml(snap, cand)
                        values |= scalars(data) | config_keys(data)
    return files, values


def vocabulary(snap: Snapshot) -> frozenset[str]:
    """Every scalar any config in the repo sets."""
    out: set[str] = set()
    for f in snap.files:
        if f.endswith((".yaml", ".yml")) and f.startswith(
            ("examples/", "tests/", "evaluations/", "toolkits/")
        ):
            out |= scalars(load_yaml(snap, f))
    return frozenset(out)
