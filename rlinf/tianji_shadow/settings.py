"""Operator-adjustable progress thresholds for the r5b Monitor variants."""
import json
import os
from pathlib import Path
import tempfile

from .parallel import PROFILES, VERSION as BASE_VERSION

FT_VERSION = "Sub-16ft_H-16v1"
VERSIONS = (BASE_VERSION, FT_VERSION)

CONFIG = Path.home() / "tianji_monitor_assets/shadow1116_operator_thresholds.json"


def overrides(values):
    if not isinstance(values, dict) or values.keys() != PROFILES.keys():
        raise ValueError("请填写单点和双点的 S1–S4 阈值")
    configured = {}
    for name, published in PROFILES.items():
        row = values[name]
        if (not isinstance(row, list) or len(row) != 4 or
                any(type(x) not in (int, float) or not 0 <= x <= 4 for x in row)):
            raise ValueError("每组需要四个 0–4 内的有效数字")
        if any(a >= b for a, b in zip(row, row[1:])):
            raise ValueError("每组阈值须按 S1 至 S4 递增")
        configured[name] = dict(thresholds=list(map(float, row)), confirm_points=published["confirm_points"])
    return configured


def values(profiles):
    return {name: list(rule["thresholds"]) for name, rule in profiles.items()}


def defaults(version=BASE_VERSION):
    if version == BASE_VERSION:
        return PROFILES
    if version != FT_VERSION:
        raise ValueError("未知 Monitor 版本")
    return overrides({"single": [0.97, 1.975, 2.97, 3.905], "two": [0.955, 1.96, 2.97, 3.815]})


def _config(version):
    return CONFIG if version == BASE_VERSION else CONFIG.with_name("shadow_sub16ft_h16v1_operator_thresholds.json")


def read(version=BASE_VERSION):
    try:
        return overrides(json.loads(_config(version).read_text())), None
    except FileNotFoundError:
        return defaults(version), None
    except (OSError, ValueError) as exc:
        return defaults(version), f"保存的阈值无法读取，已载入初始阈值：{exc}"


def write(profiles, version=BASE_VERSION):
    config = _config(version)
    with tempfile.NamedTemporaryFile("w", dir=config.parent, prefix=".monitor-thresholds-", delete=False) as file:
        path = Path(file.name)
        try:
            json.dump(values(profiles), file, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
            os.replace(path, config)
        finally:
            path.unlink(missing_ok=True)
