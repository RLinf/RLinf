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

"""Compare official RoboCasa OpenPI dumps with RLinf RoboCasa365 dumps.

The script expects .pt files produced by the opt-in debug dump helpers:

Official OpenPI side:
    DEBUG_DUMP_DIR=/tmp/openpi_dumps ...

RLinf side:
    RLINF_DEBUG_DUMP_DIR=/tmp/rlinf_dumps ...

Usage:
    python toolkits/robocasa365_check/compare_openpi_rlinf_dumps.py \
        --official-dir /tmp/openpi_dumps \
        --rlinf-dir /tmp/rlinf_dumps
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


PairSpec = tuple[str, str, list[Any], str, list[Any]]


ROBOCASA_ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)


PAIR_SPECS: list[PairSpec] = [
    (
        "policy_input_state",
        "openpi_robocasa_env_to_policy_input",
        ["payload", "policy_input", "observation/state"],
        "rlinf_openpi_obs_processor_output",
        ["payload", "policy_input", "observation/state"],
    ),
    (
        "policy_input_base_image",
        "openpi_robocasa_env_to_policy_input",
        ["payload", "policy_input", "observation/image"],
        "rlinf_openpi_obs_processor_output",
        ["payload", "policy_input", "observation/image"],
    ),
    (
        "policy_input_wrist_image",
        "openpi_robocasa_env_to_policy_input",
        ["payload", "policy_input", "observation/wrist_image"],
        "rlinf_openpi_obs_processor_output",
        ["payload", "policy_input", "observation/wrist_image"],
    ),
    (
        "model_input_state",
        "openpi_policy_model_input",
        ["payload", "inputs", "state"],
        "rlinf_openpi_sample_actions_input",
        ["payload", "observation", "state"],
    ),
    (
        "model_input_base_0_rgb",
        "openpi_policy_model_input",
        ["payload", "inputs", "image", "base_0_rgb"],
        "rlinf_openpi_sample_actions_input",
        ["payload", "observation", "image", "base_0_rgb"],
    ),
    (
        "model_input_left_wrist_0_rgb",
        "openpi_policy_model_input",
        ["payload", "inputs", "image", "left_wrist_0_rgb"],
        "rlinf_openpi_sample_actions_input",
        ["payload", "observation", "image", "left_wrist_0_rgb"],
    ),
    (
        "model_input_right_wrist_0_rgb",
        "openpi_policy_model_input",
        ["payload", "inputs", "image", "right_wrist_0_rgb"],
        "rlinf_openpi_sample_actions_input",
        ["payload", "observation", "image", "right_wrist_0_rgb"],
    ),
    (
        "model_input_image_mask",
        "openpi_policy_model_input",
        ["payload", "inputs", "image_mask"],
        "rlinf_openpi_sample_actions_input",
        ["payload", "observation", "image_mask"],
    ),
    (
        "raw_model_actions",
        "openpi_policy_model_raw_output",
        ["payload", "actions"],
        "rlinf_openpi_sample_actions_output",
        ["payload", "outputs", "actions"],
    ),
    (
        "final_policy_actions",
        "openpi_policy_final_output",
        ["payload", "actions"],
        "rlinf_openpi_output_transform_output",
        ["payload", "outputs", "actions"],
    ),
    (
        "env_step_action",
        "openpi_robocasa_env_step_input",
        ["payload", "env_action"],
        "rlinf_robocasa365_env_step_input",
        ["payload", "actions"],
    ),
]


def _load_records(root: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.pt")):
        try:
            record = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"WARNING: failed to load {path}: {exc}")
            continue
        tag = record.get("tag") if isinstance(record, dict) else None
        if not tag:
            print(f"WARNING: skipping {path}, missing tag")
            continue
        record["_path"] = str(path)
        records.setdefault(str(tag), []).append(record)
    for tag_records in records.values():
        tag_records.sort(key=lambda item: (item.get("count", 0), item.get("time", 0)))
    return records


def _get_path(value: Any, path: list[Any]) -> Any:
    cur = value
    for key in path:
        if isinstance(cur, dict):
            cur = cur[key]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(key)]
        else:
            cur = getattr(cur, str(key))
    return cur


def _to_numpy(value: Any) -> np.ndarray | None:
    if _is_robocasa_action_dict(value):
        return np.concatenate(
            [np.asarray(value[key]).reshape(-1) for key in ROBOCASA_ACTION_KEYS],
            axis=0,
        )
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value.item())
    if isinstance(value, (int, float, bool)):
        return np.asarray(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return np.asarray(value)
        except Exception:
            return None
    return None


def _is_robocasa_action_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and all(key in value for key in ROBOCASA_ACTION_KEYS)


def _looks_like_unbatched_image(arr: np.ndarray) -> bool:
    return (
        arr.ndim == 3
        and arr.shape[-1] in (1, 3, 4)
        and arr.shape[0] > 16
        and arr.shape[1] > 16
    )


def _select_rlinf_env(arr: np.ndarray, name: str, env_index: int) -> tuple[np.ndarray, bool]:
    """Select one RLinf env from batched dumps.

    Official OpenPI eval is single-env, while RLinf eval normally dumps batched
    tensors. This keeps action chunks shaped [T, D] intact when they are already
    unbatched.
    """

    if env_index < 0 or arr.ndim == 0 or _looks_like_unbatched_image(arr):
        return arr, False
    should_select = False
    if arr.ndim >= 3:
        should_select = True
    elif arr.ndim == 2 and (
        name.endswith("_state")
        or name == "env_step_action"
        or name.startswith("model_input_")
        or name.startswith("policy_input_")
    ):
        should_select = True
    if not should_select or arr.shape[0] <= env_index:
        return arr, False
    return arr[env_index], True


def _canonicalize_value(
    name: str, side: str, value: Any, *, env_index: int
) -> tuple[Any, dict[str, Any]]:
    notes: dict[str, Any] = {}
    arr = _to_numpy(value)
    if arr is None:
        return value, notes
    if side == "rlinf":
        arr, selected = _select_rlinf_env(arr, name, env_index)
        if selected:
            notes["selected_rlinf_env_index"] = env_index
    return arr, notes


def _summarize(value: Any) -> dict[str, Any]:
    arr = _to_numpy(value)
    if arr is None:
        if isinstance(value, dict):
            return {
                "type": "dict",
                "keys": sorted(str(key) for key in value.keys()),
            }
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "len": len(value)}
        return {"type": type(value).__name__, "repr": repr(value)[:240]}

    summary: dict[str, Any] = {
        "type": "array",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        arr_float = arr.astype(np.float64, copy=False)
        summary.update(
            {
                "min": float(np.nanmin(arr_float)),
                "max": float(np.nanmax(arr_float)),
                "mean": float(np.nanmean(arr_float)),
                "std": float(np.nanstd(arr_float)),
            }
        )
    return summary


def _squeeze_single_batch_dim(
    left_arr: np.ndarray, right_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    squeezed = False
    if (
        left_arr.ndim + 1 == right_arr.ndim
        and right_arr.shape[0] == 1
        and left_arr.shape == right_arr.shape[1:]
    ):
        right_arr = right_arr[0]
        squeezed = True
    elif (
        right_arr.ndim + 1 == left_arr.ndim
        and left_arr.shape[0] == 1
        and right_arr.shape == left_arr.shape[1:]
    ):
        left_arr = left_arr[0]
        squeezed = True
    return left_arr, right_arr, squeezed


def _align_common_prefix(
    left_arr: np.ndarray, right_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if (
        left_arr.shape != right_arr.shape
        and left_arr.ndim == right_arr.ndim
        and left_arr.ndim >= 1
        and left_arr.shape[1:] == right_arr.shape[1:]
    ):
        prefix = min(left_arr.shape[0], right_arr.shape[0])
        return (
            left_arr[:prefix],
            right_arr[:prefix],
            {
                "compared_common_prefix": True,
                "common_prefix_len": int(prefix),
                "original_left_shape": list(left_arr.shape),
                "original_right_shape": list(right_arr.shape),
            },
        )
    return left_arr, right_arr, {}


def _is_image_pair(name: str) -> bool:
    return name.endswith("_image") or "_rgb" in name


def _align_image_layout(
    left_arr: np.ndarray, right_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    notes: dict[str, Any] = {}
    if (
        left_arr.ndim == 3
        and right_arr.ndim == 3
        and left_arr.shape[-1] in (1, 3, 4)
        and right_arr.shape[0] == left_arr.shape[-1]
        and right_arr.shape[1:] == left_arr.shape[:2]
    ):
        right_arr = np.moveaxis(right_arr, 0, -1)
        notes["transposed_right_chw_to_hwc"] = True
    elif (
        left_arr.ndim == 3
        and right_arr.ndim == 3
        and right_arr.shape[-1] in (1, 3, 4)
        and left_arr.shape[0] == right_arr.shape[-1]
        and left_arr.shape[1:] == right_arr.shape[:2]
    ):
        left_arr = np.moveaxis(left_arr, 0, -1)
        notes["transposed_left_chw_to_hwc"] = True
    return left_arr, right_arr, notes


def _compare_dicts(left: Any, right: Any) -> dict[str, Any] | None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return None
    left_keys = {str(key) for key in left.keys()}
    right_keys = {str(key) for key in right.keys()}
    result: dict[str, Any] = {
        "dict_comparison": "ok" if left_keys == right_keys else "key_mismatch",
        "left_only_keys": sorted(left_keys - right_keys),
        "right_only_keys": sorted(right_keys - left_keys),
    }
    common_keys = sorted(left_keys & right_keys)
    value_matches: dict[str, bool] = {}
    for key in common_keys:
        left_value = left[key] if key in left else left[next(k for k in left if str(k) == key)]
        right_value = right[key] if key in right else right[next(k for k in right if str(k) == key)]
        left_arr = _to_numpy(left_value)
        right_arr = _to_numpy(right_value)
        if left_arr is None or right_arr is None:
            value_matches[key] = left_value == right_value
        else:
            value_matches[key] = bool(np.array_equal(left_arr, right_arr))
    result["value_matches"] = value_matches
    return result


def _compare_values(name: str, left: Any, right: Any) -> dict[str, Any]:
    dict_result = _compare_dicts(left, right)
    if dict_result is not None:
        return dict_result

    left_arr = _to_numpy(left)
    right_arr = _to_numpy(right)
    if left_arr is None or right_arr is None:
        return {"numeric_comparison": "skipped_non_array"}

    left_arr, right_arr, squeezed = _squeeze_single_batch_dim(left_arr, right_arr)
    layout_notes: dict[str, Any] = {}
    if _is_image_pair(name):
        left_arr, right_arr, layout_notes = _align_image_layout(left_arr, right_arr)
    left_arr, right_arr, align_notes = _align_common_prefix(left_arr, right_arr)
    if left_arr.shape != right_arr.shape:
        return {
            "numeric_comparison": "skipped_shape_mismatch",
            "left_shape": list(left_arr.shape),
            "right_shape": list(right_arr.shape),
        }
    if not (
        np.issubdtype(left_arr.dtype, np.number)
        and np.issubdtype(right_arr.dtype, np.number)
    ):
        return {"numeric_comparison": "skipped_non_numeric"}

    diff = left_arr.astype(np.float64) - right_arr.astype(np.float64)
    result = {
        "numeric_comparison": "ok",
        "squeezed_single_batch_dim": squeezed,
        **layout_notes,
        **align_notes,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "allclose_1e-5": bool(np.allclose(left_arr, right_arr, atol=1e-5, rtol=1e-5)),
    }
    if _is_image_pair(name):
        result["left_range"] = [
            float(np.nanmin(left_arr.astype(np.float64))) if left_arr.size else 0.0,
            float(np.nanmax(left_arr.astype(np.float64))) if left_arr.size else 0.0,
        ]
        result["right_range"] = [
            float(np.nanmin(right_arr.astype(np.float64))) if right_arr.size else 0.0,
            float(np.nanmax(right_arr.astype(np.float64))) if right_arr.size else 0.0,
        ]
    return result


def _select_record(
    records: dict[str, list[dict[str, Any]]], tag: str, index: int
) -> dict[str, Any] | None:
    tag_records = records.get(tag, [])
    if not tag_records:
        return None
    if index >= len(tag_records):
        return None
    return tag_records[index]


def _run_comparison(
    official_records: dict[str, list[dict[str, Any]]],
    rlinf_records: dict[str, list[dict[str, Any]]],
    *,
    index: int,
    env_index: int,
) -> list[dict[str, Any]]:
    results = []
    for name, left_tag, left_path, right_tag, right_path in PAIR_SPECS:
        left_record = _select_record(official_records, left_tag, index)
        right_record = _select_record(rlinf_records, right_tag, index)
        result: dict[str, Any] = {
            "name": name,
            "official_tag": left_tag,
            "rlinf_tag": right_tag,
            "index": index,
        }
        if left_record is None or right_record is None:
            result["status"] = "missing_record"
            result["official_found"] = left_record is not None
            result["rlinf_found"] = right_record is not None
            results.append(result)
            continue

        try:
            left_value = _get_path(left_record, left_path)
            right_value = _get_path(right_record, right_path)
            left_value, left_notes = _canonicalize_value(
                name, "official", left_value, env_index=env_index
            )
            right_value, right_notes = _canonicalize_value(
                name, "rlinf", right_value, env_index=env_index
            )
        except Exception as exc:
            result["status"] = "path_error"
            result["error"] = repr(exc)
            results.append(result)
            continue

        result["status"] = "ok"
        result["official_path"] = left_record["_path"]
        result["rlinf_path"] = right_record["_path"]
        result["canonicalization"] = {
            "official": left_notes,
            "rlinf": right_notes,
        }
        result["official_summary"] = _summarize(left_value)
        result["rlinf_summary"] = _summarize(right_value)
        result["comparison"] = _compare_values(name, left_value, right_value)
        results.append(result)
    return results


def _print_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"\n== {result['name']} ==")
        print(f"status: {result['status']}")
        if result["status"] != "ok":
            print(json.dumps(result, indent=2, sort_keys=True))
            continue
        if result["canonicalization"]["official"] or result["canonicalization"]["rlinf"]:
            print(f"canonicalization: {result['canonicalization']}")
        print(f"official: {result['official_summary']}")
        print(f"rlinf:    {result['rlinf_summary']}")
        print(f"diff:     {result['comparison']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare official RoboCasa OpenPI dumps with RLinf dumps."
    )
    parser.add_argument("--official-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rlinf-dir", type=pathlib.Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="RLinf env index to compare against the official single-env dump.",
    )
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    official_records = _load_records(args.official_dir)
    rlinf_records = _load_records(args.rlinf_dir)

    print(f"official tags: {sorted(official_records)}")
    print(f"rlinf tags:    {sorted(rlinf_records)}")

    results = _run_comparison(
        official_records,
        rlinf_records,
        index=args.index,
        env_index=args.env_index,
    )
    _print_results(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
