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
from typing import Any

import numpy as np
import torch


PairSpec = tuple[str, str, list[Any], str, list[Any]]


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
        "rlinf_openpi_input_transform_output",
        ["payload", "model_input", "state"],
    ),
    (
        "model_input_base_0_rgb",
        "openpi_policy_model_input",
        ["payload", "inputs", "image", "base_0_rgb"],
        "rlinf_openpi_input_transform_output",
        ["payload", "model_input", "image", "base_0_rgb"],
    ),
    (
        "model_input_left_wrist_0_rgb",
        "openpi_policy_model_input",
        ["payload", "inputs", "image", "left_wrist_0_rgb"],
        "rlinf_openpi_input_transform_output",
        ["payload", "model_input", "image", "left_wrist_0_rgb"],
    ),
    (
        "model_input_right_wrist_0_rgb",
        "openpi_policy_model_input",
        ["payload", "inputs", "image", "right_wrist_0_rgb"],
        "rlinf_openpi_input_transform_output",
        ["payload", "model_input", "image", "right_wrist_0_rgb"],
    ),
    (
        "model_input_image_mask",
        "openpi_policy_model_input",
        ["payload", "inputs", "image_mask"],
        "rlinf_openpi_input_transform_output",
        ["payload", "model_input", "image_mask"],
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
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value.item())
    if isinstance(value, (int, float, bool)):
        return np.asarray(value)
    return None


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


def _compare_arrays(left: Any, right: Any) -> dict[str, Any]:
    left_arr = _to_numpy(left)
    right_arr = _to_numpy(right)
    if left_arr is None or right_arr is None:
        return {"numeric_comparison": "skipped_non_array"}
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
    return {
        "numeric_comparison": "ok",
        "squeezed_single_batch_dim": squeezed,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "allclose_1e-5": bool(np.allclose(left_arr, right_arr, atol=1e-5, rtol=1e-5)),
    }


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
        except Exception as exc:
            result["status"] = "path_error"
            result["error"] = repr(exc)
            results.append(result)
            continue

        result["status"] = "ok"
        result["official_path"] = left_record["_path"]
        result["rlinf_path"] = right_record["_path"]
        result["official_summary"] = _summarize(left_value)
        result["rlinf_summary"] = _summarize(right_value)
        result["comparison"] = _compare_arrays(left_value, right_value)
        results.append(result)
    return results


def _print_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"\n== {result['name']} ==")
        print(f"status: {result['status']}")
        if result["status"] != "ok":
            print(json.dumps(result, indent=2, sort_keys=True))
            continue
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
