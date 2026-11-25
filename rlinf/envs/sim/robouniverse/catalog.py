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

"""Task catalog loading, filtering, and deterministic sharding for RoboUniverse-10K."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

VALID_ROBOTS = frozenset({"franka", "agilex"})
VALID_SPLITS = frozenset({"heldout500", "train", "all"})


@dataclass(frozen=True)
class TaskRecord:
    """One row from ``catalog.csv`` describing a packaged RoboUniverse task."""

    case: str
    robot: str
    archive: str
    sha256: str
    size_bytes: int


class RoboUniverseCatalog:
    """Load, filter, and shard RoboUniverse-10K tasks from a local dataset root."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        catalog_path = self.data_root / "catalog.csv"
        if not catalog_path.is_file():
            raise FileNotFoundError(
                f"Expected RoboUniverse catalog.csv at {catalog_path}, "
                "but the file does not exist."
            )
        self._records_by_case: dict[str, TaskRecord] = {}
        self._records: list[TaskRecord] = []
        with catalog_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"case", "robot", "archive", "sha256"}
            fieldnames = set(reader.fieldnames or ())
            missing = sorted(required - fieldnames)
            if missing:
                raise ValueError(
                    f"Expected columns {sorted(required)} in {catalog_path}, "
                    f"missing {missing}."
                )
            for row in reader:
                case = row["case"].strip()
                if not case:
                    raise ValueError(f"Empty case name in {catalog_path}.")
                if case in self._records_by_case:
                    raise ValueError(f"Duplicate case {case!r} in {catalog_path}.")
                record = TaskRecord(
                    case=case,
                    robot=row["robot"].strip(),
                    archive=row["archive"].strip(),
                    sha256=row["sha256"].strip(),
                    size_bytes=int(row.get("bytes") or 0),
                )
                self._records_by_case[case] = record
                self._records.append(record)
        if not self._records:
            raise ValueError(f"Expected at least one task in {catalog_path}, found 0.")

    def _read_split_cases(self, split_name: str) -> list[str]:
        split_path = self.data_root / "splits" / f"{split_name}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Expected split file at {split_path}, but the file does not exist."
            )
        with split_path.open("r", encoding="utf-8", newline="") as handle:
            cases = [row["case"].strip() for row in csv.DictReader(handle)]
        if len(cases) != len(set(cases)):
            raise ValueError(f"Duplicate cases found in split {split_path}.")
        unknown = [case for case in cases if case not in self._records_by_case]
        if unknown:
            raise ValueError(
                f"Split {split_path} references unknown cases not in catalog.csv: "
                f"{unknown[:5]}."
            )
        return cases

    def select(
        self,
        *,
        split: str | None = "heldout500",
        robot: str | None = None,
        cases: Sequence[str] | None = None,
    ) -> list[TaskRecord]:
        """Return the ordered list of tasks matching the requested filters.

        Args:
            split: Dataset split (``"heldout500"``, ``"train"``, or ``"all"``).
                Ignored when ``cases`` is explicitly provided.
            robot: Optional robot family filter (``"franka"`` or ``"agilex"``).
            cases: Optional explicit list of case IDs.

        Returns:
            Non-empty list of matching :class:`TaskRecord` entries.
        """
        if robot is not None and robot not in VALID_ROBOTS:
            raise ValueError(
                f"Expected robot in {sorted(VALID_ROBOTS)} or None, found {robot!r}."
            )

        if cases is not None:
            case_list = [str(case).strip() for case in cases]
            if not case_list:
                raise ValueError("Expected a non-empty cases list, found empty.")
            unknown = [c for c in case_list if c not in self._records_by_case]
            if unknown:
                raise ValueError(
                    f"Unknown RoboUniverse cases not in catalog.csv: {unknown[:5]}."
                )
            selected = [self._records_by_case[c] for c in case_list]
        else:
            split_name = split or "all"
            if split_name not in VALID_SPLITS:
                raise ValueError(
                    f"Expected split in {sorted(VALID_SPLITS)}, found {split_name!r}."
                )
            if split_name == "all":
                selected = list(self._records)
            elif split_name == "heldout500":
                selected = [
                    self._records_by_case[c]
                    for c in self._read_split_cases("heldout500")
                ]
            else:
                heldout = set(self._read_split_cases("heldout500"))
                selected = [r for r in self._records if r.case not in heldout]

        if robot is not None:
            selected = [r for r in selected if r.robot == robot]

        if not selected:
            raise ValueError(
                "No RoboUniverse tasks matched selection "
                f"(split={split!r}, robot={robot!r}, cases={cases!r})."
            )
        return selected

    @staticmethod
    def shard(
        records: Sequence[TaskRecord], rank: int, world_size: int
    ) -> list[TaskRecord]:
        """Deterministically partition ``records`` across ``world_size`` workers.

        When fewer records exist than ``world_size``, workers wrap modulo the
        available records so single-task smoke runs remain runnable.
        """
        if world_size <= 0:
            raise ValueError(f"Expected world_size > 0, found {world_size}.")
        if not 0 <= rank < world_size:
            raise ValueError(f"Expected rank in [0, {world_size}), found {rank}.")
        if not records:
            raise ValueError("Cannot shard an empty RoboUniverse task list.")
        if len(records) >= world_size:
            return list(records[rank::world_size])
        return [records[rank % len(records)]]
