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

"""What a canonical field's numbers mean.

Parts report and accept fields by name, and tasks and action channels ask for
them by name. A name on its own does not say what the numbers are: two arms
can both report ``tcp_pose`` and mean different things, and a task that scores
one would quietly misread the other. Each canonical name therefore has one
meaning here.

A part whose numbers match that meaning says nothing; a part whose numbers
differ declares its own, which turns the mismatch into a refusal when a task
is bound to it rather than a wrong reward at run time.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

#: A part's feature descriptor, as ``observation_features`` reports it.
Feature = Mapping[str, Any]

#: Key a part declares its own meaning under.
MEANING = "meaning"


@dataclass(frozen=True)
class FieldMeaning:
    """What one field's numbers are.

    Attributes:
        layout: How the numbers are arranged, such as ``"xyz+quat_xyzw"`` for
            a pose or ``"per_joint"`` for one number per joint.
        unit: The unit they are in.
        width: How many there are, or ``None`` when it follows the part, as
            one number per joint does.
        frame: The frame they are measured in, where that applies.
    """

    layout: str
    unit: str
    width: Optional[int] = None
    frame: Optional[str] = None

    def describes(self, other: "FieldMeaning") -> bool:
        """Whether a part declaring ``other`` reports what this asks for."""
        return (
            self.layout == other.layout
            and self.unit == other.unit
            and (self.width is None or other.width is None or self.width == other.width)
            and (self.frame is None or other.frame is None or self.frame == other.frame)
        )

    def __str__(self) -> str:
        where = f" in the {self.frame} frame" if self.frame else ""
        return f"{self.layout} in {self.unit}{where}"


#: What each canonical field means. A field absent from this table has no
#: agreed meaning yet, so nothing is checked for it.
CANONICAL: Mapping[str, FieldMeaning] = {
    # Arm observations.
    "tcp_pose": FieldMeaning("xyz+quat_xyzw", "m", 7, frame="base"),
    "tcp_vel": FieldMeaning("linear+angular", "m/s", 6, frame="base"),
    "tcp_force": FieldMeaning("xyz", "N", 3, frame="tool"),
    "tcp_torque": FieldMeaning("xyz", "N*m", 3, frame="tool"),
    "arm_joint_position": FieldMeaning("per_joint", "rad"),
    "arm_joint_velocity": FieldMeaning("per_joint", "rad/s"),
    # Mobile base observations.
    "pose": FieldMeaning("xy+theta", "m,rad", 3, frame="world"),
    # Commands.
    "joint_position": FieldMeaning("per_joint", "rad"),
    "velocity": FieldMeaning("linear+angular", "m/s", frame="base"),
}


def describe(name: str) -> dict[str, Any]:
    """Describe a canonical field, for a part that reports it as such."""
    meaning = CANONICAL.get(name)
    return {MEANING: meaning} if meaning is not None else {}


def declared(feature: Optional[Feature]) -> Optional[FieldMeaning]:
    """The meaning a part declared for one field, if it declared one."""
    if not isinstance(feature, Mapping):
        return None
    meaning = feature.get(MEANING)
    return meaning if isinstance(meaning, FieldMeaning) else None


def mismatch(name: str, feature: Optional[Feature]) -> Optional[str]:
    """Say how a part's field differs from the canonical meaning of its name.

    Args:
        name: The canonical field name.
        feature: What the part declares for it.

    Returns:
        A phrase naming both meanings, or ``None`` when the part reports the
        canonical one or when the name has no agreed meaning.
    """
    canonical = CANONICAL.get(name)
    if canonical is None:
        return None
    given = declared(feature)
    if given is None or canonical.describes(given):
        return None
    return f"reports {name!r} as {given}, not {canonical}"
