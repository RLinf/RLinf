# Copyright 2025 The RLinf Authors.
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

from .base import EndEffector, EndEffectorType

__all__ = ["EndEffector", "EndEffectorType", "create_end_effector"]


def create_end_effector(
    end_effector_type: str | EndEffectorType,
    **kwargs,
) -> EndEffector:
    """Factory function to create an end-effector instance.

    Args:
        end_effector_type: The type of end-effector to create.
            One of ``"franka_gripper"``, ``"aoyi_hand"``, or ``"ruiyan_hand"``.
        **kwargs: Additional keyword arguments forwarded to the end-effector
            constructor.

    Returns:
        An ``EndEffector`` instance of the requested type.

    Raises:
        ValueError: If the end-effector type is not recognized.
    """
    if isinstance(end_effector_type, str):
        end_effector_type = EndEffectorType(end_effector_type)

    if end_effector_type == EndEffectorType.FRANKA_GRIPPER:
        from .franka_gripper import FrankaGripper

        return FrankaGripper(**kwargs)
    elif end_effector_type == EndEffectorType.AOYI_HAND:
        from .aoyi_hand import AoyiHand

        return AoyiHand(**kwargs)
    elif end_effector_type == EndEffectorType.RUIYAN_HAND:
        from .ruiyan_hand import RuiyanHand

        return RuiyanHand(**kwargs)  # noqa: F811
    else:
        raise ValueError(
            f"Unknown end-effector type: {end_effector_type}. "
            f"Supported types: {[e.value for e in EndEffectorType]}"
        )
