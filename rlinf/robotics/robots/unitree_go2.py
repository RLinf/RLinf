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

"""Go2 robot composition using the official high-level SDK2 interface."""

from dataclasses import dataclass

from ..discovery import RobotConfig
from ..parts.mobility import MobileBase
from ..robot import Robot


class Go2Robot(Robot):
    """Go2 composed under a single high-level base part."""

    ROBOT_TYPE = "UnitreeGo2"

    @classmethod
    def build(
        cls,
        *,
        interface: str,
        node_rank: int | None = None,
        worker_rank: int = 0,
        env_idx: int = 0,
        enable_motion: bool = False,
        command_timeout: float = 0.5,
        state_timeout: float = 0.5,
        rpc_timeout: float = 1.0,
    ) -> "Go2Robot":
        """Declare a Go2 without connecting or changing its posture."""
        base_cls = MobileBase.backend("unitree_go2")
        return cls(
            base=base_cls(
                interface,
                enable_motion=enable_motion,
                command_timeout=command_timeout,
                state_timeout=state_timeout,
                rpc_timeout=rpc_timeout,
                node_rank=node_rank,
                worker_name=f"{base_cls.__name__}-{worker_rank}-{env_idx}",
            )
        )


@dataclass
class Go2Config(RobotConfig):
    """Hardware config; explicit interface and motion opt-in are required."""

    interface: str = ""
    enable_motion: bool = False
    command_timeout: float = 0.5
    state_timeout: float = 0.5
    rpc_timeout: float = 1.0


Go2Robot.register_type(Go2Config)
