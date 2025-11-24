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

from dataclasses import dataclass
from typing import ClassVar, Optional, TypeVar

import yaml

HardwareType = TypeVar("HardwareType")


@dataclass
class HardwareConfig:
    """Base class for hardware configuration dataclasses."""

    node_rank: int
    """The rank of the node that has this hardware."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in hardware config must be an integer. But got {type(self.node_rank)}."
        )


@dataclass
class NodeHardwareConfig:
    """This represents hardware configs for a set of nodes."""

    type: str
    """Hardware type"""

    configs: list[HardwareConfig]
    """List of hardware configurations."""

    _hardware_config_registry: ClassVar[dict[str, "type[HardwareConfig]"]] = {}

    @classmethod
    def register_hardware_config(cls, type: str):
        """Register a hardware config into the global registry.

        Args:
            type (str): The type of the hardware. This type is not case sensitive.
        """

        def hardware_config_decorator(hardware_config):
            cls._hardware_config_registry[type] = hardware_config
            return hardware_config

        return hardware_config_decorator

    def __post_init__(self):
        """Post-initialization to convert hardware_configs dicts to their respective dataclass instances."""
        self.type = str(self.type)
        hardware_config_class = NodeHardwareConfig._hardware_config_registry.get(
            self.type
        )
        if hardware_config_class is None:
            raise ValueError(
                f"Unsupported hardware type: {self.type}. Currently supported types only include: {list(self._hardware_config_registry.keys())}"
            )

        from ..cluster import dataclass_arg_check

        # Arg check
        for config in self.configs:
            assert hasattr(config, "keys"), (
                f"Each hardware config must be a dictionary. But got {type(config)}: {config}"
            )
            dataclass_arg_check(
                hardware_config_class,
                config,
                error_suffix="in cluster node_group hardware yaml config",
            )

        # Ensure all configs are unique
        config_strs = [
            yaml.dump(dict(config), sort_keys=True) for config in self.configs
        ]
        assert len(config_strs) == len(set(config_strs)), (
            "Duplicate hardware configs found in node hardware config: \n"
            + "\n".join(
                [yaml.dump(dict(config), sort_keys=False) for config in self.configs]
            )
        )

        self.configs = [hardware_config_class(**config) for config in self.configs]


@dataclass
class HardwareInfo:
    """Dataclass representing a hardware resource information."""

    type: str
    """Type of the hardware resource (e.g., Accelerator, Robot)."""

    model: str
    """Model of the hardware resource (e.g., 4090, A100, H100, Franka)."""


@dataclass
class HardwareResource:
    """A list of HardwareInfo of the same type."""

    type: str
    """Type of the hardware resource (e.g., Accelerator, Robot)."""

    infos: list[HardwareInfo]
    """The HardwareInfo list."""

    @property
    def count(self) -> int:
        """Get the count of hardware infos."""
        return len(self.infos)


class Hardware:
    """Hardware policy for a type of hardware resource.

    This is the base class for different hardware to implement their basic type and enumeration policies.
    """

    HW_TYPE: str = None
    DEFAULT_HW_TYPE: str = None
    hw_types: set[str] = set()
    policy_registry: list[type["Hardware"]] = []

    @classmethod
    def register(cls, is_default_hw: bool = False):
        """Register a new enumeration policy.

        This is to be used as a decorator for subclasses of EnumerationPolicy.

        Args:
            is_default_hw (bool): Whether this hardware type is the default hardware type.
        """

        def hardware_policy_decorator(
            policy: type["Hardware" | HardwareType],
        ) -> type[HardwareType]:
            cls.hw_types.add(policy.HW_TYPE)
            cls.policy_registry.append(policy)
            if is_default_hw:
                assert cls.DEFAULT_HW_TYPE is None, (
                    f"Default hardware type is already set to {cls.DEFAULT_HW_TYPE}. Cannot set it again to {policy.HW_TYPE}."
                )
                cls.DEFAULT_HW_TYPE = policy.HW_TYPE
            return policy

        return hardware_policy_decorator

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the hardware resources on a node.

        Args:
            node_rank (int): The rank of the node being enumerated.
            configs (Optional[list[HardwareConfig]]): The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: A list of HardwareInfo representing the hardware resources. None if no hardware is found.
        """
        raise NotImplementedError
