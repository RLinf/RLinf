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

"""Policy-interface record stored next to every SuperMimic checkpoint.

Several settings change what the networks read or predict without changing
any tensor shape, so a mismatched checkpoint would load silently and act on
the wrong inputs. The actor writes these settings beside its checkpoint;
resuming and evaluation compare them with the config and refuse a mismatch.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from pathlib import Path

from omegaconf import DictConfig

INTERFACE_FILE = "supermimic_interface.json"


@dataclasses.dataclass(frozen=True)
class PolicyInterface:
    """Interface settings of one SuperMimic checkpoint or config.

    The first seven fields come from ``actor.model`` / ``rollout.model``; the
    last three come from ``algorithm`` and shape what the trained networks
    predict.
    """

    goal_encoding: str
    goal_anchor: bool
    obs_dim: int
    velocity_tracking: bool
    critic_features: str
    input_norm: str
    wm_state_loss: str
    reward_anchor: str
    residual_frame: str
    termination_body: str

    def __post_init__(self) -> None:
        for name, kind in typing.get_type_hints(type(self)).items():
            value = getattr(self, name)
            if type(value) is not kind:
                raise TypeError(
                    f"SuperMimic interface field {name} must be {kind.__name__}, "
                    f"got {value!r}"
                )

    @classmethod
    def from_config(
        cls, model_cfg: DictConfig, algorithm_cfg: DictConfig
    ) -> PolicyInterface:
        """Interface of a model config trained or run with ``algorithm_cfg``."""
        return cls(
            goal_encoding=str(model_cfg.goal_encoding),
            goal_anchor=bool(model_cfg.goal_anchor),
            obs_dim=int(model_cfg.obs_dim),
            velocity_tracking=bool(model_cfg.velocity_tracking),
            critic_features=str(model_cfg.critic_features),
            input_norm=str(model_cfg.input_norm),
            wm_state_loss=str(model_cfg.wm_state_loss),
            reward_anchor=str(algorithm_cfg.reward_anchor),
            residual_frame=str(algorithm_cfg.residual_frame),
            termination_body=str(algorithm_cfg.termination_body),
        )

    def save(self, directory: str | os.PathLike) -> None:
        """Write this record into ``directory``."""
        path = Path(directory) / INTERFACE_FILE
        record = dataclasses.asdict(self)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | os.PathLike, levels: int = 3) -> PolicyInterface | None:
        """Record beside ``path`` or in up to ``levels`` parent directories.

        ``path`` is an actor checkpoint directory or a file inside it, such as
        ``actor/model_state_dict/full_weights.pt``.
        """
        current = Path(path)
        if current.is_file():
            current = current.parent
        for _ in range(levels + 1):
            candidate = current / INTERFACE_FILE
            if candidate.is_file():
                record = json.loads(candidate.read_text())
                names = {field.name for field in dataclasses.fields(cls)}
                if set(record) != names:
                    raise ValueError(
                        f"{candidate} has the fields {sorted(record)}, not the "
                        f"SuperMimic interface fields {sorted(names)}"
                    )
                return cls(**record)
            current = current.parent
        return None

    def validate(self, found: PolicyInterface | None, source: str) -> None:
        """Raise when ``found``, the record of checkpoint ``source``, differs.

        A checkpoint without a record predates this format and is rejected.
        """
        if found is None:
            raise ValueError(
                f"{source} has no {INTERFACE_FILE}; it was written before the current "
                "SuperMimic checkpoint format and cannot be loaded"
            )
        for field in dataclasses.fields(self):
            value, trained = getattr(self, field.name), getattr(found, field.name)
            if trained != value:
                raise ValueError(
                    f"{source} was trained with {field.name}={trained!r}, but this "
                    f"config has {field.name}={value!r}"
                )
