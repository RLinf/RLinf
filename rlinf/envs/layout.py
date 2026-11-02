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

"""Assignment of envs to env worker ranks.

An env mode config (``env.train`` or ``env.eval``) is either one env config or
lists several named env configs under ``envs``. Every env worker rank hosts
exactly one of them.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from omegaconf import DictConfig

from rlinf.envs.policy_obs import PolicyObsSpec


@dataclass(frozen=True)
class EnvAssignment:
    """The env that one env worker rank hosts in one mode.

    Attributes:
        name: Key of the env under ``envs``, or ``None`` when the mode is one env.
        cfg: Config of the env.
        rank: Index of this worker among the workers hosting the env.
        world_size: Number of workers hosting the env.
    """

    name: str | None
    cfg: DictConfig
    rank: int
    world_size: int

    def scope_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Return ``metrics`` plus a ``<name>/`` copy of each, so every env reports its own."""
        if self.name is None:
            return metrics
        scoped = {f"{self.name}/{key}": value for key, value in metrics.items()}
        return {**metrics, **scoped}


class EnvLayout:
    """Splits the env worker ranks of one mode among the envs it lists.

    Each listed env gets a contiguous block of ranks, in listing order, sized by
    its share of the mode's envs. Every rank hosts the same number of envs
    because the env-rollout channel splits batches evenly across ranks.
    """

    ENVS_KEY: ClassVar[str] = "envs"
    # One rollout loop steps every env of a mode and the actor masks dones with
    # one rule, so the envs must agree on these.
    SHARED_FIELDS: ClassVar[tuple[str, ...]] = (
        "max_steps_per_rollout_epoch",
        "auto_reset",
        "ignore_terminations",
    )
    # What a multi-env mode config may hold beside its envs. Anything else
    # belongs to an env and would be ignored at the mode level.
    MODE_FIELDS: ClassVar[tuple[str, ...]] = (
        ENVS_KEY,
        "rollout_epoch",
        "total_num_envs",
        "max_episode_steps",
        PolicyObsSpec.CFG_KEY,
        *SHARED_FIELDS,
    )

    def __init__(self, mode_cfg: DictConfig, world_size: int, path: str):
        """Assign the env worker ranks of a mode to its envs.

        Args:
            mode_cfg: ``env.train`` or ``env.eval``.
            world_size: Number of env worker ranks.
            path: Config path of ``mode_cfg``, used in error messages.

        Raises:
            ValueError: If a multi-env mode cannot be split across the ranks.
        """
        self._path = path
        self._world_size = world_size
        self._mode_fields: dict[str, Any] = {}
        if not self.is_multi(mode_cfg):
            self._blocks = [(None, mode_cfg, range(world_size))]
            return

        envs = self._check_envs(mode_cfg)
        total_num_envs = sum(env_cfg.total_num_envs for env_cfg in envs.values())
        if total_num_envs % world_size != 0:
            raise ValueError(
                f"{path}.envs hold {total_num_envs} envs in total, which must be "
                f"divisible by the {world_size} env workers."
            )
        envs_per_rank = total_num_envs // world_size
        self._blocks = []
        first_rank = 0
        for name, env_cfg in envs.items():
            if env_cfg.total_num_envs % envs_per_rank != 0:
                raise ValueError(
                    f"{path}.envs.{name}.total_num_envs must be a multiple of "
                    f"{envs_per_rank}, the number of envs each of the {world_size} "
                    f"env workers hosts, got {env_cfg.total_num_envs}. Each worker "
                    "hosts one env, so give every env a whole number of workers "
                    "(several workers can share a GPU, e.g. 'env: 0:0-1')."
                )
            num_ranks = env_cfg.total_num_envs // envs_per_rank
            self._blocks.append(
                (name, env_cfg, range(first_rank, first_rank + num_ranks))
            )
            first_rank += num_ranks

        self._mode_fields["total_num_envs"] = total_num_envs
        for field in self.SHARED_FIELDS:
            self._mode_fields[field] = self._agreed_value(envs, field, required=True)
        max_episode_steps = self._agreed_value(
            envs, "max_episode_steps", required=False
        )
        if max_episode_steps is not None:
            self._mode_fields["max_episode_steps"] = max_episode_steps
        for field, value in self._mode_fields.items():
            if mode_cfg.get(field) is not None and mode_cfg[field] != value:
                raise ValueError(
                    f"{path}.{field} is {mode_cfg[field]!r}, but the envs under "
                    f"{path}.envs give {value!r}. Remove {path}.{field} or make "
                    "the two agree."
                )

    @classmethod
    def is_multi(cls, mode_cfg: DictConfig) -> bool:
        """Return whether ``mode_cfg`` lists its envs under ``envs``."""
        return mode_cfg.get(cls.ENVS_KEY) is not None

    @classmethod
    def env_cfgs(cls, mode_cfg: DictConfig | None, path: str) -> dict[str, DictConfig]:
        """Return the env configs of a mode keyed by their config path.

        Args:
            mode_cfg: ``env.train`` or ``env.eval``; ``None`` yields no envs.
            path: Config path of ``mode_cfg``.

        Returns:
            ``{path: mode_cfg}`` for a single-env mode, otherwise one
            ``{path}.envs.<name>`` entry per listed env.
        """
        if mode_cfg is None:
            return {}
        if not cls.is_multi(mode_cfg):
            return {path: mode_cfg}
        return {
            f"{path}.{cls.ENVS_KEY}.{name}": env_cfg
            for name, env_cfg in mode_cfg[cls.ENVS_KEY].items()
        }

    def mode_fields(self) -> dict[str, Any]:
        """Return the fields a multi-env mode config takes from its envs.

        Rollout, actor, and runner read these from the mode config. The result
        is empty for a single-env mode, whose config holds them itself.
        """
        return dict(self._mode_fields)

    def assign(self, rank: int) -> EnvAssignment:
        """Return the env that env worker ``rank`` hosts."""
        for name, env_cfg, ranks in self._blocks:
            if rank in ranks:
                return EnvAssignment(
                    name=name,
                    cfg=env_cfg,
                    rank=rank - ranks.start,
                    world_size=len(ranks),
                )
        raise ValueError(
            f"Env worker rank {rank} is outside the {self._world_size} ranks "
            f"{self._path} is laid out on."
        )

    def _check_envs(self, mode_cfg: DictConfig) -> dict[str, DictConfig]:
        path = self._path
        stray = [key for key in mode_cfg.keys() if key not in self.MODE_FIELDS]
        if stray:
            raise ValueError(
                f"{path} lists its envs under {path}.envs, so {path}.{stray[0]} "
                f"would apply to none of them; set it on each env under "
                f"{path}.envs instead. Fields {path} may hold beside envs: "
                f"{list(self.MODE_FIELDS)}; got {stray}."
            )
        envs = mode_cfg[self.ENVS_KEY]
        if not isinstance(envs, DictConfig) or len(envs) == 0:
            raise ValueError(
                f"{path}.envs must map env names to env configs, got {envs!r}."
            )
        for name, env_cfg in envs.items():
            if not isinstance(env_cfg, DictConfig) or env_cfg.get("env_type") is None:
                raise ValueError(
                    f"{path}.envs.{name} must be an env config with env_type."
                )
            total_num_envs = env_cfg.get("total_num_envs")
            if not isinstance(total_num_envs, int) or total_num_envs <= 0:
                raise ValueError(
                    f"{path}.envs.{name}.total_num_envs must be a positive int, "
                    f"got {total_num_envs!r}."
                )
        return dict(envs.items())

    def _agreed_value(
        self, envs: dict[str, DictConfig], field: str, *, required: bool
    ) -> Any:
        values = {name: env_cfg.get(field) for name, env_cfg in envs.items()}
        distinct = set(values.values())
        if len(distinct) == 1 and None not in distinct:
            return distinct.pop()
        if required:
            raise ValueError(
                f"Every env under {self._path}.envs must set the same {field}, "
                f"because one rollout loop steps them all; got {values}."
            )
        return None
